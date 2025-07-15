import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from data_provider.data_factory import data_provider
from utils.metrics import metric as forecast_metric
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import random
from FE import FE_model
from F_models import PatchTST
import matplotlib.pyplot as plt 
from datetime import datetime
import copy
from utils.injection import *
from .base import Base_Solver
from torch.optim import lr_scheduler 
from AAFN import AAFN, train_cross_attn
from shared_model import SharedModel
from utils.utils import *
from vus.metrics import get_metrics

import time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Solver(Base_Solver):
    DEFAULTS = {}

    def __init__(self, config,args):
        super(Solver, self).__init__()
        
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.args=args
        # fix_seed(args.random_seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.train_loader = data_provider(args, flag = "train", task="F",step=args.train_step) 
        self.pred_loader = data_provider(args, flag = "test", task="F",step=args.pred_len) 
        self.test_loader = data_provider(args, flag = "test", task="AD",step=args.win_size) 
        self.criterion = nn.MSELoss()
        self.criterion_no_reduc = nn.MSELoss(reduction='none')
        
        self.cos_sim = nn.CosineSimilarity()
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.build_models(args)
        
    def build_models(self, args):
        self.model = SharedModel(args)
        if self.args.noise_injection or self.args.cross_attn:
            self.fe = FE_model(input_ch = self.input_c,d_model=self.d_model)
            self.fe_optimizer = torch.optim.Adam(self.fe.parameters(), lr=0.0001)
            self.fe_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.fe_optimizer, T_max=50, eta_min=0)
            self.fe.cuda()
            # self.train_noise()
        else:
            self.fe=None
            
        if self.args.cross_attn:
            self.AAFN=AAFN(self.args)  
            self.calibrator = nn.Linear((self.args.seq_len + self.args.pred_len)*self.args.input_c, self.args.pred_len*self.args.input_c)
        
        if torch.cuda.is_available():
            self.model.cuda()
            if self.args.cross_attn:
                self.AAFN.cuda()
                self.calibrator.cuda()

    def train_noise_and_cross(self, model,  AAFN, args, epochs=100):
        for n,p in self.model.named_parameters():
            p.requires_grad=True
        self.model.train()
        
        model.train()

        optimizer = torch.optim.Adam(list(self.model.parameters())+list(self.model.AD_model.prompt.parameters()),lr=self.args.lr)
        aafn_optimizer = torch.optim.Adam(AAFN.parameters(),lr=self.args.lr)
        pretrain_loader=self.train_loader
        aafn_criterion = nn.MSELoss()
        for epoch in range(self.joint_epochs): 
            iter_count = 0
            epoch_time = time.time()
            
            total_train_loss = []
            abnormal_instance_count=0
            running_loss = 0.0
            aafn_loss1, aafn_loss2= [], []
            losses_aafn=[]
            losses_recon=[]

            signal_adaptive_loss_list, pt_energy_loss_list, contrastive_loss_list, app_loss_list = [], [], [], []
            f_loss_list1, f_loss_list2=[],[]
            
            
            
            for i, batch in enumerate(pretrain_loader):
                if not batch: continue
                x_in, x_out, x_in_mark, x_out_mark,labels_prior, labels_pred = batch
                signal_adaptive_loss, pt_energy_loss, contrastive_loss,app_loss=0,0,0,0
           
                optimizer.zero_grad()
                aafn_optimizer.zero_grad()
                
                x_in = x_in.float().to(self.device)
                x_out = x_out[:,-self.args.pred_len:].float().to(self.device)
                x_in_mark = x_in_mark.float().to(self.device)
                x_out_mark = x_out_mark[:,-self.args.pred_len:].float().to(self.device)
                labels_pred = labels_pred[:,-self.args.pred_len:].float().to(self.device)
                
                normal_indices = torch.nonzero(torch.sum(labels_prior,dim=1)==0)[:,0]
                abnormal_indices = torch.nonzero(torch.sum(labels_prior,dim=1)!=0)[:,0]
                abnormal_instance_count += len(abnormal_indices)
                abnormal_x_gt = x_in[abnormal_indices]
                abnormal_l_gt = labels_prior[abnormal_indices]
                x_in_z, injected_prior_l = inject_amplify_learnable(x_in,model,args.amplify_type, self.AAFN.global_scale,self.AAFN.contextual_scale,self.AAFN.trend_scale,self.AAFN.shapelet_scale,self.AAFN.seasonal_scale)
                x_out_z, injected_pred_l = inject_amplify_learnable(x_out,model,args.amplify_type, self.AAFN.global_scale,self.AAFN.contextual_scale,self.AAFN.trend_scale,self.AAFN.shapelet_scale,self.AAFN.seasonal_scale)
                   

                ftr_idx = self.args.ftr_idx
                x_in_r, series_gt, prior, _, _,_ = self.model.AD_model(x_in)
                gt_an_ftr = series_gt[ftr_idx]

                feature_in, _ = self.fe(x_in_r)
                _, series_n, prior, _, _,_ = self.model.AD_model(x_in_r)
                psd_n_ftr = series_n[ftr_idx]
                
                syn_an_signal, _, _, _, _,prompt_in = self.model.AD_model(x_in_r, feature=feature_in)
                sim, noise_in = prompt_in['reduce_sim'], prompt_in['batched_prompt']
                final_signal_adaptive_loss = self.sa_loss_coeff * sim
                signal_adaptive_loss_list.append(final_signal_adaptive_loss.item())
                
                x_in_p_r, series_an, prior_an, _, _,_ = self.model.AD_model(x_in_r, noise=noise_in)
                psd_an_ftr = series_an[ftr_idx] 
                
                
                x_out_hat = self.model.forward_f_model(x_in)
                f_loss1 = self.criterion(x_out_hat, x_out)
                
                x_z_out_hat = self.model.forward_f_model(x_in_z)
                f_loss2 = self.criterion(x_z_out_hat, x_out_z)
                
                f_loss1 = self.args.forecast_loss_coeff * f_loss1 
                f_loss2 = self.args.forecast_loss_coeff * f_loss2
                
                f_loss = f_loss1 +f_loss2
                

                f_loss_list1.append(f_loss1.item())
                f_loss_list2.append(f_loss2.item())
              
                stacked_labels_pred = labels_pred.unsqueeze(-1).repeat(1,1,args.input_c)
                stacked_injected_pred_l = injected_pred_l.unsqueeze(-1).repeat(1,1,args.input_c)

            
                attn_output1 = AAFN(x_out_z, x_in_z) 
                cross_attn_loss1 = aafn_criterion(attn_output1, stacked_injected_pred_l)

                
                cross_attn_loss1 = self.args.cross_attn_loss_coeff * cross_attn_loss1 

                aafn_loss1.append(cross_attn_loss1.item())
                final_loss_aafn = cross_attn_loss1 
                losses_aafn.append(final_loss_aafn.item())

                self.fe_optimizer.zero_grad()
                cls_token,recon=model(x_in)
                abnormal_x_syn, abnormal_l_syn = inject_amplify_learnable(x_in[normal_indices],model,args.amplify_type, self.AAFN.global_scale,self.AAFN.contextual_scale,self.AAFN.trend_scale,self.AAFN.shapelet_scale,self.AAFN.seasonal_scale)
                
                abnormal_x = abnormal_x_syn.to(self.device)
                abnormal_l = abnormal_l_syn.to(self.device)
                pos_n = abnormal_l==0
                pos_an = abnormal_l==1
                contrastive_loss = -torch.mean(my_kl_loss(psd_n_ftr,psd_an_ftr))
                contrastive_loss_list.append(contrastive_loss.item())
                if self.args.contrastive_loss:
                    final_contrastive_loss = self.args.contrastive_loss_coeff * contrastive_loss
                    app_loss = final_contrastive_loss + signal_adaptive_loss
                    app_loss_list.append(app_loss.item())
                    final_app_loss = app_loss

                if self.args.forecast_loss:
                    f_loss.backward(retain_graph=True)
                else:
                    f_loss = 0
                if self.args.contrastive_loss:
                    final_app_loss.backward(retain_graph=True)
                else:
                    final_app_loss = 0
                if self.args.cross_attn:
                    final_loss_aafn.backward(retain_graph=True)
                else:
                    final_loss_aafn = 0

                optimizer.step()
                aafn_optimizer.step()
                
                total_loss = f_loss + final_loss_aafn + final_app_loss
                total_train_loss.append(total_loss.item())
            
            print(f"Epoch {epoch+1}")
            print(f"Abnormal instance: {abnormal_instance_count}")
            print(f"(f loss) : {np.mean(f_loss_list1):.4f} + {np.mean(f_loss_list2):.4f}")
            print(f"(AAFN loss) : {np.mean(aafn_loss1):.4f}, {np.mean(aafn_loss2):.4f}")
            print(f"(APP loss = Constrastive + Signaladaptive): {np.mean(app_loss_list):.4f}  = {np.mean(contrastive_loss_list):.4f} + {np.mean(signal_adaptive_loss_list):.4f}")
            print(f"(final loss) train loss: {np.average(total_train_loss):.6f}")
            adjust_learning_rate(optimizer, epoch + 1, self.lr)
    
        
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_FE_checkpoint.pth'))
        print("Feature extractor saved")  

        for n,p in AAFN.named_parameters():
            p.requires_grad=False

        print(">>>>>>> end pre-training >>>>>>>>>")
         
    
    def train_FE(self, model, epochs=100):
        print(">>> Train FE for extracting feature from the signal...")
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                if not batch: continue
                x_in, x_out, x_in_mark, x_out_mark,_,_ = batch
                input_data = x_in

                self.fe_optimizer.zero_grad()
                input = input_data.float().to(self.device)
                cls_token,recon=model(input)
                loss = self.criterion(recon, input)
                loss.backward()
                self.fe_optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(self.train_loader)
            self.fe_scheduler.step()
            if epoch%5==0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_FE_checkpoint.pth'))
        print("Feature extractor saved")
    def train_ftr_extractor(self):
        print(">>> Start training feature extractor...")
        filename=os.path.join(str(self.model_save_path), str(self.dataset) + '_FE_checkpoint.pth')
        if not os.path.exists(filename):
            self.train_FE(self.fe, 10)
        else:
            print("Use trained feature extractor")
            self.fe.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.dataset) + '_FE_checkpoint.pth')), strict=False) ##
        for p in self.fe.parameters():
                p.requires_grad=False 
        self.fe.eval()
        self.model.fe = self.fe
    
    def train(self,epoch=-1,fe=None):
        
        time_now = time.time()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        train_steps = len(self.train_loader)
        for p in self.model.parameters(): p.requires_grad=True
        
        if self.args.noise_injection:
            if self.args.pretrain_noise:
                for n,p in self.model.F_model.named_parameters():
                    if 'prompt' in n:
                        p.requires_grad=False 
                    else:
                        p.requires_grad=True
                for n,p in self.model.AD_model.named_parameters():
                    if 'prompt' in n:
                        p.requires_grad=False 
                    else:
                        p.requires_grad=True
            else: 
                for n,p in self.model.F_model.named_parameters():
                    p.requires_grad=True
                for n,p in self.model.AD_model.named_parameters():
                    p.requires_grad=True    
                
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)

        for ep in range(epoch):
            epoch_time = time.time()
            
            train_loss = []
            for i, batch in enumerate(self.train_loader):
                if not batch: continue
                x_in, x_out, x_in_mark, x_out_mark,labels_prior, labels_pred = batch
                optimizer.zero_grad()
                x_in = x_in.float().to(self.device)
                x_out = x_out[:,-self.args.pred_len:].float().to(self.device)
                x_in_mark = x_in_mark.float().to(self.device)
                x_out_mark = x_out_mark.float().to(self.device)
                labels_pred = labels_pred.float().to(self.device)
                f_output, ad_output, ad_loss, ad_loss_dict  = self.model(x_in, x_out)
  
                if self.args.cross_attn:
                    F_loss = self.criterion_no_reduc(f_output, x_out)
                    an_score = self.AAFN(x_out, x_in)
                    F_loss = F_loss * an_score
                    F_loss = torch.mean(F_loss)
                else:
                    F_loss = self.criterion(f_output, x_out)

                ad_loss = self.args.recon_loss_coeff * ad_loss 
                F_loss = self.args.af_loss_coeff * F_loss 

                total_loss = ad_loss + F_loss 
                total_loss.backward()
                optimizer.step()
            adjust_learning_rate(optimizer, epoch + 1, self.lr)  
            print("Epoch: {} cost time: {:.2f} loss: F {:3f}, AD {:3f}".format(ep + 1, time.time() - epoch_time,F_loss,ad_loss_dict['rec_loss']))
            train_loss = np.average(train_loss)
       
        # if self.args.learnable and self.args.cross_attn:
        #     print("Global Scale:", self.AAFN.global_scale.item())
        #     print("Contextual Scale:", self.AAFN.contextual_scale.item())
        #     print("Trend scale:", self.AAFN.trend_scale.item())
        #     print("Shapelet scale:", self.AAFN.shapelet_scale.item())
        #     print("Seasonal scale:", self.AAFN.seasonal_scale.item())

    def test(self, thresh=None, metric=None):
        print(">>> Start test...")
        for n,p in self.model.named_parameters(): p.requires_grad=False
        predicted_signals, predict_part_labels, (test_labels_prior, test_labels_pred, gt_prior_signals, gt_pred_signals, pred_signals) = self.predict(thresh,metric)
        self.predicted_data = predicted_signals
        self.original_test_loader  = self.test_loader
        self.test_loader.dataset.data = predicted_signals
        self.test_loader.dataset.test_label = predict_part_labels
        
        thresh, metric = self.get_threshold()
        if self.args.test_thresh:
            return_val = self.test_from_predicted(thresh,metric)
        else:
            return_val = self.test_from_predicted(thresh,metric)
        return return_val
    
    def predict(self, thresh=0.0,metric=None):
        for n,p in self.model.named_parameters(): p.requires_grad=False
        preds = []
        gt_signals, prior_signals= [], []
        test_labels_prior, test_labels_pred = [],[]
        with torch.no_grad():
            for i, data in enumerate(self.pred_loader):
                if data==None: continue
                x_in, x_out, x_in_mark, x_out_mark, labels_prior, labels_pred = data
                x_in = x_in.float().to(self.device)
                x_out = x_out.float().to(self.device)
                x_in_mark = x_in_mark.float().to(self.device)
                x_out_mark = x_out_mark.float().to(self.device)
                labels_prior = labels_prior.float().to(self.device)
                labels_pred = labels_pred.float().to(self.device)
                
                labels_all = torch.cat([labels_prior,labels_pred],dim=1)
                gt_signals.append(x_out[:,-self.args.pred_len:,:].detach().cpu().numpy())
                prior_signals.append(x_in.detach().cpu().numpy())
                dec_inp = torch.zeros_like(x_out[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([x_out[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            
                outputs = self.model.F_model(x_in)
                pred = outputs.detach().cpu().numpy() 
                pred = pred.reshape(-1, self.args.input_c)
                preds.append(pred)
                test_labels_prior.append(labels_prior.detach().cpu().numpy())
                test_labels_pred.append(labels_pred.detach().cpu().numpy())

        predicted_data = np.concatenate(preds,axis=0)
        label_data = np.concatenate(test_labels_pred).flatten()
        # print(f"gt signals: {np.sum(np.concatenate(gt_signals,axis=0))}")
        self.gt_signals = np.concatenate(gt_signals,axis=0).reshape(-1, gt_signals[0].shape[-1])
        print(f"predicted shape: {predicted_data.shape}")
        return predicted_data, label_data, (np.concatenate(test_labels_prior), np.concatenate(test_labels_pred), np.concatenate(prior_signals), np.concatenate(gt_signals), np.concatenate(preds))
        
    def test_from_predicted(self,thresh=0.0,metric=None):
        criterion = nn.MSELoss(reduce=False)
        temperature = 50
        recon_preds, attens_energy_pred, test_labels_pred, attens_energy_pred_gt = [], [], [], []
        test_loss=[]
        an_score_tensor_list = []
        an_score_gt_tensor_list = []
        an_score_combined_list = []
        an_score_combined_gt_list = []

        attens_energy_pred_list=[]
        for i, ((input_data, labels), (input_data_gt, _)) in enumerate(zip(self.test_loader,self.original_test_loader)):
            input_data = input_data.float().to(self.device)
            input_data_gt = input_data_gt.float().to(self.device)
            test_labels_pred.append(labels.detach().cpu().numpy())
           
            AD_input = input_data
            AD_outputs = self.model.AD_model(AD_input)
            AD_output, series, prior, _ , _,_ = AD_outputs
            loss = torch.mean(criterion(AD_input, AD_output), dim=-1)
            
            series_loss, prior_loss = self.calc_series_prior_loss_test(series,prior)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            
            cri_pred  = cri.detach().cpu().numpy()
            attens_energy_pred.append(cri_pred[:,-self.args.pred_len:])
            if AD_output != None: recon_preds.append(AD_output.detach().cpu().numpy())
            
            AT_input = input_data_gt
            AT_output, series, prior, _ , _,_= self.model.AD_model(AT_input)
            loss = torch.mean(self.criterion(AT_input, AT_output), dim=-1)
            series_loss, prior_loss = self.calc_series_prior_loss_test(series,prior)
            metric_gt = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri_pred_gt = metric_gt * loss
            attens_energy_pred_gt.append(cri_pred_gt[:,-self.args.pred_len:].detach().cpu().numpy())
            
            if self.args.test_thresh:
                attens_energy_pred2 = copy.deepcopy(cri_pred[:,-self.args.pred_len:])
                attens_energy_pred_list.append(attens_energy_pred2)
                key_tensor, _ = inject(input_data)

                query_tensor = input_data
                an_score = self.AAFN(query_tensor, key_tensor) 
                an_score_tensor = an_score.mean(dim=(-1))
                an_score_tensor_list.append(an_score_tensor) 
                an_score_tensor_list = np.array([x.cpu().numpy() for x in an_score_tensor_list])

                key_tensor, _ = inject(input_data_gt)
                
                query_tensor = input_data_gt
                an_score_gt = self.AAFN(query_tensor, key_tensor) 
                an_score_gt_tensor = an_score_gt.mean(dim=(-1))
                an_score_gt_tensor_list.append(an_score_gt_tensor) 
                an_score_gt_tensor_list = np.array([x.cpu().numpy() for x in an_score_gt_tensor_list])

                attens_energy_pred = np.array(attens_energy_pred)
                attens_energy_pred_gt = np.array(attens_energy_pred_gt)

                if self.args.test_thresh_type == 'linear':
                    an_score_combined_list.append(self.args.coeff_energy * attens_energy_pred + self.args.coeff_score * an_score_tensor_list)
                    an_score_combined_gt_list.append(self.args.coeff_energy * attens_energy_pred_gt + self.args.coeff_score * an_score_gt_tensor_list)
                elif self.args.test_thresh_type == 'var':
                    var_energy = attens_energy_pred.var()
                    var_score = an_score_tensor_list.var()

                    total_var = var_energy + var_score
                    weight_energy = var_energy / total_var
                    weight_score = var_score / total_var

                    an_score_combined_list.append(weight_energy * attens_energy_pred + weight_score * an_score_tensor_list)

                    var_energy = attens_energy_pred_gt.var()
                    var_score = an_score_gt_tensor_list.var()

                    total_var = var_energy + var_score
                    weight_energy = var_energy / total_var
                    weight_score = var_score / total_var

                    an_score_combined_gt_list.append(weight_energy * attens_energy_pred + weight_score * an_score_tensor_list)

                elif self.args.test_thresh_type == 'regu':
                    normalized_energy = np.abs((attens_energy_pred - attens_energy_pred.mean()) / attens_energy_pred.std())
                    normalized_score = np.abs((an_score_tensor_list - an_score_tensor_list.mean()) / an_score_tensor_list.std())
                    an_score_combined_list.append(self.args.coeff_energy * normalized_energy + self.args.coeff_score * normalized_score)

                    normalized_energy = np.abs((attens_energy_pred_gt - attens_energy_pred_gt.mean()) / attens_energy_pred_gt.std())
                    normalized_score = np.abs((an_score_gt_tensor_list - an_score_gt_tensor_list.mean()) / an_score_gt_tensor_list.std())
                    an_score_combined_gt_list.append(self.args.coeff_energy * normalized_energy + self.args.coeff_score * normalized_score)

                elif self.args.test_thresh_type == 'multiplication':
                    an_score_combined_list.append(attens_energy_pred * an_score_tensor_list)
                    an_score_combined_gt_list.append(attens_energy_pred_gt * an_score_gt_tensor_list)

                attens_energy_pred, attens_energy_pred_gt = [], []
                an_score_tensor_list, an_score_gt_tensor_list = [], []

        if self.args.test_thresh:
            if self.args.dataset == "MBA":

                expected_length = len(an_score_combined_list[0][0])
                an_score_combined_list_filtered = [array for array in an_score_combined_list if len(array[0]) == expected_length]
                an_score_combined_list_removed = [array for array in an_score_combined_list if len(array[0]) != expected_length]
                an_score_combined_list = [np.squeeze(array) for array in an_score_combined_list_filtered]
                an_score_combined_list_extra = [np.squeeze(array) for array in an_score_combined_list_removed]

                expected_length = len(an_score_combined_gt_list[0][0])
                an_score_combined_gt_list_filtered = [array for array in an_score_combined_gt_list if len(array[0]) == expected_length]
                an_score_combined_gt_list_removed = [array for array in an_score_combined_gt_list if len(array[0]) != expected_length]
                an_score_combined_gt_list = [np.squeeze(array) for array in an_score_combined_gt_list_filtered]
                an_score_combined_gt_list_extra = [np.squeeze(array) for array in an_score_combined_gt_list_removed]

            else:

                an_score_combined_list = [element[0] for element in an_score_combined_list] 
                an_score_combined_gt_list = [element[0] for element in an_score_combined_gt_list] 
                    
        #! pred AD
        if self.args.test_thresh:
            AT_detected_pred, gt_pred = self.get_AT_pred(an_score_combined_list, test_labels_pred, thresh)
            print(f"Output (detected, gt): ({AT_detected_pred.shape}, {gt_pred.shape})")
            AT_detected_pred_gt, gt_pred = self.get_AT_pred(an_score_combined_gt_list, test_labels_pred, thresh)
        else:
            AT_detected_pred, gt_pred = self.get_AT_pred(attens_energy_pred, test_labels_pred, thresh)
            print(f"Output (detected, gt): ({AT_detected_pred.shape}, {gt_pred.shape})")
            AT_detected_pred_gt, gt_pred = self.get_AT_pred(attens_energy_pred_gt, test_labels_pred, thresh)          
            
        
        #! pred AD
        if self.args.test_thresh:
            print(f"thresh : {thresh}")
            if self.args.dataset == "MBA":
                AT_detected_pred1, gt_pred1 = self.get_AT_pred(an_score_combined_list, test_labels_pred[:-1], thresh)
                AT_detected_pred2, gt_pred2 = self.get_AT_pred(an_score_combined_list_extra, test_labels_pred[-1], thresh)
                AT_detected_pred = np.concatenate((AT_detected_pred1, AT_detected_pred2))
                gt_pred = np.concatenate((gt_pred1, gt_pred2))

                AT_detected_pred_gt1, gt_pred1 = self.get_AT_pred(an_score_combined_gt_list, test_labels_pred[:-1], thresh)
                AT_detected_pred_gt2, gt_pred2 = self.get_AT_pred(an_score_combined_gt_list_extra, test_labels_pred[-1], thresh)
                AT_detected_pred_gt = np.concatenate((AT_detected_pred_gt1, AT_detected_pred_gt2))
                gt_pred = np.concatenate((gt_pred1, gt_pred2))


            else:
                AT_detected_pred, gt_pred = self.get_AT_pred(an_score_combined_list, test_labels_pred, thresh)
                AT_detected_pred_gt, gt_pred = self.get_AT_pred(an_score_combined_gt_list, test_labels_pred, thresh)

            print(f"AT_detected_pred Number of 0/1: {np.sum(AT_detected_pred == 0)}/{np.sum(AT_detected_pred == 1)}")
            print(f"gt_pred Number of 0/1: {np.sum(gt_pred == 0)}/{np.sum(gt_pred == 1)}")
            print(f"Output (detected, gt): ({AT_detected_pred.shape}, {gt_pred.shape})")

            
        else:
            AT_detected_pred, gt_pred = self.get_AT_pred(attens_energy_pred, test_labels_pred, thresh)
            print(f"Output (detected, gt): ({AT_detected_pred.shape}, {gt_pred.shape})")
            AT_detected_pred_gt, gt_pred = self.get_AT_pred(attens_energy_pred_gt, test_labels_pred, thresh)   

        if self.args.adjustment:
            if self.args.test_thresh and self.args.dataset=="MBA":
                AT_detected_pred_gt_ = AT_detected_pred_gt
                AT_detected_pred_ = AT_detected_pred
            
            else:
                AT_detected_pred_original = copy.deepcopy(AT_detected_pred)
                AT_detected_pred = self.detection_adjustment(AT_detected_pred_original ,gt_pred, self.args.adj_tolerance)
                AT_detected_pred_ = self.detection_adjustment_original(AT_detected_pred_original,gt_pred)
                
                AT_detected_pred_gt_original = copy.deepcopy(AT_detected_pred_gt)
                AT_detected_pred_gt = self.detection_adjustment(AT_detected_pred_gt_original, gt_pred, self.args.adj_tolerance)
                AT_detected_pred_gt_ = self.detection_adjustment_original(AT_detected_pred_gt_original,gt_pred)

        #! F         
        mae, mse, rmse, mape, mspe, rse, corr = forecast_metric(self.test_loader.dataset.data, self.gt_signals)
        print('[Prediction] mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        print("--------------------------------------------------------------------------------------------")

        #! AD GT
        AT_detected_pred_gt = np.array(AT_detected_pred_gt)
        gt_pred = np.array(gt_pred)
        accuracy, precision, recall, f_score_gt =self.get_scores(gt_pred, AT_detected_pred_gt)
        # print("  [GT]        A : {:0.6f}, P : {:0.6f}, R : {:0.6f}, F1 : {:0.6f} ".format(accuracy, precision, recall, f_score_gt))
        AT_detected_pred_ = np.array(AT_detected_pred_)
        gt_pred = np.array(gt_pred)
        accuracy, precision, recall, f_score_gt_adj =self.get_scores(gt_pred, AT_detected_pred_gt_)
        # print("  [GT + PA]   A : {:0.6f}, P : {:0.6f}, R : {:0.6f}, F1 : {:0.6f} ".format(accuracy, precision, recall, f_score_gt_adj))

        

        #! AD Pred
        gt_pred, AT_detected_pred = np.array(gt_pred), np.array(AT_detected_pred)
        accuracy, precision, recall, f_score_pred = self.get_scores(gt_pred.reshape(-1), AT_detected_pred.reshape(-1))
        
        if self.args.test_thresh:
            attens_energy_pred_list=np.concatenate(attens_energy_pred_list).reshape(-1)
            
            print("  [Pred]      A : {:0.6f}, P : {:0.6f}, R : {:0.6f}, F1 : {:0.6f} ".format(accuracy, precision, recall, f_score_pred))
            accuracy, precision, recall, f_score_pred_adj = self.get_scores(gt_pred.reshape(-1), AT_detected_pred_)
            # print("  [Pred + PA] A : {:0.6f}, P : {:0.6f}, R : {:0.6f}, F1 : {:0.6f} ".format(accuracy, precision, recall, f_score_pred_adj))
            return_val = mse, f_score_gt, f_score_gt_adj, f_score_pred, f_score_pred_adj
            print("--------------------------------------------------------------------------------------------")  

            return_val = mse, precision, recall, f_score_gt, f_score_gt_adj, f_score_pred, f_score_pred_adj,0,0,0,0
            return return_val
        else:
            attens_energy_pred=np.concatenate(attens_energy_pred).reshape(-1)
            print("  [Pred]      A : {:0.6f}, P : {:0.6f}, R : {:0.6f}, F1 : {:0.6f} ".format(accuracy, precision, recall, f_score_pred))
            accuracy, precision, recall, f_score_pred_adj = self.get_scores(gt_pred.reshape(-1), AT_detected_pred_)
            # print("  [Pred + PA] A : {:0.6f}, P : {:0.6f}, R : {:0.6f}, F1 : {:0.6f} ".format(accuracy, precision, recall, f_score_pred_adj))
            return_val = mse, f_score_gt, f_score_gt_adj, f_score_pred, f_score_pred_adj
            print("--------------------------------------------------------------------------------------------")  

            if f_score_pred !=0:
                score_metric_results = get_metrics(attens_energy_pred, gt_pred, metric='all', slidingWindow=self.args.seq_len)
                print("  [Score Metric] ")
                for metric_name, metric_value in score_metric_results.items():
                    print(f"  {metric_name} : {metric_value:.6f}")
                vus_roc = score_metric_results["VUS_ROC"]
                vus_pr = score_metric_results["VUS_PR"]
                r_auc_roc = score_metric_results["R_AUC_ROC"]
                r_auc_pr = score_metric_results["R_AUC_PR"]
                return_val = mse, precision, recall, f_score_gt, f_score_gt_adj, f_score_pred, f_score_pred_adj, vus_roc, vus_pr, r_auc_roc, r_auc_pr
                print("--------------------------------------------------------------------------------------------") 
                return return_val
            else:
                return_val = mse,0,0,0,0,0,0,0,0,0,0
                return return_val
            
  