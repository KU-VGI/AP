import torch
import torch.nn as nn
import torch.nn.functional as F
from AD_models.AT_model.AnomalyTransformer import AnomalyTransformer
from F_models import PatchTST
from utils.injection import *


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class SharedModel(nn.Module):
    def __init__(self, args):
        super(SharedModel, self).__init__()
        self.args=args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        qkv_modules=[]
        if self.args.share:
            for i in range(args.e_layers):
                if i in args.shared_layer_list:
                    query_projection = torch.nn.Linear(args.d_model, args.d_model)
                    key_projection = torch.nn.Linear(args.d_model, args.d_model)
                    value_projection = torch.nn.Linear(args.d_model, args.d_model)
                    qkv_modules.append([query_projection, key_projection, value_projection])
                else:
                    qkv_modules.append(None)
        else:
            qkv_modules=[None,None,None]
        self.AD_model = AnomalyTransformer(d_model=args.d_model,win_size=args.win_size, enc_in=args.input_c, c_out=args.output_c, n_heads=args.n_heads, d_ff=args.d_ff,
                                                    e_layers=args.e_layers,qkv_modules=qkv_modules, prompt_num=args.prompt_num, pool_size=args.pool_size, top_k=args.top_k,
                                                    args=args)
        self.F_model = PatchTST.Model(self.args, qkv_modules=qkv_modules).float()


        self.criterion = nn.MSELoss()

    def forward_f_model(self,x):
        forecasted = self.F_model(x)
        forecasted = forecasted[:, -self.args.pred_len:]
        return forecasted
    
    def forward_ad_model(self,x,y):
        recon_output = torch.zeros_like(x)
        if self.args.pred_len>=self.args.seq_len:
            assert self.args.pred_len%self.args.seq_len==0
            iter_num=int(self.args.pred_len/self.args.seq_len)
            rec_loss,series_loss,prior_loss,energy_loss,signal_adaptive_loss =0,0,0,0,0
            for i in range(iter_num):
                start, end = self.args.seq_len*(i), self.args.seq_len*(i+1)
                sliced_input_gt = y[:,start:end]
                if self.args.noise_injection: 
                    feature, _ = self.fe(sliced_input_gt)
                else: feature=None
                output, series, prior, _, finch_normal, prompt_out = self.AD_model(sliced_input_gt, feature=feature)
                recon_output[:,start:end] = output
                losses = self.calc_losses(sliced_input_gt, output, series,prior,prompt_out)
                
                rec_loss += losses['rec_loss']
                series_loss += losses['series_loss']
                prior_loss += losses['prior_loss']
                if 'energy_loss' in losses.keys(): 
                    energy_loss += losses['energy_loss']
                if 'signal_adaptive_loss' in losses.keys():
                    signal_adaptive_loss += losses['signal_adaptive_loss']

        total_loss = 2* rec_loss \
            - self.args.k * series_loss \
            - self.args.k * prior_loss \
            + energy_loss \
            + signal_adaptive_loss
        loss_dict={
            'rec_loss':rec_loss,
            'series_loss':series_loss,
            'prior_loss':prior_loss,
            'energy_loss':energy_loss,
            'signal_adaptive_loss':signal_adaptive_loss
        }
        return recon_output, total_loss, loss_dict
                

    def forward(self,x,y):
        #* FLOPs: 338763776 (MBA)
        #* FLOPs: 20833972224 (WADI)
        #* FLOPs: 21816397824 (WADI_400)
        f_output = self.forward_f_model(x)
        #* FLOPs: 1393479680 (MBA)
        #* FLOPs: 1494016000 (WADI)
        ad_output, ad_loss, ad_loss_dict = self.forward_ad_model(f_output,y)
        return f_output, ad_output, ad_loss, ad_loss_dict

    def calc_losses(self,gt, output,series,prior,prompt_out):
        losses=dict()
        losses['series_loss'], losses['prior_loss'] = self.calc_series_prior_loss(series, prior) 
        losses['rec_loss'] = self.criterion(output,gt)
        if self.args.noise_injection:
            sim, noise = prompt_out['reduce_sim'], prompt_out['batched_prompt']
            losses['signal_adaptive_loss'] = self.args.sa_loss_coeff * sim


            pos_n, pos_an= select_abnormal_positions(gt,w_ab_ratio=0.5,continual=True)
            # output, series, prior, _, prompt_out = self.AD_model(input_x, feature=feature)
            
            output1, series1, prior1, _, finch_abnormal, _ = self.AD_model(gt, noise=noise,finch_normal=None)
            losses['energy_loss'] = self.args.energy_loss_coeff * self.get_energy_loss(series1,pos_n,pos_an)
            losses['rec_loss'] += self.criterion(output1, gt)

        return losses

    def get_energy_loss(self,series1, pos_n,pos_an):
        if len(pos_n.shape)==1:
            energy_n = -torch.logsumexp(series1[0][:,:,pos_n,:] + series1[0][:,:,:,pos_n].permute(0,1,3,2), dim=-1)
            energy_an = -torch.logsumexp(series1[0][:,:,pos_an,:] + series1[0][:,:,:,pos_an].permute(0,1,3,2), dim=-1)
            energy_loss = (torch.mean(energy_an)/ torch.mean(energy_n))
        else: #! noise injection
            # series: [instance, head, win, win]
            energy_n, energy_an = [], []
            for i, (inst, n, an) in enumerate(zip(series1[self.args.ftr_idx], pos_n, pos_an)):
                if torch.sum(n)>0:
                    energy_n.append(torch.mean(-torch.logsumexp(inst[:,n,:]+inst[:,:,n].permute(0,2,1), dim=-1)))
                energy_an.append(torch.mean(-torch.logsumexp(inst[:,an,:]+inst[:,:,an].permute(0,2,1), dim=-1)))
            if len(energy_n)>0:
                energy_loss = (torch.mean(torch.stack(energy_an)) / torch.mean(torch.stack(energy_n)))
            else:
                energy_loss = torch.mean(torch.stack(energy_an))
        return energy_loss


    def calc_series_prior_loss(self, series, prior):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.win_size)).detach())) + torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.args.win_size)).detach(),
                            series[u])))
            prior_loss += (torch.mean(my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        self.args.win_size)),
                series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.args.win_size)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        return series_loss, prior_loss  