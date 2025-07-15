
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import copy
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class Base_Solver():
    def __init__(self):
        pass
    def calc_series_prior_loss(self, series, prior):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size)).detach())) + torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
            prior_loss += (torch.mean(my_kl_loss(
                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                        self.win_size)),
                series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(), (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)
        return series_loss, prior_loss  
    def calc_series_prior_loss_test(self, series, prior):
        temperature=50
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.args.win_size)),
                    series[u].detach()) * temperature
        return series_loss, prior_loss  
    def get_AT_pred(self, attens_energy, test_labels, thresh):
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        AT_detected = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)
        return AT_detected, gt
    
    def get_cri(self,AT_model, AT_input,metric):
        if self.args.AD_model=="AT":
            AT_output, series, prior, _ , _,_= AT_model(AT_input)
            mse_no_reduce = torch.nn.MSELoss(reduction='none')
            timepoint_error = mse_no_reduce(AT_input, AT_output)
            loss = torch.mean(self.criterion(AT_input, AT_output), dim=-1)
            cri = metric * loss
        return cri, AT_output, timepoint_error

    def detection_adjustment(self,AT_detected_ori,gt, tol):
        anomaly_state = False
        AT_detected = copy.deepcopy(AT_detected_ori)
        for i in range(len(gt)):
            if gt[i] == 1 and AT_detected[i] == 1 and not anomaly_state:
                anomaly_state = True
                # for j in range(i, 0, -1):
                for j in range(i, i-tol,-1):
                    if j<0: break
                    if gt[j] == 0:
                        break
                    else:
                        if AT_detected[j] == 0:
                            AT_detected[j] = 1
                # for j in range(i, len(gt)):
                for j in range(i, i+tol):
                    if j>=len(gt):break
                    if gt[j] == 0:
                        break
                    else:
                        if AT_detected[j] == 0:
                            AT_detected[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            # if anomaly_state:
            #     AT_detected[i] = 1
        return AT_detected
    
    
    def detection_adjustment_original(self,AT_detected,gt):
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and AT_detected[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                # for j in range(i, i-self.args.adj_tolerance,-1):
                    if j<0: break
                    if gt[j] == 0:
                        break
                    else:
                        if AT_detected[j] == 0:
                            AT_detected[j] = 1
                for j in range(i, len(gt)):
                # for j in range(i, i+self.args.adj_tolerance):
                    if j>=len(gt):break
                    if gt[j] == 0:
                        break
                    else:
                        if AT_detected[j] == 0:
                            AT_detected[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                AT_detected[i] = 1
        return AT_detected
    def get_scores(self,gt, AT_detected):
        accuracy = accuracy_score(gt,  AT_detected)
        precision, recall, f_score, support = precision_recall_fscore_support(gt,  AT_detected,
                                                                                average='binary')
        return accuracy, precision, recall, f_score
    
    def classify_case(self,labels_prior,labels_pred):
        if np.sum(labels_prior)==0 and np.sum(labels_pred)==0: case="nn"
        elif np.sum(labels_prior)>0 and np.sum(labels_pred)==0: case="an"
        elif np.sum(labels_prior)==0 and np.sum(labels_pred)>0: case="na"
        else: case="aa"
        return case

    def get_energy_loss(self,series1, pos_n,pos_an):
        if len(pos_n.shape)==1:
            energy_n = -torch.logsumexp(series1[0][:,:,pos_n,:] + series1[0][:,:,:,pos_n].permute(0,1,3,2), dim=-1)
            energy_an = -torch.logsumexp(series1[0][:,:,pos_an,:] + series1[0][:,:,:,pos_an].permute(0,1,3,2), dim=-1)
            energy_loss = (torch.mean(energy_n)/ torch.mean(energy_an))
        else: #! noise injection
            # series: [instance, head, win, win]
            energy_n, energy_an = [], []
            for i, (inst, n, an) in enumerate(zip(series1[self.args.ftr_idx], pos_n, pos_an)):
                if torch.sum(n)>0:
                    energy_n.append(torch.mean(-torch.logsumexp(inst[:,n,:]+inst[:,:,n].permute(0,2,1), dim=-1)))
                energy_an.append(torch.mean(-torch.logsumexp(inst[:,an,:]+inst[:,:,an].permute(0,2,1), dim=-1)))
            if len(energy_n)>0:
                energy_loss = (torch.mean(torch.stack(energy_n)) / torch.mean(torch.stack(energy_an)))
            else:
                energy_loss = 1/torch.mean(torch.stack(energy_an))
        return energy_loss

    def get_threshold(self):
        # (1) stastic on the train set
        attens_energy = []
        temperature = 50
        
        for i, batch in enumerate(self.train_loader):
            if not batch: continue
            batch_x, batch_y, batch_x_mark, batch_y_mark, labels_prior, labels = batch
            
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            labels = labels.float().to(self.device)

            signal = torch.cat([batch_x, batch_y],dim=1)
            input_x = signal[:,:self.args.win_size] 
            
            AD_model_outputs = self.model.AD_model(input_x)
         
            output, series, prior, _,_ ,_= AD_model_outputs
            loss = torch.mean(self.criterion(input_x, output), dim=-1)
            series_loss, prior_loss = self.calc_series_prior_loss_test(series,prior)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)  
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, batch in enumerate(self.test_loader):
            if not batch: continue
            batch_x,  labels = batch
            
            batch_x = batch_x.float().to(self.device)
            labels = labels.float().to(self.device)
            input_x =  batch_x
            AD_model_outputs = self.model.AD_model(input_x)
           
            output, series, prior, _,_,_ = AD_model_outputs
            loss = torch.mean(self.criterion(input_x, output), dim=-1)
            series_loss, prior_loss = self.calc_series_prior_loss_test(series,prior)
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.args.anormly_ratio)
        print("Threshold :", thresh)
        return thresh, metric