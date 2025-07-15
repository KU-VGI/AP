
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os
from torch.nn.utils import weight_norm
from utils.injection import *


def train_cross_attn (AAFN, train_loader, fe=None,device="", args=None):
        for p in fe.parameters(): p.requires_grad=False
        print(">>> Start pretraining AAFN...")
        optimizer = torch.optim.Adam(AAFN.parameters(),lr=5e-5)
        criterion = nn.MSELoss()
        for n,p in AAFN.named_parameters():
            p.requires_grad=True
        for epoch in range(args.cross_attn_epochs):
            train_loss1, train_loss2, train_loss3, train_loss4 = [], [], [], [],
            for i, batch in enumerate(train_loader):
                if not batch: continue
                optimizer.zero_grad()
                batch_x, batch_y, batch_x_mark, batch_y_mark,labels_prior, labels_pred = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_y = batch_y[:, -args.pred_len:, :]
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                labels_pred = labels_pred.float().to(device)

                injected_prior_x, injected_prior_l = inject_amplify_learnable(batch_x,fe,args.amplify_type, AAFN.global_scale, AAFN.contextual_scale, AAFN.trend_scale, AAFN.shapelet_scale, AAFN.seasonal_scale)
                injected_pred_x, injected_pred_l = inject_amplify_learnable(batch_y,fe,args.amplify_type, AAFN.global_scale, AAFN.contextual_scale, AAFN.trend_scale, AAFN.shapelet_scale, AAFN.seasonal_scale)
                    


                stacked_labels_pred = labels_pred.unsqueeze(-1).repeat(1,1,args.input_c)
                stacked_injected_pred_l = injected_pred_l.unsqueeze(-1).repeat(1,1,args.input_c)

                attn_output1 = AAFN(injected_pred_x, injected_prior_x) 
                cross_attn_loss1 = criterion(attn_output1, stacked_injected_pred_l)
                attn_output2 = AAFN(injected_pred_x, batch_x) 
                cross_attn_loss2 = criterion(attn_output2, stacked_injected_pred_l)

                attn_output3 = AAFN(batch_y, batch_x) 
                cross_attn_loss3 = criterion(attn_output3, stacked_labels_pred)

                attn_output4 = AAFN(batch_y,injected_prior_x) 
                cross_attn_loss4 = criterion(attn_output4, stacked_labels_pred)

                train_loss1.append(cross_attn_loss1.item())
                train_loss2.append(cross_attn_loss2.item())
                train_loss3.append(cross_attn_loss3.item())
                train_loss4.append(cross_attn_loss4.item())

                loss = cross_attn_loss1 +cross_attn_loss2 
                loss.backward()
                optimizer.step()
            print(f"AAFN Epoch {epoch+1} - (AN-AN, N-AN, N-N, AN-N): {np.mean(train_loss1):.4f}, {np.mean(train_loss2):.4f},  {np.mean(train_loss3):.4f},  {np.mean(train_loss4):.4f}")
        for n,p in AAFN.named_parameters():
            p.requires_grad=False



class AAFN(nn.Module):
    def __init__(self,args):
        super(AAFN,self).__init__()
        self.args= args
        self.embedding = nn.Linear(in_features = self.args.input_c, out_features = self.args.d_model)
        self.attn = nn.MultiheadAttention(embed_dim=self.args.d_model, num_heads=self.args.cross_attn_nheads,batch_first=True)
        self.out_proj = nn.Linear(in_features = self.args.d_model, out_features = self.args.input_c)
        self.activation = F.sigmoid

        if args.learnable:
            self.global_scale = nn.Parameter(torch.randn(1))
            self.contextual_scale = nn.Parameter(torch.randn(1))
            self.trend_scale = nn.Parameter(torch.randn(1))
            self.shapelet_scale = nn.Parameter(torch.randn(1))
            self.seasonal_scale = nn.Parameter(torch.randn(1))

    def forward(self, query, key, query_embedded=False):

        key, abnormal_labels = inject_learnable(key, self.global_scale,self.contextual_scale,self.trend_scale,self.shapelet_scale,self.seasonal_scale)

        if query_embedded: 
            query_embed = query
            key_embed = key
            value_embed = key
        else: 
            query_embed = self.embedding(query)
            key_embed = self.embedding(key)
            value_embed = self.embedding(key)
        o = self.attn(query_embed,key_embed,value_embed)[0]
        
  
        o = o[:,-self.args.pred_len:]
        o = self.out_proj(o)
        o = self.activation(o)
        
        return o



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
