import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .prompt import Prompt
from .clustering import cluster_dpc_knn, merge_tokens
from finch import FINCH


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0., activation="relu",prompt_num=5,top_k=3,args=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.prompt_num = prompt_num
        self.top_k = top_k
        self.args= args
        
    def forward(self, x, attn_mask=None, noise=None):
        if noise!=None:
            noise = noise.reshape(noise.shape[0],-1,noise.shape[-1])
            if self.args.noise_position=="front":
                x = torch.cat([noise,x], dim=1)
            elif self.args.noise_position=="end":
                x = torch.cat([x,noise], dim=1)
            elif self.args.noise_position=="middle":
                win_size = x.shape[1]
                x = torch.cat((x[:,:int(win_size/2)], noise, x[:,int(win_size/2):]),dim=1)
            
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
            )
        # if noise != None:
        #     new_x = new_x[:,:-self.prompt_num*self.top_k,:]
        #     x= x[:,:-self.prompt_num*self.top_k,:]
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None,noise=None,finch_normal=None):
        # x [B, L, D]
        #* FLOPS: 464076800 * 3 = 1392230400 (MBA)
        series_list = []
        prior_list = []
        sigma_list = []
        for idx,attn_layer in enumerate(self.attn_layers):
            if idx>0: noise=None
            x, series, prior, sigma = attn_layer(x, noise=noise,attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            if idx==2:
                if finch_normal is None: # store normal prototype
                    clustered_feats_lst = []
                    cluster_idx_lst = []
                    # for branch in range(3):
                    feat_finch = x
                    B,L,D = feat_finch.shape
                    which_partition = 0
                    c, num_clust, req_c = FINCH(feat_finch.reshape(B*L, -1).detach().cpu().numpy(), verbose=False) # [BxL,D] -> [BxL,p]
                    c = c[:,which_partition] # -> [BxL]
                    c = torch.Tensor(c).to(feat_finch.device).to(dtype=torch.int64)
                    clustered_feats = merge_tokens(feat_finch.reshape(1, B*L,-1), c.reshape(1, -1), num_clust[which_partition]).squeeze(0) # [1,6000,D]
                    labels =  F.one_hot(c) 

                    clustered_feats_lst.append(clustered_feats)
                    cluster_idx_lst.append(c)

                    finch_out = {'clustered_feats': clustered_feats_lst,
                                'cluster_idx': cluster_idx_lst}
                else: # abnormal
                    logits_lst = []
                    labels_lst = []
                    # for branch in range(3):
                    clustered_feats = finch_normal['clustered_feats'][0]  # prototype of N
                    c = finch_normal['cluster_idx'][0]                   # from N
                    labels =  F.one_hot(c)                              # from N

                    feat_finch = x        # from AN
                    # B=len(feat_finch)
                    feat_finch = feat_finch[:,:100]
                    B,L,D = feat_finch.shape        # from AN

                    similarity_matrix = torch.matmul(F.normalize(feat_finch.reshape(B*L,-1), dim=-1), F.normalize(clustered_feats, dim=-1).T)

                    pos = similarity_matrix[labels.bool()].unsqueeze(1)
                    neg = similarity_matrix[~labels.bool()].view(B*L,-1)

                    logits = torch.cat([pos, neg], dim=1) / 1
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(feat_finch.device)

                    logits_lst.append(logits)
                    labels_lst.append(labels)

                    finch_out = {'logits': logits_lst,
                                'labels': labels_lst}

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list, finch_out


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=16, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, qkv_modules=None, attn_layer_modules=None,
                 pool_size=10, prompt_num=5, top_k=3, args=None):
        super(AnomalyTransformer, self).__init__()
        self.win_size=win_size
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.args=args
        # Encoder
        if qkv_modules!=None:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention, top_k=top_k, prompt_num=prompt_num),
                            d_model, n_heads, qkv_module = qkv_modules[l]),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                        prompt_num=prompt_num,
                        top_k=top_k,
                        args=args
                    ) for l in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
            
        elif attn_layer_modules!=None:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        attn_layer_modules[l],
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                        prompt_num=prompt_num,
                        top_k=top_k,
                        args=args
                    ) for l in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        
        else:
            self.encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention, prompt_num=prompt_num, top_k=top_k),
                            d_model, n_heads, qkv_module = None),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation,
                        prompt_num=prompt_num,
                        top_k=top_k,
                        args=args
                    ) for l in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )

        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        if args.noise_injection:
            self.prompt = Prompt(ftr_dim=d_model, top_k=top_k, pool_size=pool_size, prompt_num=prompt_num, channel=enc_in)

    def forward_features(self,x, feature=None, noise=None,finch_normal=None):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas, finch_out = self.encoder(enc_out,noise=noise,finch_normal=finch_normal)
        if noise!=None:
            enc_out = enc_out[:,-self.args.win_size:,:]
        return enc_out
    def forward_embedding(self, enc_out):
        enc_out, series, prior, sigmas, finch_out = self.encoder(enc_out)
        # if noise!=None:
        #     enc_out = enc_out[:,-self.args.win_size:,:]
        return enc_out
    def get_anomaly_embedding(self,x,noise):
        x = self.embedding(x)
        noise = noise.reshape(noise.shape[0],-1,noise.shape[-1])
        x = torch.cat([noise,x], dim=1)
        return x
    
    def forward(self, x, feature=None, noise=None,finch_normal=None):
        #* FLOPs: 1228800 (MBA)
        #* FLOPs: 75571200 (WADI)
        enc_out = self.embedding(x)
        
        #* FLOPs: 1392230400 (MBA, WADI)
        enc_out, series, prior, sigmas, finch_out = self.encoder(enc_out,noise=noise,finch_normal=finch_normal)
        series_,prior_,sigmas_ = [], [], []
        if enc_out.shape[1]>self.win_size: # noise injection
            if self.args.noise_position=="front":
                enc_out=enc_out[:,-self.win_size:]
                for ser,pri,sig in zip(series,prior,sigmas):
                    series_.append(ser[:,:,-self.win_size:,-self.win_size:])
                    prior_.append(pri[:,:,-self.win_size:,-self.win_size:])
                    sigmas_.append(sig[:,:,-self.win_size:,-self.win_size:])
                    series,prior,sigmas = series_, prior_, sigmas_
            elif self.args.noise_position=="end":
                enc_out=enc_out[:,:self.win_size]
                for ser,pri,sig in zip(series,prior,sigmas):
                    series_.append(ser[:,:,:self.win_size,:self.win_size])
                    prior_.append(pri[:,:,:self.win_size,:self.win_size])
                    sigmas_.append(sig[:,:,:self.win_size,:self.win_size])
                    series,prior,sigmas = series_, prior_, sigmas_
            elif self.args.noise_position=="middle":
                mid_point=int(self.win_size/2)
                enc_out=torch.cat(( enc_out[:,:mid_point], enc_out[:,-mid_point:]),dim=1)
                for ser,pri,sig in zip(series,prior,sigmas):
                    # series_.append(ser[:,:,:self.win_size,:self.win_size])
                    ser_ = torch.cat((ser[:,:,:mid_point],ser[:,:,-mid_point:]), dim=2)
                    ser__ = torch.cat((ser_[:,:,:,:mid_point],ser_[:,:,:,-mid_point:]),dim=3)
                    series_.append(ser__)
                    
                    pri_ = torch.cat((pri[:,:,:mid_point],pri[:,:,-mid_point:]), dim=2)
                    pri__ = torch.cat((pri_[:,:,:,:mid_point],pri_[:,:,:,-mid_point:]),dim=3)
                    prior_.append(pri__)
                    
                    sig_ = torch.cat((sig[:,:,:mid_point],sig[:,:,-mid_point:]), dim=2)
                    sig__ = torch.cat((sig_[:,:,:,:mid_point],sig_[:,:,:,-mid_point:]),dim=3)
                    sigmas_.append(sig__)
                    series,prior,sigmas = series_, prior_, sigmas_
        enc_out = self.projection(enc_out)

        if feature is not None:
            #* FLOPs: 20480 (MBA, WADI)
            prompt_out = self.prompt(feature)
        else:
            prompt_out = None
            
        if self.output_attention:
            return enc_out, series, prior, sigmas, finch_out,prompt_out
        else:
            return enc_out, prompt_out  # [B, L, D]
