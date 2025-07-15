import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')
from utils.tools import visual
    

        
class F_AD_Dataset(Dataset):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.seq_len = self.size[0]
        self.label_len = self.size[1]
        self.pred_len = self.size[2]
        self.data_stamp = np.arange(self.data.shape[0])
    
    def __getitem__(self, index):
        if self.task == "F": 
            index = index * self.step
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            
            seq_x = self.data[s_begin:s_end]
            seq_y = np.float32(self.data[r_begin:r_end])
            seq_x_mark = np.float32(self.data_stamp[s_begin:s_end])
            seq_y_mark = np.float32(self.data_stamp[r_begin:r_end])
            
           
            anomaly_label_prior = self.test_label[s_begin:s_end]
            anomaly_label_pred = self.test_label[s_end:r_end]
            if len(anomaly_label_pred)<self.pred_len:
                return None
            return seq_x, seq_y, seq_x_mark, seq_y_mark, anomaly_label_prior, anomaly_label_pred
        
        elif self.task == "AD":            
            index = index * self.step
            seq_x = np.float32(self.data[index:index + self.win_size])
            anomaly_label = self.test_label[index:index + self.win_size]
            return seq_x, anomaly_label
        
    def __len__(self):
        if self.task=="F":
            return (self.data.shape[0] - self.pred_len - self.seq_len) // self.step + 1
        else:
            return self.data.shape[0] // self.step 

