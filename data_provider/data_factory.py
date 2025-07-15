from data_provider.data_loader import  F_AD_Dataset
from torch.utils.data import DataLoader
import torch
from functools import partial
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer


def reorganize(dataset, root_path, train_path,test_path, test_label_path, train_anomaly_ratio=0.2, scaler="standard",):
    if scaler=="minmax":
        scaler = MinMaxScaler()
    elif scaler=="normal":
        scaler = Normalizer()
    elif scaler=="robust":
        scaler = RobustScaler()
    elif scaler=="standard":
        scaler = StandardScaler()
    if train_path.endswith(".npy"):
        train_data = np.load(root_path + train_path)
        test_data = np.load(root_path+ test_path)
        test_label = np.load(root_path + test_label_path)
    elif train_path.endswith(".csv"):
        train_data = pd.read_csv(root_path + train_path)
        train_data = train_data.values[:, 1:]
        train_data = np.nan_to_num(train_data)
        
        test_data = pd.read_csv(root_path + test_path)
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_label = pd.read_csv(root_path + test_label_path).values[:, 1:]

    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    train_label = np.zeros(scaled_train_data.shape[0])
    return scaled_train_data,None,scaled_test_data,None,train_label,None, test_label, test_label

def reorganize2(dataset, root_path, train_path,test_path, test_label_path, train_anomaly_ratio=0.2, scaler="standard"):
    if scaler=="minmax":
        scaler = MinMaxScaler()
    elif scaler=="normal":
        scaler = Normalizer()
    elif scaler=="robust":
        scaler = RobustScaler()
    elif scaler=="standard":
        scaler = StandardScaler()
    if train_path.endswith(".npy"):
        train_data = np.load(root_path + train_path)
        test_data = np.load(root_path+ test_path)
        test_label = np.load(root_path + test_label_path)
    elif train_path.endswith(".csv"):
        train_data = pd.read_csv(root_path + train_path)
        train_data = train_data.values[:, 1:]
        train_data = np.nan_to_num(train_data)
        
        test_data = pd.read_csv(root_path + test_path)
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_label = pd.read_csv(root_path + test_label_path).values[:, 1:]
   
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    train_label = np.zeros(scaled_train_data.shape[0])
    return scaled_train_data,None,scaled_test_data,None,train_label,None, test_label, test_label
 
def data_provider(args, flag, task='F',step=100):
        
    Data = F_AD_Dataset
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.dataset in ["MSL","SMAP","SWAT","SMD","WADI","MBA","sine", "kpi"]:
        train_path, test_path, label_path = f"/{args.dataset}_train.npy", f"/{args.dataset}_test.npy",f"/{args.dataset}_test_label.npy"
    elif args.dataset=="PSM":
        train_path, test_path, label_path = f"/train.csv", f"/test.csv",f"/test_label.csv"
    elif "NAB" in args.dataset or "UCR" in args.dataset or "exathlon" in args.dataset:
         train_path, test_path, label_path = f"/{args.dataset}_train.npy",f"/{args.dataset}_test.npy",f"/{args.dataset}_labels.npy"
    elif "machine" in args.dataset: #SMD
        train_path, test_path, label_path = f"/{args.dataset}_train.npy", f"/{args.dataset}_test.npy",f"/{args.dataset}_labels.npy"
    
    
    train_data, new_train_data, test_data, new_test_data, train_label1, train_label2, test_label, new_test_label = reorganize2(
        dataset = args.dataset,
        root_path = args.root_path,
        train_path = train_path,
        test_path = test_path,
        test_label_path = label_path,
        train_anomaly_ratio = args.train_anomaly_ratio,
        scaler = args.scaler
    )
    if flag=="train":
        train_set1 = Data(
            root_path=args.root_path,
            data=train_data,
            task=task,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            win_size=args.win_size,
            step=step,
            test_label = train_label1,
            args=args
            )
        
        data_set = train_set1
    elif flag=="test":
        
        data_set = test_data
        test_label=test_label
            
        data_set = Data(
            root_path=args.root_path,
            data=data_set,
            test_label=test_label,
            task=task,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            win_size=args.win_size,
            step=args.pred_len,
            args=args
            )
        print(flag, len(data_set))
    
    g = torch.Generator()
    g.manual_seed(0)
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=0,
        drop_last=drop_last,
        generator=g,
        worker_init_fn=seed_worker,
        )
    if args.AD_model=="TranAD" and flag=="test": return data_loader, test_label
    return data_loader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
def collate_fn(batch):
    batch = list(filter(lambda x: (x is not None) and (len(x) >0), batch))
    return torch.utils.data.dataloader.default_collate(batch)