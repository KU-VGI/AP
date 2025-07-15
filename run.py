import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import random
import numpy as np

from solvers.joint_solver import Solver

from config.parser import get_parser, set_data_config
from utils.utils import *
from AAFN import train_cross_attn
import pandas as pd

def run_single_dataset(args):
    mse_list, f1_gt_list, f1_gt_adj_list, f1_pred_list, f1_pred_adj_list,pate_list, p_list, r_list, vus_roc_list,vus_pr_list,r_auc_roc_list,r_auc_pr_list = [], [], [], [], [], [], [], [], [], [], [], [] 
    solver = Solver(vars(args),args)
        
    if args.noise_injection or (args.cross_attn and args.aafn_amplify):
        solver.train_ftr_extractor()

    print("Start pre-training...")     
    solver.train_noise_and_cross(AAFN=solver.AAFN, model=solver.fe, args=args)   
   
    print("Start main training...")
    solver.train(epoch=args.joint_epochs)
    
    for p in solver.model.parameters(): p.requires_grad=False
    mse, precision, recall, f_score_gt, f_score_gt_adj, f_score_pred, f_score_pred_adj, vus_roc, vus_pr, r_auc_roc, r_auc_pr = solver.test()
    mse_list.append(mse)
    p_list.append(precision)
    r_list.append(recall)
    f1_gt_list.append(f_score_gt)
    f1_gt_adj_list.append(f_score_gt_adj)
    f1_pred_list.append(f_score_pred)
    f1_pred_adj_list.append(f_score_pred_adj)
    vus_roc_list.append(vus_roc)
    vus_pr_list.append(vus_pr)
    r_auc_roc_list.append(r_auc_roc)
    r_auc_pr_list.append(r_auc_pr)
    
    if args.dataset in ["MBA","SMD","PSM","MSL","SMAP","WADI","SWAT"]:
        print(f"""mse: {np.mean(mse_list):.2f} (± {np.std(mse_list)*100:.2f}),
            P: {np.mean(p_list)*100:.2f} (± {np.std(p_list)*100:.2f}), 
            R: {np.mean(r_list)*100:.2f} (± {np.std(r_list)*100:.2f}) , 
            F1: {np.mean(f1_pred_list)*100:.2f} (± {np.std(f1_pred_list)*100:.2f}),
            AD F1: {np.mean(f1_pred_adj_list)*100:.2f} (± {np.std(f1_pred_adj_list)*100:.2f}),
            gt F1: {np.mean(f1_gt_list)*100:.2f} (± {np.std(f1_gt_list)*100:.2f}),
            gt AD F1: {np.mean(f1_gt_adj_list)*100:.2f} (± {np.std(f1_gt_adj_list)*100:.2f}),
            vus_roc: {np.mean(vus_roc_list)*100:.2f} (± {np.std(vus_roc_list)*100:.2f}),
            vus_pr: {np.mean(vus_pr_list)*100:.2f} (± {np.std(vus_pr_list)*100:.2f}),
            r_auc_roc: {np.mean(r_auc_roc_list)*100:.2f} (± {np.std(r_auc_roc_list)*100:.2f}),
            r_auc_pr: {np.mean(r_auc_pr_list)*100:.2f} (± {np.std(r_auc_pr_list)*100:.2f})""")
   
    scores = {
        "f1": np.mean(f1_pred_list)*100,
        "mse": np.mean(mse_list),
        "vus_roc": np.mean(vus_roc_list)*100,
        "vus_pr": np.mean(vus_pr_list)*100,
        "r_auc_roc": np.mean(r_auc_roc_list)*100,
        "r_auc_pr": np.mean(r_auc_pr_list)*100,
    }
   
    return scores
def run_seeds(args,random_seeds):
    multi_seed_scores = dict()
    for metric in ['f1','mse','vus_roc','vus_pr','r_auc_roc','r_auc_pr']:
        multi_seed_scores[metric]=[]
    for seed in random_seeds:
        print(f"\n================================== seed: {seed} =======================================")
        args.random_seed = seed
        fix_seed(seed)      
        if args.dataset in ["MSL","PSM","SMAP","SMD","WADI","MBA","SWAT"]:
            scores=run_single_dataset(args)
            print(scores)
            for k,v in scores.items():
                multi_seed_scores[k].append(v)
        else: # sub datasets
            single_seed_scores = dict()
            for metric in ['f1','mse','vus_roc','vus_pr','r_auc_roc','r_auc_pr']:
                single_seed_scores[metric]=[]
            for i, sub_dataset in enumerate(args.sub_datasets):
                print(f"\nBegin Experiments on sub dataset #{i} - {sub_dataset}")
                args.dataset = sub_dataset  
                scores=run_single_dataset(args)
                print(scores)
                for k,v in scores.items():
                    single_seed_scores[k].append(v)
            print("Experiments on all sequences finished")
            print(f"[Result - seed {args.random_seed}]")
            for k,v in single_seed_scores.items():
                print(f"{k}: {np.mean(v):.4f} += {np.std(v):.4f}")
            for k,v in single_seed_scores.items():
                multi_seed_scores[k].append(np.mean(v))
            
    print(f"\n\n[Result over {len(random_seeds)} seeds]")
    for k,v in multi_seed_scores.items():
        print(f"{k}: {np.mean(v):.4f} += {np.std(v):.4f}")
    print("\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AT+PatchTST')
    parser = get_parser(parser)
  
    args = parser.parse_args()
    
    args=set_data_config(args)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
   
    print('Args in experiment:')
    print(args)

   
    random_seeds = [20462]

    if args.dataset in ["NAB", "UCR","exathlon"]:       
        if args.dataset == "NAB":
            args.sub_datasets =  ["NAB_machine","NAB_temperature", "NAB_cpu","NAB_ec2","NAB_key_hold","NAB_key_updown"]
        elif args.dataset == "UCR":
            args.sub_datasets = ["UCR_135", "UCR_136", "UCR_137", "UCR_138"]       
        elif args.dataset == "exathlon":
            args.sub_datasets = [f"exathlon_{app_id}" for app_id in [1,2,3,4,5,6,9,10]]   
            
    run_seeds(args, random_seeds)   