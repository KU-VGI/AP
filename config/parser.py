def get_parser(parser):
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')    
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, default="PatchTST",choices=['PatchTST','MICN','GPT2', 'iTransformer']) ##

    # data loader
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=20, help='start token length')
    parser.add_argument('--pred_len', type=int, default=100, help='prediction sequence length')


    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.0, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=10, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=321, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--F_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')


    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=1.00)
    parser.add_argument('--add_abnormal_f', action='store_true', default=False)
    parser.add_argument('--add_abnormal_ad', action='store_true', default=False)
    parser.add_argument('--train_anomaly_ratio', type=float, default=0.1)
    
    #! anomaly transformer
    parser.add_argument('--AD_lr', type=float, default=1e-4)
    parser.add_argument('--AD_epochs', type=int, default=1)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--step', type=int, default=100) # step for inference
    parser.add_argument('--train_step', type=int, default=100) # step for training
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')

    # parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    
   
    #! Anomaly prompting    
    parser.add_argument('--pool_size', type=int, default=10)
    parser.add_argument('--prompt_num', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=3)


    #! synthetic signal
    parser.add_argument('--anomaly_type', type=str, default='pg', choices=["pg", "pc", "cg", "cs", "ct"])
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--inject', action='store_true', default=False)
    
    parser.add_argument('--plotting', action='store_true', default=False)
    parser.add_argument('--develop', action='store_true', default=False)
    
    #! model selection
    parser.add_argument('--AD_model', type=str, default='AT', choices=['AT', 'DC','synaptic','Transformer'])
    
    parser.add_argument('--adjustment', action='store_true', default=True)
    parser.add_argument('--adj_tolerance', type=int, default=50)
    
    parser.add_argument('--scaler', type=str, default='standard', choices=['standard','minmax','normal','robust'])
    parser.add_argument('--share', action='store_true', default=False)
    
    #!
    parser.add_argument('--pretrain_noise', action='store_true', default=False)
    parser.add_argument('--ftr_idx', type=int, default=0)
    
    parser.add_argument('--noise_step', type=int, default=50) # step for training noise
    parser.add_argument('--joint_epochs', type=int, default=5)

    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--pretrain_epoch', type=int, default=1)
    #!!!!!    
    parser.add_argument('--noise_injection', action='store_true', default=False)
    parser.add_argument('--noise_position', type=str, default='front', choices=['front','end','middle'])
    
    
    
    
    parser.add_argument('--pt_energy_loss', action='store_true', default=False)
    parser.add_argument('--imitation_loss', action='store_true', default=False)
    parser.add_argument('--contrastive_loss', action='store_true', default=False)
    parser.add_argument('--f_loss', action='store_true', default=False)
    
    parser.add_argument('--sa_loss_coeff', type=float, default=1.0)
    parser.add_argument('--energy_loss_coeff', type=float, default=1.0)
    parser.add_argument('--pt_energy_loss_coeff', type=float, default=1.0)
    parser.add_argument('--imitation_loss_coeff', type=float, default=1.0)
    parser.add_argument('--contrastive_loss_coeff', type=float, default=1.0)
    parser.add_argument('--f_loss_coeff', type=float, default=1.0)
    
    parser.add_argument('--hp_tuning', action='store_true', default=False)
    
    #! after neurips
    parser.add_argument('--synthetic_injection', action='store_true', default=False)
    parser.add_argument('--only_synthetic_injection', action='store_true', default=False)


    parser.add_argument('--focus', action='store_true', default=False)
    parser.add_argument('--cross_attn', action='store_true', default=False)
    parser.add_argument('--cross_attn_epochs', type=int, default=5)
    parser.add_argument('--cross_attn_nheads', type=int, default=1)
    parser.add_argument('--aafn_loss', type=str, default='MSE', choices=['MSE','BCE'])
    parser.add_argument('--aafn_amplify', action='store_true', default=True)
    parser.add_argument('--amplify_type', type=str, default='5type', choices=['5type','scale2','scale5','scale10'])
    parser.add_argument('--learnable', action='store_true', default=True) 
    parser.add_argument('--test_thresh', action='store_true', default=False)
    parser.add_argument('--test_thresh_type', type=str, default='linear', choices=['linear','var','regu','multiplication'])
    parser.add_argument('--test_thresh_outlier', action='store_true', default=False)
    parser.add_argument('--coeff_energy', type=float, default=1.0)
    parser.add_argument('--coeff_score', type=float, default=1.0) 
    parser.add_argument('--pretrain_loss_type', type=str, default='linear', choices=['linear','var','regu','multiplication'])
    
    parser.add_argument('--shared_layer_list', nargs='+', type=int, default=[0,1,2], help='List of shared layers')
    
    
    parser.add_argument('--sub_datasets', nargs='+', type=str, default=[], help='List of sub datasets')

    parser.add_argument('--forecast_loss', action='store_true', default=False)
    parser.add_argument('--forecast_loss_coeff', type=float, default=1.0) 
    parser.add_argument('--cross_attn_loss_coeff', type=float, default=1.0)
    parser.add_argument('--recon_loss_coeff', type=float, default=1.0)
    parser.add_argument('--af_loss_coeff', type=float, default=1.0)

    return parser


def set_data_config(args):
    channels = {
        "MSL":55,
        "PSM":25,
        "SMAP":25,
        "SMD":38,
        "SWAT":51,
        "WADI":123,
        "kpi": 1,
        "UCR":1,
        "NAB":1,
        "MBA":2,
        "exathlon":19
    }
    args.input_c= args.output_c= args.c_out= args.enc_in= args.dec_in = channels[args.dataset]
    return args