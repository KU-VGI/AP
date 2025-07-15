import random
import numpy as np
import torch
import torch.nn as nn
import copy
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def inject(data):
    device=data.device
    data=data.cpu()
    abnormal_x = copy.deepcopy(data)
    anomaly_types = ["global", "contextual", "seasonal", "trend", "shapelet"]
    bs, win_size, ch = data.shape
    abnormal_l = torch.zeros((bs,win_size))
    pos_n, pos_an = select_abnormal_positions(data, 0.9, continual=True)
    pos_an= torch.tensor(pos_an)#.to(data.device)
    abnormal_l[:,pos_an] = 1
    anomaly_start = torch.min(pos_an).item()
    if ch>10:
        num_anomaly_ch = torch.randint(1, int(ch / 10) + 1, (1,)).item()
    else:
        num_anomaly_ch=1

    anomaly_chs = torch.randint(0, ch, (num_anomaly_ch,))
    for anomaly_ch in anomaly_chs:
        anomaly_type = random.choice(anomaly_types)
        # print(anomaly_type)
        if anomaly_type == "global":
            m, std = data[:, :, anomaly_ch].mean(dim=1), data[:, :, anomaly_ch].std(dim=1)
            g = random.randint(3, 6)
            new_data=np.random.uniform(m.detach().cpu()+g*std.detach().cpu(), m.detach().cpu()-g*std.detach().cpu())
            abnormal_x[:,anomaly_start,anomaly_ch] = torch.tensor(new_data).to(data.device)
            # abnormal_x[:, anomaly_start, anomaly_ch] = torch.empty(bs).uniform_(m + g * std, m - g * std)
        elif anomaly_type == "contextual":
            # m, std = data[:, pos_an, anomaly_ch].mean(dim=1), data[:, pos_an, anomaly_ch].std(dim=1)
            m, std = np.mean(data[:,pos_an,anomaly_ch].detach().cpu().numpy(),axis=1), np.std(data[:,pos_an,anomaly_ch].detach().cpu().numpy(),axis=1)
            x = random.randint(3, 6)
            abnormal_x[:,anomaly_start,anomaly_ch] = torch.tensor(np.random.uniform(m+x*std, m-x*std)).to(data.device)
            # abnormal_x[:, anomaly_start, anomaly_ch] = torch.empty(bs).uniform_(m + x * std, m - x * std)
        elif anomaly_type == "seasonal":
            numbers = [1/3, 1/2, 2, 3]
            n = random.choice(numbers)
            anomaly_indices = torch.arange(len(pos_an))#.to(data.device)
            if n > 1:
                abnormal_x[:, pos_an, anomaly_ch] = data[:, pos_an + (anomaly_indices * n).int() % n, anomaly_ch]
            # else:
                # abnormal_x[:, pos_an, anomaly_ch] = data[:, pos_an + (anomaly_indices * n).int(), anomaly_ch]
        elif anomaly_type == "trend":
            m, std = data[:, :, anomaly_ch].mean(dim=1), data[:, :, anomaly_ch].std(dim=1)
            b = random.randint(3, 6)
            abnormal_x[:, pos_an, anomaly_ch] += b * std[:, None].repeat(1, len(pos_an))
        elif anomaly_type == "shapelet":
            abnormal_x[:, pos_an, anomaly_ch] = data[:, anomaly_start, anomaly_ch].unsqueeze(1).repeat(1, len(pos_an))
        # print(f"{anomaly_type} anomaly:{abnormal_x.shape}")
    
    abnormal_x = abnormal_x.to(device)
    abnormal_l = abnormal_l.to(device)
    return abnormal_x, abnormal_l

def select_abnormal_positions(input_x, w_ab_ratio=0.5, continual=False):
    _,win_size,_ = input_x.shape
    noise_ratio=0
    while int(win_size*noise_ratio) ==0:
        noise_ratio = random.random() * w_ab_ratio # 0~50%
    if continual:
        anomaly_length = int(noise_ratio * win_size)
        start_max = win_size - anomaly_length
        anomaly_start = random.randint(0,start_max)
        anomaly_end = anomaly_start + anomaly_length
        pos_an = np.arange(anomaly_start, anomaly_end)
        pos_n = np.append(np.arange(0,anomaly_start), np.arange(anomaly_end, win_size))       
    else:
        positions_all = np.random.permutation(win_size)
        pos_an = positions_all[:int(win_size*noise_ratio)]
        pos_n = positions_all[int(win_size*noise_ratio):]
    # print(f"pos_an:{list(pos_an)}")
    return pos_n, pos_an


def inject_amplify(data,fe=None,w_ab_ratio=0.2,type="5type"):
    device=data.device
    # data=data.cpu()
    abnormal_x = copy.deepcopy(data)
    anomaly_types = ["global", "contextual", "seasonal", "trend", "shapelet"]
    bs, win_size, ch = data.shape
    abnormal_l = torch.zeros((bs,win_size)) ## label
    
    noise_ratio=0
    while int(win_size*noise_ratio) ==0:
        noise_ratio = random.random() * w_ab_ratio # 0~20%
    anomaly_length = int(noise_ratio * win_size)
    
    criterion = nn.MSELoss(reduction="none")
    _,win_size,_ = data.shape
    
    _,recon = fe(data)
    error = criterion(data, recon) # (bs, win, ch)
    
    # channel_wise_error = torch.sum(error, dim=[0,1]) # (ch)
    # _, max_error_ch = torch.argmax(channel_wise_error) # mse가 최대인 channel 선택
    # print(f"max error channel: {max_error_ch}")
    # max_error_values = error[:,:,max_error_ch] # (bs, win_size) mse가 최대인 channel의 mse error값
    # max_time_step = torch.argmax(max_error_values, dim=1) # (bs) 각 sample 별 특정 channel에서 mse가 최대인 time step 
    # data_ = torch.arange(data.shape[1]*data.shape[2]).reshape(data.shape[1], data.shape[2])
    
    
    
    channel_wise_error = torch.sum(error, dim=1) # (bs, ch)
    max_error_ch = torch.argmax(channel_wise_error,dim=1) # (bs) 각 sample별로 mse가 최대인 channel 선택 ## batch_size
    # print(f"max error channels: {max_error_ch}/{channel_wise_error.shape[1]}") ##
    batch_indices = torch.arange(data.size(0))
    
    # Retrieve the maximum values using advanced indexing
    max_error_values = error[batch_indices, max_error_ch] # (bs, win_size) 각 sample별 mse가 최대인 channel의 mse error값
    max_time_step = torch.argmax(max_error_values, dim=1) # (bs) 각 sample 별 특정 channel에서 mse가 최대인 time step 
    data_ = torch.arange(data.shape[1]*data.shape[2]).reshape(data.shape[1], data.shape[2])

    for idx, (sample,ch_idx, time_idx) in enumerate(zip(data,max_error_ch, max_time_step)):
        pos_an = torch.arange(time_idx-int(anomaly_length/2), time_idx+int(anomaly_length/2))
        pos_an = pos_an[(pos_an >= 0) & (pos_an <= sample.shape[0])]
        anomaly_length=len(pos_an)
        if anomaly_length==0: 
            continue
        abnormal_l[idx,pos_an]=1
        ch_data = sample[ch_idx] # (win)
        
        
        if type=="5type":
            anomaly_type = random.choice(anomaly_types)
            if anomaly_type=="global":
                
                m, std = ch_data.mean(), ch_data.std()
                g = random.randint(3, 6)
                new_data=np.random.uniform(m.detach().cpu()+g*std.detach().cpu(), m.detach().cpu()-g*std.detach().cpu())
                abnormal_x[idx,time_idx,ch_idx] = torch.tensor(new_data).to(data.device)
            elif anomaly_type=="contextual":
                m, std = np.mean(sample[pos_an,ch_idx].detach().cpu().numpy()), np.std(sample[pos_an,ch_idx].detach().cpu().numpy())
                x = random.randint(3, 6)
                abnormal_x[idx,time_idx] = torch.tensor(np.random.uniform(m+x*std, m-x*std)).to(data.device)
            elif anomaly_type=="seasonal":
                numbers = [2, 3,4]
                n = random.choice(numbers)
                anomaly_indices = torch.arange(anomaly_length)#.to(data.device)
                # if n > 1:
                abnormal_x[idx,pos_an,ch_idx] = sample[pos_an + (anomaly_indices * n).int() % n, ch_idx]
                
            elif anomaly_type=="trend":
                m, std = ch_data.mean(), ch_data.std()
                b = random.randint(3, 6)
                abnormal_x[idx,pos_an,ch_idx] += b * std
            elif anomaly_type == "shapelet":
                abnormal_x[idx, pos_an,ch_idx] = sample[time_idx,ch_idx]
        elif type=="scale2":
            abnormal_x[idx,pos_an,ch_idx] = 2*sample[pos_an,ch_idx]
        elif type=="scale5":
            abnormal_x[idx,pos_an,ch_idx] = 5*sample[pos_an,ch_idx]
            
    abnormal_x = abnormal_x.to(device)
    abnormal_l = abnormal_l.to(device)
    return abnormal_x, abnormal_l


def inject_learnable(data, global_scale, contextual_scale, trend_scale, shapelet_scale, seasonal_scale):
    device = data.device
    data = data.to(device)
    data = data + torch.randn_like(data) * 1e-6

    # abnormal_x = copy.deepcopy(data)
    abnormal_x = data.clone()  
    anomaly_types = ["global", "contextual", "seasonal", "trend", "shapelet"]
    bs, win_size, ch = data.shape
    abnormal_l = torch.zeros((bs, win_size), device=device)

    pos_n, pos_an = select_abnormal_positions(data, 0.9, continual=True)
    pos_an = torch.tensor(pos_an, device=device)

    pos_an = torch.clamp(pos_an, 0, win_size - 1)
    abnormal_l[:, pos_an] = 1

    anomaly_start = torch.min(pos_an).item()
    num_anomaly_ch = max(1, ch // 10)
    anomaly_chs = torch.randint(0, ch, (num_anomaly_ch,), device=device)

    for anomaly_ch in anomaly_chs:
        anomaly_type = random.choice(anomaly_types)
        # anomaly_type = "shapelet"

        if anomaly_type == "global":
            m = data[:, :, anomaly_ch].mean(dim=1)
            std = data[:, :, anomaly_ch].std(dim=1, unbiased=False)
            std = torch.where(std > 0, std, torch.tensor(1e-6, device=std.device))
            new_data = torch.normal(m, torch.abs(global_scale) * std).to(data.device)
            abnormal_x[:,anomaly_start,anomaly_ch] = torch.tensor(new_data).to(data.device)

        elif anomaly_type == "contextual":
            m = data[:, pos_an, anomaly_ch].mean(dim=1)
            std = data[:, pos_an, anomaly_ch].std(dim=1, unbiased=False)
            std = torch.where(std > 0, std, torch.tensor(1e-6, device=std.device))
            contextual_scale = torch.abs(contextual_scale)
            lower_bound = m[:, None] - contextual_scale * std[:, None]
            upper_bound = m[:, None] + contextual_scale * std[:, None]

            rand_vals = torch.rand_like(data[:, pos_an, anomaly_ch])  # Random values in [0, 1)
            new_data = lower_bound + rand_vals * (upper_bound - lower_bound)  # Scale to [lower_bound, upper_bound]

            # Inject anomalies
            abnormal_x[:, pos_an, anomaly_ch] = new_data

        elif anomaly_type == "seasonal":
            n = torch.abs(seasonal_scale).clamp(min=0.1, max=3)  # n: 0.1~3

            seasonal_shift = (torch.arange(len(pos_an), device=device) * n).int() % win_size
            base_data = data[:, (pos_an + seasonal_shift) % win_size, anomaly_ch]

            seasonal_noise = torch.abs(seasonal_scale).clamp(min=1e-3)  # Ensure seasonal_noise > 0

            # Generate scale factor for each element
            lower_bound = 1 - seasonal_noise  # Lower bound of scaling
            upper_bound = 1 + seasonal_noise  # Upper bound of scaling
            scale_factor = torch.rand_like(base_data)  # Random values in [0, 1)
            scale_factor = lower_bound + scale_factor * (upper_bound - lower_bound)  # Scale to [1-seasonal_noise, 1+seasonal_noise]

            # Apply scale factor to base data to create anomalies
            new_data = base_data * scale_factor

            # Inject seasonal anomalies
            abnormal_x[:, pos_an, anomaly_ch] = new_data


        elif anomaly_type == "trend":
            m = data[:, :, anomaly_ch].mean(dim=1)
            std = data[:, :, anomaly_ch].std(dim=1, unbiased=False)
            std = torch.where(std > 0, std, torch.tensor(1e-6, device=std.device))

            trend_noise = (trend_scale * std[:, None]).expand(bs, len(pos_an)).to(device)
            for b in range(bs):
                abnormal_x[b, pos_an, anomaly_ch] += trend_noise[b, :len(pos_an)]

        elif anomaly_type == "shapelet":
            shape_noise = shapelet_scale[:, None].expand(bs, len(pos_an)).to(device)
            for b in range(bs):
                abnormal_x[b, pos_an, anomaly_ch] = data[b, anomaly_start, anomaly_ch] + shape_noise[b, :len(pos_an)]

    return abnormal_x, abnormal_l



def inject_amplify_learnable(
    data,
    fe=None,
    type="5type",
    global_scale=1.0,
    contextual_scale=1.0,
    seasonal_scale=1.0,
    trend_scale=1.0,
    shapelet_scale=1.0,
    w_ab_ratio = 0.2
):

    # 1) fe(data) 로부터 MSE를 구해 가장 에러 큰 채널/시점을 골라 pos_an 선정
    # 2) anomaly 유형 랜덤으로 골라 위 scale 파라미터를 사용해 이상치 주입
    
    device = data.device
    abnormal_x = copy.deepcopy(data)
    bs, win_size, ch = data.shape
    abnormal_l = torch.zeros((bs, win_size), device=device)

    criterion = nn.MSELoss(reduction="none")

    _, recon = fe(data)
    error = criterion(data, recon)  

    channel_wise_error = torch.sum(error, dim=1)
    max_error_ch = torch.argmax(channel_wise_error, dim=1)

    batch_indices = torch.arange(bs, device=device)
    max_ch_error = error[batch_indices, :, max_error_ch]
    max_time_step = torch.argmax(max_ch_error, dim=1)

    noise_ratio = 0
    while int(win_size * noise_ratio) == 0:
        noise_ratio = random.random() * w_ab_ratio  # 0~ w_ab_ratio
    anomaly_length = int(noise_ratio * win_size)

    anomaly_types = ["global", "contextual", "seasonal", "trend", "shapelet"]

    for idx, (sample, ch_idx, time_idx) in enumerate(zip(data, max_error_ch, max_time_step)):
        start_ = time_idx - anomaly_length // 2
        end_   = time_idx + anomaly_length // 2
        pos_an = torch.arange(start_, end_+1, device=device)

        # 범위 벗어나지 않게
        pos_an = pos_an[(pos_an >= 0) & (pos_an < sample.shape[0])]
        anomaly_length = len(pos_an)

        if anomaly_length == 0:
            continue

        abnormal_l[idx, pos_an] = 1

        ch_data = sample[:, ch_idx] 

        if type == "5type":
            anomaly_type = random.choice(anomaly_types)
            # anomaly_type = "shapelet"

            if anomaly_type == "global":
                m, std = ch_data.mean(), ch_data.std(unbiased=False)
                std = std if std > 1e-6 else 1e-6
                new_data = torch.normal(m, torch.abs(global_scale) * std).to(device)
                abnormal_x[idx, time_idx, ch_idx] = new_data

            elif anomaly_type == "contextual":
                segment = ch_data[pos_an]
                m, std = segment.mean(), segment.std(unbiased=False)
                std = std if std > 1e-6 else 1e-6

                c_scale = torch.abs(contextual_scale)
                lower_bound = (m - c_scale * std).item()
                upper_bound = (m + c_scale * std).item()

                uniform_vals = torch.empty(anomaly_length, device=device).uniform_(
                    lower_bound, upper_bound
                )
                abnormal_x[idx, pos_an, ch_idx] = uniform_vals

            elif anomaly_type == "seasonal": # (주기적 shift + scale)

                n = torch.abs(seasonal_scale).clamp(min=0.1, max=3)
                shift_indices = torch.arange(anomaly_length, device=device)
                seasonal_shift = (shift_indices * n).int() % sample.shape[0] 

                base_data = ch_data[(pos_an + seasonal_shift) % sample.shape[0]]

                s_noise = torch.abs(seasonal_scale).item()
                s_noise = s_noise if s_noise > 1e-6 else 1e-3
                scale_factor = torch.empty_like(base_data).uniform_(1 - s_noise, 1 + s_noise)
                new_data = base_data * scale_factor
                abnormal_x[idx, pos_an, ch_idx] = new_data

            elif anomaly_type == "trend":
                m, std = ch_data.mean(), ch_data.std(unbiased=False)
                std = std if std > 1e-6 else 1e-6
                trend_val = trend_scale * std
                abnormal_x[idx, pos_an, ch_idx] += trend_val

            elif anomaly_type == "shapelet":
                center_val = ch_data[time_idx]
                if torch.is_tensor(shapelet_scale):
                    if shapelet_scale.ndim == 0 or shapelet_scale.size(0) == 1:  # 스칼라 또는 크기 1 텐서
                        s_val = shapelet_scale.item()
                    elif idx < shapelet_scale.size(0):  # idx가 범위 내에 있는 경우
                        s_val = shapelet_scale[idx].item()
                    else:  # idx 초과 시 기본값
                        s_val = shapelet_scale[0].item()
                else:
                    s_val = shapelet_scale  # 스칼라 값인 경우 그대로 사용

                abnormal_x[idx, pos_an, ch_idx] = center_val + s_val

        elif type == "scale2":
            abnormal_x[idx, pos_an, ch_idx] = 2 * sample[pos_an, ch_idx]

        elif type == "scale5":
            abnormal_x[idx, pos_an, ch_idx] = 5 * sample[pos_an, ch_idx]

    abnormal_x = abnormal_x.to(device)
    abnormal_l = abnormal_l.to(device)
    return abnormal_x, abnormal_l
