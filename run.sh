AD_model=AT
model=PatchTST 
joint_epochs=5
cross_attn_epochs=5
ftr_idx=0

d_model=256

contrastive_loss_coeff=1.0
forecast_loss_coeff=1.0
cross_attn_loss_coeff=1.0
recon_loss_coeff=1.0
af_loss_coeff=1.0
prompt_num=3
pool_size=10
top_k=3
cross_attn_nhead=1

data_path=/local_datasets/AD_datasets/MBA
dataset=MBA
noise_step=100
seq_len=100
pred_len=100
win_size=100
step=$pred_len

python -u run.py \
--random_seed 0 \
--root_path $data_path \
--dataset $dataset \
--model_id F+AD_${seq_len}_${pred_len} \
--seq_len $seq_len --pred_len $pred_len --win_size $win_size \
--step $step --noise_step $noise_step \
--joint_epochs $joint_epochs \
--share \
--AD_model $AD_model \
--d_model $d_model \
--noise_injection \
--pretrain_noise \
--contrastive_loss  \
--forecast_loss \
--cross_attn --cross_attn_epochs $cross_attn_epochs \
--cross_attn_nheads $cross_attn_nhead \
--ftr_idx $ftr_idx

