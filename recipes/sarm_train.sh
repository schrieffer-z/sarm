export HF_HOME=/data/zhangsy/.cache
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1 

export CUDA_VISIBLE_DEVICES=0,1,2,3
config_name=deepspeed_zero3_4gpu.yaml

# 1:2e-6
learning_rate=8e-6
per_device_train_batch_size=4
num_train_epochs=3

sae4rm_base_model=llama
model_name=/NAS/zhangsy/models/meta/Llama-3.2-3B-Instruct
base_path=/data/zhangsy/sae/SAE_models/
sae_paths+=(...)



for sae_path in ${sae_paths[@]}; do
    echo $base_path$sae_path
done
for sae_path in ${sae_paths[@]}; do
    output_path=./models/$sae_path-SARM
    
    echo using sae parameters from $sae_path
    echo run reward modeling from base model: $model_name
    echo ckpt will be saved in $output_path
    
    accelerate launch \
        --main_process_port 29994 \
        --config_file ./src/accelerate_configs/$config_name \
        ./src/bradley_terry.py \
        --output_path $output_path \
        --model_name $model_name \
        --save_only_model true \
        --max_length 4096 \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size $per_device_train_batch_size \
        --train_set_path /mnt/finder/lisihang/xAI-RLHF/Shuyi/datasets/Skywork-Reward-Preference-80K-v0.2/data/train-00000-of-00001.parquet \
        --sae_path $base_path$sae_path.pt \
        --sarm_base_model $sae4rm_base_model \
        --sarm_use_baseline false \
        --sarm_use_topk true
done
