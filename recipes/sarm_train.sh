export HF_HOME=/data/zhangsy/.cache
export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1 

export CUDA_VISIBLE_DEVICES=0,1,2,3
config_name=deepspeed_zero3_4gpu.yaml

# 1:2e-6
learning_rate=4e-6
per_device_train_batch_size=1
num_train_epochs=3

# base model path & output path
sarm_base_model=llama
model_name=
sae_path=
output_path=
train_set_path=

# sarm config
sae_use_sequence_level=false
sarm_use_baseline=false
sarm_use_topk=false
sarm_train_mode=3
sarm_rec_lambda=1


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
    --train_set_path $train_set_path \
    --sae_path $sae_path.pt \
    --sarm_base_model $sarm_base_model \
    --sae_use_sequence_level $sae_use_sequence_level \
    --sarm_use_baseline $sarm_use_baseline \
    --sarm_use_topk $sarm_use_topk \
    --sarm_train_mode $sarm_train_mode \
    --sarm_rec_lambda $sarm_rec_lambda