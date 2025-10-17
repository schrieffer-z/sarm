export CUDA_VISIBLE_DEVICES=0,1,2,3
config_name=deepspeed_zero3_4gpu.yaml

# gbs:lr = 2:1e-6 (1*4:2e-6)
per_device_train_batch_size=1
learning_rate=2e-6
num_train_epochs=3

# base model path & output path
model_name=Schrieffer/Llama-SARM-4B-PostSAEPretrain
output_path=models/Llama-SARM-4B
train_set_path=

# sarm config
sae_use_sequence_level=true
sarm_use_activation=true
# sarm_train_mode=3: sae will be updated by preference loss and reconstruction loss
# sarm_train_mode=1: sae won't not be updated
sarm_train_mode=1
sarm_rec_lambda=0


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
    --sae_use_sequence_level $sae_use_sequence_level \
    --sarm_use_activation $sarm_use_activation \
    --sarm_train_mode $sarm_train_mode \
    --sarm_rec_lambda $sarm_rec_lambda