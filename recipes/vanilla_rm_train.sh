export NCCL_P2P_DISABLE=1 
export NCCL_IB_DISABLE=1 
export CUDA_VISIBLE_DEVICES=4,5,6,7

accelerate launch \
    --main_process_port 29999 \
    --config_file ./src/accelerate_configs/deepspeed_zero3_4gpu.yaml \
    ./src/vanilla_rm_gemma.py \
    --output_path ??? \
    --model_name ??? \
    --num_train_epochs 3 \
    --learning_rate 2e-6 \
    --per_device_train_batch_size 1 \
    --save_only_model true \
    --max_length 4096 \
    --train_set_path ???
