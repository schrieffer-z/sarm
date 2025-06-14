export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=5


# accelerate launch \
#     --main_process_port 29193 \
#     --config_file ./accelerate_configs/deepspeed_zero0_reward_bench_1gpu.yaml \
#     ./run_v2.py \
#     --batch_size=4 \
#     --model /NAS/zhangsy/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/ \
#     --dataset /NAS/zhangsy/datasets/allenai/reward-bench-2 \
#     --tokenizer /NAS/zhangsy/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/

python run_v2.py \
    --batch_size=4 \
    --model /NAS/zhangsy/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/ \
    --dataset /NAS/zhangsy/datasets/allenai/reward-bench-2 \
    --tokenizer /NAS/zhangsy/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/
