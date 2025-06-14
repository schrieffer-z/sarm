CUDA_VISIBLE_DEVICES=4 python run_rm.py \
    --model /NAS/zhangsy/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2/ \
    --datapath data/total_dataset.json \
    --batch_size 1 \
    --trust_remote_code \
    --chat_template tulu