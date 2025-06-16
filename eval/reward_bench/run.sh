CUDA_VISIBLE_DEVICES=5 python eval_rewardbench_sarm_llama.py \
    --batch_size=1 \
    --model /NAS/zhangsy/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --tokenizer /NAS/zhangsy/models/Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --dataset /NAS/zhangsy/datasets/allenai/reward-bench-2 \
