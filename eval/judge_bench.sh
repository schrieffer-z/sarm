python run_judge.py \
    --judge_name reward_model \
    --judge_model /NAS/zhangsy/sarm_models/Llama-3.1-8B-Instruct_token_Latent65536_Layer16_K384_10M-SAE4RM/checkpoint-125 \
    --pairs data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl

python run_judge.py \
    --judge_name reward_model \
    --judge_model /NAS/zhangsy/sarm_models/Llama-3.1-8B-Instruct_token_Latent65536_Layer16_K384_10M-SAE4RM/checkpoint-125 \
    --pairs data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl

python run_judge.py \
    --judge_name reward_model \
    --judge_model /NAS/zhangsy/sarm_models/Llama-3.1-8B-Instruct_token_Latent65536_Layer16_K384_10M-SAE4RM/checkpoint-125 \
    --pairs data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl


CUDA_VISIBLE_DEVICES=2 python judge_bench.py \
    --judge_name reward_model \
    --judge_model /NAS/zhangsy/sarm_models/Llama-3.1-8B-Instruct_token_Latent32768_Layer16_K192_50M-SARM-TopK-Aggregated-1/checkpoint-10 \
    --pairs data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl
