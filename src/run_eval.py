



import os
import re
import argparse
import subprocess
import time
import pandas as pd
from pathlib import Path


def process_judgebench_results(base_dir, checkpoint_dir, process):
    return_code = process.wait()

    s = ""
    for line in process.stdout:
        s += line.strip()

    # 检查执行结果
    if return_code == 0:
        pattern = r'(mmlu-pro|livebench-reasoning|livebench-math|livecodebench|Overall):\s*(\d+\.\d+)%'
        matches = re.findall(pattern, s)

        # 提取结果存储到字典
        metrics_dict = {}
        for name, value in matches:
            metrics_dict[name] = float(value)

        results_leaderboard = pd.DataFrame(dict({
                'Name': [checkpoint_dir], 
                'Overall': [metrics_dict['Overall']], 
                'Knowledge': [metrics_dict['mmlu-pro']], 
                'Reasoning': [metrics_dict['livebench-reasoning']], 
                'Math': [metrics_dict['livebench-math']], 'Coding': [metrics_dict['livecodebench']],
            }))
        lpath = os.path.join(base_dir,'leaderboard.csv')
        
        print("saving leaderboard to :",lpath)
        lboard = None
        if not os.path.exists(lpath):
            lboard=pd.DataFrame(dict({
                'Name': ['Random'], 'Overall': [50.0], 'Knowledge': [50.0], 'Reasoning': [50.0], 'Math': [50.0], 'Coding': [50.0],
            }))
        else:
            lboard = pd.read_csv(lpath, index_col=0)
        df_to_save = pd.concat([lboard, results_leaderboard]).reset_index(drop=True)
        df_to_save.to_csv(lpath)
    else:
        print(f"❌ 执行失败: {checkpoint_dir} (返回码: {return_code})")






def run_batch_evaluations(base_dir, pairs_file, checkpoint_prefix="checkpoint-"):
    """
    批量运行给定目录中的所有checkpoint
    
    Args:
        base_dir: 包含所有checkpoint的目录
        pairs_file: 评估数据文件路径
        checkpoint_prefix: checkpoint目录的前缀，默认"checkpoint-"
    """
    # 获取所有checkpoint目录
    all_items = os.listdir(base_dir)
    checkpoint_dirs = sorted([
        d for d in all_items 
        if d.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(base_dir, d))
    ])
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda s: int(re.search(r'-(\d+)$', s).group(1)))
    
    if not checkpoint_dirs:
        print(f"错误：在 {base_dir} 中未找到任何以'{checkpoint_prefix}'开头的checkpoint目录")
        return
    
    print(f"找到 {len(checkpoint_dirs)} 个checkpoint，开始批量评估...")
    
    # 处理每个checkpoint
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        checkpoint_path = os.path.join(base_dir, checkpoint_dir)
        print(f"\n===== 处理checkpoint {i+1}/{len(checkpoint_dirs)}: {checkpoint_dir} =====")
            
        ##################################### Judge Bench #####################################
        env_vars = os.environ.copy()
        env_vars.update({
            "CUDA_VISIBLE_DEVICES": "0"  # 使用前两个GPU
        })
        judgebench = subprocess.Popen(
            [
                "python", "run_judge.py",
                "--judge_name", "reward_model",
                "--judge_model", checkpoint_path,
                "--pairs", "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
            ], 
            cwd='/data/zhangsy/JudgeBench/',
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env_vars
        )
        #####################################  RM Bench  ######################################
        # env_vars = os.environ.copy()
        # env_vars.update({
        #     "CUDA_VISIBLE_DEVICES": "1"  # 使用前两个GPU
        # })
        # judgebench = subprocess.Popen(
        #     [
        #         "python", "run_judge.py",
        #         "--judge_name", "reward_model",
        #         "--judge_model", checkpoint_path,
        #         "--pairs", "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
        #     ], 
        #     cwd='/data/zhangsy/JudgeBench/',
        #     stdout=subprocess.PIPE, 
        #     stderr=subprocess.STDOUT,
        #     universal_newlines=True,
        #     env=env_vars
        # )
        ################################### Reward Bench V2 ###################################
        # env_vars = os.environ.copy()
        # env_vars.update({
        #     "CUDA_VISIBLE_DEVICES": "3"  # 使用前两个GPU
        # })
        # judgebench = subprocess.Popen(
        #     [
        #         "python", "run_judge.py",
        #         "--judge_name", "reward_model",
        #         "--judge_model", checkpoint_path,
        #         "--pairs", "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
        #     ], 
        #     cwd='/data/zhangsy/JudgeBench/',
        #     stdout=subprocess.PIPE, 
        #     stderr=subprocess.STDOUT,
        #     universal_newlines=True,
        #     env=env_vars
        # )

        ###################################  wait & process  ##################################
        process_judgebench_results(base_dir, checkpoint_dir, judgebench)

    print("\n所有checkpoint处理完成!")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量运行checkpoint评估')
    parser.add_argument('--model_dir', type=str, required=True, 
                        help='包含所有checkpoint的模型目录')
    parser.add_argument('--pairs', type=str, default="data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
                        help='评估数据文件路径')
    parser.add_argument('--prefix', type=str, default="checkpoint-",
                        help='checkpoint目录前缀')
    
    args = parser.parse_args()
    
    # 检查模型目录是否存在
    if not os.path.isdir(args.model_dir):
        print(f"错误: 模型目录不存在: {args.model_dir}")
        exit(1)

    run_batch_evaluations(
        base_dir=args.model_dir,
        pairs_file=args.pairs,
        checkpoint_prefix=args.prefix
    )