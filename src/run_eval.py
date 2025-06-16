import os
import re
import argparse
import subprocess
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 配置评估任务目录
BENCH_DIRS = {
    'judge_bench': '/data/zhangsy/JudgeBench/',
    'reward_bench': '/data/zhangsy/eval/reward_bench/',
    'rm_bench': '/data/zhangsy/eval/rm_bench/'
}

def run_benchmark(checkpoint_path, benchmark, gpu_id):
    """运行单个评估任务并返回输出"""
    try:
        command = []
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        if benchmark == 'judge_bench':
            command = [
                "python", "run_judge.py",
                "--judge_name", "reward_model",
                "--judge_model", checkpoint_path,
                "--pairs", "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
            ]
        elif benchmark == 'reward_bench':
            command = [
                "python", "run_v2.py",
                "--batch_size=1",
                "--model", checkpoint_path,
                "--dataset", "/NAS/zhangsy/datasets/allenai/reward-bench-2"
            ]
        elif benchmark == 'rm_bench':
            command = [
                "python", "run_rm.py",
                "--model", checkpoint_path,
                "--datapath", "data/total_dataset.json",
                "--batch_size", "1",
                "--trust_remote_code",
                "--chat_template", "tulu"
            ]
        
        # 运行命令并捕获输出
        result = subprocess.run(
            command,
            cwd=BENCH_DIRS[benchmark],
            env=env,
            capture_output=True,
            text=True
        )
        return result.stdout
    
    except Exception as e:
        print(f"❌ {benchmark}执行失败: {checkpoint_path}\n{str(e)}")
        return None

def parse_judge_output(output):
    """解析judge_bench的输出"""
    metrics = {}
    pattern = r'([\w-]+):\s*([\d.]+)%'
    matches = re.findall(pattern, output)
    
    for name, value in matches:
        metrics[name] = float(value)
    
    # 确保关键指标存在
    for metric in ['mmlu-pro', 'livebench-reasoning', 'livebench-math', 'livecodebench', 'Overall']:
        metrics.setdefault(metric, None)
    
    # 添加指标组
    metrics['Knowledge'] = metrics['mmlu-pro']
    metrics['Math'] = metrics['livebench-math']
    metrics['Coding'] = metrics['livecodebench']
    metrics['Reasoning'] = metrics['livebench-reasoning']
    
    return metrics

def parse_reward_output(output):
    """解析reward_bench的输出"""
    metrics = {}
    pattern = r'(\w[\w\s]+):\s*[\d\.]+\/[\d.]+\s*$([\d.]+)$'
    matches = re.findall(pattern, output)
    
    for name, value in matches:
        metrics[name.strip()] = float(value)
    
    # 提取总分数
    ties_match = re.search(r'Ties: Overall score ([\d.]+)', output)
    if ties_match:
        metrics['Ties_Overall'] = float(ties_match.group(1))
    
    # 确保关键指标存在
    for metric in ['Factuality', 'Focus', 'Math', 'Precise IF', 'Safety', 'Ties_Overall']:
        metrics.setdefault(metric, None)
    
    return metrics

def parse_rm_output(output):
    """解析rm_bench的输出"""
    metrics = {}
    pattern = r"'(\w+)': ([\d.]+)"
    matches = re.findall(pattern, output)
    
    for name, value in matches:
        metrics[name] = float(value)
    
    # 确保关键指标存在
    for metric in ['chat', 'math', 'code', 'safety', 'hard_acc', 'normal_acc', 'easy_acc', 'total_avg_acc']:
        metrics.setdefault(metric, None)
    
    return metrics

def run_evaluation_for_checkpoint(base_dir, checkpoint_dir, checkpoint_idx, total_checkpoints):
    """运行一个checkpoint的所有评估任务并保存结果"""
    checkpoint_path = os.path.join(base_dir, checkpoint_dir)
    print(f"\n🚀 处理checkpoint {checkpoint_idx+1}/{total_checkpoints}: {checkpoint_dir}")
    
    # 并行运行所有评估任务
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_benchmark, checkpoint_path, 'judge_bench', 0): 'judge_bench',
            executor.submit(run_benchmark, checkpoint_path, 'reward_bench', 1): 'reward_bench',
            executor.submit(run_benchmark, checkpoint_path, 'rm_bench', 2): 'rm_bench'
        }
        
        all_metrics = {}
        
        for future in futures:
            benchmark = futures[future]
            output = future.result()
            
            if output is None:
                print(f"⚠️ {benchmark}没有输出: {checkpoint_path}")
                continue
            
            try:
                if benchmark == 'judge_bench':
                    metrics = parse_judge_output(output)
                elif benchmark == 'reward_bench':
                    metrics = parse_reward_output(output)
                elif benchmark == 'rm_bench':
                    metrics = parse_rm_output(output)
                
                all_metrics.update({f"{benchmark}_{k}": v for k, v in metrics.items()})
                print(f"✅ {benchmark}解析完成: {checkpoint_dir}")
                
            except Exception as e:
                print(f"❌ {benchmark}解析失败: {checkpoint_dir}\n{str(e)}")
        
        # 保存结果
        if all_metrics:
            all_metrics['Checkpoint'] = checkpoint_dir
            
            # 保存到汇总文件
            summary_file = os.path.join(base_dir, 'all_benchmarks_summary.csv')
            if os.path.exists(summary_file):
                df = pd.read_csv(summary_file, index_col=0)
                df = pd.concat([df, pd.DataFrame([all_metrics])], ignore_index=True)
            else:
                df = pd.DataFrame([all_metrics])
            
            df.to_csv(summary_file)
            print(f"📊 结果已保存到 {summary_file}")
            
            return True
    
    return False

def run_batch_evaluations(base_dir, checkpoint_prefix="checkpoint-"):
    """批量运行所有checkpoint的评估"""
    # 创建结果目录
    results_dir = os.path.join(base_dir, 'benchmark_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取所有checkpoint目录
    checkpoint_dirs = sorted(
        d for d in os.listdir(base_dir)
        if d.startswith(checkpoint_prefix) and os.path.isdir(os.path.join(base_dir, d))
    )
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda s: int(re.search(r'-(\d+)$', s).group(1)))
    
    if not checkpoint_dirs:
        print(f"错误：在 {base_dir} 中未找到以'{checkpoint_prefix}'开头的checkpoint目录")
        return
    
    total = len(checkpoint_dirs)
    print(f"🔍 找到 {total} 个checkpoint，开始批量评估...")
    
    # 处理每个checkpoint
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        run_evaluation_for_checkpoint(base_dir, checkpoint_dir, i, total)
    
    print("\n🎉 所有checkpoint处理完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量运行checkpoint评估')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='包含所有checkpoint的模型目录')
    parser.add_argument('--prefix', type=str, default="checkpoint-",
                        help='checkpoint目录前缀')
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"错误: 模型目录不存在: {args.model_dir}")
        exit(1)

    run_batch_evaluations(
        base_dir=args.model_dir,
        checkpoint_prefix=args.prefix
    )