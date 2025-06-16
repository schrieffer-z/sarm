import os
import subprocess
import multiprocessing
import argparse
from pathlib import Path
import time

# 基础配置
BASE_DIR = Path(os.getcwd()).absolute()
CONFIGS = {
    'judge_bench': {
        'cwd': BASE_DIR / "eval/judge_bench",
        'command_template': "CUDA_VISIBLE_DEVICES={gpu} python judge_bench.py " \
                            "--judge_name reward_model " \
                            "--judge_model {model_path} " \
                            "--pairs data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
    },
    'reward_bench': {
        'cwd': BASE_DIR / "eval/reward_bench",
        'command_template': "CUDA_VISIBLE_DEVICES={gpu} python run_v2.py " \
                            "--batch_size=1 " \
                            "--model {model_path} " \
                            "--dataset /NAS/zhangsy/datasets/allenai/reward-bench-2"
    },
    'rm_bench': {
        'cwd': BASE_DIR / "eval/rm_bench",
        'command_template': "CUDA_VISIBLE_DEVICES={gpu} python run_rm.py " \
                            "--model {model_path} " \
                            "--datapath data/total_dataset.json " \
                            "--batch_size 1 " \
                            "--trust_remote_code " \
                            "--chat_template tulu"
    }
}

def run_eval_task(model_path, benchmark, gpu_id):
    """运行单个评估任务"""
    config = CONFIGS[benchmark]
    cmd = config['command_template'].format(gpu=gpu_id, model_path=model_path)
    
    print(f"🚀 Starting {benchmark} on GPU:{gpu_id} with model: {model_path}")
    print(f"📁 Working dir: {config['cwd']}")
    print(f"💻 Command: {cmd}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=config['cwd'],
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        status = "✅ SUCCESS"
        output = result.stdout
    except subprocess.CalledProcessError as e:
        status = "❌ FAILED"
        output = e.stdout if e.stdout else str(e)
    
    elapsed = time.time() - start_time
    print(f"{status} {benchmark} on {model_path} [{elapsed:.1f}s]\n")
    
    log_entry = {
        'benchmark': benchmark,
        'model': model_path,
        'gpu': gpu_id,
        'status': status,
        'time': elapsed,
        'output': output
    }
    
    return log_entry

def process_checkpoint(model_path):
    """处理单个checkpoint的三个评估任务"""
    print(f"\n{'='*80}\n🏁 Starting evaluation for {model_path}\n{'='*80}")
    gpu_map = {'judge_bench': 0, 'reward_bench': 1, 'rm_bench': 2}
    
    with multiprocessing.Pool(processes=3) as pool:
        futures = []
        for benchmark, gpu_id in gpu_map.items():
            futures.append(
                pool.apply_async(
                    run_eval_task, 
                    args=(model_path, benchmark, gpu_id)
                )
            )
        
        # 等待所有任务完成
        results = [f.get() for f in futures]
    
    print(f"\n{'='*80}\n🏁 Completed evaluation for {model_path}\n{'='*80}\n")
    return results

def main(model_paths, log_file):
    """主处理函数"""
    all_results = []
    
    # 处理所有checkpoints
    for model_path in model_paths:
        results = process_checkpoint(model_path)
        all_results.extend(results)
    
    # 保存日志
    save_logs(all_results, log_file)

def save_logs(logs, log_file):
    """保存评估日志到文件"""
    with open(log_file, 'w') as f:
        f.write("Evaluation Report\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Evaluated {len(logs)//3} checkpoint(s): {', '.join({log['model'] for log in logs})}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 100 + "\n")
        f.write("MODEL PATH | BENCHMARK | GPU | STATUS | TIME(s) | OUTPUT\n")
        f.write("-" * 100 + "\n")
        
        for log in logs:
            # 缩短长路径显示
            model_display = log['model'] if len(log['model']) < 60 else log['model'][:30] + "..." + log['model'][-27:]
            
            # 创建输出摘要
            output_summary = log['output'].strip()
            if len(output_summary) > 500:
                output_summary = output_summary[:200] + f"\n... [TRUNCATED: {len(log['output'])} chars total] ...\n" + output_summary[-200:]
            
            f.write(
                f"{model_display} | {log['benchmark']:12} | {log['gpu']} | "
                f"{log['status']} | {log['time']:7.1f} | "
                f"{output_summary}\n"
            )
            f.write("-" * 100 + "\n")
    
    print(f"\n✅ Evaluation completed! Results saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch evaluation of checkpoints on three benchmarks.')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='List of checkpoint paths to evaluate (supports glob patterns)')
    parser.add_argument('--log', default='eval_report.txt',
                        help='Log file to save evaluation results (default: eval_report.txt)')
    
    args = parser.parse_args()
    
    # 展开可能的glob模式
    expanded_paths = []
    for pattern in args.checkpoints:
        if '*' in pattern or '?' in pattern:
            expanded_paths.extend(sorted([str(p) for p in Path().glob(pattern)]))
        else:
            expanded_paths.append(pattern)
    
    if not expanded_paths:
        print("⚠️ No valid checkpoints found. Exiting.")
        exit(1)
    
    print(f"📋 Found {len(expanded_paths)} checkpoint(s) to evaluate:")
    for path in expanded_paths:
        print(f"  - {path}")
    
    main(expanded_paths, args.log)


# import json
# import csv
# import os
# from collections import defaultdict
# from glob import glob

# # 设置结果目录和输出文件
# base_dir = "/path/to/your/checkpoints"  # 更改为您的目录路径
# output_file = "all_benchmark_results.csv"

# # 使用defaultdict自动创建嵌套字典
# results = defaultdict(lambda: defaultdict(dict))

# # 遍历所有checkpoint目录
# for checkpoint_dir in glob(os.path.join(base_dir, "checkpoint-*")):
#     checkpoint_id = os.path.basename(checkpoint_dir)
    
#     # 每个检查点的三种benchmark文件
#     benchmark_files = {
#         "judge": "judgebench.json",
#         "reward": "reward_benchv2.json",
#         "rm": "rm_bench.json"
#     }
    
#     # 读取并处理每个benchmark文件
#     for bench_type, filename in benchmark_files.items():
#         filepath = os.path.join(checkpoint_dir, filename)
        
#         if os.path.exists(filepath):
#             with open(filepath, "r") as f:
#                 try:
#                     data = json.load(f)
#                     # 将文件中的model字段统一设置为checkpoint目录名
#                     if data.get("model"):
#                         data["model"] = checkpoint_id
#                     # 移除模型类型等可能冲突的键
#                     data.pop("model_type", None)
#                     data.pop("chat_template", None)
                    
#                     # 将数据存入结果字典，加上前缀防止键冲突
#                     for key, value in data.items():
#                         if key != "model":  # 跳过模型名称键
#                             # 添加benchmark类型前缀
#                             new_key = f"{bench_type}_{key}"
#                             results[checkpoint_id][new_key] = value
#                 except json.JSONDecodeError:
#                     print(f"Error decoding JSON in {filepath}")
#                 except Exception as e:
#                     print(f"Error processing {filepath}: {str(e)}")
#         else:
#             print(f"File not found: {filepath}")

# # 准备CSV输出
# csv_rows = []
# all_columns = set()

# # 收集所有唯一列名
# for checkpoint, benchmarks in results.items():
#     # 添加基准列：模型名称
#     row = {"model": checkpoint}
    
#     # 添加各benchmark结果
#     for col, value in benchmarks.items():
#         row[col] = value
#         all_columns.add(col)
    
#     csv_rows.append(row)

# # 确保所有列名按字母顺序排序
# all_columns = sorted(all_columns)
# csv_columns = ["model"] + all_columns

# # 写入CSV文件
# with open(output_file, "w", newline="") as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#     writer.writeheader()
    
#     for row_dict in csv_rows:
#         # 确保每一行包含所有列，缺失值设为空字符串
#         full_row = {col: row_dict.get(col, "") for col in csv_columns}
#         writer.writerow(full_row)

# print(f"✅ 所有benchmark结果已保存至: {os.path.abspath(output_file)}")
# print(f"共处理了 {len(csv_rows)} 个模型")