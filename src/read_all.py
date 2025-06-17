import os
import json
import csv
import argparse
import re
from glob import glob

def extract_checkpoint_number(checkpoint_name):
    """从 checkpoint 名称中提取数字部分"""
    match = re.search(r'checkpoint-(\d+)', checkpoint_name)
    if match:
        return int(match.group(1))
    return -1  # 如果不是有效的 checkpoint 名称

def calculate_reward_average(reward_data):
    """计算 reward bench 的平均分（原始分数）"""
    scores = []
    for field in ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties"]:
        if field in reward_data:
            value = reward_data[field]
            if 0 <= value <= 1:
                scores.append(value * 100)  # 转换为百分比形式
            else:
                scores.append(value)  # 已为百分比形式
    return sum(scores) / len(scores) if scores else 0

def convert_rm_to_percentage(rm_data):
    """将 rm bench 的结果转换为百分比形式"""
    converted = {}
    for key, value in rm_data.items():
        if isinstance(value, (int, float)) and key != "model":
            if 0 <= value <= 1:
                converted[key] = value * 100
            else:
                converted[key] = value
        else:
            converted[key] = value
    return converted

def aggregate_results(base_dir):
    """汇总所有 checkpoint 的结果并计算整体得分"""
    # 获取所有 checkpoint 目录并按数字排序
    checkpoint_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and "checkpoint-" in item:
            checkpoint_dirs.append(item_path)
    
    # 按 checkpoint 数字排序
    checkpoint_dirs.sort(key=lambda x: extract_checkpoint_number(os.path.basename(x)))
    
    all_results = []
    checkpoint_count = 0
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = os.path.basename(checkpoint_dir)
        print(f"处理 checkpoint: {checkpoint_name}")
        
        result = {"checkpoint": checkpoint_name}
        scores = []
        benchmark_data = {}
        
        try:
            # Judge Bench 结果
            judge_path = os.path.join(checkpoint_dir, "judge_bench.json")
            if os.path.exists(judge_path):
                with open(judge_path) as f:
                    judge_data = json.load(f)
                    result.update({
                        "judge_Coding": judge_data.get("Coding", 0),
                        "judge_Knowledge": judge_data.get("Knowledge", 0),
                        "judge_Math": judge_data.get("Math", 0),
                        "judge_Overall": judge_data.get("Overall", 0),
                        "judge_Reasoning": judge_data.get("Reasoning", 0)
                    })
                    scores.append(judge_data.get("Overall", 0))
                    benchmark_data["judge"] = judge_data
            
            # Reward Bench v2 结果
            reward_path = os.path.join(checkpoint_dir, "reward_benchv2.json")
            if os.path.exists(reward_path):
                with open(reward_path) as f:
                    reward_data = json.load(f)
                    reward_avg = calculate_reward_average(reward_data)
                    result["reward_average"] = reward_avg
                    scores.append(reward_avg)
                    
                    # 单独字段
                    result.update({
                        "reward_Factuality": reward_data.get("Factuality", 0) * 100,
                        "reward_Focus": reward_data.get("Focus", 0) * 100,
                        "reward_Math": reward_data.get("Math", 0) * 100,
                        "reward_Precise IF": reward_data.get("Precise IF", 0) * 100,
                        "reward_Safety": reward_data.get("Safety", 0) * 100,
                        "reward_Ties": reward_data.get("Ties", 0) * 100
                    })
                    benchmark_data["reward"] = reward_data
            
            # RM Bench 结果
            rm_path = os.path.join(checkpoint_dir, "rm_bench.json")
            if os.path.exists(rm_path):
                with open(rm_path) as f:
                    rm_data = json.load(f)
                    rm_converted = convert_rm_to_percentage(rm_data)
                    result["rm_total_avg_acc"] = rm_converted.get("total_avg_acc", 0)
                    scores.append(rm_converted.get("total_avg_acc", 0))
                    
                    # 单独字段
                    result.update({
                        "rm_chat": rm_converted.get("chat", 0),
                        "rm_code": rm_converted.get("code", 0),
                        "rm_easy_acc": rm_converted.get("easy_acc", 0),
                        "rm_hard_acc": rm_converted.get("hard_acc", 0),
                        "rm_math": rm_converted.get("math", 0),
                        "rm_normal_acc": rm_converted.get("normal_acc", 0),
                        "rm_safety": rm_converted.get("safety", 0)
                    })
                    benchmark_data["rm"] = rm_data
            
            # 计算三个 benchmark 的平均分
            if scores:
                result["benchmark_average"] = sum(scores) / len(scores)
            
            all_results.append(result)
            checkpoint_count += 1
            
        except Exception as e:
            print(f"处理 {checkpoint_name} 时出错: {str(e)}")
    
    print(f"成功处理 {checkpoint_count} 个 checkpoint")
    return all_results

def save_results_as_csv(results, base_dir):
    """将结果保存为 CSV 文件（完全匹配图片中的表格结构）"""
    if not results:
        return None
    
    # 定义完全匹配图片的列顺序
    fixed_order = [
        "benchmark_average", 

        # Judge Bench 部分
        "judge_Overall",
        "judge_Knowledge",
        "judge_Reasoning",
        "judge_Math",
        "judge_Coding",
        
        # RM Bench 部分
        "rm_total_avg_acc",  # Overall
        "rm_chat",            # Chat
        "rm_math",            # Math
        "rm_code",            # Code
        "rm_safety",          # Safety
        "rm_hard_acc", 
        "rm_normal_acc", 
        "rm_easy_acc",
        
        # Reward Bench V2 部分
        "reward_average",     # Overall
        "reward_Factuality",  # Factuality
        "reward_Precise IF",  # Precise IF
        "reward_Math",        # Math
        "reward_Safety",      # Safety
        "reward_Focus",       # Focus
        "reward_Ties",        # Ties
    ]
    
    # 在最前面添加 checkpoint 列
    final_order = ["checkpoint"] + fixed_order
    
    # 创建 CSV 文件
    output_path = os.path.join(base_dir, "benchmark_results.csv")
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_order)
        writer.writeheader()
        
        # 只写入最终顺序中存在的字段
        for result in results:
            filtered_result = {k: v for k, v in result.items() if k in final_order}
            writer.writerow(filtered_result)
    
    print(f"结果已保存到: {output_path}")
    return output_path

def calculate_and_add_overall_average(results, output_path):
    """计算所有 checkpoint 的平均值并添加到 CSV"""
    if not results:
        return {}
    
    # 定义字段顺序（匹配 CSV 文件的顺序）
    final_order = [
        # Judge Bench
        "judge_Overall", "judge_Knowledge", "judge_Reasoning", "judge_Math", "judge_Coding",
        
        # RM Bench
        "rm_total_avg_acc", "rm_chat", "rm_math", "rm_code", "rm_safety","rm_hard_acc", "rm_normal_acc", "rm_easy_acc"
        
        # Reward Bench V2
        "reward_average", "reward_Factuality", "reward_Precise IF", "reward_Math", "reward_Safety", "reward_Focus", "reward_Ties"
    ]
    
    # 获取列名集合
    header_fields = ["checkpoint"] + final_order
    
    # 计算总体平均值
    overall_avg = {}
    for field in final_order:
        if field in results[0]:
            values = [res.get(field, 0) for res in results]
            overall_avg[field] = sum(values) / len(values) if values else 0
    
    # 准备要添加的行（只包含最终顺序中的字段）
    avg_row = {"checkpoint": "Overall Average"}
    for field in final_order:
        avg_row[field] = overall_avg.get(field, "")
    
    # 添加空行和平均值行到 CSV
    with open(output_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_fields)
        writer.writerow({})  # 添加空行
        writer.writerow(avg_row)
    
    print(f"已添加总体平均值到: {output_path}")
    return overall_avg


def visualize_comparison(results, overall_avg, base_dir):
    """生成可视化分析图表（英文版）"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # 创建 DataFrame
        df = pd.DataFrame(results)
        
        # 提取 checkpoint 数字用于排序
        df['checkpoint_num'] = df['checkpoint'].apply(extract_checkpoint_number)
        df = df.sort_values('checkpoint_num')
        
        # 设置输出目录
        output_dir = os.path.join(base_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. benchmark 平均分比较（按 checkpoint 顺序）
        plt.figure(figsize=(12, 6))
        if "benchmark_average" in df.columns:
            plt.plot(df["checkpoint_num"], df["benchmark_average"], "o-", markersize=8)
            plt.axhline(y=overall_avg.get("benchmark_average", 0), color="r", linestyle="--", 
                       label=f"Average Value ({overall_avg.get('benchmark_average', 0):.2f}%)")
            
            # 标记最高点
            max_index = df["benchmark_average"].idxmax()
            max_value = df.loc[max_index, "benchmark_average"]
            plt.annotate(
                f'Max: {max_value:.2f}%', 
                xy=(df.loc[max_index, "checkpoint_num"], max_value),
                xytext=(10, 20), 
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color="blue")
            )
            
            plt.title("Model Performance by Training Steps")
            plt.xlabel("Training Steps")
            plt.ylabel("Average Score (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "benchmark_trend_by_checkpoint.png"))
            plt.close()
        
        # 2. 各个 benchmark 详细分项比较
        plt.figure(figsize=(15, 8))
        if "judge_Overall" in df.columns and "reward_average" in df.columns and "rm_total_avg_acc" in df.columns:
            plt.plot(df["checkpoint_num"], df["judge_Overall"], "o-", markersize=6, label="Judge Bench")
            plt.plot(df["checkpoint_num"], df["reward_average"], "o-", markersize=6, label="Reward Bench")
            plt.plot(df["checkpoint_num"], df["rm_total_avg_acc"], "o-", markersize=6, label="RM Bench")
            
            # 添加平均线
            plt.axhline(y=overall_avg.get("judge_Overall", 0), color="r", linestyle="--", 
                       alpha=0.7, label=f"Judge Avg ({overall_avg.get('judge_Overall', 0):.2f}%)")
            plt.axhline(y=overall_avg.get("reward_average", 0), color="g", linestyle="--", 
                       alpha=0.7, label=f"Reward Avg ({overall_avg.get('reward_average', 0):.2f}%)")
            plt.axhline(y=overall_avg.get("rm_total_avg_acc", 0), color="b", linestyle="--", 
                       alpha=0.7, label=f"RM Avg ({overall_avg.get('rm_total_avg_acc', 0):.2f}%)")
            
            plt.title("Benchmark Scores by Training Steps")
            plt.xlabel("Training Steps")
            plt.ylabel("Score (%)")
            plt.legend(ncol=2, loc='upper left', bbox_to_anchor=(0, 1))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "detailed_benchmark_trends.png"))
            plt.close()
        
        # 3. Judge Bench 详细指标分析
        plt.figure(figsize=(14, 8))
        judge_cols = [col for col in df.columns if col.startswith("judge_") and col != "judge_Overall"]
        if judge_cols:
            # 提取详细指标
            judge_metrics = ["Coding", "Knowledge", "Math", "Reasoning"]
            judge_df = df[["checkpoint_num"]].copy()
            for metric in judge_metrics:
                col_name = f"judge_{metric}"
                if col_name in df.columns:
                    judge_df[metric] = df[col_name]
            
            # 创建图表
            ax = plt.subplot(111)
            for metric in judge_metrics:
                if metric in judge_df.columns:
                    ax.plot(judge_df["checkpoint_num"], judge_df[metric], 
                            marker='o', markersize=5, label=metric)
            
            # 添加总体平均线
            if "judge_Overall" in df.columns:
                ax.plot(df["checkpoint_num"], df["judge_Overall"], "ko--", 
                       linewidth=2, markersize=8, label="Overall Score")
            
            plt.title("Judge Bench Metrics Analysis")
            plt.xlabel("Training Steps")
            plt.ylabel("Score (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "judge_bench_metrics_analysis.png"))
            plt.close()
        
        print(f"Generated analysis charts in: {output_dir}")
        
    except ImportError:
        print("Matplotlib library missing, skipping chart generation. Please install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating charts: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='汇总所有 checkpoint 的评估结果')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='包含所有 checkpoint 目录的基础目录')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.base_dir):
        print(f"错误: 目录不存在 - {args.base_dir}")
        exit(1)
    
    # 汇总结果
    results = aggregate_results(args.base_dir)
    
    if not results:
        print("没有找到任何评估结果")
        return
    
    # 保存为 CSV
    csv_path = save_results_as_csv(results, args.base_dir)
    
    # 计算并添加总体平均值
    overall_avg = calculate_and_add_overall_average(results, csv_path)
    
    # 生成可视化图表
    visualize_comparison(results, overall_avg, args.base_dir)
    
    # 打印摘要
    print("\n汇总分析完成:")
    print(f"- 处理 checkpoint 数量: {len(results)}")
    if "benchmark_average" in overall_avg:
        print(f"- 所有 benchmark 平均分: {overall_avg['benchmark_average']:.2f}%")
    else:
        print("警告: 未找到 benchmark_average 值")
    
    # 找到最佳 checkpoint
    if results and "benchmark_average" in results[0]:
        best_checkpoint = max(results, key=lambda x: x["benchmark_average"])
        print(f"\n🎯 最佳表现 checkpoint: {best_checkpoint['checkpoint']}")
        print(f"- 综合评分: {best_checkpoint['benchmark_average']:.2f}%")
        if "judge_Overall" in best_checkpoint:
            print(f"- Judge Bench: {best_checkpoint['judge_Overall']:.2f}%")
        if "reward_average" in best_checkpoint:
            print(f"- Reward Bench: {best_checkpoint['reward_average']:.2f}%")
        if "rm_total_avg_acc" in best_checkpoint:
            print(f"- RM Bench: {best_checkpoint['rm_total_avg_acc']:.2f}%")

if __name__ == "__main__":
    main()