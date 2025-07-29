import os
import json
import csv
import argparse
import re
import pandas as pd
from glob import glob

def extract_checkpoint_number(checkpoint_name):
    match = re.search(r'checkpoint-(\d+)', checkpoint_name)
    if match:
        return int(match.group(1))
    return -1

def calculate_reward_average(reward_data):
    scores = []
    for field in ["Factuality", "Focus", "Math", "Precise IF", "Safety", "Ties"]:
        if field in reward_data:
            value = reward_data[field]
            if 0 <= value <= 1:
                scores.append(value * 100)
            else:
                scores.append(value)
    return sum(scores) / len(scores) if scores else 0

def convert_rm_to_percentage(rm_data):
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
    checkpoint_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and "checkpoint-" in item:
            checkpoint_dirs.append(item_path)
    
    checkpoint_dirs.sort(key=lambda x: extract_checkpoint_number(os.path.basename(x)))
    
    all_results = []
    checkpoint_count = 0
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = os.path.basename(checkpoint_dir)
        print(f"processing checkpoint: {checkpoint_name}")
        
        result = {"checkpoint": checkpoint_name}
        scores = []
        benchmark_data = {}
        
        try:
            # Judge Bench 
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
            
            # Reward Bench v2
            reward_path = os.path.join(checkpoint_dir, "reward_benchv2.json")
            if os.path.exists(reward_path):
                with open(reward_path) as f:
                    reward_data = json.load(f)
                    reward_avg = calculate_reward_average(reward_data)
                    result["reward_average"] = reward_avg
                    scores.append(reward_avg)
                    
                    result.update({
                        "reward_Factuality": reward_data.get("Factuality", 0) * 100,
                        "reward_Focus": reward_data.get("Focus", 0) * 100,
                        "reward_Math": reward_data.get("Math", 0) * 100,
                        "reward_Precise IF": reward_data.get("Precise IF", 0) * 100,
                        "reward_Safety": reward_data.get("Safety", 0) * 100,
                        "reward_Ties": reward_data.get("Ties", 0) * 100
                    })
                    benchmark_data["reward"] = reward_data
            
            # RM Bench
            rm_path = os.path.join(checkpoint_dir, "rm_bench.json")
            if os.path.exists(rm_path):
                with open(rm_path) as f:
                    rm_data = json.load(f)
                    rm_converted = convert_rm_to_percentage(rm_data)
                    result["rm_total_avg_acc"] = rm_converted.get("total_avg_acc", 0)
                    scores.append(rm_converted.get("total_avg_acc", 0))
                    
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
            
            if scores:
                result["benchmark_average"] = sum(scores) / len(scores)

            full_template = {
                "benchmark_average": 0,
                "judge_Coding": 0, "judge_Knowledge": 0, "judge_Math": 0, 
                "judge_Overall": 0, "judge_Reasoning": 0,
                "rm_total_avg_acc": 0, "rm_chat": 0, "rm_math": 0, "rm_code": 0, 
                "rm_safety": 0, "rm_hard_acc": 0, "rm_normal_acc": 0, "rm_easy_acc": 0,
                "reward_average": 0, "reward_Factuality": 0, "reward_Precise IF": 0,
                "reward_Math": 0, "reward_Safety": 0, "reward_Focus": 0, "reward_Ties": 0
            }
            for key in result.keys():
                full_template[key] = result[key]   
            all_results.append(full_template)
            checkpoint_count += 1
            
        except Exception as e:
            print(f"error on {checkpoint_name}: {str(e)}")
    
    print(f"{checkpoint_count} checkpoints done")
    return all_results

def save_results_as_csv(results, base_dir):
    if not results:
        return None
    
    fixed_order = [
        "benchmark_average", 

        # Judge Bench
        "judge_Overall",
        "judge_Knowledge",
        "judge_Reasoning",
        "judge_Math",
        "judge_Coding",
        
        # RM Bench
        "rm_total_avg_acc",  # Overall
        "rm_chat",            # Chat
        "rm_math",            # Math
        "rm_code",            # Code
        "rm_safety",          # Safety
        "rm_hard_acc", 
        "rm_normal_acc", 
        "rm_easy_acc",
        
        # Reward Bench V2
        "reward_average",     # Overall
        "reward_Factuality",  # Factuality
        "reward_Precise IF",  # Precise IF
        "reward_Math",        # Math
        "reward_Safety",      # Safety
        "reward_Focus",       # Focus
        "reward_Ties",        # Ties
    ]
    
    final_order = ["checkpoint"] + fixed_order
    
    output_path = os.path.join(base_dir, "benchmark_results.csv")
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=final_order)
        writer.writeheader()
        
        for result in results:
            filtered_result = {k: v for k, v in result.items() if k in final_order}
            writer.writerow(filtered_result)
    
    print(f"save to: {output_path}")
    return output_path

def calculate_and_add_overall_average(results, output_path):
    if not results:
        return {}
    
    fixed_order = [
        "benchmark_average", 
        "judge_Overall",
        "judge_Knowledge",
        "judge_Reasoning",
        "judge_Math",
        "judge_Coding",
        "rm_total_avg_acc",
        "rm_chat",
        "rm_math",
        "rm_code",
        "rm_safety",
        "rm_hard_acc",
        "rm_normal_acc",
        "rm_easy_acc",
        "reward_average",
        "reward_Factuality",
        "reward_Precise IF",
        "reward_Math",
        "reward_Safety",
        "reward_Focus",
        "reward_Ties"
    ]
    
    with open(output_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        header_fields = reader.fieldnames
        
    columns_to_avg = [col for col in header_fields if col != "checkpoint"]
    
    overall_avg = {}
    for column in columns_to_avg:
        valid_values = []
        for result in results:
            if column in result and not pd.isna(result.get(column)):
                valid_values.append(result[column])
        
        if valid_values:
            overall_avg[column] = sum(valid_values) / len(valid_values)
        else:
            overall_avg[column] = 0
    
    avg_row = {"checkpoint": "Overall Average"}
    for column in columns_to_avg:
        avg_row[column] = overall_avg.get(column, "")
    
    # 添加空行和平均值行到 CSV
    with open(output_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_fields)
        writer.writerow({})  # 添加空行
        writer.writerow(avg_row)
    
    print(f"Save to: {output_path}")
    return overall_avg

def visualize_comparison(results, overall_avg, base_dir):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame(results)
        
        df['checkpoint_num'] = df['checkpoint'].apply(extract_checkpoint_number)
        df = df.sort_values('checkpoint_num')
        
        output_dir = os.path.join(base_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Judge Bench
        plt.figure(figsize=(15, 8))
        judge_cols = [col for col in df.columns if col.startswith("judge_")]

        if judge_cols:
            ax = plt.subplot(111)
            for col in judge_cols:
                label = col.replace("judge_", "")
                ax.plot(df["checkpoint_num"], df[col], 
                        marker='o', markersize=5, linewidth=2, label=label)
            
            plt.title("Judge Bench Performance by Training Steps")
            plt.xlabel("Training Steps")
            plt.ylabel("Score (%)")
            plt.legend(loc='upper left', bbox_to_anchor=(0, -0.1), ncol=3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "1_judge_bench_metrics.png"))
            plt.close()
        
        # 2. Reward Bench V2 
        plt.figure(figsize=(15, 8))
        reward_cols = [col for col in df.columns if col.startswith("reward_") and col != "reward_average"]
        if reward_cols:
            ax = plt.subplot(111)
            if "reward_average" in df.columns:
                ax.plot(df["checkpoint_num"], df["reward_average"], "o-", 
                        markersize=6, linewidth=3, color="black", label="Average (Overall)")
            
            for col in reward_cols:
                label = col.replace("reward_", "")
                ax.plot(df["checkpoint_num"], df[col], 
                        marker='o', markersize=4, linewidth=1.5, label=label)
            
            plt.title("Reward Bench V2 Performance by Training Steps")
            plt.xlabel("Training Steps")
            plt.ylabel("Score (%)")
            plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "2_reward_bench_metrics.png"))
            plt.close()
        
        # 3. RM Bench
        plt.figure(figsize=(15, 8))
        rm_cols = [col for col in df.columns if col.startswith("rm_") and col != "rm_total_avg_acc"]
        if rm_cols:
            ax = plt.subplot(111)
            if "rm_total_avg_acc" in df.columns:
                ax.plot(df["checkpoint_num"], df["rm_total_avg_acc"], "o-", 
                        markersize=6, linewidth=3, color="black", label="Total Avg (Overall)")
            
            for col in rm_cols:
                label = col.replace("rm_", "")
                ax.plot(df["checkpoint_num"], df[col], 
                        marker='o', markersize=4, linewidth=1.5, label=label)
            
            plt.title("RM Bench Performance by Training Steps")
            plt.xlabel("Training Steps")
            plt.ylabel("Score (%)")
            plt.legend(loc='upper left', bbox_to_anchor=(0, -0.15), ncol=3)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "3_rm_bench_metrics.png"))
            plt.close()
        
        plt.figure(figsize=(12, 6))
        benchmarks_available = False
        labels = []
        lines = []
        
        if "judge_Overall" in df.columns:
            plt.plot(df["checkpoint_num"], df["judge_Overall"], "o-", markersize=8, linewidth=2.5, color="red")
            labels.append("Judge Bench")
            benchmarks_available = True
            
        if "reward_average" in df.columns:
            plt.plot(df["checkpoint_num"], df["reward_average"], "o-", markersize=8, linewidth=2.5, color="green")
            labels.append("Reward Bench")
            benchmarks_available = True
            
        if "rm_total_avg_acc" in df.columns:
            plt.plot(df["checkpoint_num"], df["rm_total_avg_acc"], "o-", markersize=8, linewidth=2.5, color="blue")
            labels.append("RM Bench")
            benchmarks_available = True
        
        if benchmarks_available:
            plt.title("Comparison of Benchmarks Overall Scores")
            plt.xlabel("Training Steps")
            plt.ylabel("Score (%)")
            plt.legend(labels)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "4_benchmarks_comparison.png"))
            plt.close()
        
        print(f"Generated analysis charts in: {output_dir}")
        
    except ImportError:
        print("Matplotlib library missing, skipping chart generation. Please install with: pip install matplotlib")
    except Exception as e:
        print(f"Error generating charts: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='read results of all checkpoints')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='base dir to save all checkpoints')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.base_dir):
        print(f"{args.base_dir} do not exist")
        exit(1)
    
    results = aggregate_results(args.base_dir)
    
    if not results:
        print("Not json result to read")
        return
    
    csv_path = save_results_as_csv(results, args.base_dir)
    
    overall_avg = calculate_and_add_overall_average(results, csv_path)
    
    visualize_comparison(results, overall_avg, args.base_dir)
    
    print("\nFinished:")
    print(f"- Number of checkpoint: {len(results)}")
    if "benchmark_average" in overall_avg:
        print(f"- Benchmarks Average: {overall_avg['benchmark_average']:.2f}%")
    else:
        print("error: No benchmark_average in result")
    
    # 找到最佳 checkpoint
    if results and "benchmark_average" in results[0]:
        best_checkpoint = max(results, key=lambda x: x["benchmark_average"])
        print(f"\n🎯 Best checkpoint: {best_checkpoint['checkpoint']}")
        print(f"- Average: {best_checkpoint['benchmark_average']:.2f}%")
        if "judge_Overall" in best_checkpoint:
            print(f"- Judge Bench: {best_checkpoint['judge_Overall']:.2f}%")
        if "reward_average" in best_checkpoint:
            print(f"- Reward Bench: {best_checkpoint['reward_average']:.2f}%")
        if "rm_total_avg_acc" in best_checkpoint:
            print(f"- RM Bench: {best_checkpoint['rm_total_avg_acc']:.2f}%")

if __name__ == "__main__":
    main()