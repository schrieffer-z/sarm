import os
import subprocess
import glob
import time
import argparse
import shlex
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

# # 用户配置区域 - 根据实际需求修改
# TASK_PATHS = {
#     "judge_bench": "eval/judge_bench",
#     "reward_bench": "eval/reward_bench",
#     "rm_bench": "eval/rm_bench"
# }

# # 实际使用的命令和参数
# TASK_COMMANDS = {
#     "judge_bench": [
#         "python", "judge_bench.py",
#         "--judge_name", "reward_model",
#         "--pairs", "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
#     ],
#     "reward_bench": [
#         "python", "eval_rewardbench_sarm_llama.py",
#         "--batch_size=1",
#         "--dataset", "/NAS/zhangsy/datasets/allenai/reward-bench-2"
#     ],
#     "rm_bench": [
#         "python", "eval_rmbench_sarm_llama.py",
#         "--datapath", "data/total_dataset.json",
#         "--batch_size", "1",
#         "--trust_remote_code",
#         "--chat_template", "tulu"
#     ]
# }

# # 任务超时时间（秒）
# TASK_TIMEOUTS = {
#     "judge_bench": 1800,  # 30分钟
#     "reward_bench": 1800,  # 60分钟
#     "rm_bench": 1800      # 60分钟
# }

# DEFAULT_LOG_DIR = "eval_logs"
# MAX_WAIT_RETRIES = 5  # 等待任务完成的超时次数

# def run_task(gpu, ckpt_path, task_name, base_dir):
#     """运行单个评估任务"""
#     log_file = os.path.join(
#         os.path.join(base_dir, DEFAULT_LOG_DIR),
#         f"{os.path.basename(ckpt_path)}_{task_name}.log"
#     )
    
#     # 构建命令
#     cmd = TASK_COMMANDS[task_name].copy()
    
#     # 添加模型路径参数（根据任务不同位置不同）
#     if task_name == "judge_bench":
#         cmd.append("--judge_model")
#     else:
#         cmd.append("--model")
#     cmd.append(ckpt_path)
    
#     # 添加tokenizer路径
#     if task_name in ["reward_bench", "rm_bench"]:
#         tokenizer_path = os.path.dirname(ckpt_path)
#         cmd.append("--tokenizer")
#         cmd.append(tokenizer_path)
    
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = gpu
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
#     with open(log_file, "w") as log:
#         try:
#             print(f"开始任务: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
#             print(f"命令: {' '.join(shlex.quote(arg) for arg in cmd)}")
            
#             # 使用Popen以便后续控制
#             process = subprocess.Popen(
#                 cmd,
#                 cwd=os.path.join('.', TASK_PATHS[task_name]),
#                 env=env,
#                 stdout=log,
#                 stderr=log
#             )
            
#             # 等待任务完成（带超时）
#             timeout = TASK_TIMEOUTS.get(task_name, 3600)
#             try:
#                 process.communicate(timeout=timeout)
#             except subprocess.TimeoutExpired:
#                 print(f"⏰ 任务超时: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
#                 process.terminate()
#                 try:
#                     process.communicate(timeout=30)
#                 except:
#                     pass
#                 return False
                
#             if process.returncode == 0:
#                 print(f"✅ 完成任务: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
#                 return True
#             else:
#                 print(f"❌ 任务失败: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)} | 退出码: {process.returncode}")
#                 return False
                
#         except Exception as e:
#             print(f"❌ 任务异常: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)} | 错误: {e}")
#             return False

# def main(base_dir, devices):
#     # 验证设备数量匹配任务数量
#     num_tasks = len(TASK_PATHS)
#     if len(devices) < num_tasks:
#         raise ValueError(f"需要至少 {num_tasks} 个设备ID，但只提供了 {len(devices)}: {devices}")
    
#     # 创建日志目录
#     log_dir = os.path.join(base_dir, DEFAULT_LOG_DIR)
#     os.makedirs(log_dir, exist_ok=True)
    
#     # 获取所有checkpoint路径（按数字排序）
#     checkpoints = sorted(
#         glob.glob(os.path.join(base_dir, "checkpoint-*")),
#         key=lambda x: int(x.split('-')[-1])
#     )
    
#     if not checkpoints:
#         print(f"在 {base_dir} 中没有找到任何checkpoint目录")
#         return
    
#     print(f"发现 {len(checkpoints)} 个checkpoint需要评估")
#     print(f"使用设备: {', '.join(devices[:num_tasks])} (分配给 {', '.join(TASK_PATHS.keys())})")
    
#     # 设置超时处理
#     def handler(signum, frame):
#         raise TimeoutError("任务等待超时")
    
#     original_handler = signal.signal(signal.SIGALRM, handler)
    
#     for ckpt in checkpoints:
#         ckpt_name = os.path.basename(ckpt)
#         print(f"\n{'='*60}\n开始评估 checkpoint: {ckpt_name}\n{'='*60}")
#         start_time = time.time()
        
#         # 使用线程池并行执行多个任务
#         with ThreadPoolExecutor(max_workers=num_tasks) as executor:
#             futures = {
#                 executor.submit(run_task, gpu, ckpt, task_name, base_dir): task_name
#                 for gpu, task_name in zip(devices[:num_tasks], TASK_PATHS.keys())
#             }
            
#             # 等待所有任务完成（带超时重试机制）
#             completed = 0
#             for _ in range(MAX_WAIT_RETRIES):
#                 if completed >= len(futures):
#                     break
                    
#                 signal.alarm(1800)  # 30分钟超时
#                 try:
#                     for future in as_completed(futures, timeout=1800):
#                         task_name = futures[future]
#                         try:
#                             success = future.result()
#                             completed += 1
#                         except Exception as e:
#                             print(f"❌ 任务异常: {task_name} | 错误: {e}")
#                             completed += 1
#                 except TimeoutError:
#                     print(f"⏰ 等待任务超时，将继续等待...")
#                     continue
#                 finally:
#                     signal.alarm(0)  # 取消超时设置
        
#         elapsed = time.time() - start_time
#         print(f"完成评估 {ckpt_name} | 耗时: {elapsed:.2f}秒")
    
#     print(f"\n所有评估完成！日志保存在: {log_dir}")
#     # 恢复原始信号处理
#     signal.signal(signal.SIGALRM, original_handler)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='批量评估模型checkpoint')
#     parser.add_argument('--base_dir', required=True,
#                         help='包含所有checkpoint的根目录')
#     parser.add_argument('--devices', required=True,
#                         help='GPU设备ID列表，用逗号分隔（如 "0,1,2"）')
#     args = parser.parse_args()
    
#     # 确保路径存在
#     if not os.path.exists(args.base_dir):
#         raise ValueError(f"目录不存在: {args.base_dir}")
    
#     # 解析设备列表
#     devices = [dev.strip() for dev in args.devices.split(',') if dev.strip()]
    
#     main(os.path.abspath(args.base_dir), devices)


import os
import subprocess
import glob
import time
import argparse
import shlex
import concurrent.futures
from collections import deque
import threading
import traceback
import psutil  # 添加psutil用于系统资源监控


# 用户配置区域 - 根据实际需求修改
TASK_PATHS = {
    "judge_bench": "eval/judge_bench",
    "reward_bench": "eval/reward_bench",
    "rm_bench": "eval/rm_bench"
}

# 实际使用的命令和参数
TASK_COMMANDS = {
    "judge_bench": [
        "python", "judge_bench.py",
        "--judge_name", "reward_model",
        "--pairs", "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl"
    ],
    "reward_bench": [
        "python", "eval_rewardbench_sarm_llama.py",
        "--batch_size=1",
        "--dataset", "/NAS/zhangsy/datasets/allenai/reward-bench-2"
    ],
    "rm_bench": [
        "python", "eval_rmbench_sarm_llama.py",
        "--datapath", "data/total_dataset.json",
        "--batch_size", "1",
        "--trust_remote_code",
        "--chat_template", "tulu"
    ]
}

# 任务超时时间（秒）
TASK_TIMEOUTS = {
    "judge_bench": 1800,  # 30分钟
    "reward_bench": 1800,  # 60分钟
    "rm_bench": 1800      # 60分钟
}

DEFAULT_LOG_DIR = "eval_logs"
MAX_WAIT_RETRIES = 5  # 等待任务完成的超时次数

def run_task(gpu, ckpt_path, task_name, base_dir):
    """运行单个评估任务"""
    log_file = os.path.join(
        os.path.join(base_dir, DEFAULT_LOG_DIR),
        f"{os.path.basename(ckpt_path)}_{task_name}.log"
    )
    
    # 构建命令
    cmd = TASK_COMMANDS[task_name].copy()
    
    # 添加模型路径参数（根据任务不同位置不同）
    if task_name == "judge_bench":
        cmd.append("--judge_model")
    else:
        cmd.append("--model")
    cmd.append(ckpt_path)
    
    # 添加tokenizer路径
    if task_name in ["reward_bench", "rm_bench"]:
        tokenizer_path = os.path.dirname(ckpt_path)
        cmd.append("--tokenizer")
        cmd.append(tokenizer_path)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "w") as log:
        try:
            print(f"开始任务: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
            print(f"命令: {' '.join(shlex.quote(arg) for arg in cmd)}")
            
            # 使用Popen以便后续控制
            process = subprocess.Popen(
                cmd,
                cwd=os.path.join('.', TASK_PATHS[task_name]),
                env=env,
                stdout=log,
                stderr=log
            )
            
            # 等待任务完成（带超时）
            timeout = TASK_TIMEOUTS.get(task_name, 3600)
            try:
                process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"⏰ 任务超时: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
                process.terminate()
                try:
                    process.communicate(timeout=30)
                except:
                    pass
                return False
                
            if process.returncode == 0:
                print(f"✅ 完成任务: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
                return True
            else:
                print(f"❌ 任务失败: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)} | 退出码: {process.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ 任务异常: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)} | 错误: {e}")
            return False

class TaskExecutor:
    # ... [之前的类代码保持不变] ...
    
    def run(self):
        """运行单个评估任务"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # 构建命令
        cmd = TASK_COMMANDS[self.task_name].copy()
        
        # 添加模型路径参数
        cmd.append("--model" if self.task_name != "judge_bench" else "--judge_model")
        cmd.append(self.ckpt_path)
        
        # 添加tokenizer路径
        if self.task_name in ["reward_bench", "rm_bench"]:
            tokenizer_path = os.path.dirname(self.ckpt_path)
            cmd.append("--tokenizer")
            cmd.append(tokenizer_path)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu
        
        self.start_time = time.time()
        
        try:
            print(f"⏳ 开始任务: {self.task_name} on GPU:{self.gpu} | CKPT: {os.path.basename(self.ckpt_path)}")
            print(f"    命令: {' '.join(shlex.quote(arg) for arg in cmd)}")
            print(f"    工作目录: {os.path.join(self.base_dir, TASK_PATHS[self.task_name])}")
            
            # 打印系统资源信息
            self.print_system_resources()
            
            with open(self.log_file, "w") as log:
                log.write(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}\n")
                log.write(f"命令: {' '.join(shlex.quote(arg) for arg in cmd)}\n")
                log.write(f"工作目录: {os.path.join(self.base_dir, TASK_PATHS[self.task_name])}\n")
                log.write("\n============ 系统资源状态 ============\n")
                log.write(self.get_system_resources())
                log.write("\n============ 命令输出 ============\n")
                
                # 刷新缓冲区确保写入
                log.flush()
                
                # 使用Popen启动进程
                self.process = subprocess.Popen(
                    cmd,
                    cwd=os.path.join(self.base_dir, TASK_PATHS[self.task_name]),
                    env=env,
                    stdout=log,
                    stderr=log,
                    text=True
                )
                
                # 等待进程完成（带超时）
                timeout = TASK_TIMEOUTS.get(self.task_name, 0)
                if timeout > 0:
                    try:
                        self.process.communicate(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        log.write("\n============ 任务超时终止 ============\n")
                        log.flush()
                        self.end_time = time.time()
                        duration = self.end_time - self.start_time
                        print(f"⏰ 任务超时: {self.task_name} | CKPT: {os.path.basename(self.ckpt_path)} | 超时: {timeout}秒")
                        print(f"    日志位置: {self.log_file}")
                        self.process.terminate()
                        try:
                            self.process.communicate(timeout=30)
                        except:
                            pass
                        return False
                else:
                    self.process.communicate()
                
                self.end_time = time.time()
                duration = self.end_time - self.start_time
                
                if self.process.returncode == 0:
                    self.success = True
                    print(f"✅ 完成任务: {self.task_name} | CKPT: {os.path.basename(self.ckpt_path)} | 耗时: {duration:.2f}秒")
                else:
                    log.write("\n============ 任务失败 ============\n")
                    log.write(f"退出码: {self.process.returncode}\n")
                    log.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))}\n")
                    self.error_message = f"退出码: {self.process.returncode}"
                    print(f"❌ 任务失败: {self.task_name} | CKPT: {os.path.basename(self.ckpt_path)} | {self.error_message} | 耗时: {duration:.2f}秒")
                    print(f"    详细日志: {self.log_file}")
            
        except Exception as e:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.error_message = f"异常: {str(e)}"
            print(f"❌ 任务异常: {self.task_name} | CKPT: {os.path.basename(self.ckpt_path)} | {self.error_message} | 耗时: {duration:.2f}秒")
            print(f"    详细日志: {self.log_file}")
            
            # 将完整异常堆栈写入日志
            if os.path.exists(self.log_file):
                with open(self.log_file, "a") as log:
                    log.write("\n============ 任务异常 ============\n")
                    traceback.print_exc(file=log)
            
            if self.process:
                try:
                    self.process.terminate()
                    self.process.communicate(timeout=5)
                except:
                    pass
    
    def get_system_resources(self):
        """获取系统资源状态"""
        msg = ""
        
        # 内存信息
        mem = psutil.virtual_memory()
        msg += f"内存总量: {mem.total / (1024 ** 3):.2f} GB\n"
        msg += f"已用内存: {mem.used /(1024 ** 3):.2f} GB\n"
        msg += f"可用内存: {mem.available / (1024 ** 3):.2f} GB\n"
        # msg += f"内存使用率: {mem.percent}%\n"
        
        # 磁盘空间
        disk = psutil.disk_usage('/')
        msg += f"\n磁盘总量: {disk.total / (1024 ** 3):.2f} GB\n"
        msg += f"已用空间: {disk.used / (1024 ** 3):.2f} GB\n"
        msg += f"可用空间: {disk.free / (1024 ** 3):.2f} GB\n"
        msg += f"磁盘使用率: {disk.percent}%\n"
        
        # GPU信息
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            msg += f"\nGPU数量: {device_count}\n"
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                msg += f"GPU {i}: 显存使用 {mem_info.used/(1024**2):.2f} MB / {mem_info.total/(1024**2):.2f} MB, 使用率 {util.gpu}%\n"
            pynvml.nvmlShutdown()
        except Exception as e:
            msg += f"\n无法获取GPU信息: {str(e)}\n"
        
        return msg
    
    def print_system_resources(self):
        """打印系统资源摘要"""
        resources = self.get_system_resources().split("\n")
        print("    当前资源状态:")
        for line in resources[:4]:
            print(f"        {line}")
        print(f"        更多信息见日志: {self.log_file}")


def main(base_dir, devices):
    # 验证设备数量匹配任务数量
    num_tasks = len(TASK_PATHS)
    if len(devices) < num_tasks:
        raise ValueError(f"需要至少 {num_tasks} 个设备ID，但只提供了 {len(devices)}: {devices}")
    
    # 创建日志目录
    log_dir = os.path.join(base_dir, DEFAULT_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取所有checkpoint路径（按数字排序）
    checkpoints = sorted(
        glob.glob(os.path.join(base_dir, "checkpoint-*")),
        key=lambda x: int(x.split('-')[-1])
    )
    
    if not checkpoints:
        print(f"在 {base_dir} 中没有找到任何checkpoint目录")
        return
    
    print(f"发现 {len(checkpoints)} 个checkpoint需要评估")
    print(f"使用设备: {', '.join(devices[:num_tasks])} (分配给 {', '.join(TASK_PATHS.keys())})")
    
    # 设置超时处理
    def handler(signum, frame):
        raise TimeoutError("任务等待超时")
    
    original_handler = signal.signal(signal.SIGALRM, handler)
    
    for ckpt in checkpoints:
        ckpt_name = os.path.basename(ckpt)
        print(f"\n{'='*60}\n开始评估 checkpoint: {ckpt_name}\n{'='*60}")
        start_time = time.time()
        
        # 使用线程池并行执行多个任务
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = {
                executor.submit(run_task, gpu, ckpt, task_name, base_dir): task_name
                for gpu, task_name in zip(devices[:num_tasks], TASK_PATHS.keys())
            }
            
            # 等待所有任务完成（带超时重试机制）
            completed = 0
            for _ in range(MAX_WAIT_RETRIES):
                if completed >= len(futures):
                    break
                    
                signal.alarm(1800)  # 30分钟超时
                try:
                    for future in as_completed(futures, timeout=1800):
                        task_name = futures[future]
                        try:
                            success = future.result()
                            completed += 1
                        except Exception as e:
                            print(f"❌ 任务异常: {task_name} | 错误: {e}")
                            completed += 1
                except TimeoutError:
                    print(f"⏰ 等待任务超时，将继续等待...")
                    continue
                finally:
                    signal.alarm(0)  # 取消超时设置
        
        elapsed = time.time() - start_time
        print(f"完成评估 {ckpt_name} | 耗时: {elapsed:.2f}秒")
    
    print(f"\n所有评估完成！日志保存在: {log_dir}")
    # 恢复原始信号处理
    signal.signal(signal.SIGALRM, original_handler)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量评估模型checkpoint')
    parser.add_argument('--base_dir', required=True,
                        help='包含所有checkpoint的根目录')
    parser.add_argument('--devices', required=True,
                        help='GPU设备ID列表，用逗号分隔（如 "0,1,2"）')
    args = parser.parse_args()
    
    # 确保路径存在
    if not os.path.exists(args.base_dir):
        raise ValueError(f"目录不存在: {args.base_dir}")
    
    # 解析设备列表
    devices = [dev.strip() for dev in args.devices.split(',') if dev.strip()]
    
    main(os.path.abspath(args.base_dir), devices)

# import os
# import subprocess
# import glob
# import time
# import argparse
# import shlex
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # 固定配置区域
# TASK_PATHS = {
#     "judge_bench": "eval/judge_bench",
#     "reward_bench": "eval/reward_bench",
#     "rm_bench": "eval/rm_bench"
# }
# DATASET_PATHS = {
#     "judge_bench": "data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl",
#     "reward_bench": "/NAS/zhangsy/datasets/allenai/reward-bench-2",
#     "rm_bench": "data/total_dataset.json"
# }
# DEFAULT_LOG_DIR = "eval_logs"

# def run_task(gpu, ckpt_path, task_name, base_dir):
#     """运行单个评估任务"""
#     log_file = os.path.join(
#         os.path.join(base_dir, DEFAULT_LOG_DIR),
#         f"{os.path.basename(ckpt_path)}_{task_name}.log"
#     )
    
#     if task_name == "judge_bench":
#         cmd = [
#             "python", "judge_bench.py",
#             "--judge_name", "reward_model",
#             "--judge_model", ckpt_path,
#             "--pairs", DATASET_PATHS[task_name]
#         ]
#     elif task_name == "reward_bench":
#         cmd = [
#             "python", "eval_rewardbench_sarm_llama.py",
#             "--batch_size=1",
#             "--model", ckpt_path,
#             "--tokenizer", BASE_DIR,
#             "--dataset", DATASET_PATHS[task_name]
#         ]
#     elif task_name == "rm_bench":
#         cmd = [
#             "python", "eval_rmbench_sarm_llama.py",
#             "--model", ckpt_path,
#             "--tokenizer", BASE_DIR,
#             "--datapath", DATASET_PATHS[task_name],
#             "--batch_size", "1",
#             "--trust_remote_code",
#             "--chat_template", "tulu"
#         ]
    
#     env = os.environ.copy()
#     env["CUDA_VISIBLE_DEVICES"] = gpu
    
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
#     with open(log_file, "w") as log:
#         try:
#             print(f"开始任务: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
#             print(f"命令: {' '.join(shlex.quote(arg) for arg in cmd)}")
            
#             subprocess.run(
#                 cmd,
#                 cwd=os.path.join('.', TASK_PATHS[task_name]),
#                 env=env,
#                 stdout=log,
#                 stderr=log,
#                 check=True
#             )
#             print(f"✅ 完成任务: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
#             return True
#         except subprocess.CalledProcessError as e:
#             print(f"❌ 任务失败: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)} | 错误: {e}")
#             return False

# def main(base_dir, devices):
#     # 验证设备数量匹配任务数量
#     num_tasks = len(TASK_PATHS)
#     if len(devices) < num_tasks:
#         raise ValueError(f"需要至少 {num_tasks} 个设备ID，但只提供了 {len(devices)}: {devices}")
    
#     # 创建日志目录
#     log_dir = os.path.join(base_dir, DEFAULT_LOG_DIR)
#     os.makedirs(log_dir, exist_ok=True)
    
#     # 获取所有checkpoint路径（按数字排序）
#     checkpoints = sorted(
#         glob.glob(os.path.join(base_dir, "checkpoint-*")),
#         key=lambda x: int(x.split('-')[-1])
#     )
    
#     if not checkpoints:
#         print(f"在 {base_dir} 中没有找到任何checkpoint目录")
#         return
    
#     print(f"发现 {len(checkpoints)} 个checkpoint需要评估")
#     print(f"使用设备: {', '.join(devices[:num_tasks])} (分配给 {', '.join(TASK_PATHS.keys())})")
    
#     for ckpt in checkpoints:
#         ckpt_name = os.path.basename(ckpt)
#         print(f"\n{'='*60}\n开始评估 checkpoint: {ckpt_name}\n{'='*60}")
#         start_time = time.time()
        
#         # 使用线程池并行执行多个任务
#         with ThreadPoolExecutor(max_workers=num_tasks) as executor:
#             futures = {
#                 executor.submit(run_task, gpu, ckpt, task_name, base_dir): task_name
#                 for gpu, task_name in zip(devices[:num_tasks], TASK_PATHS.keys())
#             }
            
#             # 等待所有任务完成
#             for future in as_completed(futures):
#                 task_name = futures[future]
#                 try:
#                     success = future.result()
#                 except Exception as e:
#                     print(f"❌ 任务异常: {task_name} | 错误: {e}")
        
#         elapsed = time.time() - start_time
#         print(f"完成评估 {ckpt_name} | 耗时: {elapsed:.2f}秒")
    
#     print(f"\n所有评估完成！日志保存在: {log_dir}")

# BASE_DIR = None
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='批量评估模型checkpoint')
#     parser.add_argument('--base_dir', required=True,
#                         help='包含所有checkpoint的根目录')
#     parser.add_argument('--devices', required=True,
#                         help='GPU设备ID列表，用逗号分隔（如 "0,1,2"）')
#     args = parser.parse_args()
    
#     BASE_DIR = args.base_dir
    
#     # 确保路径存在
#     if not os.path.exists(args.base_dir):
#         raise ValueError(f"目录不存在: {args.base_dir}")
    
#     # 解析设备列表
#     devices = [dev.strip() for dev in args.devices.split(',') if dev.strip()]
    
#     main(os.path.abspath(args.base_dir), devices)