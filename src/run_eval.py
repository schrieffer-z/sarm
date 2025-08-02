import os
import subprocess
import glob
import time
import argparse
import shlex
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import traceback
import psutil


TASK_PATHS = {
    "reward_bench": "eval/reward_bench",
    "rm_bench": "eval/rm_bench"
}

TASK_COMMANDS = {
    "reward_bench": [
        "python", "eval_rewardbench_sarm_llama.py",
        "--batch_size=1",
        "--dataset", "reward-bench-2"
    ],
    "rm_bench": [
        "python", "eval_rmbench_sarm_llama.py",
        "--datapath", "data/total_dataset.json",
        "--batch_size", "1",
        "--trust_remote_code",
        "--chat_template", "tulu"
    ]
}

TASK_TIMEOUTS = {
    "reward_bench": 1800, 
    "rm_bench": 1800
}

DEFAULT_LOG_DIR = "eval_logs"
MAX_WAIT_RETRIES = 3

def run_task(gpu, ckpt_path, task_name, base_dir):
    log_file = os.path.join(
        os.path.join(base_dir, DEFAULT_LOG_DIR),
        f"{os.path.basename(ckpt_path)}_{task_name}.log"
    )
    
    cmd = TASK_COMMANDS[task_name].copy()
    
    cmd.append("--model")
    cmd.append(ckpt_path)
    
    tokenizer_path = os.path.dirname(ckpt_path)
    cmd.append("--tokenizer")
    cmd.append(tokenizer_path)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "w") as log:
        try:
            print(f"Start Eval: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
            print(f"Command: {' '.join(shlex.quote(arg) for arg in cmd)}")
            
            process = subprocess.Popen(
                cmd,
                cwd=os.path.join('.', TASK_PATHS[task_name]),
                env=env,
                stdout=log,
                stderr=log
            )
            
            timeout = TASK_TIMEOUTS.get(task_name, 3600)
            try:
                process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
                process.terminate()
                try:
                    process.communicate(timeout=30)
                except:
                    pass
                return False
                
            if process.returncode == 0:
                print(f"✅ Finished: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)}")
                return True
            else:
                print(f"❌ Fail: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)} | 退出码: {process.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Fail: {task_name} on GPU:{gpu} | CKPT: {os.path.basename(ckpt_path)} | 错误: {e}")
            return False


def main(base_dir, devices):
    num_tasks = len(TASK_PATHS)
    
    log_dir = os.path.join(base_dir, DEFAULT_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoints = sorted(
        glob.glob(os.path.join(base_dir, "checkpoint-*")),
        key=lambda x: int(x.split('-')[-1])
    )
    
    if not checkpoints:
        print(f"No ckpts found in {base_dir}")
        return
    
    print(f"{len(checkpoints)} to be evaluated")
    print(f"Devices: {', '.join(devices[:num_tasks])} (allocated to {', '.join(TASK_PATHS.keys())})")
    
    def handler(signum, frame):
        raise TimeoutError("Timeout")
    
    original_handler = signal.signal(signal.SIGALRM, handler)
    
    for ckpt in checkpoints:
        ckpt_name = os.path.basename(ckpt)
        print(f"\n{'='*60}\Evaluating checkpoint: {ckpt_name}\n{'='*60}")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = {
                executor.submit(run_task, gpu, ckpt, task_name, base_dir): task_name
                for gpu, task_name in zip(devices[:num_tasks], TASK_PATHS.keys())
            }
            
            completed = 0
            for _ in range(MAX_WAIT_RETRIES):
                if completed >= len(futures):
                    break
                    
                signal.alarm(1800)
                try:
                    for future in as_completed(futures, timeout=1800):
                        task_name = futures[future]
                        try:
                            success = future.result()
                            completed += 1
                        except Exception as e:
                            print(f"❌ Error: {task_name} | {e}")
                            completed += 1
                except TimeoutError:
                    print(f"⏰ Time out, retrying...")
                    continue
                finally:
                    signal.alarm(0)
        
        elapsed = time.time() - start_time
        print(f"Finish {ckpt_name} | Time: {elapsed:.2f}s")
    
    print(f"\nAll ckpts finished! Log saved to: {log_dir}")
    signal.signal(signal.SIGALRM, original_handler)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate all checkpoints in one run')
    parser.add_argument('--base_dir', required=True,
                        help='base dir of all checkpoints')
    parser.add_argument('--devices', required=True,
                        help='GPU ddevices ID list("0,1")')
    parser.add_argument('--tasks', nargs='+', type=str, default=['reward_bench', 'rm_bench'],
                    help='11 stands for all benchmark')
    args = parser.parse_args()

    TASK_PATHS = {task:TASK_PATHS[task] for task in args.tasks}
    
    if not os.path.exists(args.base_dir):
        raise ValueError(f"{args.base_dir} not exist")
    
    devices = [dev.strip() for dev in args.devices.split(',') if dev.strip()]
    
    main(os.path.abspath(args.base_dir), devices)