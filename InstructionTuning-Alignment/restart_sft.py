"""重启SFT训练: 杀掉旧的慢训练进程 + 用新配置(epochs=2)启动"""
import subprocess
import os
import time

# 1. 杀旧训练进程(PID 18556)
OLD_PID = 18556
r = subprocess.run(["taskkill", "/PID", str(OLD_PID), "/F"],
                   capture_output=True, text=True)
print(f"停止旧进程 {OLD_PID}: {r.stdout.strip()} {r.stderr.strip()}")
time.sleep(5)  # 等GPU显存释放

# 2. 启动新训练(epochs=2)
PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
os.makedirs("logs", exist_ok=True)
log_path = os.path.abspath("logs/sft_huatuo_train.log")

cmd = [PYTHON, "scripts/run_experiment.py",
       "--config", "configs/experiment/week15_sft_tinyllama.yaml",
       "--exp_id", "week15_sft_huatuo"]

flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

with open(log_path, "w", encoding="utf-8") as log_f:
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                            creationflags=flags, cwd=os.getcwd())

print(f"新SFT训练已启动, PID={proc.pid}")
print(f"配置: epochs=2 (优化后预计~83分钟)")
print(f"日志: {log_path}")
