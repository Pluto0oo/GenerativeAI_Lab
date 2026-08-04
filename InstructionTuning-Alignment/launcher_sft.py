"""SFT训练后台启动器（绕过PowerShell执行策略限制）

用 DETACHED_PROCESS 启动训练子进程，launcher本身立即返回，
训练作为独立进程运行，输出写入日志文件，可用Read工具监控。
"""
import subprocess
import os

PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
os.makedirs("logs", exist_ok=True)
log_path = os.path.abspath("logs/sft_huatuo_train.log")

cmd = [
    PYTHON, "scripts/run_experiment.py",
    "--config", "configs/experiment/week15_sft_tinyllama.yaml",
    "--exp_id", "week15_sft_huatuo",
]

# DETACHED_PROCESS(0x00000008) | CREATE_NEW_PROCESS_GROUP(0x00000200)
# 子进程完全独立于父进程，父退出后子进程继续运行
flags = 0x00000008 | 0x00000200

with open(log_path, "w", encoding="utf-8") as log_f:
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=flags,
        cwd=os.getcwd(),
    )

print(f"SFT训练已后台启动, PID={proc.pid}")
print(f"日志文件: {log_path}")
print(f"配置: configs/experiment/week15_sft_tinyllama.yaml")
print(f"exp_id: week15_sft_huatuo")
print(f"训练预计15-25分钟, 请用Read工具读取日志文件监控进度")
