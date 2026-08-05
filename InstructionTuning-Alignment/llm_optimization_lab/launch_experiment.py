"""LLM优化实验启动器 - 后台无窗口运行"""
import os
import subprocess
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\pythonw.exe"
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "experiment.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

print(f"启动LLM优化实验（无窗口模式）...")
print(f"日志文件: {LOG_FILE}")
print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# 使用pythonw.exe + CREATE_NO_WINDOW避免弹窗
log_f = open(LOG_FILE, 'w', encoding='utf-8')
proc = subprocess.Popen(
    [PYTHON, "run_all.py"],
    cwd=PROJECT_ROOT,
    stdout=log_f,
    stderr=subprocess.STDOUT,
    creationflags=0x08000000,  # CREATE_NO_WINDOW
    close_fds=True,
)

print(f"进程已启动: PID={proc.pid}")
print(f"可通过以下命令查看进度:")
print(f"  Get-Content logs\\experiment.log -Wait")
