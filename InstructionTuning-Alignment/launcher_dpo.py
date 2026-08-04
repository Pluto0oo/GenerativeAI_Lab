"""DPO pipeline 后台启动器（绕过PowerShell执行策略）

SFT训练完成后运行本脚本，DETACHED启动DPO全流程。
"""
import subprocess
import os

PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

proc = subprocess.Popen(
    [PYTHON, "run_dpo_pipeline.py"],
    creationflags=flags,
    cwd=os.getcwd(),
)
print(f"DPO pipeline 已后台启动, PID={proc.pid}")
print("流程: 合并SFT → 生成偏好对 → 训练DPO")
print("日志: logs/dpo_pipeline.log (用Read工具监控)")
