"""监控SFT训练完成，自动启动DPO pipeline

检测 results/week15_sft_huatuo/metrics.json 出现（=SFT完成），
然后自动启动 DPO 全流程（合并→生成偏好对→DPO训练）。
"""
import subprocess
import os
import time

PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SFT_METRICS = "results/week15_sft_huatuo/metrics.json"
LOG_PATH = os.path.abspath("logs/monitor_and_dpo.log")
os.makedirs("logs", exist_ok=True)

log_f = open(LOG_PATH, "w", encoding="utf-8")


def log(msg):
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()


log("=" * 60)
log("监控SFT训练完成，完成后自动启动DPO pipeline")
log("=" * 60)

wait_start = time.time()
while not os.path.exists(SFT_METRICS):
    time.sleep(60)
    elapsed = (time.time() - wait_start) / 60
    log(f"  等待SFT完成... 已等 {elapsed:.1f} 分钟 ({time.strftime('%H:%M:%S')})")

log("✅ SFT训练完成！（检测到 metrics.json）")
time.sleep(5)  # 确保文件写完

log("启动 DPO pipeline（合并→生成偏好对→DPO训练）...")
flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
proc = subprocess.Popen(
    [PYTHON, "run_dpo_pipeline.py"],
    creationflags=flags,
    cwd=os.getcwd(),
)
log(f"✅ DPO pipeline 已启动, PID={proc.pid}")
log(f"DPO日志: logs/dpo_pipeline.log")
log("DPO完成后，请运行: python scripts/eval_full.py && python scripts/generate_final_report.py")

log_f.close()
