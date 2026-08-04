"""监控DPO训练完成，自动运行评估和报告生成

检测 results/week15_dpo_huatuo/metrics.json 出现（=DPO完成），
然后自动运行 eval_full.py（评估对比）+ generate_final_report.py（生成总结文档）。
"""
import subprocess
import os
import time

PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DPO_METRICS = "results/week15_dpo_huatuo/metrics.json"
LOG_PATH = os.path.abspath("logs/monitor_and_eval.log")
os.makedirs("logs", exist_ok=True)

log_f = open(LOG_PATH, "w", encoding="utf-8")


def log(msg):
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()


log("=" * 60)
log("监控DPO训练完成，完成后自动运行评估+报告")
log("=" * 60)

wait_start = time.time()
while not os.path.exists(DPO_METRICS):
    time.sleep(60)
    elapsed = (time.time() - wait_start) / 60
    log(f"  等待DPO完成... 已等 {elapsed:.1f} 分钟 ({time.strftime('%H:%M:%S')})")

log("✅ DPO训练完成！（检测到 metrics.json）")
time.sleep(10)  # 等GPU显存释放

# 1. 运行评估
log("\n[1/2] 运行 eval_full.py（SFT vs SFT+DPO 评估，约30分钟）...")
r = subprocess.run([PYTHON, "scripts/eval_full.py"],
                   stdout=open("logs/eval_full.log", "w", encoding="utf-8"),
                   stderr=subprocess.STDOUT, cwd=os.getcwd())
log(f"eval_full.py 退出码: {r.returncode}")

# 2. 运行报告生成
log("\n[2/2] 运行 generate_final_report.py（生成总结文档）...")
r = subprocess.run([PYTHON, "scripts/generate_final_report.py"],
                   capture_output=True, text=True, cwd=os.getcwd())
if r.stdout:
    log(r.stdout[-1500:])
log(f"generate_final_report.py 退出码: {r.returncode}")

log("\n" + "=" * 60)
log("✅ 全流程完成！")
log("评估报告: results/week15_full_eval.md")
log("评估指标: results/week15_eval_metrics.json")
log("总结文档: reports/week15_huatuo_final_report.md")
log("=" * 60)

log_f.close()
