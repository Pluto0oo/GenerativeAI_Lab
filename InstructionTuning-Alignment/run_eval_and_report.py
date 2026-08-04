"""立即运行评估 + 报告生成（eval_full.py + generate_final_report.py）

SFT 和 DPO 训练均已完成，此脚本直接运行评估对比和报告生成。
"""
import subprocess
import os
import time

PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LOG = os.path.abspath("logs/eval_and_report.log")
log_f = open(LOG, "w", encoding="utf-8")


def log(msg):
    print(msg)
    log_f.write(msg + "\n")
    log_f.flush()


log("=" * 60)
log("立即运行评估 + 报告生成")
log(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 60)

# 1. 运行评估
log("\n[1/2] 运行 eval_full.py（SFT vs SFT+DPO 评估，约30分钟）...")
r = subprocess.run(
    [PYTHON, "scripts/eval_full.py"],
    stdout=open("logs/eval_full.log", "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    cwd=os.getcwd()
)
log(f"eval_full.py 退出码: {r.returncode}")

if r.returncode != 0:
    log("⚠️ 评估失败！查看 logs/eval_full.log")
    log_f.close()
    exit(1)

log("✅ 评估完成！")

# 2. 运行报告生成
log("\n[2/2] 运行 generate_final_report.py（生成总结文档）...")
r = subprocess.run(
    [PYTHON, "scripts/generate_final_report.py"],
    capture_output=True, text=True, cwd=os.getcwd()
)
if r.stdout:
    log(r.stdout[-1500:])
if r.stderr:
    log(f"stderr: {r.stderr[-500:]}")
log(f"generate_final_report.py 退出码: {r.returncode}")

log("\n" + "=" * 60)
log(f"全流程完成！{time.strftime('%Y-%m-%d %H:%M:%S')}")
log("评估报告: results/week15_full_eval.md")
log("评估指标: results/week15_eval_metrics.json")
log("总结文档: reports/week15_huatuo_final_report.md")
log("=" * 60)

log_f.close()
