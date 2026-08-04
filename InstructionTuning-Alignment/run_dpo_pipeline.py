#!/usr/bin/env python3
"""DPO 全流程编排: 合并SFT adapter → 生成偏好对 → 训练DPO

顺序执行三步，输出写入 logs/dpo_pipeline.log。
前置条件: SFT 训练已完成 (results/week15_sft_huatuo 存在 adapter)

使用方法:
    python run_dpo_pipeline.py
"""
import subprocess
import os
import sys

PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
# 切到项目根目录（本脚本位于根目录）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

steps = [
    ("1. 合并SFT adapter -> 完整SFT模型", [PYTHON, "scripts/merge_sft_adapter.py"]),
    ("2. 生成DPO偏好对(Stanford方法: chosen=真实答案, rejected=SFT生成)",
     [PYTHON, "scripts/generate_dpo_pairs.py"]),
    ("3. 训练DPO", [PYTHON, "scripts/run_experiment.py", "--config",
                    "configs/experiment/week15_dpo_tinyllama.yaml",
                    "--exp_id", "week15_dpo_huatuo"]),
]

log_path = os.path.abspath("logs/dpo_pipeline.log")
os.makedirs("logs", exist_ok=True)

print(f"DPO pipeline 启动, 日志: {log_path}")

with open(log_path, "w", encoding="utf-8") as log_f:
    for name, cmd in steps:
        msg = f"\n{'='*60}\n{name}\n{'='*60}\n"
        print(msg)
        log_f.write(msg)
        log_f.flush()
        r = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=os.getcwd())
        msg2 = f"{name} 完成, 退出码={r.returncode}\n"
        print(msg2)
        log_f.write(msg2)
        log_f.flush()
        if r.returncode != 0:
            err = f"{name} 失败，停止后续步骤\n"
            print(err)
            log_f.write(err)
            sys.exit(1)

print("\nDPO pipeline 全部完成！可运行 eval_full.py 做评估")
