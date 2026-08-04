"""轮询等待SFT训练完成（每轮最多等9分钟）"""
import os
import time

target = "results/week15_sft_huatuo/metrics.json"
t = 0
while not os.path.exists(target) and t < 540:
    time.sleep(30)
    t += 30
done = os.path.exists(target)
print(f"waited {t}s, sft_done={done}")
if done:
    print("SFT完成！monitor_and_dpo将自动启动DPO pipeline")
