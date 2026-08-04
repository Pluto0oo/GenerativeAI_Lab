#!/usr/bin/env python3
"""下载真实中文医疗问答数据集 huatuo_encyclopedia_qa 并转为 SFT 指令格式

数据来源: FreedomIntelligence/huatuo_encyclopedia_qa (真实中文医疗百科问答)
输出格式: instruction / input / output (符合 skill 规范的 SFT 格式)

使用方法:
    python scripts/download_huatuo_sft.py
"""
import os
import json
import time

# 需要联网下载数据集
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)

from datasets import load_dataset

OUT_DIR = "data/processed/huatuo_sft"
NUM_SAMPLES = 6000  # 取子集控制训练时间（huatuo 总量约36万）


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== 1. 下载子集（带重试，应对网络不稳定）=====
    print(f"下载 huatuo_encyclopedia_qa 前 {NUM_SAMPLES} 条...")
    ds = None
    for attempt in range(6):
        try:
            ds = load_dataset(
                "FreedomIntelligence/huatuo_encyclopedia_qa",
                split=f"train[:{NUM_SAMPLES}]"
            )
            print(f"  下载成功: {len(ds)} 条")
            break
        except Exception as e:
            print(f"  尝试 {attempt+1}/6 失败: {str(e)[:150]}")
            time.sleep(5)

    if ds is None:
        raise RuntimeError("下载 huatuo 数据集失败，请检查网络")

    # ===== 2. 转为 SFT 指令格式 =====
    # huatuo 的 questions/answers 是嵌套 list（如 [['问题']]），需递归剥到字符串
    def flatten_to_str(v):
        while isinstance(v, list):
            v = v[0] if v else ""
        return str(v).strip()

    def to_instruction(example):
        return {
            "instruction": flatten_to_str(example["questions"]),
            "input": "",
            "output": flatten_to_str(example["answers"]),
        }

    ds = ds.map(to_instruction, remove_columns=ds.column_names)
    # 过滤空样本和过短样本
    ds = ds.filter(lambda x: len(x["instruction"]) > 3 and len(x["output"]) > 5)
    print(f"  过滤后: {len(ds)} 条有效样本")

    # ===== 3. 划分 train / validation =====
    split = ds.train_test_split(test_size=0.15, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    # ===== 4. 保存为 jsonl =====
    def save_jsonl(dataset, path):
        with open(path, "w", encoding="utf-8") as f:
            for x in dataset:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    save_jsonl(train_ds, os.path.join(OUT_DIR, "train.jsonl"))
    save_jsonl(val_ds, os.path.join(OUT_DIR, "validation.jsonl"))

    print(f"\nTrain: {len(train_ds)} 条 -> {OUT_DIR}/train.jsonl")
    print(f"Validation: {len(val_ds)} 条 -> {OUT_DIR}/validation.jsonl")

    # 打印样本示例
    print("\n样本示例:")
    for i in range(2):
        x = train_ds[i]
        print(f"  [{i+1}] Q: {x['instruction'][:70]}")
        print(f"      A: {x['output'][:70]}")
    print(f"\n数据已保存到 {OUT_DIR}/，可用于 SFT 训练")


if __name__ == "__main__":
    main()
