#!/usr/bin/env python3
"""构造 DPO 偏好对（Stanford CS224N 论文方法）

策略:
  - chosen  = huatuo 真实医生答案（专业、完整 = "好回答"）
  - rejected = SFT 模型对该问题的生成（相对粗糙 = "坏回答"）

这是真实数据衍生的医疗好/坏回答偏好对，符合 DPO 本质:
  让模型远离自己的差输出，靠近真实专家好输出。

前置条件: SFT 已训练并合并为 models/TinyLlama-SFT-merged

使用方法:
    python scripts/generate_dpo_pairs.py
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/TinyLlama-SFT-merged"   # 合并后的SFT模型
SFT_DATA = "data/processed/huatuo_sft/train.jsonl"
OUT_DIR = "data/processed/huatuo_dpo"
NUM_PAIRS = 1000          # 生成1000对偏好（控制时间）
MAX_NEW_TOKENS = 256


def generate(model, tokenizer, prompt):
    """用 chat template 生成回答（greedy 保证可复现）"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 读取 SFT 训练数据子集（作为 DPO 构造源）
    data = []
    with open(SFT_DATA, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    data = data[:NUM_PAIRS]
    print(f"读取 {len(data)} 条 huatuo 数据作为 DPO 构造源")

    # 2. 加载 SFT 合并模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"SFT合并模型不存在: {MODEL_PATH}，请先运行合并脚本")
    print(f"加载 SFT 合并模型: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 生成偏好对
    pairs = []
    print("开始生成偏好对...")
    for i, item in enumerate(data):
        prompt = item["instruction"]
        chosen = item["output"]
        rejected = generate(model, tokenizer, prompt)
        # 过滤空 rejected 或与 chosen 完全相同的
        if rejected and len(rejected) > 5 and rejected != chosen:
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(data)}, 有效偏好对 {len(pairs)}")

    print(f"\n总有效偏好对: {len(pairs)}")

    # 4. 划分 train / validation
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(50, len(pairs) // 10)
    val = pairs[:n_val]
    train = pairs[n_val:]

    def save_jsonl(lst, path):
        with open(path, "w", encoding="utf-8") as f:
            for x in lst:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    save_jsonl(train, os.path.join(OUT_DIR, "train.jsonl"))
    save_jsonl(val, os.path.join(OUT_DIR, "validation.jsonl"))
    print(f"Train: {len(train)} -> {OUT_DIR}/train.jsonl")
    print(f"Validation: {len(val)} -> {OUT_DIR}/validation.jsonl")

    # 打印示例
    print("\n偏好对示例:")
    p = pairs[0]
    print(f"  Prompt:   {p['prompt'][:60]}")
    print(f"  Chosen:   {p['chosen'][:60]}")
    print(f"  Rejected: {p['rejected'][:60]}")


if __name__ == "__main__":
    main()
