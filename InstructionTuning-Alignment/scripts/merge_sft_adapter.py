#!/usr/bin/env python3
"""合并 SFT 阶段的 LoRA adapter 到 base model，生成 SFT+DPO 阶段的基座模型。

流程: base(TinyLlama) + SFT LoRA adapter  --merge_and_unload-->  完整 SFT 模型
之后 DPO 配置 model.name 指向该合并模型，DPO 再套一层 LoRA 做偏好对齐，
ref_model 用同一合并模型(frozen)作为参考，符合 "SFT 基础上做 DPO" 的标准设定。

使用方法:
    python scripts/merge_sft_adapter.py
"""
import os
# 离线模式：避免联网检查（本地模型 + 本地 adapter）
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 路径（相对项目根目录运行）
BASE = "models/TinyLlama-1.1B-Chat-v1.0"      # 原始 base model
ADAPTER = "results/week15_sft_huatuo"         # SFT 阶段输出的 LoRA adapter (huatuo真实数据)
OUT = "models/TinyLlama-SFT-merged"           # 合并后的 SFT 模型输出目录


def main():
    print("=" * 60)
    print("合并 SFT LoRA adapter -> 完整 SFT 模型")
    print("=" * 60)
    print(f"Base model : {BASE}")
    print(f"SFT adapter: {ADAPTER}")
    print(f"Output     : {OUT}")

    # 1. 加载 base model
    print("\n[1/4] 加载 base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. 加载 SFT 的 LoRA adapter
    print("[2/4] 加载 SFT LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER)

    # 3. 合并 adapter 到 base（得到完整 SFT 模型，不再是 PEFT 模型）
    print("[3/4] 合并 adapter (merge_and_unload)...")
    model = model.merge_and_unload()

    # 4. 保存合并后的完整模型 + tokenizer
    print(f"[4/4] 保存合并模型到 {OUT}...")
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    tokenizer.save_pretrained(OUT)

    # 打印参数量确认
    n = sum(p.numel() for p in model.parameters())
    print(f"\n合并完成！模型参数量: {n:,}")
    print(f"输出目录: {OUT}")
    print("下一步: DPO 配置 model.name 指向该目录，跑 SFT+DPO 对齐")
    print("=" * 60)


if __name__ == "__main__":
    main()
