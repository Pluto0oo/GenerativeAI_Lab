#!/usr/bin/env python3
"""综合评估: SFT-only vs SFT+DPO 在 huatuo 真实医疗问答上的表现

指标(全部本地计算, 字符级, 中文友好):
  - BLEU-4 (字符级 n-gram 精度 + brevity penalty)
  - ROUGE-L (最长公共子序列 F1)
  - 平均生成长度
  - 重复度 (重复字符占比)
  - 与真实答案的字符重叠率

使用方法:
    python scripts/eval_full.py
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import math
import torch
from collections import Counter
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== 路径配置 =====
SFT_BASE = "models/TinyLlama-1.1B-Chat-v1.0"
SFT_ADAPTER = "results/week15_sft_huatuo"
DPO_BASE = "models/TinyLlama-SFT-merged"
DPO_ADAPTER = "results/week15_dpo_huatuo"
VAL_DATA = "data/processed/huatuo_sft/validation.jsonl"
NUM_EVAL = 100
MAX_NEW_TOKENS = 256
OUTPUT_MD = "results/week15_full_eval.md"


# ===== 手动指标实现(字符级, 适合中文) =====
def char_ngrams(text, n):
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def bleu_char(pred, ref, max_n=4):
    """字符级 BLEU-4, 单参考"""
    if not pred or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        pred_ng = Counter(char_ngrams(pred, n))
        ref_ng = Counter(char_ngrams(ref, n))
        overlap = sum((pred_ng & ref_ng).values())
        total = sum(pred_ng.values())
        precisions.append(overlap / total if total > 0 else 0.0)
    # brevity penalty
    bp = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / max(len(pred), 1))
    if min(precisions) == 0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return bp * geo_mean


def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(pred, ref):
    """字符级 ROUGE-L F1"""
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    p = lcs / len(pred)
    r = lcs / len(ref)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def repetition_ratio(text):
    """重复字符占比(去重后字符数/总字符数, 越小越重复)"""
    if not text:
        return 0.0
    return 1 - len(set(text)) / len(text)


def char_overlap(pred, ref):
    """与真实答案的字符集合重叠率(Jaccard)"""
    if not pred or not ref:
        return 0.0
    sp, sr = set(pred), set(ref)
    return len(sp & sr) / len(sp | sr)


# ===== 模型加载与生成 =====
def load_model(base, adapter):
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def free_model(model):
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# ===== 主流程 =====
def main():
    print("=" * 60)
    print("SFT-only vs SFT+DPO 综合评估 (huatuo 真实医疗问答)")
    print("=" * 60)

    # 读取验证集
    data = []
    with open(VAL_DATA, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    data = data[:NUM_EVAL]
    prompts = [x["instruction"] for x in data]
    references = [x["output"] for x in data]
    print(f"评估样本数: {len(data)}")

    # 1. SFT-only 生成
    print("\n[1/3] 加载 SFT-only 模型并生成...")
    model, tok = load_model(SFT_BASE, SFT_ADAPTER)
    sft_preds = []
    for i, p in enumerate(prompts):
        sft_preds.append(generate(model, tok, p))
        if (i + 1) % 20 == 0:
            print(f"  SFT 生成 {i+1}/{len(prompts)}")
    free_model(model)
    del tok

    # 2. SFT+DPO 生成
    print("\n[2/3] 加载 SFT+DPO 模型并生成...")
    model, tok = load_model(DPO_BASE, DPO_ADAPTER)
    dpo_preds = []
    for i, p in enumerate(prompts):
        dpo_preds.append(generate(model, tok, p))
        if (i + 1) % 20 == 0:
            print(f"  DPO 生成 {i+1}/{len(prompts)}")
    free_model(model)
    del tok

    # 3. 计算指标
    print("\n[3/3] 计算指标...")
    def aggregate(preds):
        bleus = [bleu_char(p, r) for p, r in zip(preds, references)]
        rouges = [rouge_l(p, r) for p, r in zip(preds, references)]
        overlaps = [char_overlap(p, r) for p, r in zip(preds, references)]
        reps = [repetition_ratio(p) for p in preds]
        lengths = [len(p) for p in preds]
        return {
            "bleu4": sum(bleus) / len(bleus),
            "rouge_l": sum(rouges) / len(rouges),
            "char_overlap": sum(overlaps) / len(overlaps),
            "repetition": sum(reps) / len(reps),
            "avg_length": sum(lengths) / len(lengths),
        }

    sft_m = aggregate(sft_preds)
    dpo_m = aggregate(dpo_preds)

    print("\n===== 指标对比 =====")
    print(f"{'指标':<16} {'SFT-only':<14} {'SFT+DPO':<14} {'变化':<10}")
    for k in ["bleu4", "rouge_l", "char_overlap", "repetition", "avg_length"]:
        s, d = sft_m[k], dpo_m[k]
        delta = d - s
        print(f"{k:<16} {s:<14.4f} {d:<14.4f} {delta:+.4f}")

    # 4. 保存JSON指标(供报告脚本读取)
    eval_metrics = {"sft_only": sft_m, "sft_dpo": dpo_m, "num_eval": len(prompts)}
    with open("results/week15_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, ensure_ascii=False, indent=2)

    # 5. 写报告
    write_report(sft_m, dpo_m, prompts, references, sft_preds, dpo_preds)
    print(f"\n报告已生成: {OUTPUT_MD}")
    print(f"指标JSON: results/week15_eval_metrics.json")


def write_report(sft_m, dpo_m, prompts, references, sft_preds, dpo_preds):
    lines = []
    lines.append("# SFT-only vs SFT+DPO 综合评估报告（真实医疗数据）\n")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **评估数据**: huatuo_encyclopedia_qa 验证集 {len(prompts)} 条真实医疗问答")
    lines.append(f"- **解码方式**: greedy (do_sample=False)")
    lines.append("- **指标**: 字符级 BLEU-4 / ROUGE-L / 字符重叠 / 重复度 / 平均长度\n")
    lines.append("---\n")

    # 指标对比表
    lines.append("## 一、指标对比\n")
    lines.append("| 指标 | SFT-only | SFT+DPO | 变化 | 说明 |")
    lines.append("|------|----------|---------|------|------|")
    desc = {
        "bleu4": "与真实答案的n-gram精度(越高越好)",
        "rouge_l": "与真实答案的最长公共子序列F1(越高越好)",
        "char_overlap": "与真实答案字符重叠率(越高越好)",
        "repetition": "生成重复度(越低越好)",
        "avg_length": "平均生成长度(字符)",
    }
    for k in ["bleu4", "rouge_l", "char_overlap", "repetition", "avg_length"]:
        s, d = sft_m[k], dpo_m[k]
        delta = d - s
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        lines.append(f"| {k} | {s:.4f} | {d:.4f} | {delta:+.4f} {arrow} | {desc[k]} |")
    lines.append("")

    # 样本对比
    lines.append("## 二、生成样本对比（前8条）\n")
    for i in range(min(8, len(prompts))):
        lines.append(f"### 样本 {i+1}")
        lines.append(f"- **问题**: {prompts[i]}")
        lines.append(f"- **真实答案**: {references[i][:120]}...")
        lines.append("")
        lines.append("**SFT-only 输出**:")
        lines.append("```")
        lines.append(sft_preds[i][:300])
        lines.append("```")
        lines.append("**SFT+DPO 输出**:")
        lines.append("```")
        lines.append(dpo_preds[i][:300])
        lines.append("```")
        lines.append("---\n")

    # 结论
    lines.append("## 三、结论\n")
    lines.append("- BLEU/ROUGE 反映模型生成与真实医生答案的契合度")
    lines.append("- 重复度反映生成质量（低=少退化）")
    lines.append("- DPO 目标: 让模型靠近真实好答案(chosen)、远离SFT差生成(rejected)")
    lines.append("- 若 SFT+DPO 的 BLEU/ROUGE 高于 SFT-only，说明 DPO 偏好对齐生效")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
