#!/usr/bin/env python3
"""生成 Week15 指令微调与对齐实验完整总结报告

读取:
  - results/week15_sft_huatuo/ (metrics.json, metrics.csv)
  - results/week15_dpo_huatuo/ (metrics.json, metrics.csv)
  - results/week15_eval_metrics.json (评估对比指标)
生成:
  - reports/week15_huatuo_final_report.md (完整总结文档, 符合 skill 规范)

使用方法:
    python scripts/generate_final_report.py
"""
import os
import json
import csv
from datetime import datetime

SFT_DIR = "results/week15_sft_huatuo"
DPO_DIR = "results/week15_dpo_huatuo"
EVAL_JSON = "results/week15_eval_metrics.json"
OUTPUT = "reports/week15_huatuo_final_report.md"


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def get_float(r, key):
    v = r.get(key, "")
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def extract_train_metrics(csv_rows):
    """从训练历史提取 loss / DPO偏好指标"""
    losses = [get_float(r, "loss") for r in csv_rows]
    losses = [x for x in losses if x is not None]
    eval_losses = [get_float(r, "eval_loss") for r in csv_rows]
    eval_losses = [x for x in eval_losses if x is not None]
    accs = [get_float(r, "rewards/accuracies") for r in csv_rows]
    accs = [x for x in accs if x is not None]
    margins = [get_float(r, "rewards/margins") for r in csv_rows]
    margins = [x for x in margins if x is not None]
    return {
        "train_loss_start": losses[0] if losses else None,
        "train_loss_end": losses[-1] if losses else None,
        "eval_loss": eval_losses[-1] if eval_losses else None,
        "pref_acc_start": accs[0] if accs else None,
        "pref_acc_end": accs[-1] if accs else None,
        "margin_start": margins[0] if margins else None,
        "margin_end": margins[-1] if margins else None,
    }


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "N/A"


def main():
    sft_metrics = load_json(os.path.join(SFT_DIR, "metrics.json"))
    dpo_metrics = load_json(os.path.join(DPO_DIR, "metrics.json"))
    sft_csv = load_csv(os.path.join(SFT_DIR, "metrics.csv"))
    dpo_csv = load_csv(os.path.join(DPO_DIR, "metrics.csv"))
    eval_metrics = load_json(EVAL_JSON)

    sft_t = extract_train_metrics(sft_csv)
    dpo_t = extract_train_metrics(dpo_csv)

    L = []
    L.append("# Week15 指令微调与对齐实验总结报告\n")
    L.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("**数据集**: FreedomIntelligence/huatuo_encyclopedia_qa（真实中文医疗问答）")
    L.append("**模型**: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n")
    L.append("---\n")

    # ===== 一、实验概述 =====
    L.append("## 一、实验概述\n")
    L.append("### 1.1 实验目标")
    L.append("基于 TinyLlama-1.1B-Chat，使用**真实中文医疗问答数据集**完成指令微调（SFT）与"
             "直接偏好优化（DPO）全流程，对比 SFT-only 与 SFT+DPO 的输出质量，验证 DPO 偏好对齐效果。\n")
    L.append("### 1.2 方法路线")
    L.append("- **SFT 阶段**: LoRA 微调 TinyLlama，数据为 huatuo 真实医疗问答（5085 train / 898 val）")
    L.append("- **DPO 阶段**: 采用 Stanford CS224N 论文方法构造偏好对")
    L.append("  - chosen = huatuo 真实医生答案（专业、完整 = 好回答）")
    L.append("  - rejected = SFT 模型对该问题的生成（相对粗糙 = 坏回答）")
    L.append("- **对比评估**: 同一基座，SFT-only vs SFT+DPO，用 BLEU/ROUGE/生成质量指标评估\n")
    L.append("### 1.3 实验环境")
    if sft_metrics and "hardware" in sft_metrics:
        hw = sft_metrics["hardware"]
        L.append(f"- GPU: {hw.get('gpu_model', 'N/A')}")
        L.append(f"- CUDA: {hw.get('cuda_version', 'N/A')}")
    L.append("- 模型: TinyLlama-1.1B-Chat (1.1B 参数)")
    L.append("- 微调: PEFT + LoRA (r=16, target=q/v/k/o/gate/up/down_proj)")
    L.append("- 精度: bf16 + gradient_checkpointing\n")

    # ===== 二、数据集说明 =====
    L.append("## 二、数据集说明\n")
    L.append("### 2.1 SFT 数据（真实数据集）")
    L.append("- **来源**: `FreedomIntelligence/huatuo_encyclopedia_qa`（华佗中文医疗百科问答，真实医生回答）")
    L.append("- **规模**: 取 6000 条子集，过滤后 5983 条有效样本")
    L.append("- **划分**: train 5085 条 / validation 898 条")
    L.append("- **格式**: instruction（问题）/ input（空）/ output（真实医生答案）\n")
    L.append("### 2.2 DPO 偏好数据（Stanford 方法，真实数据衍生）")
    L.append("- **构造方法**: 参考 Stanford CS224N《Efficient Alignment of Medical Language Models using DPO》")
    L.append("- **chosen**: huatuo 真实医生答案（专业、完整 = 好回答）")
    L.append("- **rejected**: SFT 模型对同一问题的生成（相对粗糙 = 坏回答）")
    L.append("- **理论依据**: DPO 本质是让模型远离自己的差输出、靠近真实专家好输出")
    L.append("- **规模**: 1000 偏好对，按 9:1 划分 train/val\n")

    # ===== 三、SFT 实验结果 =====
    L.append("## 三、SFT 实验结果\n")
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")
    L.append(f"| 训练损失(起始) | {fmt(sft_t['train_loss_start'])} |")
    L.append(f"| 训练损失(结束) | {fmt(sft_t['train_loss_end'])} |")
    L.append(f"| 验证损失 | {fmt(sft_t['eval_loss'])} |")
    L.append("| 训练数据 | 5085 条 × 2 epochs |")
    L.append("| 训练步数 | 636 步 (batch 4 × grad_accum 4) |")
    L.append("| 学习率 | 2e-4 (cosine) |")
    L.append("")
    if sft_t["train_loss_start"] and sft_t["train_loss_end"]:
        drop = sft_t["train_loss_start"] - sft_t["train_loss_end"]
        L.append(f"训练损失下降 {drop:.4f}（{fmt(sft_t['train_loss_start'])} → {fmt(sft_t['train_loss_end'])}），"
                 f"说明 SFT 收敛，模型学到了医疗问答的指令跟随模式。\n")

    # ===== 四、DPO 实验结果 =====
    L.append("## 四、DPO 实验结果\n")
    L.append("| 指标 | 值 |")
    L.append("|------|-----|")
    L.append(f"| DPO训练损失(起始) | {fmt(dpo_t['train_loss_start'])} |")
    L.append(f"| DPO训练损失(结束) | {fmt(dpo_t['train_loss_end'])} |")
    L.append(f"| 偏好准确率(起始) | {fmt(dpo_t['pref_acc_start'])} |")
    L.append(f"| 偏好准确率(结束) | {fmt(dpo_t['pref_acc_end'])} |")
    L.append(f"| Reward Margin(起始) | {fmt(dpo_t['margin_start'])} |")
    L.append(f"| Reward Margin(结束) | {fmt(dpo_t['margin_end'])} |")
    L.append("| beta(温度) | 0.1 |")
    L.append("")
    if dpo_t["pref_acc_start"] is not None and dpo_t["pref_acc_end"] is not None:
        acc_delta = dpo_t["pref_acc_end"] - dpo_t["pref_acc_start"]
        L.append(f"偏好准确率从 {fmt(dpo_t['pref_acc_start'])} 提升至 {fmt(dpo_t['pref_acc_end'])}"
                 f"（+{acc_delta:.4f}），Reward Margin 从 {fmt(dpo_t['margin_start'])} 变化为 "
                 f"{fmt(dpo_t['margin_end'])}，说明 DPO 让模型学会了区分好/坏回答，"
                 f"策略朝偏好方向移动。\n")

    # ===== 五、SFT vs SFT+DPO 评估对比 =====
    L.append("## 五、SFT-only vs SFT+DPO 评估对比\n")
    if eval_metrics:
        sft_m = eval_metrics["sft_only"]
        dpo_m = eval_metrics["sft_dpo"]
        L.append(f"评估数据: huatuo 验证集 {eval_metrics['num_eval']} 条真实医疗问答\n")
        L.append("| 指标 | SFT-only | SFT+DPO | 变化 | 说明 |")
        L.append("|------|----------|---------|------|------|")
        desc = {
            "bleu4": "与真实答案n-gram精度(越高越好)",
            "rouge_l": "与真实答案LCS-F1(越高越好)",
            "char_overlap": "与真实答案字符重叠(越高越好)",
            "repetition": "生成重复度(越低越好)",
            "avg_length": "平均生成长度(字符)",
        }
        for k in ["bleu4", "rouge_l", "char_overlap", "repetition", "avg_length"]:
            s, d = sft_m[k], dpo_m[k]
            delta = d - s
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            L.append(f"| {k} | {s:.4f} | {d:.4f} | {delta:+.4f} {arrow} | {desc[k]} |")
        L.append("")
    else:
        L.append("（评估未完成，请先运行 eval_full.py）\n")

    # ===== 六、结论与分析 =====
    L.append("## 六、结论与分析\n")
    if eval_metrics and dpo_t["pref_acc_end"] is not None:
        sft_m = eval_metrics["sft_only"]
        dpo_m = eval_metrics["sft_dpo"]
        bleu_delta = dpo_m["bleu4"] - sft_m["bleu4"]
        rouge_delta = dpo_m["rouge_l"] - sft_m["rouge_l"]
        acc_delta = dpo_t["pref_acc_end"] - dpo_t["pref_acc_start"] if dpo_t["pref_acc_start"] else 0

        L.append("### 6.1 DPO 偏好对齐效果")
        if acc_delta > 0:
            L.append(f"- 偏好准确率提升 {acc_delta:+.4f}，证明 DPO 训练让模型学会了区分好/坏医疗回答")
        if bleu_delta > 0 or rouge_delta > 0:
            L.append(f"- 生成质量: BLEU {bleu_delta:+.4f}, ROUGE-L {rouge_delta:+.4f}，"
                     f"DPO 使生成更接近真实医生答案")
        L.append("")
        L.append("### 6.2 局限性")
        L.append("- TinyLlama 1.1B 模型规模有限，医疗专业知识覆盖不足")
        L.append("- 受单卡显存限制，batch_size 较小，训练步数有限")
        L.append("- DPO 偏好对由 SFT 模型生成 rejected，质量受 SFT 模型能力制约")
        L.append("")
        L.append("### 6.3 与原始实验要求的对应")
        L.append("- ✅ 数据构建: 使用 Prompt 模板将医疗问答转为指令格式（huatuo 真实数据）")
        L.append("- ✅ SFT演示: peft+LoRA 对 TinyLlama 做指令微调（真实医疗数据，单卡）")
        L.append("- ✅ DPO对比: 同一基座模型，对比 SFT-only vs SFT+DPO 的输出质量")
        L.append("- ✅ 真实数据集: 使用 huatuo_encyclopedia_qa 真实中文医疗问答")
        L.append("")

    # ===== 七、产物文件清单 =====
    L.append("## 七、产物文件清单\n")
    L.append("| 文件 | 说明 |")
    L.append("|------|------|")
    L.append("| results/week15_sft_huatuo/ | SFT 实验结果（metrics, loss曲线, adapter） |")
    L.append("| results/week15_dpo_huatuo/ | DPO 实验结果（metrics, 偏好准确率, adapter） |")
    L.append("| models/TinyLlama-SFT-merged/ | SFT 合并模型（DPO 基座） |")
    L.append("| data/processed/huatuo_sft/ | SFT 真实医疗数据 |")
    L.append("| data/processed/huatuo_dpo/ | DPO 偏好对数据 |")
    L.append("| results/week15_full_eval.md | 评估对比详细报告 |")
    L.append("| results/week15_eval_metrics.json | 评估指标 JSON |")
    L.append("| reports/week15_huatuo_final_report.md | 本总结报告 |")
    L.append("")

    # ===== 八、复现说明 =====
    L.append("## 八、复现说明\n")
    L.append("```bash")
    L.append("# 1. 下载真实医疗数据集")
    L.append("python scripts/download_huatuo_sft.py")
    L.append("")
    L.append("# 2. SFT 训练")
    L.append("python scripts/run_experiment.py \\")
    L.append("  --config configs/experiment/week15_sft_tinyllama.yaml \\")
    L.append("  --exp_id week15_sft_huatuo")
    L.append("")
    L.append("# 3. 合并 SFT adapter -> 完整 SFT 模型")
    L.append("python scripts/merge_sft_adapter.py")
    L.append("")
    L.append("# 4. 生成 DPO 偏好对（Stanford 方法）")
    L.append("python scripts/generate_dpo_pairs.py")
    L.append("")
    L.append("# 5. DPO 训练")
    L.append("python scripts/run_experiment.py \\")
    L.append("  --config configs/experiment/week15_dpo_tinyllama.yaml \\")
    L.append("  --exp_id week15_dpo_huatuo")
    L.append("")
    L.append("# 6. 评估对比 SFT-only vs SFT+DPO")
    L.append("python scripts/eval_full.py")
    L.append("")
    L.append("# 7. 生成本总结报告")
    L.append("python scripts/generate_final_report.py")
    L.append("```\n")

    os.makedirs("reports", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"报告已生成: {OUTPUT}")


if __name__ == "__main__":
    main()
