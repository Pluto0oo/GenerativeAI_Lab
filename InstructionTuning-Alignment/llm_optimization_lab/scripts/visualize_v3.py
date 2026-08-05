#!/usr/bin/env python3
"""v3可视化 - SFT微调前后对比"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_PATH = os.path.join(PROJECT_ROOT, "results", "experiment_v2_real", "metrics_summary.json")
V3_PATH = os.path.join(PROJECT_ROOT, "results", "experiment_v3_sft", "metrics_summary.json")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    with open(V2_PATH, 'r', encoding='utf-8') as f:
        v2 = json.load(f)
    with open(V3_PATH, 'r', encoding='utf-8') as f:
        v3 = json.load(f)

    # 统一策略名映射
    strategy_map = {
        'zero_shot': ('zero_shot_v2', 'zero_shot'),
        'few_shot': ('few_shot_v2', 'few_shot'),
        'cot': ('cot_v2', 'cot'),
        'self_consistency': ('self_consistency_v2', 'self_consistency'),
        'tot_v2': ('tot_v2', 'tot_v2'),
        'cot_verifier': ('cot_verifier', 'cot_verifier'),
    }

    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Cons.', 'ToT v2', 'CoT+Ver']
    v2_accs = [v2[strategy_map[s][0]]['accuracy'] * 100 for s in strategy_map]
    v3_accs = [v3[strategy_map[s][1]]['accuracy'] * 100 for s in strategy_map]
    v3_hals = [v3[strategy_map[s][1]]['hallucination_rate'] * 100 for s in strategy_map]

    # ===== 图1: SFT前后准确率对比柱状图 =====
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, v2_accs, width, label='v2 微调前 (TinyLlama-SFT-merged)',
                   color='#90A4AE', edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + width/2, v3_accs, width, label='v3 MedQA-SFT微调后',
                   color='#2196F3', edgecolor='black', linewidth=0.6)

    for bar, v in zip(bars1, v2_accs):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.5, f'{v:.0f}%',
                ha='center', fontsize=10, color='#546E7A')
    for bar, v in zip(bars2, v3_accs):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.5, f'{v:.0f}%',
                ha='center', fontsize=11, fontweight='bold', color='#1565C0')

    # 标注变化幅度
    for i in range(len(labels)):
        diff = v3_accs[i] - v2_accs[i]
        color = '#2E7D32' if diff > 0 else ('#C62828' if diff < 0 else '#757575')
        sign = '+' if diff > 0 else ''
        ax.annotate(f'{sign}{diff:.0f}', xy=(i, max(v2_accs[i], v3_accs[i]) + 5),
                    ha='center', fontsize=10, fontweight='bold', color=color)

    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='随机基线 (25%)')
    ax.set_xlabel('Prompt 策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('MedQA-SFT微调前后准确率对比\n(真实MedQA 100题 · 6种Prompt策略)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v3_sft_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v3_sft_comparison.png")

    # ===== 图2: SFT前后幻觉率对比 =====
    fig, ax = plt.subplots(figsize=(14, 7))
    v2_hals = [v2[strategy_map[s][0]]['hallucination_rate'] * 100 for s in strategy_map]
    bars1 = ax.bar(x - width/2, v2_hals, width, label='v2 微调前',
                   color='#EF9A9A', edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + width/2, v3_hals, width, label='v3 微调后',
                   color='#66BB6A', edgecolor='black', linewidth=0.6)

    for bar, v in zip(bars1, v2_hals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2., v + 1.5, f'{v:.0f}%',
                    ha='center', fontsize=10, color='#C62828')
    for bar, v in zip(bars2, v3_hals):
        ax.text(bar.get_x() + bar.get_width()/2., v + 1.5, f'{v:.0f}%',
                ha='center', fontsize=10, fontweight='bold', color='#2E7D32')

    ax.set_xlabel('Prompt 策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('幻觉率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('MedQA-SFT微调前后幻觉率对比\n(SFT后所有策略幻觉率降至0%)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v3_hallucination_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v3_hallucination_comparison.png")

    # ===== 图3: v3策略排名 =====
    v3_items = sorted(zip(labels, v3_accs, v3_hals), key=lambda x: x[1], reverse=True)
    v3_labels = [x[0] for x in v3_items]
    v3_sorted_accs = [x[1] for x in v3_items]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#795548']

    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(v3_labels, v3_sorted_accs, color=colors, edgecolor='black', linewidth=0.6, width=0.6)
    for bar, acc in zip(bars, v3_sorted_accs):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 0.8, f'{acc:.0f}%',
                ha='center', fontsize=13, fontweight='bold')
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='随机基线 (25%)')
    ax.set_xlabel('Prompt 策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('MedQA-SFT模型 6策略准确率排名\n(所有策略幻觉率均为0%)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, max(v3_sorted_accs) * 1.4 + 5)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v3_strategy_ranking.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v3_strategy_ranking.png")

    # ===== 图4: 综合仪表板 =====
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) SFT前后准确率对比
    ax = axes[0, 0]
    bars1 = ax.bar(x - width/2, v2_accs, width, label='v2微调前', color='#90A4AE', edgecolor='black')
    bars2 = ax.bar(x + width/2, v3_accs, width, label='v3微调后', color='#2196F3', edgecolor='black')
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('(a) SFT前后准确率对比', fontsize=12, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 45)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) SFT前后幻觉率对比
    ax = axes[0, 1]
    bars1 = ax.bar(x - width/2, v2_hals, width, label='v2微调前', color='#EF9A9A', edgecolor='black')
    bars2 = ax.bar(x + width/2, v3_hals, width, label='v3微调后', color='#66BB6A', edgecolor='black')
    ax.set_title('(b) SFT前后幻觉率对比', fontsize=12, fontweight='bold')
    ax.set_ylabel('幻觉率 (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) 准确率vs幻觉率散点
    ax = axes[1, 0]
    for i in range(len(labels)):
        ax.scatter(v2_hals[i], v2_accs[i], s=200, c='#EF5350', edgecolors='black',
                   linewidth=1, alpha=0.7, zorder=3, marker='o')
        ax.scatter(v3_hals[i], v3_accs[i], s=200, c='#2196F3', edgecolors='black',
                   linewidth=1, alpha=0.8, zorder=3, marker='^')
        # 箭头从v2到v3
        ax.annotate('', xy=(v3_hals[i], v3_accs[i]), xytext=(v2_hals[i], v2_accs[i]),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, alpha=0.5))
        ax.annotate(labels[i], xy=(v3_hals[i], v3_accs[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, fontweight='bold')
    ax.scatter([], [], c='#EF5350', marker='o', s=100, label='v2微调前', edgecolors='black')
    ax.scatter([], [], c='#2196F3', marker='^', s=100, label='v3微调后', edgecolors='black')
    ax.set_title('(c) 准确率 vs 幻觉率 (SFT效果)', fontsize=12, fontweight='bold')
    ax.set_xlabel('幻觉率 (%)', fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 45)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) 提升幅度
    ax = axes[1, 1]
    diffs = [v3_accs[i] - v2_accs[i] for i in range(len(labels))]
    colors_diff = ['#4CAF50' if d > 0 else '#F44336' for d in diffs]
    bars = ax.bar(labels, diffs, color=colors_diff, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, d in zip(bars, diffs):
        sign = '+' if d > 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2., d + (0.5 if d >= 0 else -1.5),
                f'{sign}{d:.0f}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_title('(d) SFT带来的准确率变化', fontsize=12, fontweight='bold')
    ax.set_ylabel('准确率变化 (百分点)', fontsize=11)
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylim(min(diffs) - 5, max(diffs) + 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('LLM优化实验 v3 综合分析仪表板\n(MedQA-SFT微调 vs 微调前 · 真实MedQA 100题)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v3_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v3_dashboard.png")

    print("\n" + "=" * 60)
    print("全部v3图表生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
