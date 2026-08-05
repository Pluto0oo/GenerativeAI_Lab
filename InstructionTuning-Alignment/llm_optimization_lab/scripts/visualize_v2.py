#!/usr/bin/env python3
"""LLM优化实验 v2 可视化 - 真实MedQA数据集结果"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "experiment_v2_real", "metrics_summary.json")
OLD_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "experiment_optimized", "metrics_summary.json")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS_6 = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#795548']
COLOR_ACC = '#4CAF50'
COLOR_HAL = '#F44336'


def main():
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        new = json.load(f)
    with open(OLD_RESULTS_PATH, 'r', encoding='utf-8') as f:
        old = json.load(f)

    # ===== 图1: 6策略准确率排名柱状图 =====
    items = sorted(new.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    names = [k for k, _ in items]
    accs = [v['accuracy'] * 100 for _, v in items]
    labels = ['ToT v2\n(分段生成)', 'Zero-Shot v2\n(角色+约束)', 'CoT v2\n(2步推理)',
              'CoT+Verifier\n(验证)', 'SC v2\n(5路径投票)', 'Few-Shot v2\n(2示例)']
    labels = [labels[i] for i in range(len(items))]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(labels, accs, color=COLORS_6, edgecolor='black', linewidth=0.6, width=0.65)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.6, label='随机基线 (25%)')
    ax.set_xlabel('Prompt 策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('真实MedQA数据集上6种Prompt策略准确率对比\n(100题真实USMLE医疗选择题)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylim(0, max(accs) * 1.4 + 5)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v2_accuracy_ranking.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v2_accuracy_ranking.png")

    # ===== 图2: 准确率vs幻觉率散点图 =====
    fig, ax = plt.subplots(figsize=(11, 7.5))
    short_names = {
        'tot_v2': 'ToT v2', 'zero_shot_v2': 'ZS v2', 'cot_v2': 'CoT v2',
        'cot_verifier': 'CoT+Ver', 'self_consistency_v2': 'SC v2', 'few_shot_v2': 'FS v2'
    }
    for i, (k, v) in enumerate(items):
        acc = v['accuracy'] * 100
        hal = v['hallucination_rate'] * 100
        ax.scatter(hal, acc, s=420, c=COLORS_6[i], edgecolors='black',
                   linewidth=1.3, alpha=0.88, zorder=3)
        offset_x, offset_y = 3, 2
        if k == 'tot_v2':
            offset_x, offset_y = -20, 3
        elif k == 'zero_shot_v2':
            offset_x, offset_y = 3, 3
        elif k == 'few_shot_v2':
            offset_x, offset_y = 3, -5
        ax.annotate(short_names.get(k, k), xy=(hal, acc),
                    xytext=(hal + offset_x, acc + offset_y),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.4)
    ax.text(3, 28, '理想区域\n(高准确率·低幻觉)', fontsize=10, color='#2E7D32', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.7))
    ax.set_xlabel('幻觉率 / 无效答案率 (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('真实MedQA：准确率 vs 幻觉率\n(左上角为理想区域，ToT v2 最优)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 40)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v2_acc_vs_hal.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v2_acc_vs_hal.png")

    # ===== 图3: ToT v1 vs v2 对比（核心优化成果） =====
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 子图a: ToT准确率对比
    ax = axes[0]
    tot_accs = [old['tot_v1']['accuracy'] * 100, new['tot_v2']['accuracy'] * 100]
    tot_hals = [old['tot_v1']['hallucination_rate'] * 100, new['tot_v2']['hallucination_rate'] * 100]
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width/2, tot_accs, width, label='准确率', color='#4CAF50', edgecolor='black')
    bars2 = ax.bar(x + width/2, tot_hals, width, label='幻觉率', color='#F44336', edgecolor='black', alpha=0.8)
    for bar, v in zip(bars1, tot_accs):
        ax.text(bar.get_x() + bar.get_width()/2., v + 1, f'{v:.0f}%', ha='center', fontweight='bold')
    for bar, v in zip(bars2, tot_hals):
        ax.text(bar.get_x() + bar.get_width()/2., v + 1, f'{v:.0f}%', ha='center', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['ToT v1\n(单次生成3分支)', 'ToT v2\n(分段:分析→选择)'], fontsize=11)
    ax.set_ylabel('百分比 (%)', fontsize=12, fontweight='bold')
    ax.set_title('(a) ToT策略优化效果\n准确率 0%→27%，幻觉率 100%→0%', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 子图b: 所有策略准确率对比(v1构造数据 vs v2真实数据)
    ax = axes[1]
    common_strategies = ['zero_shot_v2', 'few_shot_v2', 'cot_v2', 'self_consistency_v2']
    common_labels = ['ZS v2', 'FS v2', 'CoT v2', 'SC v2']
    v1_accs = [old[s]['accuracy'] * 100 for s in common_strategies]
    v2_accs = [new[s]['accuracy'] * 100 for s in common_strategies]
    x = np.arange(len(common_labels))
    bars1 = ax.bar(x - width/2, v1_accs, width, label='v1(构造数据40题)', color='#B0BEC5', edgecolor='black')
    bars2 = ax.bar(x + width/2, v2_accs, width, label='v2(真实MedQA100题)', color='#2196F3', edgecolor='black')
    for bar, v in zip(bars1, v1_accs):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.5, f'{v:.0f}', ha='center', fontsize=9)
    for bar, v in zip(bars2, v2_accs):
        ax.text(bar.get_x() + bar.get_width()/2., v + 0.5, f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_labels, fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=12, fontweight='bold')
    ax.set_title('(b) v1 vs v2 准确率对比\n(数据集从构造改为真实MedQA)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(v1_accs), max(v2_accs)) * 1.5 + 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v2_tot_improvement.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v2_tot_improvement.png")

    # ===== 图4: 综合仪表板 =====
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) 准确率排名
    ax = axes[0, 0]
    bars = ax.bar([short_names.get(k, k) for k, _ in items], accs,
                  color=COLORS_6, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 0.5, f'{acc:.0f}%',
                ha='center', fontsize=10, fontweight='bold')
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('(a) 6策略准确率排名', fontsize=12, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_ylim(0, max(accs) * 1.3 + 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) 幻觉率对比
    ax = axes[0, 1]
    hals = [v['hallucination_rate'] * 100 for _, v in items]
    bars = ax.bar([short_names.get(k, k) for k, _ in items], hals,
                  color=COLORS_6, edgecolor='black', linewidth=0.5, width=0.6, alpha=0.8)
    for bar, h in zip(bars, hals):
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f'{h:.0f}%',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_title('(b) 6策略幻觉率对比', fontsize=12, fontweight='bold')
    ax.set_ylabel('幻觉率 (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) 准确率vs幻觉率散点
    ax = axes[1, 0]
    for i, (k, v) in enumerate(items):
        acc = v['accuracy'] * 100
        hal = v['hallucination_rate'] * 100
        ax.scatter(hal, acc, s=300, c=COLORS_6[i], edgecolors='black', linewidth=1.2, alpha=0.88, zorder=3)
        ax.annotate(short_names.get(k, k), xy=(hal, acc), xytext=(hal + 3, acc + 2), fontsize=10, fontweight='bold')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.4)
    ax.set_title('(c) 准确率 vs 幻觉率', fontsize=12, fontweight='bold')
    ax.set_xlabel('幻觉率 (%)', fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 40)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) 正确题数堆叠
    ax = axes[1, 1]
    correct = [v['correct'] for _, v in items]
    wrong = [v['num_samples'] - v['correct'] - int(v['hallucination_rate'] * v['num_samples']) for _, v in items]
    hallucinated = [int(v['hallucination_rate'] * v['num_samples']) for _, v in items]
    x = np.arange(len(items))
    ax.bar(x, correct, label='正确', color='#4CAF50', edgecolor='black', linewidth=0.5)
    ax.bar(x, wrong, bottom=correct, label='错误(有答案)', color='#FF9800', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.bar(x, hallucinated, bottom=[c + w for c, w in zip(correct, wrong)],
           label='幻觉(无答案)', color='#F44336', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_names.get(k, k) for k, _ in items], fontsize=10, rotation=15)
    ax.set_title('(d) 答题结果分布（正确/错误/幻觉）', fontsize=12, fontweight='bold')
    ax.set_ylabel('题数', fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('LLM优化实验 v2 综合分析仪表板\n(真实MedQA 100题 · 6种Prompt策略 · TinyLlama-SFT)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'v2_dashboard.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  保存: v2_dashboard.png")

    print("\n" + "=" * 60)
    print("全部图表生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
