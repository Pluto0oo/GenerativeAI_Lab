#!/usr/bin/env python3
"""生成实验结果可视化图表"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色
COLORS = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#B279A2']


def load_results():
    """加载最新实验结果"""
    results_path = os.path.join(RESULTS_DIR, "experiment_final", "metrics.json")
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def plot_accuracy_comparison(results):
    """准确率对比柱状图"""
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']
    accuracies = [results[s]['accuracy'] for s in strategies if s in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, accuracies, color=COLORS[:4], edgecolor='black', linewidth=0.5, width=0.6)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Prompt策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('准确率 (Accuracy)', fontsize=13, fontweight='bold')
    ax.set_title('不同Prompt策略的医疗问答准确率对比', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, max(accuracies) * 1.3 if max(accuracies) > 0 else 0.3)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'accuracy_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: accuracy_comparison.png")


def plot_hallucination_comparison(results):
    """幻觉率对比柱状图"""
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']
    hallucination = [results[s]['hallucination_rate'] for s in strategies if s in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, hallucination, color=COLORS[:4], edgecolor='black', linewidth=0.5, width=0.6, alpha=0.8)

    for bar, rate in zip(bars, hallucination):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Prompt策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('幻觉率 (Hallucination Rate)', fontsize=13, fontweight='bold')
    ax.set_title('不同Prompt策略的幻觉率对比（越低越好）', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%基准线')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'hallucination_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: hallucination_comparison.png")


def plot_accuracy_vs_hallucination(results):
    """准确率vs幻觉率散点图"""
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, s in enumerate(strategies):
        if s in results:
            ax.scatter(results[s]['accuracy'], results[s]['hallucination_rate'],
                      s=300, c=COLORS[i], edgecolors='black', linewidth=1.5,
                      zorder=5, alpha=0.9)
            ax.annotate(labels[i],
                       (results[s]['accuracy'], results[s]['hallucination_rate']),
                       textcoords="offset points", xytext=(15, 10),
                       fontsize=12, fontweight='bold')

    ax.set_xlabel('准确率（越右越好）', fontsize=13, fontweight='bold')
    ax.set_ylabel('幻觉率（越下越好）', fontsize=13, fontweight='bold')
    ax.set_title('准确率 vs 幻觉率 散点图\n（右下角为理想区域）', fontsize=15, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle='--')

    # 标记理想区域
    ax.axvspan(0.1, 1.0, alpha=0.1, color='green')
    ax.axhspan(0, 0.5, alpha=0.1, color='green')
    ax.text(0.5, 0.25, '理想区域', fontsize=12, alpha=0.5, ha='center', color='green')

    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'accuracy_vs_hallucination.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: accuracy_vs_hallucination.png")


def plot_radar_chart(results):
    """雷达图综合对比"""
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']
    metrics = ['准确率', '低幻觉率']

    # 准备数据
    data = []
    for s in strategies:
        if s in results:
            accuracy = results[s]['accuracy']
            low_halluc = 1 - results[s]['hallucination_rate']
            data.append([accuracy, low_halluc])

    # 雷达图设置
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    for i, (d, label) in enumerate(zip(data, labels)):
        d += d[:1]
        ax.plot(angles, d, 'o-', linewidth=2, markersize=8, label=label, color=COLORS[i])
        ax.fill(angles, d, alpha=0.15, color=COLORS[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    ax.set_title('Prompt策略综合性能雷达图', fontsize=15, fontweight='bold', pad=30)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'radar_chart.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: radar_chart.png")


def plot_combined_analysis(results):
    """综合分析子图"""
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']
    accuracies = [results[s]['accuracy'] for s in strategies if s in results]
    hallucinations = [results[s]['hallucination_rate'] for s in strategies if s in results]
    corrects = [results[s]['correct'] for s in strategies if s in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('LLM优化实验综合分析报告', fontsize=18, fontweight='bold', y=0.98)

    # 1. 正确答题数
    ax = axes[0, 0]
    bars = ax.bar(labels, corrects, color=COLORS[:4], edgecolor='black', linewidth=0.5)
    for bar, c in zip(bars, corrects):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{c}', ha='center', va='bottom', fontweight='bold')
    ax.set_title('正确答题数 (20题)', fontsize=13, fontweight='bold')
    ax.set_ylabel('正确数')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 2. 准确率
    ax = axes[0, 1]
    ax.pie(accuracies, labels=labels, colors=COLORS[:4], autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 11}, pctdistance=0.75)
    centre_circle = plt.Circle((0, 0), 0.50, fc='white')
    ax.add_artist(centre_circle)
    ax.set_title('准确率分布', fontsize=13, fontweight='bold')

    # 3. 幻觉率
    ax = axes[1, 0]
    bars = ax.bar(labels, hallucinations, color=COLORS[:4], edgecolor='black', linewidth=0.5, alpha=0.8)
    for bar, h in zip(bars, hallucinations):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{h:.0%}', ha='center', va='bottom', fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50%基准')
    ax.set_title('幻觉率对比（低=好）', fontsize=13, fontweight='bold')
    ax.set_ylabel('幻觉率')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4. 性能指数 (准确率 * (1-幻觉率))
    ax = axes[1, 1]
    performance_idx = [a * (1 - h) for a, h in zip(accuracies, hallucinations)]
    bars = ax.bar(labels, performance_idx, color=COLORS[:4], edgecolor='black', linewidth=0.5)
    for bar, p in zip(bars, performance_idx):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{p:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.set_title('综合性能指数\n(准确率 × (1-幻觉率))', fontsize=13, fontweight='bold')
    ax.set_ylabel('性能指数')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, 'combined_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: combined_analysis.png")


def main():
    print("=" * 50)
    print("生成实验结果可视化图表")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print()

    results = load_results()
    if not results:
        print("错误: 未找到实验结果数据")
        return

    print(f"找到实验结果，包含 {len(results)} 组数据")
    print()
    print("生成图表中...")
    print()

    plot_accuracy_comparison(results)
    plot_hallucination_comparison(results)
    plot_accuracy_vs_hallucination(results)
    plot_radar_chart(results)
    plot_combined_analysis(results)

    print()
    print("=" * 50)
    print(f"全部完成！图表保存在: {FIGURES_DIR}")
    print("生成的图表:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(FIGURES_DIR, f)) / 1024
            print(f"  - {f} ({size:.1f} KB)")
    print("=" * 50)


if __name__ == "__main__":
    main()
