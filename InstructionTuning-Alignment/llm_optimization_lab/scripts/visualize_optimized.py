"""
LLM优化实验 - 优化版可视化脚本
生成优化前后对比图表 + 优化版策略综合分析
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "experiment_final", "metrics.json")
NEW_RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "experiment_optimized", "metrics_summary.json")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 配色方案
COLOR_BEFORE = '#B0BEC5'   # 优化前-灰
COLOR_AFTER = '#2196F3'    # 优化后-蓝
COLOR_ACC = '#4CAF50'      # 准确率-绿
COLOR_HAL = '#F44336'      # 幻觉率-红
COLORS_5 = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']


def load_data():
    """加载优化前后数据"""
    with open(OLD_RESULTS_PATH, 'r', encoding='utf-8') as f:
        old = json.load(f)
    with open(NEW_RESULTS_PATH, 'r', encoding='utf-8') as f:
        new = json.load(f)
    return old, new


def plot_before_after_accuracy(old, new):
    """优化前后准确率对比柱状图"""
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']

    old_acc = [old[s]['accuracy'] * 100 for s in strategies]
    new_acc = [new[s + '_v2']['accuracy'] * 100 for s in strategies]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars1 = ax.bar(x - width/2, old_acc, width, label='优化前 (Baseline)',
                   color=COLOR_BEFORE, edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + width/2, new_acc, width, label='优化后 (Optimized)',
                   color=COLOR_AFTER, edgecolor='black', linewidth=0.6)

    # 数值标签
    for bar, acc in zip(bars1, old_acc):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, color='#546E7A')
    for bar, acc in zip(bars2, new_acc):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='#1565C0')

    # 提升幅度标注
    for i, (o, n) in enumerate(zip(old_acc, new_acc)):
        delta = n - o
        if delta > 0:
            ax.annotate(f'+{delta:.1f}', xy=(i, max(o, n) + 8),
                        ha='center', fontsize=9, color='#2E7D32', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=0.8))
        elif delta < 0:
            ax.annotate(f'{delta:.1f}', xy=(i, max(o, n) + 8),
                        ha='center', fontsize=9, color='#C62828', fontweight='bold')

    ax.set_xlabel('Prompt 策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prompt 优化前后准确率对比\n(Zero-Shot 与 CoT 显著提升)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.set_ylim(0, max(max(old_acc), max(new_acc)) * 1.4 + 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'before_after_accuracy.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {os.path.basename(out)}")


def plot_before_after_hallucination(old, new):
    """优化前后幻觉率对比柱状图"""
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']

    old_h = [old[s]['hallucination_rate'] * 100 for s in strategies]
    new_h = [new[s + '_v2']['hallucination_rate'] * 100 for s in strategies]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars1 = ax.bar(x - width/2, old_h, width, label='优化前 (Baseline)',
                   color=COLOR_BEFORE, edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + width/2, new_h, width, label='优化后 (Optimized)',
                   color=COLOR_HAL, edgecolor='black', linewidth=0.6, alpha=0.75)

    for bar, h in zip(bars1, old_h):
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=10, color='#546E7A')
    for bar, h in zip(bars2, new_h):
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5,
                f'{h:.0f}%', ha='center', va='bottom', fontsize=11,
                fontweight='bold', color='#B71C1C')

    ax.set_xlabel('Prompt 策略', fontsize=13, fontweight='bold')
    ax.set_ylabel('幻觉率 / 无效答案率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prompt 优化前后幻觉率对比\n(Zero-Shot 幻觉率从 45% 降至 0%)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, frameon=True)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'before_after_hallucination.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {os.path.basename(out)}")


def plot_optimized_strategy_ranking(new):
    """优化版5种策略准确率排名（横向条形图）"""
    items = [(k, v['accuracy'] * 100, v['correct'], v['num_samples'])
             for k, v in new.items()]
    items.sort(key=lambda x: x[1], reverse=True)

    names = [it[0] for it in items]
    accs = [it[1] for it in items]
    correct = [it[2] for it in items]
    total = [it[3] for it in items]

    # 友好显示名
    display_names = {
        'zero_shot_v2': 'Zero-Shot v2\n(角色+格式约束)',
        'cot_v2': 'CoT v2\n(3步推理)',
        'few_shot_v2': 'Few-Shot v2\n(3示例)',
        'self_consistency_v2': 'Self-Consistency v2\n(5路径投票)',
        'tot_v1': 'ToT v1\n(思维树)',
    }
    labels = [display_names.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(labels))
    colors = [COLORS_5[i] for i in range(len(labels))]
    bars = ax.barh(y, accs, color=colors, edgecolor='black', linewidth=0.6, height=0.55)

    for i, (bar, acc, c, t) in enumerate(zip(bars, accs, correct, total)):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2.,
                f'{acc:.1f}%  ({c}/{t})', va='center', fontsize=11, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('优化版 5 种 Prompt 策略准确率排名',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, max(accs) * 1.5 + 5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'optimized_strategy_ranking.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {os.path.basename(out)}")


def plot_accuracy_vs_hallucination(new):
    """准确率 vs 幻觉率 散点图（优化版5种策略）"""
    fig, ax = plt.subplots(figsize=(10, 7))

    display_names = {
        'zero_shot_v2': 'Zero-Shot v2',
        'cot_v2': 'CoT v2',
        'few_shot_v2': 'Few-Shot v2',
        'self_consistency_v2': 'Self-Consistency v2',
        'tot_v1': 'ToT v1',
    }

    for i, (k, v) in enumerate(new.items()):
        acc = v['accuracy'] * 100
        hal = v['hallucination_rate'] * 100
        ax.scatter(hal, acc, s=380, c=COLORS_5[i], edgecolors='black',
                   linewidth=1.2, alpha=0.85, zorder=3)
        # 标注位置调整
        offset_x, offset_y = 3, 2
        if k == 'tot_v1':
            offset_x, offset_y = -15, 3
        if k == 'zero_shot_v2':
            offset_x, offset_y = 3, 3
        ax.annotate(display_names.get(k, k),
                    xy=(hal, acc), xytext=(hal + offset_x, acc + offset_y),
                    fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # 象限分隔线
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.axhline(y=20, color='gray', linestyle='--', alpha=0.4, linewidth=1)

    # 象限标注
    ax.text(5, 42, '理想区域\n(高准确率·低幻觉)', fontsize=10,
            color='#2E7D32', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', edgecolor='#2E7D32', alpha=0.7))
    ax.text(75, 5, '需改进区域\n(低准确率·高幻觉)', fontsize=10,
            color='#C62828', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828', alpha=0.7))

    ax.set_xlabel('幻觉率 / 无效答案率 (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold')
    ax.set_title('优化版策略：准确率 vs 幻觉率\n(左上角为理想区域)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 50)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'optimized_acc_vs_hal.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {os.path.basename(out)}")


def plot_combined_dashboard(old, new):
    """综合分析仪表板（2x2子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # ---- 子图1: 优化前后准确率对比 ----
    ax = axes[0, 0]
    strategies = ['zero_shot', 'few_shot', 'cot', 'self_consistency']
    labels = ['Zero-Shot', 'Few-Shot', 'CoT', 'Self-Consistency']
    old_acc = [old[s]['accuracy'] * 100 for s in strategies]
    new_acc = [new[s + '_v2']['accuracy'] * 100 for s in strategies]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, old_acc, width, label='优化前', color=COLOR_BEFORE, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, new_acc, width, label='优化后', color=COLOR_AFTER, edgecolor='black', linewidth=0.5)
    for i, (o, n) in enumerate(zip(old_acc, new_acc)):
        ax.text(i - width/2, o + 1, f'{o:.0f}%', ha='center', fontsize=9, color='#546E7A')
        ax.text(i + width/2, n + 1, f'{n:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#1565C0')
    ax.set_title('(a) 准确率：优化前后对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(old_acc), max(new_acc)) * 1.5 + 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- 子图2: 优化版5策略准确率 ----
    ax = axes[0, 1]
    items = [(k, v['accuracy'] * 100) for k, v in new.items()]
    items.sort(key=lambda t: t[1], reverse=True)
    names = [it[0] for it in items]
    accs = [it[1] for it in items]
    short_names = {
        'zero_shot_v2': 'ZS v2', 'cot_v2': 'CoT v2', 'few_shot_v2': 'FS v2',
        'self_consistency_v2': 'SC v2', 'tot_v1': 'ToT v1'
    }
    short_labels = [short_names.get(n, n) for n in names]
    bars = ax.bar(short_labels, accs, color=COLORS_5, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 1, f'{acc:.1f}%',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_title('(b) 优化版 5 策略准确率排名', fontsize=12, fontweight='bold')
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_ylim(0, max(accs) * 1.3 + 5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- 子图3: 幻觉率对比 ----
    ax = axes[1, 0]
    old_h = [old[s]['hallucination_rate'] * 100 for s in strategies]
    new_h = [new[s + '_v2']['hallucination_rate'] * 100 for s in strategies]
    ax.bar(x - width/2, old_h, width, label='优化前', color=COLOR_BEFORE, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, new_h, width, label='优化后', color=COLOR_HAL, edgecolor='black', linewidth=0.5, alpha=0.75)
    for i, (o, n) in enumerate(zip(old_h, new_h)):
        ax.text(i - width/2, o + 1.5, f'{o:.0f}%', ha='center', fontsize=9, color='#546E7A')
        ax.text(i + width/2, n + 1.5, f'{n:.0f}%', ha='center', fontsize=9, fontweight='bold', color='#B71C1C')
    ax.set_title('(c) 幻觉率：优化前后对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('幻觉率 (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ---- 子图4: 准确率vs幻觉率散点 ----
    ax = axes[1, 1]
    for i, (k, v) in enumerate(new.items()):
        acc = v['accuracy'] * 100
        hal = v['hallucination_rate'] * 100
        ax.scatter(hal, acc, s=300, c=COLORS_5[i], edgecolors='black', linewidth=1.2, alpha=0.85, zorder=3)
        short = short_names.get(k, k)
        ax.annotate(short, xy=(hal, acc), xytext=(hal + 3, acc + 2),
                    fontsize=10, fontweight='bold')
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(y=20, color='gray', linestyle='--', alpha=0.4)
    ax.set_title('(d) 准确率 vs 幻觉率 (优化版)', fontsize=12, fontweight='bold')
    ax.set_xlabel('幻觉率 (%)', fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 50)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('LLM 优化实验综合分析仪表板\n(优化前后对比 + 5种策略综合评估)',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'optimized_dashboard.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {os.path.basename(out)}")


def main():
    print("=" * 60)
    print("LLM优化实验 - 生成可视化图表")
    print("=" * 60)
    print(f"输出目录: {FIGURES_DIR}")
    print()

    old, new = load_data()

    print("[1] 优化前后准确率对比...")
    plot_before_after_accuracy(old, new)

    print("[2] 优化前后幻觉率对比...")
    plot_before_after_hallucination(old, new)

    print("[3] 优化版策略排名...")
    plot_optimized_strategy_ranking(new)

    print("[4] 准确率vs幻觉率散点图...")
    plot_accuracy_vs_hallucination(new)

    print("[5] 综合分析仪表板...")
    plot_combined_dashboard(old, new)

    print()
    print("=" * 60)
    print("全部图表生成完成！")
    print("=" * 60)
    print()
    print("优化效果总结：")
    for s in ['zero_shot', 'few_shot', 'cot', 'self_consistency']:
        o_acc = old[s]['accuracy'] * 100
        n_acc = new[s + '_v2']['accuracy'] * 100
        delta = n_acc - o_acc
        sign = '+' if delta >= 0 else ''
        print(f"  {s:20s}: {o_acc:5.1f}% -> {n_acc:5.1f}%  ({sign}{delta:.1f})")


if __name__ == "__main__":
    main()
