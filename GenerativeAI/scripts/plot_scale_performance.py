"""
规模-性能关系图生成脚本

对比三种生成模型 (VAE, DCGAN, Diffusion) 的:
  1. 模型规模 (参数量) vs FID
  2. 模型规模 (参数量) vs IS
  3. 模型规模 (参数量) vs Precision
  4. 训练损失 vs 模型规模
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def generate_scale_performance_chart():
    base_dir = r'c:\Users\17456\Documents\GitHub\Deep_learningPractice\GenerativeAI'
    results_dir = os.path.join(base_dir, 'results')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    models = {
        'VAE': {
            'params': 253648,
            'params_M': 0.25,
            'FID': 1472.65,
            'IS': 2.23,
            'Precision': 0.0767,
            'Recall': 0.0000,
            'final_loss': 115.10,
            'epochs': 5,
            'color': '#2196F3',
            'marker': 'o'
        },
        'DCGAN': {
            'params': 6085888,
            'params_M': 6.09,
            'FID': 351.53,
            'IS': 3.61,
            'Precision': 0.4000,
            'Recall': 0.7033,
            'final_loss': 3.67,
            'epochs': 20,
            'color': '#4CAF50',
            'marker': 's'
        },
        'Diffusion': {
            'params': 15724931,
            'params_M': 15.72,
            'FID': 978.10,
            'IS': 2.03,
            'Precision': 0.8100,
            'Recall': 0.0000,
            'final_loss': 0.04,
            'epochs': 5,
            'color': '#FF9800',
            'marker': '^'
        }
    }

    model_names = list(models.keys())
    params_M = [models[m]['params_M'] for m in model_names]
    fid_scores = [models[m]['FID'] for m in model_names]
    is_scores = [models[m]['IS'] for m in model_names]
    precision_scores = [models[m]['Precision'] for m in model_names]
    recall_scores = [models[m]['Recall'] for m in model_names]
    final_losses = [models[m]['final_loss'] for m in model_names]
    epochs = [models[m]['epochs'] for m in model_names]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Scale-Performance Relationship Analysis\n生成模型规模-性能关系分析', fontsize=16, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    for name in model_names:
        m = models[name]
        ax1.scatter(m['params_M'], m['FID'], c=m['color'], marker=m['marker'],
                    s=200, zorder=5, edgecolors='black', linewidths=1.5)
    for i, name in enumerate(model_names):
        ax1.annotate(name, (params_M[i], fid_scores[i]),
                     textcoords="offset points", xytext=(12, 8), fontsize=11, fontweight='bold')
    ax1.set_xlabel('Model Parameters (Million)', fontsize=11)
    ax1.set_ylabel('FID (Fréchet Inception Distance)', fontsize=11)
    ax1.set_title('Model Scale vs FID\n模型规模 vs FID', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(2, 3, 2)
    for name in model_names:
        m = models[name]
        ax2.scatter(m['params_M'], m['IS'], c=m['color'], marker=m['marker'],
                    s=200, zorder=5, edgecolors='black', linewidths=1.5)
    for i, name in enumerate(model_names):
        ax2.annotate(name, (params_M[i], is_scores[i]),
                     textcoords="offset points", xytext=(12, 5), fontsize=11, fontweight='bold')
    ax2.set_xlabel('Model Parameters (Million)', fontsize=11)
    ax2.set_ylabel('IS (Inception Score)', fontsize=11)
    ax2.set_title('Model Scale vs IS\n模型规模 vs IS', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    ax3 = fig.add_subplot(2, 3, 3)
    for name in model_names:
        m = models[name]
        ax3.scatter(m['params_M'], m['Precision'], c=m['color'], marker=m['marker'],
                    s=200, zorder=5, edgecolors='black', linewidths=1.5, label=name)
        ax3.scatter(m['params_M'], m['Recall'], c=m['color'], marker=m['marker'],
                    s=200, zorder=5, edgecolors='black', linewidths=1.5, facecolors='none')
    for i, name in enumerate(model_names):
        ax3.annotate(f'{name}(P)', (params_M[i], precision_scores[i]),
                     textcoords="offset points", xytext=(12, 8), fontsize=10, fontweight='bold')
        ax3.annotate(f'{name}(R)', (params_M[i], recall_scores[i]),
                     textcoords="offset points", xytext=(12, -15), fontsize=10)
    ax3.set_xlabel('Model Parameters (Million)', fontsize=11)
    ax3.set_ylabel('Precision / Recall', fontsize=11)
    ax3.set_title('Model Scale vs Precision/Recall\n模型规模 vs Precision/Recall', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=12, label='VAE'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4CAF50', markersize=12, label='DCGAN'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#FF9800', markersize=12, label='Diffusion'),
        Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=8, label='Precision (solid)'),
        Line2D([0], [0], marker='o', color='black', markerfacecolor='none', markersize=8, label='Recall (hollow)')
    ]
    ax3.legend(handles=legend_elements, fontsize=8, loc='upper right')

    ax4 = fig.add_subplot(2, 3, 4)
    x_pos = np.arange(len(model_names))
    bars = ax4.bar(x_pos, final_losses, color=[models[m]['color'] for m in model_names],
                   edgecolor='black', linewidth=1.5, width=0.6)
    for bar, loss in zip(bars, final_losses):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + max(final_losses) * 0.02,
                 f'{loss:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, fontsize=12, fontweight='bold')
    ax4.set_ylabel('Final Training Loss', fontsize=11)
    ax4.set_title('Final Training Loss by Model\n各模型最终训练Loss', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = fig.add_subplot(2, 3, 5)
    scatter = ax5.scatter(params_M, is_scores, c=[models[m]['FID'] for m in model_names],
                          cmap='RdYlGn_r', s=[1500, 3000, 4500],
                          alpha=0.8, edgecolors='black', linewidths=1.5, vmin=0, vmax=1500)
    for i, name in enumerate(model_names):
        ax5.annotate(name, (params_M[i], is_scores[i]),
                     textcoords="offset points", xytext=(0, -20), fontsize=11, fontweight='bold', ha='center')
    cbar = plt.colorbar(scatter, ax=ax5, shrink=0.8)
    cbar.set_label('FID (color)', fontsize=10)
    ax5.set_xlabel('Model Parameters (Million)', fontsize=11)
    ax5.set_ylabel('IS (Inception Score)', fontsize=11)
    ax5.set_title('Bubble Chart: Params vs IS (size=FID)\n气泡图: 参数 vs IS (大小=FID)', fontsize=11, fontweight='bold')
    ax5.set_xscale('log')
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    table_data = [
        ['VAE', '0.25', '1472.65', '2.23', '0.08', '0.00', '115.10', '5'],
        ['DCGAN', '6.09', '351.53', '3.61', '0.40', '0.70', '3.67', '20'],
        ['Diffusion', '15.72', '978.10', '2.03', '0.81', '0.00', '0.04', '5']
    ]
    table = ax6.table(cellText=table_data,
                     colLabels=['Model', 'Params(M)', 'FID', 'IS', 'Prec', 'Rec', 'Loss', 'Epochs'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
    ax6.axis('off')
    ax6.set_title('Performance Summary Table\n性能汇总表', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(plots_dir, 'scale_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'规模-性能关系图已保存至: {os.path.join(plots_dir, "scale_performance.png")}')

    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Scale-Performance Analysis (Individual)\n规模-性能关系分析（独立视图）', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for name in model_names:
        m = models[name]
        ax.scatter(m['params_M'], m['FID'], c=m['color'], marker=m['marker'],
                   s=300, zorder=5, edgecolors='black', linewidths=1.5)
    for i, name in enumerate(model_names):
        ax.annotate(name, (params_M[i], fid_scores[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight='bold')
    ax.set_xlabel('Parameters (M)', fontsize=10)
    ax.set_ylabel('FID', fontsize=10)
    ax.set_title('FID vs Model Size', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name in model_names:
        m = models[name]
        ax.scatter(m['params_M'], m['IS'], c=m['color'], marker=m['marker'],
                   s=300, zorder=5, edgecolors='black', linewidths=1.5)
    for i, name in enumerate(model_names):
        ax.annotate(name, (params_M[i], is_scores[i]),
                    textcoords="offset points", xytext=(10, 3), fontsize=10, fontweight='bold')
    ax.set_xlabel('Parameters (M)', fontsize=10)
    ax.set_ylabel('IS', fontsize=10)
    ax.set_title('IS vs Model Size', fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, precision_scores, width, label='Precision',
                   color='#2196F3', edgecolor='black')
    bars2 = ax.bar(x + width/2, recall_scores, width, label='Recall',
                   color='#FF9800', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Precision/Recall by Model', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

    ax = axes[1, 1]
    for name in model_names:
        m = models[name]
        sizes = [100, 300, 500]
        for s in sizes:
            ax.scatter(m['params_M'], m['final_loss'], c=m['color'], marker=m['marker'],
                       s=s * 3, alpha=0.3, edgecolors='none')
    for i, name in enumerate(model_names):
        m = models[name]
        ax.scatter(m['params_M'], m['final_loss'], c=m['color'], marker=m['marker'],
                   s=250, zorder=5, edgecolors='black', linewidths=1.5)
        ax.annotate(f'{name}\nLoss={m["final_loss"]:.2f}',
                     (params_M[i], final_losses[i]),
                     textcoords="offset points", xytext=(15, 10), fontsize=10, fontweight='bold')
    ax.set_xlabel('Parameters (M)', fontsize=10)
    ax.set_ylabel('Final Loss', fontsize=10)
    ax.set_title('Final Loss vs Model Size', fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scale_performance_detailed.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'详细规模-性能图已保存至: {os.path.join(plots_dir, "scale_performance_detailed.png")}')

    return models


if __name__ == '__main__':
    generate_scale_performance_chart()
