import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

SCIENCE_COLORS = ['#0077B6', '#00B4D8', '#48CAE4', '#90E0EF', '#CAF0F8',
                  '#F72585', '#7209B7', '#3A0CA3', '#4361EE', '#4CC9F0',
                  '#06D6A0', '#118AB2', '#073B4C', '#FFD166', '#EF476F']


def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(SCIENCE_COLORS)
    plt.rcParams.update({
        'text.usetex': False,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.8,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'figure.figsize': (8, 5),
    })


def plot_training_curve(metrics_history: List[Dict], save_path: str, 
                        title_prefix: str = ""):
    set_publication_style()
    df = pd.DataFrame(metrics_history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.lineplot(data=df, x='epoch', y='loss', ax=ax1, linewidth=2.5, 
                 color=SCIENCE_COLORS[0], marker='o', markersize=5, 
                 markevery=5)
    ax1.set_title(f'{title_prefix}Training Loss', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12, width=1.2)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    sns.lineplot(data=df, x='epoch', y='accuracy', ax=ax2, linewidth=2.5, 
                 color=SCIENCE_COLORS[4], marker='s', markersize=5, 
                 markevery=5)
    ax2.set_title(f'{title_prefix}Training Accuracy', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.set_ylim(0.0, 1.05)
    ax2.tick_params(axis='both', labelsize=12, width=1.2)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_path, 'training_curve.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'training_curve.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], 
                          save_path: str, title: str = 'Confusion Matrix'):
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar=True, cbar_kws={'label': 'Count', 'shrink': 0.8},
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.tick_params(axis='both', labelsize=12, width=1.2)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison(results: Dict[str, Dict], save_path: str, 
                    title: str = 'Method Comparison - Accuracy'):
    set_publication_style()
    
    methods = list(results.keys())
    accuracies = []
    stds = []
    
    for m in methods:
        acc = results[m].get('test_accuracy', results[m].get('accuracy', 0))
        if isinstance(acc, dict):
            accuracies.append(acc.get('mean', 0))
            stds.append(acc.get('std', 0))
        else:
            accuracies.append(acc)
            stds.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(methods))
    bar_width = 0.6
    
    bars = ax.bar(x, accuracies, yerr=stds, capsize=8, 
                  color=SCIENCE_COLORS[:len(methods)],
                  edgecolor='#333333', linewidth=1.2,
                  error_kw={'elinewidth': 1.5, 'capsize': 8})
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12, rotation=15, ha='right')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='both', labelsize=12, width=1.2)
    ax.grid(True, alpha=0.4, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, acc, std in zip(bars, accuracies, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{acc:.4f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
        if std > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.04,
                    f'±{std:.4f}', ha='center', va='bottom', 
                    fontsize=10, color='#666666')
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_path, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_shot_comparison(shot_results: Dict[int, Dict], save_path: str):
    set_publication_style()
    
    shots = sorted(shot_results.keys())
    methods = list(shot_results[shots[0]].keys())
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for idx, method in enumerate(methods):
        accuracies = []
        stds = []
        for s in shots:
            acc = shot_results[s][method].get('test_accuracy', shot_results[s][method].get('accuracy', 0))
            if isinstance(acc, dict):
                accuracies.append(acc.get('mean', 0))
                stds.append(acc.get('std', 0))
            else:
                accuracies.append(acc)
                stds.append(0)
        
        x_vals = np.array(shots)
        y_vals = np.array(accuracies)
        y_err = np.array(stds)
        
        sns.lineplot(x=x_vals, y=y_vals, marker='o', label=method, 
                     linewidth=2.5, markersize=10,
                     color=SCIENCE_COLORS[idx % len(SCIENCE_COLORS)])
        
        ax.fill_between(x_vals, y_vals - 1.96 * y_err, y_vals + 1.96 * y_err,
                        alpha=0.15, color=SCIENCE_COLORS[idx % len(SCIENCE_COLORS)])
    
    ax.set_title('Accuracy vs Number of Shots', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Shots', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xticks(shots)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, frameon=True, edgecolor='#333333', loc='lower right')
    ax.tick_params(axis='both', labelsize=12, width=1.2)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_path, 'shot_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'shot_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_way_comparison(way_results: Dict[int, Dict], save_path: str):
    set_publication_style()
    
    ways = sorted(way_results.keys())
    methods = list(way_results[ways[0]].keys())
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for idx, method in enumerate(methods):
        accuracies = []
        stds = []
        for w in ways:
            acc = way_results[w][method].get('test_accuracy', way_results[w][method].get('accuracy', 0))
            if isinstance(acc, dict):
                accuracies.append(acc.get('mean', 0))
                stds.append(acc.get('std', 0))
            else:
                accuracies.append(acc)
                stds.append(0)
        
        x_vals = np.array(ways)
        y_vals = np.array(accuracies)
        y_err = np.array(stds)
        
        sns.lineplot(x=x_vals, y=y_vals, marker='s', label=method, 
                     linewidth=2.5, markersize=10,
                     color=SCIENCE_COLORS[idx % len(SCIENCE_COLORS)])
        
        ax.fill_between(x_vals, y_vals - 1.96 * y_err, y_vals + 1.96 * y_err,
                        alpha=0.15, color=SCIENCE_COLORS[idx % len(SCIENCE_COLORS)])
    
    ax.set_title('Accuracy vs Number of Ways', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Ways', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xticks(ways)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, frameon=True, edgecolor='#333333', loc='upper right')
    ax.tick_params(axis='both', labelsize=12, width=1.2)
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_path, 'way_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'way_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_nway_kshot_heatmap(results: Dict[tuple, float], save_path: str):
    set_publication_style()
    
    ways = sorted(set(k[0] for k in results.keys()))
    shots = sorted(set(k[1] for k in results.keys()))
    
    data = np.zeros((len(shots), len(ways)))
    for i, s in enumerate(shots):
        for j, w in enumerate(ways):
            data[i, j] = results.get((w, s), 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                xticklabels=ways, yticklabels=shots, ax=ax,
                cbar=True, cbar_kws={'label': 'Accuracy', 'shrink': 0.8},
                annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_title('N-way K-shot Accuracy Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Ways', fontsize=14)
    ax.set_ylabel('Number of Shots', fontsize=14)
    ax.tick_params(axis='both', labelsize=12, width=1.2)
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_path, 'nway_kshot_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'nway_kshot_heatmap.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curve_comparison(curves: Dict[str, List[Dict]], save_path: str):
    set_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (method, history) in enumerate(curves.items()):
        df = pd.DataFrame(history)
        color = SCIENCE_COLORS[idx % len(SCIENCE_COLORS)]
        
        sns.lineplot(data=df, x='epoch', y='loss', ax=ax1, linewidth=2.5, 
                     label=method, color=color, marker='o', markersize=5, markevery=10)
        
        sns.lineplot(data=df, x='epoch', y='accuracy', ax=ax2, linewidth=2.5, 
                     label=method, color=color, marker='s', markersize=5, markevery=10)
    
    ax1.set_title('Training Loss Comparison', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.legend(fontsize=10, frameon=True, edgecolor='#333333')
    ax1.tick_params(axis='both', labelsize=12, width=1.2)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.set_title('Training Accuracy Comparison', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(fontsize=10, frameon=True, edgecolor='#333333')
    ax2.tick_params(axis='both', labelsize=12, width=1.2)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    plt.savefig(os.path.join(save_path, 'learning_curve_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'learning_curve_comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
