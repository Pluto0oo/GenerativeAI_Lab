import os
import json
import torch
import pandas as pd
from typing import Dict, List
from .visualization import plot_training_curve


def save_metrics(metrics_history: List[Dict], save_path: str):
    df = pd.DataFrame(metrics_history)
    df.to_csv(os.path.join(save_path, 'metrics.csv'), index=False)


def save_final_metrics(final_metrics: Dict, save_path: str):
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)


def generate_summary(config: Dict, final_metrics: Dict, save_path: str):
    summary = f"""# Experiment Summary: {config['experiment']['name']}

## Configuration
- **Experiment ID**: {config['experiment'].get('exp_id', 'N/A')}
- **Seed**: {config['experiment']['seed']}
- **Repeat Times**: {config['experiment']['repeat_times']}
- **Device**: {config['experiment']['device']}

## Data Configuration
- **Dataset**: {config['data']['dataset_name']}
- **Train Ways**: {config['data']['train_ways']}
- **Train Shots**: {config['data']['train_shots']}
- **Test Ways**: {config['data']['test_ways']}
- **Test Shots**: {config['data']['test_shots']}

## Model Configuration
- **Model Type**: {config['model']['type']}
- **Backbone**: {config['model']['backbone']}
- **Hidden Size**: {config['model']['hidden_size']}
- **Embedding Dim**: {config['model']['embedding_dim']}

## Training Configuration
- **Method**: {config['training']['method']}
- **Meta LR**: {config['training']['meta_lr']}
- **Fast LR**: {config['training']['fast_lr']}
- **Epochs**: {config['training']['epochs']}
- **Meta Batch Size**: {config['training']['meta_batch_size']}

## Results
"""
    
    if config['experiment']['repeat_times'] > 1:
        for metric, values in final_metrics.items():
            if isinstance(values, dict) and 'mean' in values:
                summary += f"- **{metric}**: {values['mean']:.4f} (±{values['std']:.4f})\n"
            else:
                summary += f"- **{metric}**: {values}\n"
    else:
        for metric, value in final_metrics.items():
            summary += f"- **{metric}**: {value:.4f}\n"
    
    with open(os.path.join(save_path, 'summary.md'), 'w') as f:
        f.write(summary)


def save_checkpoint(model, save_path: str, epoch: int = None):
    os.makedirs(save_path, exist_ok=True)
    filename = 'best_model.pth' if epoch is None else f'model_epoch_{epoch}.pth'
    torch.save(model.state_dict(), os.path.join(save_path, filename))


def setup_results_dir(exp_id: str, results_dir: str = "./results") -> str:
    exp_dir = os.path.join(results_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    return exp_dir


def aggregate_repeat_results(repeat_results: List[Dict]) -> Dict:
    metrics = repeat_results[0].keys()
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in repeat_results]
        aggregated[metric] = {
            'mean': float(sum(values) / len(values)),
            'std': float(pd.Series(values).std()),
            'values': values,
        }
    
    return aggregated


def save_repeat_results(repeat_results: List[Dict], save_path: str):
    repeats_dir = os.path.join(save_path, 'repeats')
    os.makedirs(repeats_dir, exist_ok=True)
    
    for i, result in enumerate(repeat_results):
        repeat_dir = os.path.join(repeats_dir, f'repeat_{i+1:03d}')
        os.makedirs(repeat_dir, exist_ok=True)
        save_final_metrics(result, repeat_dir)
    
    aggregated = aggregate_repeat_results(repeat_results)
    stats_dir = os.path.join(repeats_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    save_final_metrics(aggregated, stats_dir)
    
    return aggregated


def save_experiment_results(config: Dict, metrics_history: List[Dict], final_metrics: Dict, exp_dir: str):
    save_metrics(metrics_history, exp_dir)
    save_final_metrics(final_metrics, exp_dir)
    generate_summary(config, final_metrics, exp_dir)
    
    if config['results'].get('save_plots', True) and len(metrics_history) > 0:
        plot_training_curve(metrics_history, os.path.join(exp_dir, 'plots'))
