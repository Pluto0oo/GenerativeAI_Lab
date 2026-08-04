import argparse
import sys
import os
import json
import yaml
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import (
    plot_comparison,
    plot_learning_curve_comparison,
    plot_shot_comparison,
    plot_way_comparison,
)


def load_all_results(results_dir: str = "./results"):
    results = {}
    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        config_path = os.path.join(exp_path, 'config_used.yaml')
        metrics_path = os.path.join(exp_path, 'metrics.json')
        metrics_csv_path = os.path.join(exp_path, 'metrics.csv')
        
        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            csv_data = None
            if os.path.exists(metrics_csv_path):
                csv_data = pd.read_csv(metrics_csv_path)
            
            results[exp_dir] = {
                'config': config,
                'metrics': metrics,
                'csv_data': csv_data,
            }
        except Exception as e:
            print(f"Error loading {exp_dir}: {e}")
    
    return results


def extract_accuracy(result):
    acc = result['metrics'].get('test_accuracy', 0)
    if isinstance(acc, dict):
        return acc.get('mean', 0), acc.get('std', 0)
    return acc, 0


def generate_all_visualizations(results, output_dir: str = "./results/comprehensive_comparison"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualization 1: Method Comparison (5-way 1-shot)...")
    method_results = {}
    for exp_id, result in results.items():
        config = result['config']
        if config['data']['test_ways'] == 5 and config['data']['test_shots'] == 1:
            name = config['experiment']['name']
            acc, std = extract_accuracy(result)
            method_results[name] = {'test_accuracy': acc, 'std': std}
    
    if method_results:
        plot_comparison(method_results, output_dir, 'Method Comparison - 5-way 1-shot')
        print(f"  -> comparison.png saved")
    
    print("\nGenerating visualization 2: Method Comparison (5-way 5-shot)...")
    method_results_5shot = {}
    for exp_id, result in results.items():
        config = result['config']
        if config['data']['test_ways'] == 5 and config['data']['test_shots'] == 5:
            name = config['experiment']['name']
            acc, std = extract_accuracy(result)
            method_results_5shot[name] = {'test_accuracy': acc, 'std': std}
    
    if method_results_5shot:
        plot_comparison(method_results_5shot, output_dir, 'Method Comparison - 5-way 5-shot')
        print(f"  -> comparison_5shot.png saved")
    
    print("\nGenerating visualization 3: Architecture Comparison (Experiment A)...")
    arch_results = {}
    for exp_id, result in results.items():
        config = result['config']
        name = config['experiment']['name']
        if name.startswith('expA_'):
            acc, std = extract_accuracy(result)
            arch_name = name.replace('expA_', '')
            arch_results[arch_name] = {'test_accuracy': acc, 'std': std}
    
    if arch_results:
        plot_comparison(arch_results, output_dir, 'Architecture Comparison - Experiment A')
        print(f"  -> architecture_comparison.png saved")
    
    print("\nGenerating visualization 4: Augmentation Comparison (Experiment B)...")
    aug_results = {}
    for exp_id, result in results.items():
        config = result['config']
        name = config['experiment']['name']
        if name.startswith('expB_'):
            acc, std = extract_accuracy(result)
            aug_name = name.replace('expB_', '')
            aug_results[aug_name] = {'test_accuracy': acc, 'std': std}
    
    if aug_results:
        plot_comparison(aug_results, output_dir, 'Augmentation Comparison - Experiment B')
        print(f"  -> augmentation_comparison.png saved")
    
    print("\nGenerating visualization 5: Distance Metric Comparison (Experiment D)...")
    dist_results = {}
    for exp_id, result in results.items():
        config = result['config']
        name = config['experiment']['name']
        if name.startswith('expD_'):
            acc, std = extract_accuracy(result)
            dist_name = name.replace('expD_', '')
            dist_results[dist_name] = {'test_accuracy': acc, 'std': std}
    
    if dist_results:
        plot_comparison(dist_results, output_dir, 'Distance Metric Comparison - Experiment D')
        print(f"  -> distance_comparison.png saved")
    
    print("\nGenerating visualization 6: Learning Curve Comparison...")
    learning_curves = {}
    for exp_id, result in results.items():
        config = result['config']
        name = config['experiment']['name']
        if result['csv_data'] is not None and len(result['csv_data']) > 0:
            learning_curves[name] = result['csv_data'].to_dict('records')
    
    if learning_curves:
        plot_learning_curve_comparison(learning_curves, output_dir)
        print(f"  -> learning_curve_comparison.png saved")
    
    print("\nGenerating visualization 7: Shot Comparison...")
    shot_results = {}
    for exp_id, result in results.items():
        config = result['config']
        shots = config['data']['test_shots']
        model_type = config['model']['type']
        
        if shots not in shot_results:
            shot_results[shots] = {}
        
        acc, std = extract_accuracy(result)
        shot_results[shots][model_type] = {'test_accuracy': acc, 'std': std}
    
    if shot_results and len(shot_results) > 1:
        plot_shot_comparison(shot_results, output_dir)
        print(f"  -> shot_comparison.png saved")
    
    print("\nGenerating summary JSON...")
    summary = {}
    for exp_id, result in results.items():
        config = result['config']
        acc, std = extract_accuracy(result)
        summary[exp_id] = {
            'name': config['experiment']['name'],
            'model_type': config['model']['type'],
            'backbone': config['model'].get('backbone', 'convnet'),
            'train_ways': config['data']['train_ways'],
            'train_shots': config['data']['train_shots'],
            'test_accuracy': acc,
            'test_accuracy_std': std,
        }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"  -> summary.json saved")
    
    print("\nGenerating results table...")
    df = pd.DataFrame([{
        'Experiment': r['config']['experiment']['name'],
        'Model': r['config']['model']['type'],
        'Backbone': r['config']['model'].get('backbone', 'convnet'),
        'Ways': r['config']['data']['test_ways'],
        'Shots': r['config']['data']['test_shots'],
        'Accuracy': f"{extract_accuracy(r)[0]:.4f}",
        'Std': f"{extract_accuracy(r)[1]:.4f}",
    } for r in results.values()])
    
    df.to_csv(os.path.join(output_dir, 'results_table.csv'), index=False)
    print(f"  -> results_table.csv saved")
    
    print("\nAll visualizations generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualizations from experiment results")
    parser.add_argument('--results_dir', default="./results", help="Directory containing results")
    parser.add_argument('--output_dir', default="./results/comprehensive_comparison", help="Output directory for visualizations")
    args = parser.parse_args()
    
    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)
    print(f"Loaded {len(results)} experiments")
    
    generate_all_visualizations(results, args.output_dir)


if __name__ == "__main__":
    main()
