#!/usr/bin/env python3
"""汇总所有实验结果脚本

使用方法:
    # 汇总指定目录下的所有实验
    python scripts/aggregate_results.py --results_dir results/ --output reports/
    
    # 指定特定实验ID
    python scripts/aggregate_results.py --exp_ids exp_001 exp_002 exp_003
"""

import os
import sys
import argparse
import json
import glob
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logger, get_timestamp
from src.utils.io_utils import load_metrics_json, save_metrics_json


def find_experiment_dirs(results_dir: str) -> list:
    """查找所有实验结果目录"""
    exp_dirs = []
    
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.') and not item.startswith('_'):
            # 检查是否包含metrics.json
            if os.path.exists(os.path.join(item_path, 'metrics.json')):
                exp_dirs.append(item_path)
            # 检查repeats目录
            repeats_dir = os.path.join(item_path, 'repeats')
            if os.path.exists(repeats_dir):
                for repeat_item in os.listdir(repeats_dir):
                    repeat_path = os.path.join(repeats_dir, repeat_item)
                    if os.path.isdir(repeat_path) and os.path.exists(os.path.join(repeat_path, 'metrics.json')):
                        exp_dirs.append(repeat_path)
    
    return exp_dirs


def load_all_metrics(exp_dirs: list, logger: logging.Logger = None) -> list:
    """加载所有实验的metrics"""
    all_metrics = []
    
    for exp_dir in exp_dirs:
        metrics_path = os.path.join(exp_dir, 'metrics.json')
        config_path = os.path.join(exp_dir, 'config_used.yaml')
        
        if os.path.exists(metrics_path):
            try:
                metrics = load_metrics_json(metrics_path)
                metrics['exp_dir'] = exp_dir
                
                # 尝试加载配置
                if os.path.exists(config_path):
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    metrics['config'] = config
                
                all_metrics.append(metrics)
                
            except Exception as e:
                if logger:
                    logger.warning(f"加载 {metrics_path} 失败: {e}")
    
    return all_metrics


def compute_aggregate_stats(all_metrics: list) -> dict:
    """计算汇总统计"""
    import numpy as np
    
    if not all_metrics:
        return {}
    
    stats = {
        'total_experiments': len(all_metrics),
        'timestamp': get_timestamp(),
        'experiments': [],
        'aggregate': {},
    }
    
    # 收集每个实验的关键指标
    metric_values = {
        'train_loss': [],
        'eval_loss': [],
        'accuracy': [],
        'bleu': [],
    }
    
    for metrics in all_metrics:
        exp_info = {
            'exp_id': metrics.get('experiment_id', 'N/A'),
            'method': metrics.get('training_method', 'N/A'),
            'model': metrics.get('model_info', {}).get('base_model', 'N/A'),
            'completed_at': metrics.get('completed_at', 'N/A'),
        }
        
        # 提取指标
        for key in metric_values:
            val = metrics.get('metrics', {}).get(key)
            if val is not None:
                if isinstance(val, dict) and 'value' in val:
                    exp_info[key] = val['value']
                    metric_values[key].append(val['value'])
                elif isinstance(val, (int, float)):
                    exp_info[key] = val
                    metric_values[key].append(val)
        
        stats['experiments'].append(exp_info)
    
    # 计算汇总统计
    for key, values in metric_values.items():
        if values:
            stats['aggregate'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)) if len(values) > 1 else 0.0,
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values),
            }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='汇总所有实验结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/aggregate_results.py --results_dir results/ --output reports/
    python scripts/aggregate_results.py --exp_ids exp_001 exp_002
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='实验结果根目录'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='汇总结果输出目录'
    )
    
    parser.add_argument(
        '--exp_ids',
        type=str,
        nargs='+',
        default=None,
        help='指定要汇总的实验ID列表（可选）'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_file = os.path.join(project_root, 'logs', f'aggregate_{get_timestamp()}.log')
    logger = setup_logger('aggregate', log_file, getattr(logging, args.log_level))
    
    logger.info(f"{'='*60}")
    logger.info(f"实验结果汇总工具")
    logger.info(f"{'='*60}")
    
    # 查找实验目录
    if args.exp_ids:
        exp_dirs = [os.path.join(args.results_dir, exp_id) for exp_id in args.exp_ids]
    else:
        exp_dirs = find_experiment_dirs(args.results_dir)
    
    logger.info(f"发现 {len(exp_dirs)} 个实验目录")
    
    if not exp_dirs:
        logger.warning("未找到任何实验结果！")
        sys.exit(0)
    
    # 加载所有metrics
    all_metrics = load_all_metrics(exp_dirs, logger)
    logger.info(f"成功加载 {len(all_metrics)} 个实验的指标")
    
    # 计算汇总统计
    aggregate_stats = compute_aggregate_stats(all_metrics)
    
    # 保存汇总结果
    os.makedirs(args.output, exist_ok=True)
    
    output_path = os.path.join(args.output, f'aggregate_stats_{get_timestamp()}.json')
    save_metrics_json(aggregate_stats, output_path)
    
    logger.info(f"汇总统计已保存: {output_path}")
    
    # 打印摘要
    logger.info(f"\n{'='*60}")
    logger.info(f"汇总摘要")
    logger.info(f"{'='*60}")
    logger.info(f"实验总数: {aggregate_stats['total_experiments']}")
    
    for key, stats in aggregate_stats.get('aggregate', {}).items():
        logger.info(f"  {key}:")
        logger.info(f"    均值: {stats['mean']:.4f}")
        logger.info(f"    标准差: {stats['std']:.4f}")
        logger.info(f"    范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
