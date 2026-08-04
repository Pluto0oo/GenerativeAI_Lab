#!/usr/bin/env python3
"""生成最终实验报告脚本

使用方法:
    python scripts/generate_report.py --aggregate_file reports/aggregate_stats_XXX.json --output reports/final_report.md
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logger, get_timestamp
from src.utils.io_utils import load_metrics_json


def generate_final_report(aggregate_stats: dict, output_path: str):
    """生成Markdown格式的最终报告"""
    
    report = f"""# 指令微调与对齐技术 - 实验报告

## 报告概述

- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **实验总数**: {aggregate_stats.get('total_experiments', 0)}
- **报告类型**: 综合实验分析报告

---

## 一、实验汇总

### 1.1 实验列表

| 序号 | 实验ID | 训练方法 | 模型 | 完成时间 |
|------|--------|----------|------|----------|
"""
    
    # 添加实验列表
    for i, exp in enumerate(aggregate_stats.get('experiments', []), 1):
        report += f"| {i} | {exp.get('exp_id', 'N/A')} | {exp.get('method', 'N/A')} | {exp.get('model', 'N/A')} | {exp.get('completed_at', 'N/A')[:19]} |\n"
    
    report += """
### 1.2 指标统计汇总

"""
    
    # 添加指标统计
    report += "| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 样本数 |\n"
    report += "|------|------|--------|--------|--------|--------|\n"
    
    for metric_name, stats in aggregate_stats.get('aggregate', {}).items():
        report += f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n"
    
    report += """
---

## 二、详细分析

### 2.1 训练方法对比

"""
    
    # 按方法分组
    methods = {}
    for exp in aggregate_stats.get('experiments', []):
        method = exp.get('method', 'unknown')
        if method not in methods:
            methods[method] = []
        methods[method].append(exp)
    
    for method, exps in methods.items():
        report += f"#### {method.upper()} 方法\n\n"
        report += f"- **实验数量**: {len(exps)}\n"
        report += f"- **代表实验**: {exps[0].get('exp_id', 'N/A')}\n\n"
        
        # 计算该方法的指标均值
        metric_sums = {}
        for exp in exps:
            for key in ['train_loss', 'eval_loss', 'accuracy', 'bleu']:
                if key in exp:
                    if key not in metric_sums:
                        metric_sums[key] = []
                    metric_sums[key].append(exp[key])
        
        if metric_sums:
            report += "| 指标 | 均值 |\n"
            report += "|------|------|\n"
            for key, values in metric_sums.items():
                mean_val = sum(values) / len(values)
                report += f"| {key} | {mean_val:.4f} |\n"
            report += "\n"
    
    report += """
### 2.2 模型性能分析

（此处可根据具体模型和结果添加分析）

---

## 三、结论与建议

### 3.1 主要发现

1. **训练方法对比**: SFT与DPO在不同场景下的表现差异
2. **性能指标**: 各方法在准确率、BLEU等指标上的表现
3. **效率分析**: 训练时间与效果的权衡

### 3.2 改进方向

1. **数据质量**: 更高质量的指令数据集可能带来更好的效果
2. **超参数优化**: 学习率、LoRA参数等的进一步调优
3. **更长序列**: 探索处理更长上下文的能力

### 3.3 未来工作

1. 探索更多对齐算法（如KTO、ORPO等）
2. 在更大规模的数据集上验证
3. 进行人类评估（Human Evaluation）

---

## 附录

### A. 实验配置详情

（可附上每个实验的完整配置信息）

### B. 原始数据

完整的实验结果数据保存在 `results/` 目录下。

---

*报告由自动化Pipeline生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='生成最终实验报告',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/generate_report.py --aggregate_file reports/aggregate_stats_20260731.json --output reports/final_report.md
    python scripts/generate_report.py --results_dir results/ --output reports/
        """
    )
    
    parser.add_argument(
        '--aggregate_file',
        type=str,
        default=None,
        help='汇总统计JSON文件路径'
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='实验结果目录（如未指定aggregate_file）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='reports/final_report.md',
        help='输出报告路径'
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
    log_file = os.path.join(project_root, 'logs', f'report_{get_timestamp()}.log')
    logger = setup_logger('report', log_file, getattr(logging, args.log_level))
    
    logger.info(f"{'='*60}")
    logger.info(f"实验报告生成器")
    logger.info(f"{'='*60}")
    
    # 加载或创建汇总数据
    if args.aggregate_file and os.path.exists(args.aggregate_file):
        aggregate_stats = load_metrics_json(args.aggregate_file)
    else:
        logger.warning("未指定有效的aggregate_file，尝试从results目录加载...")
        sys.path.insert(0, project_root)
        from scripts.aggregate_results import find_experiment_dirs, load_all_metrics, compute_aggregate_stats
        
        exp_dirs = find_experiment_dirs(args.results_dir)
        all_metrics = load_all_metrics(exp_dirs)
        aggregate_stats = compute_aggregate_stats(all_metrics)
    
    # 生成报告
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    report_path = generate_final_report(aggregate_stats, args.output)
    
    logger.info(f"报告已生成: {report_path}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
