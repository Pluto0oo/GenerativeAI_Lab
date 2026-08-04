#!/usr/bin/env python3
"""对比实验脚本 - 同时运行多个配置并生成对比报告

使用方法:
    python scripts/run_comparison.py \
        --configs configs/experiment/sft_baseline.yaml configs/experiment/dpo_improved.yaml \
        --output_dir results/comparison_20260731
"""

import os
import sys
import argparse
import json
import time
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.logger import setup_logger, get_timestamp
from src.utils.io_utils import load_config, save_metrics_json
from src.pipeline.alignment_pipeline import AlignmentPipeline


def generate_comparison_report(results: list, output_dir: str):
    """生成对比报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'comparison_report.md')
    
    report = """# 对比实验报告

## 实验概述

"""
    report += f"**生成时间**: {get_timestamp()}\n\n"
    report += f"**实验数量**: {len(results)}\n\n"
    
    # 对比表格
    report += "## 结果对比\n\n"
    report += "| 实验ID | 方法 | 训练Loss | 评估Loss | 准确率 | BLEU | 训练时间 |\n"
    report += "|--------|------|----------|----------|--------|------|----------|\n"
    
    for result in results:
        metrics = result.get('final_metrics', {})
        exp_id = result.get('exp_id', 'N/A')
        method = result.get('method', 'N/A')
        train_loss = metrics.get('metrics', {}).get('train_loss', 'N/A')
        eval_loss = metrics.get('metrics', {}).get('eval_loss', 'N/A')
        accuracy = metrics.get('metrics', {}).get('accuracy', 'N/A')
        bleu = metrics.get('metrics', {}).get('bleu', 'N/A')
        
        train_time = 'N/A'
        if 'train_metrics' in result:
            train_time = result['train_metrics'].get('training_time_seconds', 'N/A')
            if isinstance(train_time, (int, float)):
                train_time = f"{train_time:.1f}s"
        
        report += f"| {exp_id} | {method} | {train_loss} | {eval_loss} | {accuracy} | {bleu} | {train_time} |\n"
    
    # 详细分析
    report += "\n## 详细分析\n\n"
    
    for i, result in enumerate(results):
        report += f"### 实验 {i+1}: {result.get('exp_id', 'N/A')}\n\n"
        report += f"- **配置**: {result.get('config_path', 'N/A')}\n"
        report += f"- **方法**: {result.get('method', 'N/A')}\n"
        report += f"- **模型**: {result.get('model', 'N/A')}\n\n"
    
    report += "\n## 结论与建议\n\n"
    report += "（请根据实际结果填写分析内容）\n"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"对比报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='运行对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/run_comparison.py \\
        --configs configs/exp1.yaml configs/exp2.yaml configs/exp3.yaml \\
        --output_dir results/comparison_001
        """
    )
    
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        required=True,
        help='多个YAML配置文件路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='对比结果输出目录'
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
    timestamp = get_timestamp()
    output_dir = args.output_dir or f"results/comparison_{timestamp}"
    log_file = os.path.join(project_root, 'logs', f'comparison_{timestamp}.log')
    logger = setup_logger('comparison', log_file, getattr(logging, args.log_level))
    
    logger.info(f"{'='*60}")
    logger.info(f"对比实验系统")
    logger.info(f"{'='*60}")
    logger.info(f"配置文件数量: {len(args.configs)}")
    logger.info(f"输出目录: {output_dir}")
    
    results = []
    
    for i, config_path in enumerate(args.configs):
        logger.info(f"\n{'*'*40}")
        logger.info(f"运行实验 {i+1}/{len(args.configs)}: {config_path}")
        logger.info(f"{'*'*40}")
        
        try:
            # 加载配置获取基本信息
            config = load_config(config_path)
            exp_name = config.get('experiment', {}).get('name', f'exp_{i}')
            exp_id = f"{timestamp}_{exp_name}"
            
            # 创建Pipeline并运行
            pipeline = AlignmentPipeline(config_path, exp_id)
            pipeline.prepare_data()
            pipeline.prepare_model()
            train_metrics = pipeline.train()
            final_metrics = pipeline.evaluate()
            pipeline.save_results()
            
            results.append({
                'exp_id': exp_id,
                'config_path': config_path,
                'method': config.get('training', {}).get('method', 'unknown'),
                'model': config.get('model', {}).get('name', 'unknown'),
                'config': config,
                'train_metrics': train_metrics,
                'final_metrics': final_metrics,
            })
            
            logger.info(f"实验 {exp_id} 完成")
            
        except Exception as e:
            logger.error(f"实验 {config_path} 失败: {e}")
            results.append({
                'exp_id': f"failed_{i}",
                'config_path': config_path,
                'error': str(e),
            })
    
    # 保存所有结果
    os.makedirs(output_dir, exist_ok=True)
    save_metrics_json(results, os.path.join(output_dir, 'all_results.json'))
    
    # 生成对比报告
    generate_comparison_report(results, output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"所有对比实验完成！")
    logger.info(f"结果保存在: {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
