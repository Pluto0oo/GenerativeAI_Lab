#!/usr/bin/env python3
"""主实验运行脚本 - 指令微调与对齐技术

使用方法:
    # 运行单次实验
    python scripts/run_experiment.py --config configs/experiment/sft_llama3.yaml
    
    # 指定实验ID
    python scripts/run_experiment.py --config configs/experiment/dpo_alignment.yaml --exp_id my_exp_001
    
    # 多次重复实验
    python scripts/run_experiment.py --config configs/experiment/sft_llama3.yaml --repeat_times 3
"""

import os
import sys
import argparse
import logging

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 离线模式：使用本地模型时避免 transformers/huggingface_hub 联网检查（网络不稳定时必需）
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from src.utils.logger import setup_logger, get_timestamp
from src.pipeline.alignment_pipeline import AlignmentPipeline


def main():
    parser = argparse.ArgumentParser(
        description='运行指令微调与对齐实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础用法
    python scripts/run_experiment.py --config configs/experiment/sft_llama3.yaml
    
    # 指定实验ID和重复次数
    python scripts/run_experiment.py --config configs/experiment/dpo_alignment.yaml --exp_id test_001 --repeat_times 3
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='YAML配置文件路径'
    )
    
    parser.add_argument(
        '--exp_id',
        type=str,
        default=None,
        help='实验ID (如未指定则自动生成时间戳)'
    )
    
    parser.add_argument(
        '--repeat_times',
        type=int,
        default=None,
        help='重复实验次数 (覆盖配置文件中的设置)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    exp_id = args.exp_id or f"exp_{get_timestamp()}"
    log_file = os.path.join(project_root, 'logs', f'{exp_id}.log')
    logger = setup_logger('alignment', log_file, getattr(logging, args.log_level))
    
    logger.info(f"{'='*60}")
    logger.info(f"指令微调与对齐实验系统")
    logger.info(f"{'='*60}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"实验ID: {exp_id}")
    
    try:
        # 创建Pipeline
        pipeline = AlignmentPipeline(args.config, exp_id)
        
        # 多次重复实验
        if args.repeat_times and args.repeat_times > 1:
            logger.info(f"运行 {args.repeat_times} 次重复实验...")
            all_metrics = pipeline.run_with_repeats(args.repeat_times)
            logger.info(f"所有重复实验完成！")
        else:
            # 单次完整Pipeline
            final_metrics = pipeline.run_full_pipeline()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"实验成功完成！")
        logger.info(f"结果保存在: results/{exp_id}/")
        logger.info(f"详细日志保存在: logs/{exp_id}.log")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"实验失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
