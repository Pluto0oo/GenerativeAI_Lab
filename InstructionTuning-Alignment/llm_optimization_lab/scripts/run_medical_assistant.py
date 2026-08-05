#!/usr/bin/env python3
"""医疗问答助手 - 全流程优化

使用方法:
    python scripts/run_medical_assistant.py --config configs/experiment/medical_assistant.yaml
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import yaml
import torch
from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.pipeline.optimization_pipeline import OptimizationPipeline


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='医疗问答助手')
    parser.add_argument('--config', required=True, help='配置文件')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--evaluate', action='store_true', help='评估模式')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    exp_id = f"medical_assistant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = os.path.join(project_root, 'logs', f'{exp_id}.log')
    logger = setup_logger('medical_assistant', log_file)
    
    logger.info("=" * 60)
    logger.info("医疗问答助手 - 全流程优化系统")
    logger.info("=" * 60)
    logger.info(f"配置: {config['experiment']['name']}")
    logger.info(f"实验ID: {exp_id}")
    
    seed_everything(config['experiment'].get('seed', 42))
    
    if not torch.cuda.is_available():
        logger.error("必须使用GPU运行！")
        sys.exit(1)
    
    pipeline = OptimizationPipeline(config)
    
    if args.interactive:
        logger.info("进入交互式模式")
        pipeline.run_interactive()
    elif args.evaluate:
        logger.info("运行评估模式")
        results = pipeline.evaluate()
        
        output_dir = os.path.join(project_root, config['output']['save_dir'], exp_id)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果保存在: {output_dir}")
    else:
        logger.info("运行全流程优化")
        start_time = time.time()
        results = pipeline.run_full_pipeline()
        elapsed = time.time() - start_time
        
        output_dir = os.path.join(project_root, config['output']['save_dir'], exp_id)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(output_dir, 'summary.md'), 'w', encoding='utf-8') as f:
            f.write(f"# 医疗问答助手实验摘要\n\n")
            f.write(f"- 实验ID: {exp_id}\n")
            f.write(f"- 耗时: {elapsed:.1f}s\n")
            f.write(f"- 准确率: {results.get('accuracy', 'N/A')}\n")
            f.write(f"- 忠实度: {results.get('faithfulness', 'N/A')}\n")
            f.write(f"- 幻觉率: {results.get('hallucination_rate', 'N/A')}\n")
        
        logger.info(f"全流程完成，耗时 {elapsed:.1f}s")
        logger.info(f"结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
