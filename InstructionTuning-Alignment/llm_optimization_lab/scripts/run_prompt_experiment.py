#!/usr/bin/env python3
"""运行Prompt工程实验

使用方法:
    python scripts/run_prompt_experiment.py --config configs/prompt/cot.yaml
    python scripts/run_prompt_experiment.py --config configs/prompt/cot.yaml --compare_configs zero_shot.yaml few_shot.yaml
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
from src.prompt.strategies import (
    ZeroShotStrategy, FewShotStrategy, ChainOfThoughtStrategy,
    TreeOfThoughtStrategy, SelfConsistencyStrategy
)
from src.evaluation.evaluator import PromptEvaluator


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_strategy(config):
    strategy_map = {
        'zero_shot': ZeroShotStrategy,
        'few_shot': FewShotStrategy,
        'cot': ChainOfThoughtStrategy,
        'tot': TreeOfThoughtStrategy,
        'self_consistency': SelfConsistencyStrategy,
    }
    strategy_name = config['prompt']['strategy']
    strategy_class = strategy_map.get(strategy_name)
    if strategy_class is None:
        raise ValueError(f"未知的策略: {strategy_name}")
    return strategy_class(config['prompt'])


def main():
    parser = argparse.ArgumentParser(description='运行Prompt工程实验')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--compare_configs', nargs='*', help='对比配置文件')
    parser.add_argument('--exp_id', help='实验ID')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    exp_id = args.exp_id or f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = os.path.join(project_root, 'logs', f'{exp_id}.log')
    logger = setup_logger('prompt_experiment', log_file)
    
    logger.info(f"开始实验: {config['experiment']['name']}")
    logger.info(f"实验ID: {exp_id}")
    
    seed_everything(config['experiment'].get('seed', 42))
    
    if not torch.cuda.is_available():
        logger.error("必须使用GPU运行！")
        sys.exit(1)
    
    strategy = get_strategy(config)
    logger.info(f"使用策略: {strategy.__class__.__name__}")
    
    evaluator = PromptEvaluator(config)
    
    logger.info("开始评估...")
    start_time = time.time()
    results = evaluator.evaluate(strategy)
    elapsed = time.time() - start_time
    
    logger.info(f"评估完成，耗时 {elapsed:.1f}s")
    
    # 保存结果
    output_dir = os.path.join(project_root, config['output']['save_dir'], exp_id)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, 'config_used.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    logger.info(f"结果保存在: {output_dir}")
    logger.info(f"准确率: {results.get('accuracy', 'N/A')}")
    logger.info(f"幻觉率: {results.get('hallucination_rate', 'N/A')}")


if __name__ == '__main__':
    main()
