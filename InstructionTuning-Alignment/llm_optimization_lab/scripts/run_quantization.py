#!/usr/bin/env python3
"""运行量化实验

使用方法:
    python scripts/run_quantization.py --config configs/quantization/int8.yaml
    python scripts/run_quantization.py --config configs/quantization/int8.yaml --compare_configs int4.yaml
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
from src.compression.quantizer import ModelQuantizer
from src.evaluation.evaluator import QuantizationEvaluator


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='运行量化实验')
    parser.add_argument('--config', required=True, help='配置文件')
    parser.add_argument('--compare_configs', nargs='*', help='对比配置')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    exp_id = f"quant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file = os.path.join(project_root, 'logs', f'{exp_id}.log')
    logger = setup_logger('quantization', log_file)
    
    logger.info(f"开始量化实验: {config['experiment']['name']}")
    
    seed_everything(config['experiment'].get('seed', 42))
    
    if not torch.cuda.is_available():
        logger.error("必须使用GPU运行！")
        sys.exit(1)
    
    quantizer = ModelQuantizer(config)
    
    logger.info("加载模型...")
    model = quantizer.load_model()
    
    logger.info("应用量化...")
    start_time = time.time()
    quantized_model = quantizer.quantize(model)
    quant_time = time.time() - start_time
    
    logger.info(f"量化完成，耗时 {quant_time:.1f}s")
    
    logger.info("运行基准测试...")
    evaluator = QuantizationEvaluator(config)
    results = evaluator.benchmark(quantized_model)
    
    output_dir = os.path.join(project_root, config['output']['save_dir'], exp_id)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果保存在: {output_dir}")
    logger.info(f"精度: {results.get('accuracy', 'N/A')}")
    logger.info(f"推理速度: {results.get('latency_ms', 'N/A')}ms")
    logger.info(f"显存占用: {results.get('memory_gb', 'N/A')}GB")
    
    # 对比实验
    if args.compare_configs:
        logger.info("运行对比实验...")
        comparisons = {}
        for comp_config_path in args.compare_configs:
            comp_config = load_config(os.path.join(os.path.dirname(args.config), comp_config_path))
            comp_quantizer = ModelQuantizer(comp_config)
            comp_model = comp_quantizer.load_model()
            comp_quantized = comp_quantizer.quantize(comp_model)
            comp_results = evaluator.benchmark(comp_quantized)
            comparisons[comp_config['experiment']['name']] = comp_results
        
        comparisons['baseline'] = results
        comparisons_path = os.path.join(output_dir, 'comparisons.json')
        with open(comparisons_path, 'w', encoding='utf-8') as f:
            json.dump(comparisons, f, ensure_ascii=False, indent=2)
        logger.info(f"对比结果保存在: {comparisons_path}")


if __name__ == '__main__':
    main()
