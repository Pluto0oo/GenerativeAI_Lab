#!/usr/bin/env python3
"""运行RAG医疗问答演示

使用方法:
    # 构建知识库
    python scripts/run_rag_demo.py --config configs/rag/medical_rag.yaml --build_knowledge_base
    
    # 交互式问答
    python scripts/run_rag_demo.py --config configs/rag/medical_rag.yaml --interactive
    
    # 评估RAG效果
    python scripts/run_rag_demo.py --config configs/rag/medical_rag.yaml --evaluate
"""
import os
import sys
import argparse
import json
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import yaml
import torch
from src.utils.seed import seed_everything
from src.utils.logger import setup_logger
from src.rag.retriever import MedicalRAG
from src.evaluation.evaluator import RAGEvaluator


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_knowledge_base(rag, logger):
    logger.info("开始构建知识库...")
    kb_path = rag.config['knowledge_base']['path']
    
    if not os.path.exists(kb_path):
        logger.error(f"知识库路径不存在: {kb_path}")
        return
    
    files = [f for f in os.listdir(kb_path) if f.endswith('.txt') or f.endswith('.md')]
    
    documents = []
    for file in files:
        with open(os.path.join(kb_path, file), 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'id': file,
                'content': content,
                'source': file
            })
    
    logger.info(f"加载了 {len(documents)} 个文档")
    
    rag.build_index(documents)
    logger.info("知识库构建完成！")


def interactive_mode(rag, logger):
    logger.info("进入交互式问答模式（输入 'quit' 退出）")
    print("\n" + "="*60)
    print("医疗问答助手 - RAG Demo")
    print("输入问题开始问答，输入 'quit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("问题: ").strip()
            if question.lower() == 'quit':
                print("再见！")
                break
            
            if not question:
                continue
            
            start_time = time.time()
            result = rag.answer(question)
            elapsed = time.time() - start_time
            
            print(f"\n回答: {result['answer']}")
            print(f"耗时: {elapsed:.2f}s")
            
            if result.get('sources'):
                print(f"参考来源: {result['sources']}")
            
            if result.get('faithfulness') is not None:
                faith = "可信" if result['faithfulness'] > 0.7 else "存疑"
                print(f"忠实度: {faith} ({result['faithfulness']:.2f})")
            
            print()
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            logger.error(f"错误: {e}")
            print(f"发生错误: {e}")


def evaluate_rag(rag, config, logger):
    logger.info("开始评估RAG效果...")
    
    evaluator = RAGEvaluator(config)
    results = evaluator.evaluate(rag)
    
    output_dir = os.path.join(project_root, config.get('output', {}).get('save_dir', 'results'), 'rag_evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估完成，结果保存在: {output_dir}")
    logger.info(f"准确率: {results.get('accuracy', 'N/A')}")
    logger.info(f"忠实度: {results.get('faithfulness', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='RAG医疗问答演示')
    parser.add_argument('--config', required=True, help='配置文件')
    parser.add_argument('--build_knowledge_base', action='store_true', help='构建知识库')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--evaluate', action='store_true', help='评估模式')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    log_file = os.path.join(project_root, 'logs', 'rag_demo.log')
    logger = setup_logger('rag_demo', log_file)
    
    logger.info(f"RAG医疗问答助手启动")
    logger.info(f"配置: {config['experiment']['name']}")
    
    seed_everything(config['experiment'].get('seed', 42))
    
    if not torch.cuda.is_available():
        logger.error("必须使用GPU运行！")
        sys.exit(1)
    
    rag = MedicalRAG(config)
    
    if args.build_knowledge_base:
        build_knowledge_base(rag, logger)
    
    if args.interactive:
        interactive_mode(rag, logger)
    
    if args.evaluate:
        evaluate_rag(rag, config, logger)
    
    if not any([args.build_knowledge_base, args.interactive, args.evaluate]):
        # 默认：构建 + 交互
        build_knowledge_base(rag, logger)
        interactive_mode(rag, logger)


if __name__ == '__main__':
    main()
