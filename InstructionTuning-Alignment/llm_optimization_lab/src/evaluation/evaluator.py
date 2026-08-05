"""
综合评估器模块

提供一站式模型评估接口，整合所有评估指标：
- 准确性评估
- 生成质量评估
- 幻觉检测
- 性能基准测试
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field

from .metrics import (
    accuracy_score,
    f1_score,
    bleu_score,
    rouge_l,
    PerformanceMetrics,
)
from .hallucination import HallucinationDetector


@dataclass
class EvaluationReport:
    """评估报告数据类。"""
    model_name: str
    task: str
    metrics: Dict[str, float] = field(default_factory=dict)
    hallucination: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "metrics": self.metrics,
            "hallucination": self.hallucination,
            "performance": self.performance,
            "timestamp": self.timestamp,
        }


class PromptEvaluator:
    """
    Prompt工程评估器。

    专门用于评估不同Prompt策略在问答任务上的效果。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('model', {}).get('name', 'TinyLlama')
        self.device = config.get('model', {}).get('device', 'cuda')
        self.max_new_tokens = config.get('model', {}).get('max_new_tokens', 256)
        self.temperature = config.get('model', {}).get('temperature', 0.0)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载模型。"""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # 优先使用 path 字段，否则使用 name
            model_path = self.config.get('model', {}).get('path', '') or self.config.get('model', {}).get('name', 'TinyLlama')
            torch_dtype = self.config.get('model', {}).get('torch_dtype', 'bfloat16')

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
            )
            self._model.eval()
        return self._model, self._tokenizer

    def _generate(self, prompt: str) -> str:
        """使用模型生成回答。"""
        import torch
        model, tokenizer = self._load_model()

        # 尝试使用chat template，如果不支持则直接使用prompt
        try:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # 只提取生成的部分
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        result = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # 如果生成结果为空或太短，尝试直接解码全部输出
        if len(result) < 2:
            result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        return result

    def _load_test_data(self) -> List[Dict]:
        """加载测试数据。"""
        data_path = self.config.get('data', {}).get('path', '')
        num_samples = self.config.get('data', {}).get('num_samples', 100)

        if not data_path or not os.path.exists(data_path):
            return self._generate_fallback_data()

        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(samples) >= num_samples:
                    break
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples if samples else self._generate_fallback_data()

    def _generate_fallback_data(self) -> List[Dict]:
        """生成备用测试数据。"""
        return [
            {
                "id": i,
                "question": f"这是第{i+1}个医学问题：高血压的标准治疗药物是什么？",
                "options": {"A": "氨氯地平", "B": "阿莫西林", "C": "维生素C", "D": "布洛芬"},
                "answer": "A",
            }
            for i in range(20)
        ]

    def evaluate(self, strategy) -> Dict[str, Any]:
        """
        评估Prompt策略。

        Args:
            strategy: Prompt策略实例（需有build_prompt方法）

        Returns:
            评估结果字典
        """
        import torch

        # 加载数据
        samples = self._load_test_data()
        num_samples = min(len(samples), self.config.get('data', {}).get('num_samples', 50))
        samples = samples[:num_samples]

        if not torch.cuda.is_available():
            return {"error": "CUDA不可用"}

        # 加载模型
        self._load_model()

        predictions = []
        references = []
        latencies = []

        for idx, sample in enumerate(samples):
            question = sample.get('question', '')
            options = sample.get('options', {})
            answer = sample.get('answer', '')

            # 构造options文本
            options_text = "\n".join(
                f"{k}. {v}" for k, v in options.items()
            )

            # 使用策略构建Prompt
            try:
                prompt = strategy.build_prompt(
                    question=question,
                    options=options_text,
                )
            except Exception:
                prompt = f"问题：{question}\n选项：{options_text}\n答案："

            # 生成
            start = time.perf_counter()
            try:
                response = self._generate(prompt)
            except Exception as e:
                response = f"[生成错误: {e}]"
            latency = (time.perf_counter() - start) * 1000

            predictions.append(response)
            references.append(answer)
            latencies.append(latency)

            if (idx + 1) % 10 == 0:
                print(f"  评估进度: {idx + 1}/{len(samples)}")

        # 计算指标
        metrics = self._compute_metrics(predictions, references)
        metrics['avg_latency_ms'] = float(np.mean(latencies))
        metrics['num_samples'] = len(samples)
        metrics['strategy'] = strategy.__class__.__name__

        # 简单幻觉检测
        hallucination_rate = self._estimate_hallucination(predictions)
        metrics['hallucination_rate'] = hallucination_rate

        return metrics

    def _compute_metrics(self, predictions, references):
        """计算评估指标。"""
        # 提取预测的答案选项
        pred_answers = []
        for pred in predictions:
            pred_upper = pred.upper()
            if 'A' in pred_upper and len(pred) < 50:
                pred_answers.append('A')
            elif 'B' in pred_upper and len(pred) < 50:
                pred_answers.append('B')
            elif 'C' in pred_upper and len(pred) < 50:
                pred_answers.append('C')
            elif 'D' in pred_upper and len(pred) < 50:
                pred_answers.append('D')
            else:
                # 尝试从文本中提取选项
                for opt in ['A', 'B', 'C', 'D']:
                    if f'{opt}.' in pred or f'{opt})' in pred:
                        pred_answers.append(opt)
                        break
                else:
                    pred_answers.append('')

        acc = accuracy_score(pred_answers, references) if pred_answers else 0.0

        return {
            "accuracy": float(acc),
            "accuracy_letter": float(acc),
        }

    def _estimate_hallucination(self, predictions):
        """估算幻觉率（简化版）。"""
        if not predictions:
            return 0.0

        hallucination_count = 0
        for pred in predictions:
            # 如果预测非常短（< 10字符）且不是有效答案，可能是幻觉
            if len(pred) < 10 and not any(c in pred.upper() for c in 'ABCD'):
                hallucination_count += 1
            # 如果包含重复文本，可能是幻觉
            elif len(pred) > 0:
                words = pred.split()
                if len(words) > 0:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.3:  # 重复率过高
                        hallucination_count += 1

        return float(hallucination_count / len(predictions))


class RAGEvaluator:
    """RAG评估器。"""

    def __init__(self, config: Dict):
        self.config = config

    def evaluate(self, rag_system) -> Dict[str, Any]:
        """评估RAG系统。"""
        return {
            "accuracy": 0.0,
            "faithfulness": 0.0,
            "note": "RAG评估需要完整知识库，请使用 --evaluate 模式",
        }


class QuantizationEvaluator:
    """量化评估器。"""

    def __init__(self, config: Dict):
        self.config = config

    def benchmark(self, model) -> Dict[str, Any]:
        """基准测试量化模型。"""
        import torch

        # 计算模型大小
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = param_count * 4 / 1024 / 1024  # FP32估算

        # 显存占用
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # 简单推理测试
        try:
            start = time.perf_counter()
            _ = model(torch.randn(1, 32, model.config.hidden_size).to(model.device))
            latency = (time.perf_counter() - start) * 1000
        except Exception:
            latency = 0.0

        return {
            "model_size_mb": float(model_size_mb),
            "parameters": int(param_count),
            "inference_latency_ms": float(latency),
            "accuracy": None,  # 需要完整测试数据
        }


class ComprehensiveEvaluator:
    """综合评估器（保留兼容性）。"""

    def __init__(self, model_name: str = "unnamed_model", task: str = "text_generation"):
        self.model_name = model_name
        self.task = task

    def evaluate(self, predictions, references, **kwargs):
        metrics = {
            "accuracy": accuracy_score(predictions, references),
            "f1": f1_score(predictions, references),
        }
        try:
            metrics["bleu"] = bleu_score(predictions, references)
        except Exception:
            metrics["bleu"] = 0.0
        return metrics