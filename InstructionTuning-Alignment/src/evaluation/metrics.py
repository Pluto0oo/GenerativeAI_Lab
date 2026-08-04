import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self, config: dict):
        self.config = config
        self.metrics_config = config['evaluation']['metrics']
        self.metrics = {}
        self._load_metrics()

    def _load_metrics(self):
        """加载所需的评估指标（使用evaluate库）"""
        try:
            if 'bleu' in self.metrics_config:
                import evaluate
                self.metrics['bleu'] = evaluate.load('bleu')
            if 'rouge' in self.metrics_config:
                import evaluate
                self.metrics['rouge'] = evaluate.load('rouge')
        except Exception as e:
            logger.warning(f"Failed to load some metrics: {e}")

    def compute(self, predictions: List[str], references: List[str]) -> Dict:
        """计算所有评估指标"""
        results = {}

        # 基础统计
        results['length'] = self._compute_length_metrics(predictions)
        results['accuracy'] = self._compute_accuracy(predictions, references)

        # BLEU
        if 'bleu' in self.metrics:
            results['bleu'] = self._compute_bleu(predictions, references)

        # ROUGE
        if 'rouge' in self.metrics:
            results['rouge'] = self._compute_rouge(predictions, references)

        return results

    def _compute_length_metrics(self, predictions: List[str]) -> Dict:
        """计算长度相关指标"""
        if not predictions:
            return {'mean_length': 0, 'max_length': 0, 'min_length': 0}
        lengths = [len(p.split()) for p in predictions]
        return {
            'mean_length': float(np.mean(lengths)),
            'max_length': int(np.max(lengths)),
            'min_length': int(np.min(lengths)),
        }

    def _compute_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """计算准确率（精确匹配）"""
        if not predictions:
            return 0.0
        correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        return correct / len(predictions)

    def _compute_bleu(self, predictions: List[str], references: List[str]) -> Dict:
        """计算BLEU分数"""
        try:
            result = self.metrics['bleu'].compute(
                predictions=predictions,
                references=[[r] for r in references]
            )
            return {
                'bleu': result['bleu'] * 100,
            }
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            return {'bleu': 0.0}

    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict:
        """计算ROUGE分数"""
        try:
            result = self.metrics['rouge'].compute(
                predictions=predictions,
                references=[[r] for r in references]
            )
            return {
                'rouge1': result['rouge1'] * 100,
                'rouge2': result['rouge2'] * 100,
                'rougeL': result['rougeL'] * 100,
            }
        except Exception as e:
            logger.warning(f"ROUGE computation failed: {e}")
            return {'rougeL': 0.0}

    def compute_statistics(self, all_metrics: List[Dict]) -> Dict:
        """计算多次实验的统计信息"""
        stats = {}

        if not all_metrics:
            return stats

        for metric_key in ['train_loss', 'eval_loss', 'accuracy', 'bleu']:
            values = []
            for m in all_metrics:
                if metric_key in m:
                    val = m[metric_key]
                    if isinstance(val, dict) and 'value' in val:
                        values.append(val['value'])
                    elif isinstance(val, (int, float)):
                        values.append(val)

            if values:
                stats[metric_key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) if len(values) > 1 else 0.0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }

        return stats
