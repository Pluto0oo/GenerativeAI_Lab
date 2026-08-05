"""
Prompt 效果评估模块

提供多种 Prompt 评估指标，包括：
- 准确性评估
- 多样性评估
- 连贯性评估
- 成本效益分析
- A/B 测试
"""

import time
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """
    评估结果数据类。

    Attributes:
        strategy_name: 策略名称
        metric_name: 指标名称
        value: 指标值
        details: 详细信息
        timestamp: 评估时间戳
    """
    strategy_name: str
    metric_name: str
    value: float
    details: str = ""
    timestamp: float = field(default_factory=time.time)


class PromptEvaluator:
    """
    Prompt 评估器基类。

    提供统一的评估接口，支持多种评估方式。
    """

    def __init__(self):
        self._results: List[EvalResult] = []

    def evaluate(
        self,
        strategy_name: str,
        ground_truth: List[str],
        predictions: List[str],
        **kwargs,
    ) -> List[EvalResult]:
        """
        运行所有评估指标。

        Args:
            strategy_name: 策略名称
            ground_truth: 真实标签列表
            predictions: 预测结果列表

        Returns:
            评估结果列表
        """
        results = []
        results.append(self.compute_accuracy(strategy_name, ground_truth, predictions))
        results.append(self.compute_diversity(strategy_name, predictions))
        self._results.extend(results)
        return results

    def compute_accuracy(
        self,
        strategy_name: str,
        ground_truth: List[str],
        predictions: List[str],
    ) -> EvalResult:
        """
        计算准确率（Exact Match）。
        """
        if not ground_truth or not predictions:
            return EvalResult(strategy_name, "accuracy", 0.0, "空数据")

        correct = sum(
            1 for gt, pred in zip(ground_truth, predictions)
            if gt.strip().lower() == pred.strip().lower()
        )
        accuracy = correct / len(ground_truth)
        return EvalResult(
            strategy_name, "accuracy", accuracy,
            f"{correct}/{len(ground_truth)} 正确"
        )

    def compute_diversity(
        self,
        strategy_name: str,
        predictions: List[str],
    ) -> EvalResult:
        """
        计算预测结果多样性（唯一答案比例）。
        """
        if not predictions:
            return EvalResult(strategy_name, "diversity", 0.0, "空数据")

        unique_ratio = len(set(predictions)) / len(predictions)
        return EvalResult(
            strategy_name, "diversity", unique_ratio,
            f"{len(set(predictions))}/{len(predictions)} 唯一"
        )

    def compute_consistency(
        self,
        strategy_name: str,
        answers: List[str],
    ) -> EvalResult:
        """
        计算自洽性（多次采样答案一致性）。
        """
        if len(answers) < 2:
            return EvalResult(strategy_name, "consistency", 1.0, "单次采样")

        from collections import Counter
        counter = Counter(answers)
        most_common_count = counter.most_common(1)[0][1]
        consistency = most_common_count / len(answers)
        return EvalResult(
            strategy_name, "consistency", consistency,
            f"众数出现 {most_common_count}/{len(answers)} 次"
        )

    def compute_latency(
        self,
        strategy_name: str,
        start_time: float,
        end_time: float,
    ) -> EvalResult:
        """
        计算延迟。
        """
        latency = end_time - start_time
        return EvalResult(
            strategy_name, "latency_ms", latency * 1000,
            f"耗时 {latency:.3f}s"
        )

    def get_results(self) -> List[EvalResult]:
        """获取所有评估结果。"""
        return self._results.copy()

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        生成评估结果摘要。

        Returns:
            按策略名称分组的指标字典
        """
        summary: Dict[str, Dict[str, float]] = {}
        for result in self._results:
            if result.strategy_name not in summary:
                summary[result.strategy_name] = {}
            summary[result.strategy_name][result.metric_name] = result.value
        return summary


class PromptABTest:
    """
    Prompt A/B 测试框架。

    比较两种 Prompt 策略的效果差异。
    """

    def __init__(self, evaluator: Optional[PromptEvaluator] = None):
        self.evaluator = evaluator or PromptEvaluator()
        self._history: List[Dict] = []

    def run_ab_test(
        self,
        strategy_a_name: str,
        strategy_b_name: str,
        ground_truth: List[str],
        predictions_a: List[str],
        predictions_b: List[str],
    ) -> Dict:
        """
        执行 A/B 测试。

        Args:
            strategy_a_name: 策略 A 名称
            strategy_b_name: 策略 B 名称
            ground_truth: 真实标签
            predictions_a: 策略 A 的预测
            predictions_b: 策略 B 的预测

        Returns:
            A/B 测试结果
        """
        results_a = self.evaluator.evaluate(
            strategy_a_name, ground_truth, predictions_a
        )
        results_b = self.evaluator.evaluate(
            strategy_b_name, ground_truth, predictions_b
        )

        comparison = self._compare_results(results_a, results_b)
        self._history.append({
            "strategy_a": strategy_a_name,
            "strategy_b": strategy_b_name,
            "comparison": comparison,
        })

        return comparison

    def _compare_results(
        self,
        results_a: List[EvalResult],
        results_b: List[EvalResult],
    ) -> Dict:
        """
        比较两组结果。
        """
        metrics_a = {r.metric_name: r.value for r in results_a}
        metrics_b = {r.metric_name: r.value for r in results_b}

        comparison = {}
        all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())

        for metric in all_metrics:
            val_a = metrics_a.get(metric, 0)
            val_b = metrics_b.get(metric, 0)
            diff = val_b - val_a
            comparison[metric] = {
                "A": val_a,
                "B": val_b,
                "diff": diff,
                "winner": "B" if diff > 0 else ("A" if diff < 0 else "tie"),
            }
        return comparison

    def get_history(self) -> List[Dict]:
        """获取 A/B 测试历史。"""
        return self._history.copy()