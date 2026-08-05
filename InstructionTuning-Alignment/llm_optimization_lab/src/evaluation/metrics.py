"""
评估指标模块

提供全面的模型评估指标，包括：
- 准确性指标（Accuracy, Exact Match, F1）
- 文本生成指标（BLEU, ROUGE, METEOR）
- 性能指标（延迟、吞吐量、内存占用）
"""

import time
import numpy as np
from typing import List, Dict, Optional, Union
from collections import Counter


# ============ 准确性指标 ============

def accuracy_score(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> float:
    """
    计算准确率（Exact Match）。

    Args:
        predictions: 预测结果列表
        references: 参考结果列表
        normalize: 是否在比较前规范化文本

    Returns:
        准确率 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions 和 references 长度不匹配")

    if not predictions:
        return 0.0

    correct = 0
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = pred.strip().lower()
            ref = ref.strip().lower()
        if pred == ref:
            correct += 1

    return correct / len(predictions)


def f1_score_single(prediction: str, reference: str) -> float:
    """
    计算单个样本的 F1 分数（基于 token 级别）。

    Args:
        prediction: 预测文本
        reference: 参考文本

    Returns:
        F1 分数 (0-1)
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def f1_score(
    predictions: List[str],
    references: List[str],
) -> float:
    """
    计算 F1 分数（宏平均）。

    Args:
        predictions: 预测结果列表
        references: 参考结果列表

    Returns:
        平均 F1 分数 (0-1)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions 和 references 长度不匹配")
    if not predictions:
        return 0.0

    f1_scores = [
        f1_score_single(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    return float(np.mean(f1_scores))


# ============ BLEU 指标 ============

def _ngrams(tokens: List[str], n: int) -> Counter:
    """生成 n-gram Counter。"""
    return Counter(
        tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)
    )


def bleu_score(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
    weights: Optional[List[float]] = None,
    smooth: bool = True,
) -> float:
    """
    计算 BLEU 分数。

    Args:
        predictions: 预测结果列表
        references: 参考结果列表
        max_n: 最大 n-gram 阶数
        weights: 各阶 n-gram 权重
        smooth: 是否使用平滑

    Returns:
        BLEU 分数 (0-1)
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n

    if len(predictions) != len(references):
        raise ValueError("predictions 和 references 长度不匹配")
    if not predictions:
        return 0.0

    log_avg_precision = 0.0
    valid_n = 0

    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        clipped_counts = 0
        total_counts = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            pred_ngrams = _ngrams(pred_tokens, n)
            ref_ngrams = _ngrams(ref_tokens, n)

            total_counts += sum(pred_ngrams.values())
            clipped = sum(
                min(pred_ngrams[ng], ref_ngrams.get(ng, 0))
                for ng in pred_ngrams
            )
            clipped_counts += clipped

        if total_counts > 0 and clipped_counts > 0:
            precision = clipped_counts / total_counts
            log_avg_precision += weights[n - 1] * np.log(precision)
            valid_n += 1
        elif smooth and total_counts > 0 and n == 1:
            log_avg_precision += weights[n - 1] * np.log(1.0 / total_counts)
            valid_n += 1

    if valid_n == 0:
        return 0.0

    bp = 1.0
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    avg_pred_len = np.mean(pred_lengths)
    avg_ref_len = np.mean(ref_lengths)

    if avg_pred_len <= avg_ref_len:
        bp = np.exp(1 - avg_ref_len / avg_pred_len)
    else:
        bp = 1.0

    bleu = bp * np.exp(log_avg_precision)
    return float(max(0.0, min(1.0, bleu)))


# ============ ROUGE 指标 ============

def _lcs_length(x: List[str], y: List[str]) -> int:
    """计算最长公共子序列长度。"""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def rouge_l(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    计算 ROUGE-L 分数。

    Args:
        predictions: 预测结果列表
        references: 参考结果列表

    Returns:
        包含 precision, recall, f1 的字典
    """
    if len(predictions) != len(references):
        raise ValueError("predictions 和 references 长度不匹配")
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precisions = []
    recalls = []
    f1s = []

    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()

        lcs_len = _lcs_length(pred_tokens, ref_tokens)

        precision = lcs_len / len(pred_tokens) if pred_tokens else 0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }


# ============ 性能指标 ============

class PerformanceMetrics:
    """
    性能指标收集器。

    用于测量模型推理的延迟、吞吐量和内存占用。
    """

    def __init__(self):
        self._latencies: List[float] = []
        self._throughputs: List[float] = []
        self._memory_usages: List[float] = []

    def measure_latency(self, fn, *args, **kwargs):
        """
        测量函数执行延迟。

        Args:
            fn: 待测量的函数
            *args, **kwargs: 函数参数

        Returns:
            (函数返回值, 延迟毫秒数)
        """
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        latency = (end - start) * 1000
        self._latencies.append(latency)
        return result, latency

    def add_latency(self, latency_ms: float) -> None:
        """添加延迟记录。"""
        self._latencies.append(latency_ms)

    def add_throughput(self, requests_per_second: float) -> None:
        """添加吞吐量记录。"""
        self._throughputs.append(requests_per_second)

    def add_memory_usage(self, memory_mb: float) -> None:
        """添加内存使用记录。"""
        self._memory_usages.append(memory_mb)

    @property
    def avg_latency(self) -> float:
        """平均延迟（毫秒）。"""
        return float(np.mean(self._latencies)) if self._latencies else 0.0

    @property
    def p50_latency(self) -> float:
        """P50 延迟（毫秒）。"""
        return float(np.percentile(self._latencies, 50)) if self._latencies else 0.0

    @property
    def p95_latency(self) -> float:
        """P95 延迟（毫秒）。"""
        return float(np.percentile(self._latencies, 95)) if self._latencies else 0.0

    @property
    def avg_throughput(self) -> float:
        """平均吞吐量。"""
        return float(np.mean(self._throughputs)) if self._throughputs else 0.0

    @property
    def avg_memory(self) -> float:
        """平均内存使用（MB）。"""
        return float(np.mean(self._memory_usages)) if self._memory_usages else 0.0

    def summary(self) -> Dict[str, float]:
        """生成性能摘要。"""
        return {
            "avg_latency_ms": self.avg_latency,
            "p50_latency_ms": self.p50_latency,
            "p95_latency_ms": self.p95_latency,
            "avg_throughput_rps": self.avg_throughput,
            "avg_memory_mb": self.avg_memory,
        }

    def reset(self) -> None:
        """重置所有记录。"""
        self._latencies.clear()
        self._throughputs.clear()
        self._memory_usages.clear()