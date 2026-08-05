"""
剪枝实现模块

提供模型剪枝功能，支持：
- 结构化剪枝（按通道/层剪枝）
- 非结构化剪枝（按权重剪枝）
- 自动剪枝策略
- 剪枝后精调
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod


class BasePruner(ABC):
    """
    剪枝器基类。

    所有剪枝器实现都应继承此类。
    """

    @abstractmethod
    def prune(self, model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        对模型进行剪枝。

        Args:
            model: 待剪枝的模型
            amount: 剪枝比例

        Returns:
            剪枝后的模型
        """
        pass


class UnstructuredPruner(BasePruner):
    """
    非结构化剪枝器。

    按权重重要性（绝对值大小）剪枝，不改变张量形状。

    Args:
        target_modules: 剪枝目标模块类型
    """

    def __init__(self, target_modules: Optional[List[type]] = None):
        self.target_modules = target_modules or [nn.Linear, nn.Conv2d]
        self._pruning_history: List[Dict] = []

    def prune(self, model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        对模型进行非结构化剪枝。
        """
        model.eval()
        pruned_count = 0
        total_count = 0

        for name, module in model.named_modules():
            if any(isinstance(module, t) for t in self.target_modules):
                if prune.is_pruned(module):
                    prune.remove(module, 'weight')

                prune.l1_unstructured(module, name='weight', amount=amount)

                sparsity = (module.weight == 0).float().mean().item()
                total_count += module.weight.numel()
                pruned_count += int(sparsity * module.weight.numel())

                self._pruning_history.append({
                    "module": name,
                    "sparsity": sparsity,
                    "amount": amount,
                })

        overall_sparsity = pruned_count / total_count if total_count > 0 else 0
        return model

    def get_sparsity(self) -> float:
        """
        获取整体稀疏率。
        """
        if not self._pruning_history:
            return 0.0
        return np.mean([h["sparsity"] for h in self._pruning_history])


class StructuredPruner(BasePruner):
    """
    结构化剪枝器。

    按通道/层剪枝，改变张量形状以实现实际加速。

    Args:
        target_modules: 剪枝目标模块类型
        pruning_dim: 剪枝维度（0=输出通道, 1=输入通道）
    """

    def __init__(
        self,
        target_modules: Optional[List[type]] = None,
        pruning_dim: int = 0,
    ):
        self.target_modules = target_modules or [nn.Conv2d, nn.Linear]
        self.pruning_dim = pruning_dim
        self._pruning_history: List[Dict] = []

    def prune(self, model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        对模型进行结构化剪枝。
        """
        model.eval()

        for name, module in model.named_modules():
            if any(isinstance(module, t) for t in self.target_modules):
                if prune.is_pruned(module):
                    prune.remove(module, 'weight')

                prune.ln_structured(
                    module, name='weight',
                    amount=amount, dim=self.pruning_dim, n=1
                )

                if isinstance(module, nn.Conv2d):
                    out_channels = module.weight.shape[0]
                    remaining = int(out_channels * (1 - amount))
                    self._pruning_history.append({
                        "module": name,
                        "type": "conv2d",
                        "original_channels": out_channels,
                        "remaining_channels": remaining,
                    })
                elif isinstance(module, nn.Linear):
                    out_features = module.weight.shape[0]
                    remaining = int(out_features * (1 - amount))
                    self._pruning_history.append({
                        "module": name,
                        "type": "linear",
                        "original_features": out_features,
                        "remaining_features": remaining,
                    })

        return model


class AutomatedPruner(BasePruner):
    """
    自动剪枝器。

    基于重要性评分自动选择剪枝目标。

    Args:
        importance_fn: 重要性评估函数
        max_sparsity: 最大目标稀疏率
        incremental: 是否增量式剪枝
    """

    def __init__(
        self,
        importance_fn: Optional[Callable] = None,
        max_sparsity: float = 0.5,
        incremental: bool = True,
    ):
        self.importance_fn = importance_fn or self._default_importance
        self.max_sparsity = max_sparsity
        self.incremental = incremental
        self._current_sparsity = 0.0

    def _default_importance(self, weight: torch.Tensor) -> torch.Tensor:
        """
        默认重要性评分：基于 L1 范数。
        """
        return weight.abs()

    def _compute_importance_scores(
        self, model: nn.Module
    ) -> List[Tuple[str, float]]:
        """
        计算各层的重要性得分。
        """
        scores = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                importance = self.importance_fn(module.weight).mean().item()
                scores.append((name, importance))
        return sorted(scores, key=lambda x: x[1])

    def prune(self, model: nn.Module, amount: float = 0.3) -> nn.Module:
        """
        自动剪枝。
        """
        target_sparsity = min(self._current_sparsity + amount, self.max_sparsity)
        actual_amount = target_sparsity - self._current_sparsity

        if actual_amount <= 0:
            return model

        importance_scores = self._compute_importance_scores(model)
        num_to_prune = max(1, int(len(importance_scores) * actual_amount))
        layers_to_prune = [name for name, _ in importance_scores[:num_to_prune]]

        for name, module in model.named_modules():
            if name in layers_to_prune and isinstance(module, nn.Linear):
                if prune.is_pruned(module):
                    prune.remove(module, 'weight')
                prune.l1_unstructured(
                    module, name='weight', amount=actual_amount
                )

        self._current_sparsity = target_sparsity
        return model

    def get_current_sparsity(self) -> float:
        """获取当前稀疏率。"""
        return self._current_sparsity


class PruningConfig:
    """
    剪枝配置类。

    Attributes:
        method: 剪枝方法
        amount: 剪枝比例
        target_modules: 目标模块
    """

    def __init__(
        self,
        method: str = "unstructured",
        amount: float = 0.3,
        target_modules: Optional[List[type]] = None,
        **kwargs,
    ):
        self.method = method
        self.amount = amount
        self.target_modules = target_modules or [nn.Linear, nn.Conv2d]
        self.extra_params = kwargs

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            "method": self.method,
            "amount": self.amount,
            "target_modules": [t.__name__ for t in self.target_modules],
            **self.extra_params,
        }


def create_pruner(config: PruningConfig) -> BasePruner:
    """
    工厂函数：根据配置创建剪枝器。

    Args:
        config: 剪枝配置

    Returns:
        剪枝器实例
    """
    if config.method == "unstructured":
        return UnstructuredPruner(target_modules=config.target_modules)
    elif config.method == "structured":
        return StructuredPruner(
            target_modules=config.target_modules,
            pruning_dim=config.extra_params.get("pruning_dim", 0),
        )
    elif config.method == "automated":
        return AutomatedPruner(**config.extra_params)
    else:
        raise ValueError(f"未知的剪枝方法: {config.method}")