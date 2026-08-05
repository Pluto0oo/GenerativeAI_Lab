"""
蒸馏实现模块

提供知识蒸馏功能，支持：
- Logits 蒸馏（ softened logits）
- Feature 蒸馏（中间层特征）
- Attention 蒸馏（注意力权重）
- 自蒸馏（无需教师模型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod


class DistillationLoss(nn.Module):
    """
    蒸馏损失基类。

    所有蒸馏损失都应继承此类。
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        计算蒸馏损失。

        Args:
            student_output: 学生模型输出
            teacher_output: 教师模型输出
            **kwargs: 其他参数

        Returns:
            损失值
        """
        pass


class LogitsDistillationLoss(DistillationLoss):
    """
    Logits 蒸馏损失。

    基于 softened logits 的 KL 散度损失。

    Args:
        temperature: 温度参数（越大越平滑）
        alpha: 蒸馏损失权重
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        计算 logits 蒸馏损失。
        """
        soft_teacher = F.log_softmax(teacher_output / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output / self.temperature, dim=-1)

        distill_loss = F.kl_div(
            soft_student, soft_teacher, reduction='batchmean'
        ) * (self.temperature ** 2)

        if labels is not None:
            hard_loss = F.cross_entropy(student_output, labels)
            return self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        return distill_loss


class FeatureDistillationLoss(DistillationLoss):
    """
    特征蒸馏损失。

    对齐学生和教师的中间层特征。

    Args:
        projection_dim: 投影维度（若学生和教师特征维度不同）
        layer_weights: 各层权重
    """

    def __init__(
        self,
        projection_dim: Optional[int] = None,
        layer_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.layer_weights = layer_weights
        self._projection = None

    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失。

        Args:
            student_features: 学生中间层特征列表
            teacher_features: 教师中间层特征列表
        """
        if len(student_features) != len(teacher_features):
            min_len = min(len(student_features), len(teacher_features))
            student_features = student_features[-min_len:]
            teacher_features = teacher_features[-min_len:]

        total_loss = torch.tensor(0.0, device=student_features[0].device)
        num_layers = len(student_features)

        for i, (sf, tf) in enumerate(zip(student_features, teacher_features)):
            if sf.shape != tf.shape:
                if self._projection is None or self._projection.in_features != sf.shape[-1]:
                    self._projection = nn.Linear(
                        sf.shape[-1], tf.shape[-1]
                    ).to(sf.device)
                sf = self._projection(sf)

            layer_loss = F.mse_loss(sf, tf.detach())
            weight = self.layer_weights[i] if self.layer_weights else 1.0 / num_layers
            total_loss = total_loss + weight * layer_loss

        return total_loss


class AttentionDistillationLoss(DistillationLoss):
    """
    注意力蒸馏损失。

    对齐学生和教师的注意力权重。

    Args:
        alpha: 注意力损失权重
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        student_attention: torch.Tensor,
        teacher_attention: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        计算注意力蒸馏损失。

        Args:
            student_attention: 学生注意力矩阵 (batch, heads, seq, seq)
            teacher_attention: 教师注意力矩阵
        """
        if student_attention.shape != teacher_attention.shape:
            student_attention = student_attention.mean(dim=1, keepdim=True)
            teacher_attention = teacher_attention.mean(dim=1, keepdim=True)

        loss = F.kl_div(
            F.log_softmax(student_attention, dim=-1),
            F.softmax(teacher_attention, dim=-1),
            reduction='batchmean',
        )
        return self.alpha * loss


class SelfDistillationLoss(DistillationLoss):
    """
    自蒸馏损失。

    使用模型自身的输出作为软标签，无需独立的教师模型。

    Args:
        temperature: 温度参数
        num_augmentations: 增强次数
    """

    def __init__(self, temperature: float = 3.0, num_augmentations: int = 2):
        super().__init__()
        self.temperature = temperature
        self.num_augmentations = num_augmentations

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        计算自蒸馏损失。
        """
        soft_target = F.log_softmax(teacher_output / self.temperature, dim=-1)
        soft_input = F.log_softmax(student_output / self.temperature, dim=-1)

        loss = F.kl_div(
            soft_input, soft_target, reduction='batchmean'
        ) * (self.temperature ** 2)

        return loss


class DistillationConfig:
    """
    蒸馏配置类。

    Attributes:
        method: 蒸馏方法
        temperature: 温度参数
        alpha: 损失权重
    """

    def __init__(
        self,
        method: str = "logits",
        temperature: float = 4.0,
        alpha: float = 0.7,
        **kwargs,
    ):
        self.method = method
        self.temperature = temperature
        self.alpha = alpha
        self.extra_params = kwargs

    def to_dict(self) -> Dict:
        """转换为字典。"""
        return {
            "method": self.method,
            "temperature": self.temperature,
            "alpha": self.alpha,
            **self.extra_params,
        }


def create_distillation_loss(config: DistillationConfig) -> DistillationLoss:
    """
    工厂函数：根据配置创建蒸馏损失。

    Args:
        config: 蒸馏配置

    Returns:
        蒸馏损失实例
    """
    if config.method == "logits":
        return LogitsDistillationLoss(
            temperature=config.temperature, alpha=config.alpha
        )
    elif config.method == "feature":
        return FeatureDistillationLoss(**config.extra_params)
    elif config.method == "attention":
        return AttentionDistillationLoss(alpha=config.alpha)
    elif config.method == "self":
        return SelfDistillationLoss(
            temperature=config.temperature, **config.extra_params
        )
    else:
        raise ValueError(f"未知的蒸馏方法: {config.method}")