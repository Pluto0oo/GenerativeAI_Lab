"""
固定随机种子工具模块

提供统一的随机种子设置功能，确保实验可复现性。
支持 Python random、NumPy、PyTorch、CUDA 等多个随机源。
"""

import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    固定所有随机种子，确保实验可复现。

    Args:
        seed: 随机种子值，默认为 42
        deterministic: 是否启用确定性模式（可能影响性能）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """seed_everything 别名，保持兼容性。"""
    set_seed(seed, deterministic)


def get_seed() -> int:
    """
    获取当前环境中设置的随机种子。

    Returns:
        当前随机种子值
    """
    return int(os.environ.get('PYTHONHASHSEED', '42'))