import torch
import random
import numpy as np
from typing import Dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: Dict) -> torch.device:
    device_str = config['experiment'].get('device', 'cuda')
    if device_str == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: Dict) -> str:
    return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
