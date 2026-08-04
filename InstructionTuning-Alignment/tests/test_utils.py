import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.utils import set_seed, get_device, count_parameters
from src.models import ConvNet


def test_set_seed():
    seed = 42
    set_seed(seed)
    
    np_random = np.random.rand()
    torch_random = torch.rand(1).item()
    
    set_seed(seed)
    
    assert np.random.rand() == np_random
    assert torch.rand(1).item() == torch_random


def test_get_device():
    config = {'experiment': {'device': 'cuda'}}
    device = get_device(config)
    
    if torch.cuda.is_available():
        assert str(device) == 'cuda'
    else:
        assert str(device) == 'cpu'


def test_count_parameters():
    model = ConvNet(hidden_size=64, num_layers=4, embedding_dim=64)
    params = count_parameters(model)
    assert params > 0


def test_format_metrics():
    from src.utils import format_metrics
    metrics = {'accuracy': 0.95, 'loss': 0.1234}
    formatted = format_metrics(metrics)
    assert 'accuracy: 0.9500' in formatted
    assert 'loss: 0.1234' in formatted
