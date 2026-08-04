import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import ProtoNet, ConvNet, create_model


def test_convnet_output_shape():
    model = ConvNet(hidden_size=64, num_layers=4, embedding_dim=64)
    x = torch.randn(16, 1, 28, 28)
    output = model(x)
    assert output.shape == (16, 64)


def test_protonet_forward():
    model = ProtoNet(backbone='convnet', hidden_size=64, embedding_dim=64)
    support_images = torch.randn(5, 1, 28, 28)
    support_labels = torch.tensor([0, 1, 2, 3, 4])
    query_images = torch.randn(10, 1, 28, 28)
    
    logits = model(support_images, support_labels, query_images)
    assert logits.shape == (10, 5)


def test_create_model():
    config = {
        'model': {
            'type': 'protonet',
            'backbone': 'convnet',
            'hidden_size': 64,
            'embedding_dim': 64,
            'num_layers': 4,
        }
    }
    model = create_model(config)
    assert isinstance(model, ProtoNet)


def test_model_device():
    model = ConvNet()
    model = model.to('cpu')
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    assert output.device == torch.device('cpu')
