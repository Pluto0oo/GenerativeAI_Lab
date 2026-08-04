import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ConvNet(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 4, embedding_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        layers = []
        in_channels = 1
        pool_count = min(num_layers, 3)
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_size))
            layers.append(nn.ReLU())
            if i < pool_count:
                layers.append(nn.MaxPool2d(2))
            in_channels = hidden_size
        
        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_size, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ProtoNet(nn.Module):
    def __init__(self, backbone: str = "convnet", **kwargs):
        super().__init__()
        encoder_kwargs = {k: v for k, v in kwargs.items() if k != 'distance_metric'}
        if backbone == "convnet":
            self.encoder = ConvNet(**encoder_kwargs)
        elif backbone == "resnet18":
            from torchvision.models import resnet18
            self.encoder = resnet18(pretrained=False)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, kwargs.get('embedding_dim', 512))
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.distance_metric = kwargs.get('distance_metric', 'euclidean')
    
    def compute_distance(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        if self.distance_metric == 'euclidean':
            return torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance_metric == 'manhattan':
            return torch.cdist(query_embeddings, prototypes, p=1)
        elif self.distance_metric == 'cosine':
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            prototype_norm = F.normalize(prototypes, p=2, dim=1)
            cosine_sim = query_norm @ prototype_norm.T
            return 1 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def forward(self, support_images: torch.Tensor, support_labels: torch.Tensor, query_images: torch.Tensor) -> torch.Tensor:
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)
        
        num_classes = torch.unique(support_labels).size(0)
        prototypes = []
        
        for c in range(num_classes):
            class_mask = (support_labels == c)
            class_embeddings = support_embeddings[class_mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)
        distances = self.compute_distance(query_embeddings, prototypes)
        logits = -distances
        
        return logits


class FinetuneNet(nn.Module):
    def __init__(self, backbone: str = "convnet", **kwargs):
        super().__init__()
        if backbone == "convnet":
            self.encoder = ConvNet(**kwargs)
        elif backbone == "resnet18":
            from torchvision.models import resnet18
            self.encoder = resnet18(pretrained=False)
            self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, kwargs.get('embedding_dim', 512))
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.classifier = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        if self.classifier is not None:
            return self.classifier(embeddings)
        return embeddings
    
    def set_classifier(self, num_classes: int, embedding_dim: int):
        self.classifier = nn.Linear(embedding_dim, num_classes)


def create_model(config: Dict) -> nn.Module:
    model_type = config['model']['type']
    backbone = config['model']['backbone']
    
    kwargs = {
        'hidden_size': config['model']['hidden_size'],
        'embedding_dim': config['model']['embedding_dim'],
        'num_layers': config['model'].get('num_layers', 4),
        'distance_metric': config['model'].get('distance_metric', 'euclidean'),
    }
    
    if model_type == "protonet":
        model = ProtoNet(backbone=backbone, **kwargs)
    elif model_type == "finetune":
        model = FinetuneNet(backbone=backbone, **kwargs)
    elif model_type == "maml":
        model = ConvNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
