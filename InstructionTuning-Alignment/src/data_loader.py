import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot
import learn2learn as l2l
from typing import Tuple, Dict


def get_omniglot_dataset(config: Dict) -> Tuple[l2l.data.MetaDataset, l2l.data.MetaDataset]:
    data_path = config['data']['data_path']
    download = config['data']['download']
    
    augment = config['data'].get('augment', False)
    augmentation_methods = config['data'].get('augmentation_methods', [])
    
    train_transform_list = [
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
    ]
    
    if augment:
        if 'random_rotation' in augmentation_methods:
            train_transform_list.append(transforms.RandomRotation(15))
        if 'random_shift' in augmentation_methods:
            train_transform_list.append(transforms.RandomAffine(0, translate=(0.1, 0.1)))
    
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.92206], std=[0.08426]),
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    
    test_transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.92206], std=[0.08426]),
    ])
    
    train_dataset = Omniglot(
        root=data_path,
        background=True,
        download=download,
        transform=train_transform,
    )
    
    test_dataset = Omniglot(
        root=data_path,
        background=False,
        download=download,
        transform=test_transform,
    )
    
    meta_train_dataset = l2l.data.MetaDataset(train_dataset)
    meta_test_dataset = l2l.data.MetaDataset(test_dataset)
    
    return meta_train_dataset, meta_test_dataset


def create_task_loader(dataset: l2l.data.MetaDataset, config: Dict, mode: str = "train") -> DataLoader:
    if mode == "train":
        ways = config['data']['train_ways']
        shots = config['data']['train_shots']
        queries = config['data']['train_queries']
    else:
        ways = config['data']['test_ways']
        shots = config['data']['test_shots']
        queries = config['data']['test_queries']
    
    task_transforms = [
        l2l.data.transforms.NWays(dataset, ways),
        l2l.data.transforms.KShots(dataset, shots + queries),
        l2l.data.transforms.LoadData(dataset),
        l2l.data.transforms.RemapLabels(dataset),
        l2l.data.transforms.ConsecutiveLabels(dataset),
    ]
    
    taskset = l2l.data.TaskDataset(
        dataset,
        task_transforms=task_transforms,
        num_tasks=-1,
    )
    
    num_workers = config['experiment'].get('num_workers', 4)
    task_loader = DataLoader(taskset, batch_size=config['training']['meta_batch_size'], shuffle=True, num_workers=num_workers)
    
    return task_loader


def split_support_query(batch: Tuple[torch.Tensor, torch.Tensor], shots: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images, labels = batch
    
    if images.dim() == 5:
        images = images.squeeze(0)
    if labels.dim() == 2:
        labels = labels.squeeze(0)
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    unique_labels = torch.unique(labels)
    num_ways = unique_labels.size(0)
    
    for idx, label in enumerate(unique_labels):
        way_mask = (labels == label)
        way_indices = way_mask.nonzero(as_tuple=True)[0]
        permuted_indices = way_indices[torch.randperm(way_indices.size(0))]
        
        support_indices = permuted_indices[:shots]
        query_indices = permuted_indices[shots:]
        
        support_images.append(images[support_indices])
        support_labels.append(torch.full((len(support_indices),), idx, dtype=torch.long))
        
        if len(query_indices) > 0:
            query_images.append(images[query_indices])
            query_labels.append(torch.full((len(query_indices),), idx, dtype=torch.long))
    
    support_images = torch.cat(support_images) if len(support_images) > 0 else images[:0]
    support_labels = torch.cat(support_labels) if len(support_labels) > 0 else torch.tensor([], dtype=torch.long)
    query_images = torch.cat(query_images) if len(query_images) > 0 else images[:0]
    query_labels = torch.cat(query_labels) if len(query_labels) > 0 else torch.tensor([], dtype=torch.long)
    
    return support_images, support_labels, query_images, query_labels
