"""
数据预处理模块

支持数据集: MNIST, CIFAR-10
功能: 数据加载、变换、统计分析、可视化
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter


class DataPreprocessor:
    def __init__(self, data_dir='./data', image_size=32):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transforms = {}
        self.datasets = {}
        self._setup_transforms()

    def _setup_transforms(self):
        self.transforms['mnist'] = transforms.Compose([
            transforms.ToTensor()
        ])
        self.transforms['cifar'] = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms['mnist_3ch'] = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms['diffusion'] = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def load_mnist(self, train=True, download=True):
        dataset = datasets.MNIST(
            self.data_dir, train=train, download=download,
            transform=self.transforms['mnist']
        )
        self.datasets['mnist'] = dataset
        return dataset

    def load_cifar10(self, train=True, download=False):
        try:
            dataset = datasets.CIFAR10(
                self.data_dir, train=train, download=download,
                transform=self.transforms['cifar']
            )
            self.datasets['cifar10'] = dataset
            return dataset
        except Exception:
            print('CIFAR-10 not found, falling back to MNIST-3ch')
            return self.load_mnist_3ch(train=train)

    def load_mnist_3ch(self, train=True):
        dataset = datasets.MNIST(
            self.data_dir, train=train, download=False,
            transform=self.transforms['mnist_3ch']
        )
        self.datasets['mnist_3ch'] = dataset
        return dataset

    def create_dataloader(self, dataset, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            pin_memory=True
        )

    def compute_statistics(self, dataset):
        loader = DataLoader(dataset, batch_size=512, shuffle=False)
        all_images = []
        for images, _ in loader:
            all_images.append(images)
        all_images = torch.cat(all_images, dim=0)
        stats = {
            'mean': all_images.mean(dim=[0, 2, 3]).tolist(),
            'std': all_images.std(dim=[0, 2, 3]).tolist(),
            'min': all_images.min().item(),
            'max': all_images.max().item(),
            'shape': tuple(all_images.shape[1:]),
            'num_samples': len(dataset)
        }
        return stats

    def show_samples(self, dataset, num_samples=25, save_path=None):
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        for i, idx in enumerate(indices):
            img, label = dataset[idx]
            ax = axes[i // 5][i % 5]
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap='gray')
            else:
                ax.imshow((img.permute(1, 2, 0) + 1) / 2)
            ax.set_title(f'Label: {label}')
            ax.axis('off')
        plt.suptitle('Dataset Samples', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def compute_class_distribution(self, dataset, num_classes=10):
        loader = DataLoader(dataset, batch_size=1000, shuffle=False)
        labels = []
        for _, batch_labels in loader:
            labels.extend(batch_labels.tolist())
        counter = Counter(labels)
        distribution = {i: counter.get(i, 0) for i in range(num_classes)}
        return distribution

    def plot_class_distribution(self, distribution, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        classes = list(distribution.keys())
        counts = list(distribution.values())
        ax.bar(classes, counts, color='steelblue', edgecolor='white')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.set_xticks(classes)
        for i, count in enumerate(counts):
            ax.text(i, count + max(counts) * 0.01, str(count), ha='center')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def run_data_preprocessing():
    base_dir = r'c:\Users\17456\Documents\GitHub\Deep_learningPractice\GenerativeAI'
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    preprocessor = DataPreprocessor(data_dir=data_dir, image_size=32)

    print('=' * 60)
    print('数据预处理报告')
    print('=' * 60)

    datasets_info = []

    print('\n[1] 加载MNIST数据集...')
    mnist_train = preprocessor.load_mnist(train=True, download=True)
    mnist_test = preprocessor.load_mnist(train=False, download=True)
    mnist_stats = preprocessor.compute_statistics(mnist_train)
    print(f'  训练集: {mnist_stats["num_samples"]} 样本')
    print(f'  图像形状: {mnist_stats["shape"]}')
    print(f'  均值: {mnist_stats["mean"]}')
    print(f'  标准差: {mnist_stats["std"]}')
    mnist_dist = preprocessor.compute_class_distribution(mnist_train)
    print(f'  类别分布: {mnist_dist}')

    datasets_info.append({
        'name': 'MNIST',
        'samples': 60000,
        'shape': (1, 28, 28),
        'classes': 10,
        'type': 'Grayscale'
    })

    print('\n[2] 加载CIFAR-10数据集...')
    try:
        cifar_train = preprocessor.load_cifar10(train=True, download=False)
        cifar_stats = preprocessor.compute_statistics(cifar_train)
        print(f'  训练集: {cifar_stats["num_samples"]} 样本')
        print(f'  图像形状: {cifar_stats["shape"]}')
        print(f'  均值: {cifar_stats["mean"]}')
        print(f'  标准差: {cifar_stats["std"]}')
        cifar_dist = preprocessor.compute_class_distribution(cifar_train)
        print(f'  类别分布: {cifar_dist}')

        datasets_info.append({
            'name': 'CIFAR-10',
            'samples': 50000,
            'shape': (3, 32, 32),
            'classes': 10,
            'type': 'RGB'
        })

        preprocessor.show_samples(
            cifar_train, num_samples=25,
            save_path=os.path.join(results_dir, 'plots', 'cifar_samples.png')
        )
        preprocessor.plot_class_distribution(
            cifar_dist,
            save_path=os.path.join(results_dir, 'plots', 'cifar_distribution.png')
        )
    except Exception as e:
        print(f'  CIFAR-10不可用: {e}')
        print('  使用MNIST-3ch作为替代')
        mnist3ch_train = preprocessor.load_mnist_3ch(train=True)
        mnist3ch_stats = preprocessor.compute_statistics(mnist3ch_train)
        print(f'  MNIST-3ch样本数: {mnist3ch_stats["num_samples"]}')

        datasets_info.append({
            'name': 'MNIST-3ch (替代CIFAR-10)',
            'samples': 60000,
            'shape': (3, 32, 32),
            'classes': 10,
            'type': 'RGB (from grayscale)'
        })

    print('\n[3] 保存样本可视化...')
    preprocessor.show_samples(
        mnist_train, num_samples=25,
        save_path=os.path.join(results_dir, 'plots', 'mnist_samples.png')
    )
    preprocessor.plot_class_distribution(
        mnist_dist,
        save_path=os.path.join(results_dir, 'plots', 'mnist_distribution.png')
    )

    print('\n[4] 数据集信息汇总:')
    print('-' * 80)
    print(f'{"数据集":<25} {"样本数":>10} {"形状":>15} {"类别":>6} {"类型":>15}')
    print('-' * 80)
    for info in datasets_info:
        shape_str = f'{info["shape"][0]}x{info["shape"][1]}x{info["shape"][2]}'
        print(f'{info["name"]:<25} {info["samples"]:>10} {shape_str:>15} {info["classes"]:>6} {info["type"]:>15}')
    print('-' * 80)

    print('\n[5] 数据预处理完成!')
    print('    输出文件:')
    print('      - plots/mnist_samples.png')
    print('      - plots/mnist_distribution.png')
    if os.path.exists(os.path.join(results_dir, 'plots', 'cifar_samples.png')):
        print('      - plots/cifar_samples.png')
        print('      - plots/cifar_distribution.png')

    return datasets_info


if __name__ == '__main__':
    run_data_preprocessing()
