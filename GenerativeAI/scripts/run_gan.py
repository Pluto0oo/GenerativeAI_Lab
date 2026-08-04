import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

def compute_diversity(images):
    """计算生成图像多样性指标，用于检测模式崩溃"""
    flat = images.view(images.size(0), -1)
    flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    sim_matrix = flat @ flat.t()
    n = sim_matrix.size(0)
    sim_matrix.fill_diagonal_(0)
    avg_sim = sim_matrix.sum() / (n * (n - 1))
    return 1.0 - avg_sim.item()

def train():
    base_dir = r'c:\Users\17456\Documents\GitHub\Deep_learningPractice\GenerativeAI'
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    log_file = os.path.join(results_dir, 'gan_train.log')

    log = open(log_file, 'w', encoding='utf-8')
    def log_print(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f'Using device: {device}')
    if torch.cuda.is_available():
        log_print(f'GPU: {torch.cuda.get_device_name(0)}')
        torch.backends.cudnn.benchmark = True

    G = Generator(latent_dim=100).to(device)
    D = Discriminator().to(device)
    log_print(f'Generator params: {sum(p.numel() for p in G.parameters()):,}')
    log_print(f'Discriminator params: {sum(p.numel() for p in D.parameters()):,}')

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 尝试加载CIFAR-10，失败则使用MNIST转3通道RGB作为替代方案
    train_loader = None
    dataset_name = ''
    try:
        log_print('Loading CIFAR-10 dataset...')
        cifar_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform)
        _ = cifar_dataset[0]
        train_loader = DataLoader(cifar_dataset, batch_size=128, shuffle=True,
                                  num_workers=0, pin_memory=True)
        dataset_name = 'CIFAR-10'
        log_print(f'CIFAR-10 loaded: {len(train_loader.dataset)} samples')
    except Exception as e:
        log_print(f'CIFAR-10 load failed: {e}')
        log_print('Falling back to MNIST (3-channel RGB) as substitute...')

        mnist_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        mnist_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=mnist_transform)
        train_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=True,
                                   num_workers=0, pin_memory=True)
        dataset_name = 'MNIST-3ch'
        log_print(f'MNIST-3ch loaded: {len(train_loader.dataset)} samples')

    log_print(f'Using dataset: {dataset_name}')

    os.makedirs(os.path.join(results_dir, 'images', 'gan'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    d_losses, g_losses, diversities = [], [], []

    num_epochs = 20
    fixed_z = torch.randn(64, 100, 1, 1, device=device)

    log_print('Starting DCGAN training...')
    for epoch in range(num_epochs):
        try:
            G.train()
            D.train()
            d_loss_sum, g_loss_sum = 0.0, 0.0

            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                bs = x.size(0)

                # Train Discriminator
                opt_D.zero_grad()
                real_label = torch.ones(bs, device=device)
                fake_label = torch.zeros(bs, device=device)

                output = D(x)
                loss_D_real = criterion(output, real_label)
                loss_D_real.backward()

                z = torch.randn(bs, 100, 1, 1, device=device)
                fake = G(z)
                output = D(fake.detach())
                loss_D_fake = criterion(output, fake_label)
                loss_D_fake.backward()
                opt_D.step()
                d_loss_sum += (loss_D_real + loss_D_fake).item()

                # Train Generator
                opt_G.zero_grad()
                output = D(fake)
                loss_G = criterion(output, real_label)
                loss_G.backward()
                opt_G.step()
                g_loss_sum += loss_G.item()

                if (batch_idx + 1) % 100 == 0:
                    log_print(f'  Epoch {epoch+1} Batch {batch_idx+1}/{len(train_loader)}: '
                              f'D={loss_D_real.item()+loss_D_fake.item():.4f} G={loss_G.item():.4f}')

            avg_d = d_loss_sum / len(train_loader)
            avg_g = g_loss_sum / len(train_loader)
            d_losses.append(avg_d)
            g_losses.append(avg_g)

            # 模式崩溃检测：用固定噪声生成样本并计算多样性
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z).cpu()
            diversity = compute_diversity(samples)
            diversities.append(diversity)

            log_print(f'Epoch {epoch+1}/{num_epochs} | D Loss: {avg_d:.4f} | '
                      f'G Loss: {avg_g:.4f} | Diversity: {diversity:.4f}')

            # 每5个epoch保存一次样本
            if (epoch + 1) % 5 == 0 or epoch == 0:
                G.eval()
                with torch.no_grad():
                    samples = G(fixed_z).cpu()
                fig, ax = plt.subplots(8, 8, figsize=(8, 8))
                for i, axi in enumerate(ax.flat):
                    axi.imshow((samples[i].permute(1, 2, 0) + 1) / 2)
                    axi.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'images', 'gan', f'epoch_{epoch+1}.png'), dpi=120)
                plt.close()

        except Exception as e:
            log_print(f'Error in epoch {epoch+1}: {e}')
            import traceback
            log_print(traceback.format_exc())
            break

    log_print('Training complete. Saving final samples and metrics...')

    # 保存最终生成样本
    G.eval()
    with torch.no_grad():
        final_samples = G(fixed_z).cpu()
    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i, axi in enumerate(ax.flat):
        axi.imshow((final_samples[i].permute(1, 2, 0) + 1) / 2)
        axi.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'images', 'gan', 'final.png'), dpi=150)
    plt.close()
    log_print('Final samples saved')

    # 保存损失曲线
    np.save(os.path.join(results_dir, 'gan_d_losses.npy'), np.array(d_losses))
    np.save(os.path.join(results_dir, 'gan_g_losses.npy'), np.array(g_losses))
    np.save(os.path.join(results_dir, 'gan_diversities.npy'), np.array(diversities))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(d_losses, label='Discriminator Loss')
    axes[0].plot(g_losses, label='Generator Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('DCGAN Training Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(diversities, label='Sample Diversity', color='green')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Diversity Score')
    axes[1].set_title('Mode Collapse Indicator (Diversity)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plots', 'gan_loss.png'), dpi=150)
    plt.close()
    log_print('Loss & diversity plots saved')
    
    # 保存模型checkpoint
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'optimizer_G_state_dict': opt_G.state_dict(),
        'optimizer_D_state_dict': opt_D.state_dict(),
        'latent_dim': 100,
        'd_losses': d_losses,
        'g_losses': g_losses,
        'diversities': diversities
    }, os.path.join(checkpoint_dir, 'gan_model.pth'))
    log_print('GAN checkpoints saved')
    
    # 模式崩溃分析报告
    final_div = diversities[-1] if diversities else 0
    max_div = max(diversities) if diversities else 0
    log_print(f'\n=== Mode Collapse Analysis ===')
    log_print(f'Max diversity: {max_div:.4f}')
    log_print(f'Final diversity: {final_div:.4f}')
    if final_div < 0.1:
        log_print('WARNING: Severe mode collapse detected (diversity < 0.1)')
    elif final_div < max_div * 0.5:
        log_print('WARNING: Moderate mode collapse detected')
    else:
        log_print('Diversity stable, no obvious mode collapse')

    log_print('GAN experiment complete!')
    log.close()

if __name__ == '__main__':
    train()
