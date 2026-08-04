import torch
from diffusers import DDPMPipeline, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

def train():
    base_dir = r'c:\Users\17456\Documents\GitHub\Deep_learningPractice\GenerativeAI'
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    log_file = os.path.join(results_dir, 'diffusion_train.log')

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

    # 数据集加载
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    log_print('Loading CIFAR-10 dataset...')
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    log_print(f'Dataset loaded: {len(train_dataset)} samples')

    # 模型定义
    model = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log_print(f'Model params: {total_params:,}')

    # 调度器
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    num_epochs = 5
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * num_epochs),
    )

    os.makedirs(os.path.join(results_dir, 'images', 'diffusion'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    losses = []

    log_print('Starting diffusion model training...')
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')

        for step, (clean_images, _) in enumerate(train_loader):
            clean_images = clean_images.to(device)
            batch_size = clean_images.shape[0]

            noise = torch.randn(clean_images.shape, device=device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (batch_size,), device=device).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            losses.append(loss.item())
            global_step += 1

            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr_scheduler.get_last_lr()[0]:.6f}'})

            if (step + 1) % 100 == 0:
                log_print(f'  Epoch {epoch+1} Step {step+1}/{len(train_loader)}: loss={loss.item():.4f}')

        avg_loss = epoch_loss / len(train_loader)
        progress_bar.close()
        log_print(f'Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}')

        # 每epoch生成样本
        log_print(f'  Generating samples after epoch {epoch+1}...')
        model.eval()
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        pipeline.set_progress_bar_config(disable=True)
        with torch.no_grad():
            generated = pipeline(batch_size=16, num_inference_steps=50,
                                 generator=torch.manual_seed(42)).images

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(generated[i])
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'images', 'diffusion', f'epoch_{epoch+1}.png'), dpi=120)
        plt.close()
        log_print(f'  Samples saved')

    log_print('Training complete. Saving final results...')

    # 保存最终样本（更多）
    model.eval()
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.set_progress_bar_config(disable=True)
    with torch.no_grad():
        final_images = pipeline(batch_size=64, num_inference_steps=100,
                                generator=torch.manual_seed(42)).images

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(final_images[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'images', 'diffusion', 'final.png'), dpi=150)
    plt.close()
    log_print('Final 64 samples saved')

    # 保存loss曲线
    np.save(os.path.join(results_dir, 'diffusion_losses.npy'), np.array(losses))
    plt.figure(figsize=(10, 5))
    plt.plot(losses, alpha=0.3, label='Step Loss')
    window = min(100, len(losses)//10)
    if window > 0:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=f'Smoothed (w={window})', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Diffusion Model Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'plots', 'diffusion_loss.png'), dpi=150)
    plt.close()
    log_print('Loss plot saved')
    
    # 保存模型checkpoint
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(os.path.join(checkpoint_dir, 'diffusion_unet'))
    noise_scheduler.save_pretrained(os.path.join(checkpoint_dir, 'diffusion_scheduler'))
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }, os.path.join(checkpoint_dir, 'diffusion_optimizer.pth'))
    log_print('Diffusion checkpoints saved')
    
    log_print('Diffusion experiment complete!')
    log.close()

if __name__ == '__main__':
    train()
