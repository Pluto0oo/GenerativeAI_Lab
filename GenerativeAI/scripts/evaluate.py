import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import linalg
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = r'c:\Users\17456\Documents\GitHub\Deep_learningPractice\GenerativeAI'
results_dir = os.path.join(base_dir, 'results')
data_dir = os.path.join(base_dir, 'data')
log_file = os.path.join(results_dir, 'evaluation.log')

log = open(log_file, 'w', encoding='utf-8')
def log_print(msg):
    print(msg, flush=True)
    log.write(msg + '\n')
    log.flush()

def get_inception_model(device):
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, transform_input=False).to(device)
    model.eval()
    return model

def compute_activations(images_tensor, model, device, batch_size=16):
    dataset = TensorDataset(images_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    activations = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            if x.shape[2] != 299 or x.shape[3] != 299:
                x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            feat = model(x)
            if isinstance(feat, tuple):
                feat = feat[0]
            feat = feat[:, :2048]
            activations.append(feat.cpu().numpy())
    return np.concatenate(activations, axis=0)

def compute_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_is(probs, num_splits=10):
    scores = []
    split_size = len(probs) // num_splits
    for i in range(num_splits):
        part = probs[i*split_size:(i+1)*split_size]
        marginal = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-16) - np.log(marginal + 1e-16))
        kl = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl)))
    return float(np.mean(scores)), float(np.std(scores))

def compute_precision_recall(real_feat, fake_feat, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn_real = NearestNeighbors(n_neighbors=k).fit(real_feat)
    nn_fake = NearestNeighbors(n_neighbors=k).fit(fake_feat)
    
    real_dists, _ = nn_real.kneighbors(real_feat)
    real_radii = real_dists[:, -1]
    
    fake_dists, _ = nn_fake.kneighbors(fake_feat)
    fake_radii = fake_dists[:, -1]
    
    fake_to_real, _ = nn_real.kneighbors(fake_feat)
    precision = np.mean(np.min(fake_to_real, axis=1) <= real_radii[np.argmin(fake_to_real, axis=1)])
    
    real_to_fake, _ = nn_fake.kneighbors(real_feat)
    recall = np.mean(np.min(real_to_fake, axis=1) <= fake_radii[np.argmin(real_to_fake, axis=1)])
    
    return float(precision), float(recall)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f'Using device: {device}')
    if torch.cuda.is_available():
        log_print(f'GPU: {torch.cuda.get_device_name(0)}')
        torch.backends.cudnn.benchmark = True

    log_print('\n=== Loading Inception v3 ===')
    inception = get_inception_model(device)
    log_print('Inception v3 loaded')

    log_print('\n=== Loading real data (CIFAR-10) ===')
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    real_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=100, shuffle=True, num_workers=0)
    
    real_samples = []
    for x, _ in real_loader:
        real_samples.append(x)
        if len(real_samples) * 100 >= 1000:
            break
    real_images = torch.cat(real_samples, dim=0)[:1000]
    log_print(f'Real samples: {real_images.shape}')

    log_print('\n=== Computing real data features ===')
    real_feat = compute_activations(real_images, inception, device, batch_size=16)
    mu_real = np.mean(real_feat, axis=0)
    sigma_real = np.cov(real_feat, rowvar=False)
    log_print(f'Real: mu={mu_real.shape}, sigma={sigma_real.shape}')

    results = {}
    NUM_SAMPLES = 500
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')

    # VAE
    log_print('\n=== Evaluating VAE ===')
    try:
        sys.path.insert(0, os.path.join(base_dir, 'scripts'))
        from run_vae import VAE
        
        vae = VAE(latent_dim=32).to(device)
        vae_checkpoint = os.path.join(checkpoint_dir, 'vae_model.pth')
        if os.path.exists(vae_checkpoint):
            checkpoint = torch.load(vae_checkpoint, map_location=device)
            vae.load_state_dict(checkpoint['model_state_dict'])
            log_print(f'VAE checkpoint loaded from {vae_checkpoint}')
        else:
            log_print(f'WARNING: VAE checkpoint not found at {vae_checkpoint}, using random weights')
        vae.eval()
        
        vae_samples = []
        with torch.no_grad():
            for _ in range(0, NUM_SAMPLES, 64):
                bs = min(64, NUM_SAMPLES - len(vae_samples))
                z = torch.randn(bs, 32).to(device)
                s = vae.decoder(z).view(-1, 1, 28, 28)
                vae_samples.append(s.cpu())
        vae_samples = torch.cat(vae_samples, dim=0)
        
        if vae_samples.shape[1] == 1:
            vae_samples = vae_samples.repeat(1, 3, 1, 1)
        vae_samples = nn.functional.interpolate(vae_samples, size=(32, 32), mode='bilinear', align_corners=False)
        log_print(f'VAE samples: {vae_samples.shape}')
        
        vae_feat = compute_activations(vae_samples, inception, device, batch_size=16)
        mu_vae = np.mean(vae_feat, axis=0)
        sigma_vae = np.cov(vae_feat, rowvar=False)
        fid_vae = compute_fid(mu_real, sigma_real, mu_vae, sigma_vae)
        
        vae_logits = []
        with torch.no_grad():
            for i in range(0, NUM_SAMPLES, 16):
                x = vae_samples[i:i+16].to(device)
                x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                out = inception(x)
                if isinstance(out, tuple):
                    out = out[0]
                vae_logits.append(out.cpu().numpy())
        vae_probs = np.concatenate(vae_logits, axis=0)
        vae_probs = nn.functional.softmax(torch.tensor(vae_probs), dim=1).numpy()
        is_mean_vae, is_std_vae = compute_is(vae_probs)
        
        prec_vae, rec_vae = compute_precision_recall(real_feat[:300], vae_feat[:300])
        
        results['VAE'] = {'FID': fid_vae, 'IS_mean': is_mean_vae, 'IS_std': is_std_vae,
                          'Precision': prec_vae, 'Recall': rec_vae}
        log_print(f'VAE - FID: {fid_vae:.2f}, IS: {is_mean_vae:.2f}±{is_std_vae:.2f}, '
                  f'P: {prec_vae:.4f}, R: {rec_vae:.4f}')
        
        del vae, vae_samples, vae_feat, mu_vae, sigma_vae, vae_logits, vae_probs
        torch.cuda.empty_cache()
    except Exception as e:
        log_print(f'VAE evaluation failed: {e}')
        import traceback
        log_print(traceback.format_exc())

    # GAN
    log_print('\n=== Evaluating GAN ===')
    try:
        from run_gan import Generator
        
        gan_G = Generator(latent_dim=100).to(device)
        gan_checkpoint = os.path.join(checkpoint_dir, 'gan_model.pth')
        if os.path.exists(gan_checkpoint):
            checkpoint = torch.load(gan_checkpoint, map_location=device)
            gan_G.load_state_dict(checkpoint['generator_state_dict'])
            log_print(f'GAN checkpoint loaded from {gan_checkpoint}')
        else:
            log_print(f'WARNING: GAN checkpoint not found at {gan_checkpoint}, using random weights')
        gan_G.eval()
        
        gan_samples = []
        with torch.no_grad():
            for _ in range(0, NUM_SAMPLES, 64):
                bs = min(64, NUM_SAMPLES - len(gan_samples))
                z = torch.randn(bs, 100, 1, 1).to(device)
                s = gan_G(z)
                gan_samples.append(s.cpu())
        gan_samples = torch.cat(gan_samples, dim=0)
        log_print(f'GAN samples: {gan_samples.shape}')
        
        gan_feat = compute_activations(gan_samples, inception, device, batch_size=16)
        mu_gan = np.mean(gan_feat, axis=0)
        sigma_gan = np.cov(gan_feat, rowvar=False)
        fid_gan = compute_fid(mu_real, sigma_real, mu_gan, sigma_gan)
        
        gan_logits = []
        with torch.no_grad():
            for i in range(0, NUM_SAMPLES, 16):
                x = gan_samples[i:i+16].to(device)
                x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                out = inception(x)
                if isinstance(out, tuple):
                    out = out[0]
                gan_logits.append(out.cpu().numpy())
        gan_probs = np.concatenate(gan_logits, axis=0)
        gan_probs = nn.functional.softmax(torch.tensor(gan_probs), dim=1).numpy()
        is_mean_gan, is_std_gan = compute_is(gan_probs)
        
        prec_gan, rec_gan = compute_precision_recall(real_feat[:300], gan_feat[:300])
        
        results['GAN'] = {'FID': fid_gan, 'IS_mean': is_mean_gan, 'IS_std': is_std_gan,
                          'Precision': prec_gan, 'Recall': rec_gan}
        log_print(f'GAN - FID: {fid_gan:.2f}, IS: {is_mean_gan:.2f}±{is_std_gan:.2f}, '
                  f'P: {prec_gan:.4f}, R: {rec_gan:.4f}')
        
        del gan_G, gan_samples, gan_feat, mu_gan, sigma_gan, gan_logits, gan_probs
        torch.cuda.empty_cache()
    except Exception as e:
        log_print(f'GAN evaluation failed: {e}')
        import traceback
        log_print(traceback.format_exc())

    # Diffusion
    log_print('\n=== Evaluating Diffusion ===')
    try:
        from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
        
        diff_unet_path = os.path.join(checkpoint_dir, 'diffusion_unet')
        diff_scheduler_path = os.path.join(checkpoint_dir, 'diffusion_scheduler')
        
        if os.path.exists(diff_unet_path):
            diff_model = UNet2DModel.from_pretrained(diff_unet_path).to(device)
            log_print(f'Diffusion UNet loaded from {diff_unet_path}')
        else:
            log_print(f'WARNING: Diffusion UNet not found at {diff_unet_path}, using random weights')
            diff_model = UNet2DModel(
                sample_size=32, in_channels=3, out_channels=3, layers_per_block=2,
                block_out_channels=(64, 128, 256),
                down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            ).to(device)
        
        if os.path.exists(diff_scheduler_path):
            diff_scheduler = DDPMScheduler.from_pretrained(diff_scheduler_path)
            log_print(f'Diffusion scheduler loaded from {diff_scheduler_path}')
        else:
            diff_scheduler = DDPMScheduler(num_train_timesteps=1000)
        
        diff_model.eval()
        pipeline = DDPMPipeline(unet=diff_model, scheduler=diff_scheduler)
        pipeline.set_progress_bar_config(disable=True)
        
        diff_samples = []
        with torch.no_grad():
            for _ in range(0, NUM_SAMPLES, 32):
                bs = min(32, NUM_SAMPLES - len(diff_samples))
                imgs = pipeline(batch_size=bs, num_inference_steps=50,
                                generator=torch.manual_seed(42)).images
                for img in imgs:
                    t = transforms.ToTensor()(img).unsqueeze(0)
                    diff_samples.append(t)
        diff_samples = torch.cat(diff_samples, dim=0)
        diff_samples = (diff_samples - 0.5) / 0.5
        log_print(f'Diffusion samples: {diff_samples.shape}')
        
        diff_feat = compute_activations(diff_samples, inception, device, batch_size=16)
        mu_diff = np.mean(diff_feat, axis=0)
        sigma_diff = np.cov(diff_feat, rowvar=False)
        fid_diff = compute_fid(mu_real, sigma_real, mu_diff, sigma_diff)
        
        diff_logits = []
        with torch.no_grad():
            for i in range(0, NUM_SAMPLES, 16):
                x = diff_samples[i:i+16].to(device)
                x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
                out = inception(x)
                if isinstance(out, tuple):
                    out = out[0]
                diff_logits.append(out.cpu().numpy())
        diff_probs = np.concatenate(diff_logits, axis=0)
        diff_probs = nn.functional.softmax(torch.tensor(diff_probs), dim=1).numpy()
        is_mean_diff, is_std_diff = compute_is(diff_probs)
        
        prec_diff, rec_diff = compute_precision_recall(real_feat[:300], diff_feat[:300])
        
        results['Diffusion'] = {'FID': fid_diff, 'IS_mean': is_mean_diff, 'IS_std': is_std_diff,
                                'Precision': prec_diff, 'Recall': rec_diff}
        log_print(f'Diffusion - FID: {fid_diff:.2f}, IS: {is_mean_diff:.2f}±{is_std_diff:.2f}, '
                  f'P: {prec_diff:.4f}, R: {rec_diff:.4f}')
        
        del diff_model, diff_scheduler, diff_samples, diff_feat, mu_diff, sigma_diff, diff_logits, diff_probs
        torch.cuda.empty_cache()
    except Exception as e:
        log_print(f'Diffusion evaluation failed: {e}')
        import traceback
        log_print(traceback.format_exc())

    del inception
    torch.cuda.empty_cache()

    # Summary
    log_print('\n=== Summary ===')
    log_print(f'{"Model":<12} {"FID":<10} {"IS":<15} {"Precision":<12} {"Recall":<12}')
    log_print('-' * 65)
    for name, r in results.items():
        log_print(f'{name:<12} {r["FID"]:<10.2f} {r["IS_mean"]:.2f}±{r["IS_std"]:.2f}     {r["Precision"]:<12.4f} {r["Recall"]:<12.4f}')

    # Plot comparison
    if results:
        names = list(results.keys())
        fids = [results[n]['FID'] for n in names]
        iss = [results[n]['IS_mean'] for n in names]
        precs = [results[n]['Precision'] for n in names]
        recs = [results[n]['Recall'] for n in names]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        bars = axes[0,0].bar(names, fids, color=['#4C72B0', '#DD8452', '#55A467'])
        axes[0,0].set_ylabel('FID ↓'); axes[0,0].set_title('FID Comparison')
        axes[0,0].grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, fids):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, v, f'{v:.1f}', ha='center', va='bottom')
        
        bars = axes[0,1].bar(names, iss, color=['#4C72B0', '#DD8452', '#55A467'])
        axes[0,1].set_ylabel('IS ↑'); axes[0,1].set_title('Inception Score Comparison')
        axes[0,1].grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, iss):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, v, f'{v:.2f}', ha='center', va='bottom')
        
        x = np.arange(len(names))
        w = 0.35
        axes[1,0].bar(x - w/2, precs, w, label='Precision', color='#4C72B0')
        axes[1,0].bar(x + w/2, recs, w, label='Recall', color='#DD8452')
        axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(names)
        axes[1,0].set_ylabel('Score'); axes[1,0].set_title('Precision & Recall')
        axes[1,0].legend(); axes[1,0].grid(axis='y', alpha=0.3)
        
        axes[1,1].axis('off')
        table_data = [['Model', 'FID', 'IS', 'Precision', 'Recall']]
        for name, r in results.items():
            table_data.append([name, f'{r["FID"]:.2f}', f'{r["IS_mean"]:.2f}',
                               f'{r["Precision"]:.4f}', f'{r["Recall"]:.4f}'])
        table = axes[1,1].table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(10)
        table.scale(1, 1.8)
        axes[1,1].set_title('Evaluation Summary', y=0.95)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plots', 'comparison.png'), dpi=150)
        plt.close()
        log_print(f'\nComparison plot saved to results/plots/comparison.png')

    log_print('\nEvaluation complete!')
    log.close()

if __name__ == '__main__':
    main()
