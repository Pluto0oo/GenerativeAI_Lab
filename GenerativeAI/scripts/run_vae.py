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
import sys

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU())
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 256), nn.ReLU(),
                                     nn.Linear(256, 784), nn.Sigmoid())

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x.view(-1, 784))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train():
    base_dir = r'c:\Users\17456\Documents\GitHub\Deep_learningPractice\GenerativeAI'
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    log_file = os.path.join(results_dir, 'vae_train.log')
    
    log = open(log_file, 'w', encoding='utf-8')
    def log_print(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f'Using device: {device}')
    
    model = VAE(latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    transform = transforms.Compose([transforms.ToTensor()])
    
    log_print('Loading MNIST dataset...')
    train_loader = DataLoader(datasets.MNIST(data_dir, train=True, download=True, transform=transform),
                              batch_size=128, shuffle=True)
    log_print(f'Dataset loaded: {len(train_loader.dataset)} samples')
    
    os.makedirs(os.path.join(results_dir, 'images', 'vae'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    losses = []
    
    log_print('Starting training...')
    num_epochs = 5
    for epoch in range(num_epochs):
        try:
            model.train()
            total_loss = 0
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar = model(x)
                loss = loss_fn(recon_x, x, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader.dataset)
            losses.append(avg_loss)
            log_print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        except Exception as e:
            log_print(f'Error in epoch {epoch+1}: {e}')
            break
    
    log_print('Training complete. Generating samples...')
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, 32).to(device)
        samples = model.decoder(z).view(-1, 1, 28, 28).cpu()
    
    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i, axi in enumerate(ax.flat):
        axi.imshow(samples[i][0], cmap='gray')
        axi.axis('off')
    plt.savefig(os.path.join(results_dir, 'images', 'vae', 'generated.png'), dpi=150)
    log_print('Generated samples saved')
    
    np.save(os.path.join(results_dir, 'vae_losses.npy'), np.array(losses))
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss')
    plt.savefig(os.path.join(results_dir, 'plots', 'vae_loss.png'), dpi=150)
    log_print('Loss plot saved')
    
    # 保存模型checkpoint
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'latent_dim': 32,
        'losses': losses
    }, os.path.join(checkpoint_dir, 'vae_model.pth'))
    log_print('Model checkpoint saved')
    
    log_print('VAE experiment complete!')
    log.close()

if __name__ == '__main__':
    train()