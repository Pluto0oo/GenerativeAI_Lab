import torch
import os

base_dir = r'c:\Users\17456\Documents\GitHub\Deep_learningPractice\GenerativeAI'
log_path = os.path.join(base_dir, 'vae_test.log')
results_dir = os.path.join(base_dir, 'results')

with open(log_path, 'w') as f:
    f.write(f'Base dir: {base_dir}\n')
    f.write(f'CUDA available: {torch.cuda.is_available()}\n')
    if torch.cuda.is_available():
        f.write(f'GPU: {torch.cuda.get_device_name(0)}\n')
    
    os.makedirs(os.path.join(results_dir, 'images', 'vae'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
    
    f.write('Directories created\n')
    f.write('Test complete!\n')

print('Test complete! Check vae_test.log')
