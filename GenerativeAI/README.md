# 生成式人工智能实验项目

## 项目概述

本项目实现并对比了三种主流生成模型（VAE、GAN、Diffusion），使用PyTorch框架进行训练和评估，支持GPU加速训练。

## 目录结构

```
GenerativeAI/
├── configs/
│   └── base.yaml              # 基础配置文件
├── docs/
│   └── experiment_principle.md # 实验原理与结果文档
├── scripts/
│   ├── run_vae.py             # VAE训练脚本
│   ├── run_gan.py             # DCGAN训练脚本
│   ├── run_diffusion.py       # Diffusion模型训练脚本
│   ├── evaluate.py            # 统一评估脚本（FID/IS/Precision/Recall）
│   ├── compare_results.py     # 结果对比可视化
│   ├── test_gpu.py            # GPU环境检测
│   └── test_vae.py            # VAE功能测试
├── src/
│   ├── __init__.py
│   └── config.py              # 配置管理模块
├── results/
│   ├── images/                # 生成样本图像
│   │   ├── vae/
│   │   ├── gan/
│   │   └── diffusion/
│   ├── plots/                 # 训练曲线与对比图
│   ├── checkpoints/           # 模型检查点
│   └── evaluation.log         # 评估日志
├── data/                      # 数据集存储目录
├── requirements.txt           # 依赖列表
├── .gitignore
└── README.md
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 主要依赖

| 库 | 版本要求 | 说明 |
|----|----------|------|
| Python | 3.9+ | 推荐使用conda环境 |
| PyTorch | 2.5.1+ | 深度学习框架（CUDA版本） |
| torchvision | 0.20.1+ | 视觉数据处理 |
| diffusers | 0.30.0+ | 扩散模型实现 |
| numpy | 2.0+ | 数值计算 |
| matplotlib | 3.9+ | 可视化绘图 |
| scikit-learn | 1.7+ | Precision/Recall计算 |
| scipy | 1.15+ | FID计算 |

### 3. GPU环境检测

```bash
python scripts/test_gpu.py
```

## 快速开始

### 训练VAE模型

```bash
python scripts/run_vae.py
```

**训练参数**：
- 数据集：MNIST（手写数字）
- 潜在维度：32
- 训练轮数：5 epochs
- 优化器：Adam (lr=1e-3)
- 批大小：128

**输出**：
- 生成样本：`results/images/vae/generated.png`
- Loss曲线：`results/plots/vae_loss.png`
- 模型检查点：`results/checkpoints/vae_model.pth`

### 训练DCGAN模型

```bash
python scripts/run_gan.py
```

**训练参数**：
- 数据集：CIFAR-10（若不可用则自动使用MNIST-3ch）
- 潜在维度：100
- 训练轮数：20 epochs
- 优化器：Adam (lr=2e-4, betas=0.5, 0.999)
- 批大小：128
- 生成器参数：3,448,576
- 判别器参数：2,637,312

**输出**：
- 生成样本：`results/images/gan/final.png`（每5轮保存）
- Loss曲线：`results/plots/gan_loss.png`
- 多样性曲线：同上（模式崩溃检测）
- 模型检查点：`results/checkpoints/gan_model.pth`

### 训练Diffusion模型

```bash
python scripts/run_diffusion.py
```

**训练参数**：
- 数据集：CIFAR-10
- 模型：UNet2DModel (15,724,931 参数)
- 训练轮数：5 epochs
- 优化器：AdamW (lr=1e-4, weight_decay=1e-6)
- 调度器：DDPMScheduler (1000 timesteps)
- 学习率调度：余弦退火 + 预热
- 批大小：128

**输出**：
- 生成样本：`results/images/diffusion/final.png`（每轮保存）
- Loss曲线：`results/plots/diffusion_loss.png`
- 模型检查点：
  - UNet权重：`results/checkpoints/diffusion_unet/`
  - 调度器：`results/checkpoints/diffusion_scheduler/`
  - 优化器：`results/checkpoints/diffusion_optimizer.pth`

### 运行统一评估

```bash
python scripts/evaluate.py
```

**评估指标**：
- **FID** (Fréchet Inception Distance)：衡量生成图像与真实图像分布的距离，越低越好
- **IS** (Inception Score)：衡量生成图像的质量和多样性，越高越好
- **Precision**：生成样本中"有效"的比例
- **Recall**：真实样本中可被生成的比例

**输出**：
- 评估日志：`results/evaluation.log`
- 对比图表：`results/plots/comparison.png`

### 结果可视化

```bash
python scripts/compare_results.py
```

## 实验结果

### 训练过程记录

#### VAE训练记录
| Epoch | Loss |
|-------|------|
| 1 | 192.90 |
| 2 | 143.53 |
| 3 | 126.48 |
| 4 | 119.20 |
| 5 | 115.10 |

#### DCGAN训练记录
| Epoch | D Loss | G Loss | Diversity |
|-------|--------|--------|-----------|
| 1 | 0.55 | 4.82 | 0.6975 |
| 5 | 0.67 | 3.07 | 0.9319 |
| 10 | 0.49 | 3.43 | 0.9285 |
| 15 | 0.53 | 3.43 | 0.8987 |
| 20 | 0.39 | 3.67 | 0.8830 |

#### Diffusion训练记录
| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.3119 |
| 2 | 0.0541 |
| 3 | 0.0441 |
| 4 | 0.0409 |
| 5 | 0.0396 |

### 评估结果对比

| 模型 | FID ↓ | IS ↑ | Precision | Recall |
|------|-------|------|-----------|--------|
| VAE | 1472.65 | 2.23±0.10 | 0.0767 | 0.0000 |
| GAN | **351.53** | **3.61±0.41** | 0.4000 | **0.7033** |
| Diffusion | 978.10 | 2.03±0.06 | **0.8100** | 0.0000 |

### 关键发现

1. **GAN在综合指标上表现最优**：FID最低（351.53），IS最高（3.61），说明在当前训练配置下DCGAN生成的图像质量最好、多样性最佳。

2. **Diffusion在Precision上表现突出**：Precision达到0.81，说明生成的样本中有很高比例是"有效"的，即接近真实数据分布。

3. **VAE的FID较高**：可能由于训练轮数较少（仅5 epochs），或MNIST与CIFAR-10的跨数据集评估存在分布差异。

4. **模式崩溃分析**：GAN的多样性从0.6975（初始）提升至0.9319（第5轮），最终保持在0.8830，表明训练稳定，无明显模式崩溃。

## 技术架构

### VAE架构
```
编码器：Linear(784→256) → ReLU → Linear(256→128) → ReLU
潜在空间：
  - fc_mu: Linear(128→32)
  - fc_logvar: Linear(128→32)
重参数化：z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)
解码器：Linear(32→128) → ReLU → Linear(128→256) → ReLU → Linear(256→784) → Sigmoid
损失函数：BCE(重建损失) + KL散度
```

### DCGAN架构
```
生成器 (Generator):
  ConvTranspose2d(100→512, 4x4) → BN → ReLU
  ConvTranspose2d(512→256, 4x4) → BN → ReLU
  ConvTranspose2d(256→128, 4x4) → BN → ReLU
  ConvTranspose2d(128→3, 4x4) → Tanh

判别器 (Discriminator):
  Conv2d(3→128, 4x4) → LeakyReLU(0.2)
  Conv2d(128→256, 4x4) → BN → LeakyReLU(0.2)
  Conv2d(256→512, 4x4) → BN → LeakyReLU(0.2)
  Conv2d(512→1, 4x4) → Sigmoid
```

### Diffusion架构
```
UNet2DModel (15,724,931 参数):
  输入尺寸：32x32
  输入通道：3 (RGB)
  输出通道：3
  每层block数：2
  通道数：(64, 128, 256)
  下采样块：DownBlock2D × 2 + AttnDownBlock2D × 1
  上采样块：AttnUpBlock2D × 1 + UpBlock2D × 2
```

## 参数配置指南

### 训练参数说明

#### VAE参数
| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| latent_dim | 32 | 潜在空间维度 | 16-128 |
| epochs | 5 | 训练轮数 | 10-50 |
| lr | 1e-3 | 学习率 | 1e-4 ~ 5e-3 |
| batch_size | 128 | 批大小 | 64-256 |

#### GAN参数
| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| latent_dim | 100 | 噪声维度 | 64-256 |
| epochs | 20 | 训练轮数 | 20-100 |
| lr_G | 2e-4 | 生成器学习率 | 1e-4 ~ 1e-3 |
| lr_D | 2e-4 | 判别器学习率 | 1e-4 ~ 1e-3 |
| beta1 | 0.5 | Adam beta1 | 0.3-0.7 |
| batch_size | 128 | 批大小 | 64-256 |

#### Diffusion参数
| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| epochs | 5 | 训练轮数 | 5-20 |
| lr | 1e-4 | 学习率 | 5e-5 ~ 5e-4 |
| num_timesteps | 1000 | 扩散步数 | 1000 (固定) |
| img_size | 32 | 图像尺寸 | 32-64 |
| batch_size | 128 | 批大小 | 32-128 |

### 评估参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_samples | 500 | 每个模型生成的样本数 |
| batch_size | 16 | 评估批大小 |
| k_neighbors | 5 | Precision/Recall的k值 |
| inception_input_size | 299 | Inception v3输入尺寸 |

## 常见问题解答 (FAQ)

### Q1: 如何处理Windows上的多进程DataLoader卡死问题？

**问题描述**：在Windows系统上，设置`num_workers>0`可能导致DataLoader卡死。

**解决方案**：
```python
# 将num_workers设置为0
train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
```

### Q2: CIFAR-10数据集下载失败怎么办？

**问题描述**：由于网络原因，CIFAR-10数据集可能下载失败或下载不完整。

**解决方案**：
1. 手动下载数据集：
   - 使用国内镜像：清华大学/阿里云镜像站
   - 或从其他来源获取已下载的数据集
2. 将数据集文件放到`data/`目录下
3. 确保文件结构正确：
   ```
   data/
   ├── cifar-10-batches-py/
   │   ├── data_batch_1
   │   ├── data_batch_2
   │   ├── ...
   │   └── test_batch
   ```

### Q3: GPU内存不足（CUDA Out of Memory）怎么办？

**问题描述**：训练或评估时GPU显存不足。

**解决方案**：
1. 减小批大小：
   ```python
   batch_size = 64  # 从128减小到64
   ```
2. 在评估脚本中减小样本量：
   ```python
   NUM_SAMPLES = 250  # 从500减小到250
   ```
3. 使用混合精度训练（需修改代码）
4. 及时清理中间变量：
   ```python
   del intermediate_tensor
   torch.cuda.empty_cache()
   ```

### Q4: 如何重新加载已训练的模型？

```python
import torch

# 加载VAE
from run_vae import VAE
vae = VAE(latent_dim=32)
checkpoint = torch.load('results/checkpoints/vae_model.pth')
vae.load_state_dict(checkpoint['model_state_dict'])
vae.eval()

# 加载GAN生成器
from run_gan import Generator
gan_G = Generator(latent_dim=100)
checkpoint = torch.load('results/checkpoints/gan_model.pth')
gan_G.load_state_dict(checkpoint['generator_state_dict'])
gan_G.eval()

# 加载Diffusion模型
from diffusers import UNet2DModel
diff_model = UNet2DModel.from_pretrained('results/checkpoints/diffusion_unet')
diff_model.eval()
```

### Q5: 如何延长训练轮数？

修改对应脚本中的`num_epochs`参数：
```python
# run_vae.py
num_epochs = 50  # 从5改为50

# run_gan.py
num_epochs = 100  # 从20改为100

# run_diffusion.py
num_epochs = 20  # 从5改为20
```

### Q6: Diffusion模型训练太慢怎么办？

**优化建议**：
1. 减少图像尺寸（如从32降到28）
2. 减少训练步数（`num_inference_steps`）
3. 使用更轻量的UNet架构
4. 考虑使用DDIM进行更快采样

### Q7: 如何保存训练中间结果？

代码已支持checkpoint保存，保存路径为`results/checkpoints/`。如需额外保存：
```python
# 每N轮保存一次
if (epoch + 1) % 5 == 0:
    torch.save(model.state_dict(), f'results/checkpoints/model_epoch_{epoch+1}.pth')
```

## 版本信息

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v1.0.0 | 2026-07-28 | 初始版本，包含VAE/GAN/Diffusion三种模型实现与评估框架 |
| v1.1.0 | 2026-07-28 | 添加模型checkpoint保存与加载功能，更新评估脚本使用训练权重 |

## 参考文献

1. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.

2. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., ... & Bengio, Y. (2014). Generative Adversarial Nets.

3. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.

4. Heusel, M., Ramsauer, H., Unterthiner, T., Stehfest, E., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.

5. Kynkäänniemi, T., Karras, T., Laine, S., & Aila, T. (2019). Improved Precision and Recall Metric for Assessing Generative Models.

## 许可证

本项目仅供学术研究使用。
