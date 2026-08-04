# 指令微调与对齐技术 (Instruction Tuning & Alignment)

一个完整的指令微调与对齐实验框架，支持 SFT（监督微调）、DPO（直接偏好优化）等对齐技术。

## 目录

- [项目概述](#项目概述)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [配置说明](#配置说明)

---

## 项目概述

本项目是一个用于研究大语言模型（LLM）指令微调与对齐技术的完整实验框架。

### 核心特性

- **多种对齐算法**：支持 SFT、DPO 等主流对齐方法
- **参数高效微调**：集成 LoRA、QLoRA 等高效微调技术
- **可复现实验**：固定随机种子、完整配置记录、结果自动归档
- **对比实验**：支持多配置并行对比，自动生成对比报告
- **结果汇总**：一键汇总所有实验结果，生成最终分析报告
- **GPU强制校验**：自动检测CUDA环境，确保在GPU上运行

### 技术栈

- PyTorch 2.0+
- HuggingFace Transformers, PEFT, TRL
- LoRA / QLoRA 参数高效微调
- YAML 配置管理

---

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
conda create -n alignment python=3.10
conda activate alignment

# 安装PyTorch (根据CUDA版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 准备数据

**SFT数据格式** (保存为 `data/processed/your_dataset/train.json`):
```json
[
  {
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  }
]
```

**DPO数据格式** (保存为 `data/processed/preference_dataset/train.json`):
```json
[
  {
    "prompt": "什么是Transformer?",
    "chosen": "Transformer是一种基于自注意力机制的神经网络...",
    "rejected": "Transformer就是一个RNN的变种。"
  }
]
```

### 3. 运行实验

```bash
# 运行SFT实验
python scripts/run_experiment.py --config configs/experiment/sft_llama3.yaml

# 运行DPO实验
python scripts/run_experiment.py --config configs/experiment/dpo_alignment.yaml

# 多次重复实验
python scripts/run_experiment.py --config configs/experiment/sft_llama3.yaml --repeat_times 5
```

### 4. 查看结果

实验完成后，结果自动保存在 `results/exp_*/` 目录：
```
results/exp_20260731_220000/
├── config_used.yaml       # 实际使用的配置
├── metrics.csv            # 每个epoch的指标
├── metrics.json           # 最终汇总指标
├── summary.md             # 自动生成的摘要
├── plots/                 # 训练图表
│   ├── loss_curve.png
│   └── eval_loss_curve.png
└── checkpoints/           # 模型权重
```

---

## 使用方法

### 单次实验

```bash
python scripts/run_experiment.py \
  --config configs/experiment/sft_llama3.yaml \
  --exp_id my_first_exp
```

### 多次重复实验

```bash
python scripts/run_experiment.py \
  --config configs/experiment/sft_llama3.yaml \
  --repeat_times 5
```

### 对比实验

```bash
python scripts/run_comparison.py \
  --configs \
    configs/experiment/sft_baseline.yaml \
    configs/experiment/dpo_improved.yaml \
  --output_dir results/comparison_001
```

### 汇总结果并生成报告

```bash
# Step 1: 汇总所有实验结果
python scripts/aggregate_results.py

# Step 2: 生成最终报告
python scripts/generate_report.py
```

---

## 项目结构

```
├── configs/                    # 配置文件
│   ├── base.yaml              # 基础配置
│   └── experiment/            # 实验配置
├── data/
│   ├── raw/                   # 原始数据
│   └── processed/             # 预处理数据
├── src/
│   ├── data/                  # 数据处理
│   ├── models/                # 模型管理
│   ├── training/             # 训练逻辑
│   ├── evaluation/           # 评估指标
│   ├── pipeline/             # 完整Pipeline
│   └── utils/                # 工具函数
├── scripts/
│   ├── run_experiment.py      # 主实验脚本
│   ├── run_comparison.py      # 对比实验
│   ├── aggregate_results.py  # 结果汇总
│   └── generate_report.py     # 报告生成
├── results/                   # 实验结果 (gitignored)
├── logs/                      # 日志文件 (gitignored)
├── reports/                   # 生成的报告
├── tests/                     # 测试文件
├── requirements.txt
└── README.md
```

---

## 配置说明

### 配置文件结构

```yaml
experiment:
  name: "sft_llama3"           # 实验名称
  seed: 42                     # 随机种子
  repeat_times: 3              # 重复次数

data:
  name: "alpaca"               # 数据集名称
  path: "data/processed/..."  # 数据路径
  max_length: 2048             # 最大序列长度

model:
  name: "meta-llama/Meta-Llama-3-8B"  # 模型名称
  lora:
    enabled: true
    r: 16
    target_modules: ["q_proj", "v_proj"]

training:
  method: "sft"                # sft, dpo
  epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5
  fp16: true

evaluation:
  metrics: ["accuracy", "bleu", "rouge"]
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `training.method` | 训练方法 | `sft`, `dpo` |
| `model.lora.enabled` | 是否启用LoRA | `true` (节省显存) |
| `model.quantization.enabled` | 是否启用量化 | `false` (16G+显存) |
| `training.epochs` | 训练轮数 | SFT: 2-5, DPO: 1-3 |
| `training.learning_rate` | 学习率 | SFT: 2e-5, DPO: 5e-6 |

### SFT vs DPO 对比

| 配置项 | SFT | DPO |
|--------|-----|-----|
| 数据格式 | instruction/output | prompt/chosen/rejected |
| 学习率 | 2e-5 | 5e-6 |
| Epochs | 3 | 1 |
| 额外参数 | - | `beta: 0.1` |

---

## 技术文档

### 核心论文

- [Instruction-Following Evaluation for LLMs](https://arxiv.org/abs/2311.07911)
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)
- [RLHF: Training LLMs to Follow Instructions](https://arxiv.org/abs/2203.02155)

### 关键库

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT: Parameter-Efficient Fine-Tuning](https://peft.readthedocs.io/)
- [TRL: Transformer Reinforcement Learning](https://trl.readthedocs.io/)

---

## 常见问题

### Q1: 显存不足怎么办？

1. 启用量化：设置 `model.quantization.enabled: true`
2. 使用更小的LoRA rank：`model.lora.r: 8`
3. 减小batch_size和sequence_length
4. 启用梯度检查点：`training.gradient_checkpointing: true`

### Q2: 如何添加自定义数据集？

1. 准备数据为JSON格式
2. 修改 `configs/experiment/your_config.yaml` 中的 `data.path`
3. 运行 `python scripts/run_experiment.py --config configs/experiment/your_config.yaml`

### Q3: 支持哪些模型？

支持所有 HuggingFace Transformers 兼容的因果语言模型：
- LLaMA / LLaMA-2 / LLaMA-3
- Mistral / Mixtral
- GPT-NeoX / Pythia
- 其他 decoder-only 架构模型

---

## 许可证

MIT License
