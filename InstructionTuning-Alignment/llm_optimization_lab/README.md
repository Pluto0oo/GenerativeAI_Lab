# LLM Optimization Lab - 大模型优化技术实验

基于 Prompt工程、RAG、模型压缩三大优化方向的轻量级医疗问答助手系统。

## 概述

本项目旨在解决**如何让大模型在资源受限/专业场景下可靠工作**这一核心问题，通过三种互补的优化技术路线：

| 优化方向 | 核心技术 | 解决问题 |
|----------|----------|----------|
| **Prompt工程** | Zero/Few-Shot, CoT, ToT, Self-Consistency | 无需训练，直接提升推理能力 |
| **RAG检索增强** | 向量检索+上下文拼接+生成 | 缓解幻觉，注入实时知识 |
| **模型压缩** | INT8/INT4量化, LoRA微调 | 降低显存占用，加速推理 |

## 快速开始

### 环境准备

```bash
# 创建虚拟环境
conda create -n llm-opt python=3.10
conda activate llm-opt

# 安装依赖
pip install -r requirements.txt
```

### 1. 下载数据

```bash
# 下载MedQA医疗问答数据集
python scripts/download_medqa.py

# 构建医学知识库
python scripts/build_knowledge_base.py
```

### 2. Prompt工程实验

```bash
# 运行Zero-Shot基线
python scripts/run_prompt_experiment.py --config configs/prompt/zero_shot.yaml

# 运行CoT思维链
python scripts/run_prompt_experiment.py --config configs/prompt/cot.yaml

# 运行Self-Consistency
python scripts/run_prompt_experiment.py --config configs/prompt/self_consistency.yaml

# 对比多种策略
python scripts/run_prompt_experiment.py \
  --config configs/prompt/cot.yaml \
  --compare_configs zero_shot.yaml few_shot.yaml self_consistency.yaml
```

### 3. RAG医疗问答

```bash
# 构建知识库
python scripts/run_rag_demo.py --config configs/rag/medical_rag.yaml --build_knowledge_base

# 交互式问答
python scripts/run_rag_demo.py --config configs/rag/medical_rag.yaml --interactive

# 评估RAG效果
python scripts/run_rag_demo.py --config configs/rag/medical_rag.yaml --evaluate
```

### 4. 量化实验

```bash
# INT8量化
python scripts/run_quantization.py --config configs/quantization/int8.yaml

# INT4量化
python scripts/run_quantization.py --config configs/quantization/int4.yaml

# 对比不同量化精度
python scripts/run_quantization.py \
  --config configs/quantization/int8.yaml \
  --compare_configs int4.yaml
```

### 5. 完整医疗助手

```bash
# 全流程优化（Prompt + RAG + 量化）
python scripts/run_medical_assistant.py --config configs/experiment/medical_assistant.yaml

# 交互式使用
python scripts/run_medical_assistant.py \
  --config configs/experiment/medical_assistant.yaml \
  --interactive
```

## 项目结构

```
llm_optimization_lab/
├── configs/                    # 配置文件
│   ├── base.yaml              # 基础配置
│   ├── prompt/                # Prompt实验配置
│   ├── rag/                   # RAG实验配置
│   ├── quantization/          # 量化实验配置
│   └── experiment/            # 综合实验配置
├── src/                        # 核心代码
│   ├── prompt/                # Prompt工程模块
│   ├── rag/                   # RAG模块
│   ├── compression/           # 模型压缩模块
│   ├── models/                # 模型模块
│   ├── evaluation/            # 评估模块
│   ├── utils/                 # 工具函数
│   └── pipeline/              # 完整Pipeline
├── scripts/                    # 运行脚本
├── notebooks/                 # Jupyter笔记
├── data/                      # 数据目录
├── knowledge_base/            # RAG知识库
├── results/                   # 实验结果
├── logs/                      # 日志
├── reports/                   # 报告
├── requirements.txt           # 依赖
└── README.md                  # 本文件
```

## 评估指标

| 维度 | 指标 | 说明 |
|------|------|------|
| **精度** | Accuracy, EM, F1 | 回答正确性 |
| **效率** | Latency(ms), Throughput | 推理速度 |
| **安全** | Hallucination Rate, Faithfulness | 幻觉率与忠实度 |
| **资源** | Memory(GB), Model Size | 显存与模型大小 |

## 技术栈

- **基础框架**: PyTorch, HuggingFace Transformers
- **RAG**: FAISS, ChromaDB, LangChain
- **量化**: bitsandbytes, HuggingFace Optimum
- **嵌入**: BAAI/bge-small-zh-v1.5
- **模型**: TinyLlama-1.1B-Chat-v1.0

## 文献参考

1. **Prompt Engineering**:
   - Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (NeurIPS 2022)
   - Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (ICLR 2023)

2. **RAG**:
   - Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
   - Ding et al., "Parameter-Efficient Prompt Tuning Makes Generalized and Calibrated Medical Question Answering Models" (2023)

3. **模型压缩**:
   - Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (ICLR 2023)
   - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (ACL 2024)

## 验收标准

- [x] 独立设计针对专业场景的推理优化方案
- [x] 实现可演示的端到端原型
- [x] 完成技术权衡分析（精度-效率-安全三角）
- [x] 对比多种优化方法的效果
- [x] 输出完整实验报告

## 许可证

MIT License
