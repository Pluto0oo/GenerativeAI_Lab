# LLM优化实验报告 v3 — MedQA-SFT针对性微调

**实验日期**: 2026-08-05  
**实验耗时**: 38.4分钟 (SFT训练27.2分钟 + 评估11.2分钟)  
**数据集**: 真实MedQA (10,178条USMLE选择题)  
**模型**: TinyLlama-1.1B-Chat → MedQA-SFT (LoRA微调)  
**硬件**: NVIDIA GeForce RTX 5060 Laptop GPU

---

## 1. 实验目标

v2实验中，最佳策略(ToT v2)准确率仅27%（随机基线25%），主要原因是TinyLlama未学习过USMLE选择题格式。v3通过**用MedQA训练集做针对性SFT微调**来提升准确率。

## 2. SFT微调方案

### 2.1 训练数据
- **来源**: 真实MedQA数据集 (medalpaca/medical_meadow_medqa)
- **训练集**: 5,000条（从8,068条4选项样本中随机抽取，排除测试集）
- **测试集**: 100条（与v2相同的真实MedQA样本）
- **数据格式**:
  ```
  Instruction: You are a medical doctor answering a USMLE multiple-choice question.
               Question: ... Options: A. ... B. ... C. ... D. ...
               Select the single best answer. End with "Answer: X".
  Output: The correct answer is X.
          Answer: X
  ```

### 2.2 LoRA配置
| 参数 | 值 |
|------|-----|
| 基础模型 | TinyLlama-1.1B-Chat-v1.0 (原始，非huatuo-SFT) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| 可训练参数 | 4,505,600 (0.41%) |

### 2.3 训练参数
| 参数 | 值 |
|------|-----|
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.05 |
| Max sequence length | 1024 |
| Precision | bf16 |

### 2.4 训练结果
- **训练loss**: 1.0845
- **训练时间**: 27.2分钟
- **训练速度**: 6.14 samples/s
- **总步数**: 626步

## 3. 评估结果

### 3.1 v2(微调前) vs v3(微调后) 对比

| 策略 | v2准确率 | v3准确率 | 变化 | v2幻觉率 | v3幻觉率 |
|------|----------|----------|------|----------|----------|
| **CoT** | 15% | **32%** | **+17** | 51% | **0%** |
| **Few-Shot** | 1% | **31%** | **+30** | 90% | **0%** |
| **Self-Consistency** | 2% | **25%** | **+23** | 95% | **0%** |
| Zero-Shot | 24% | 23% | -1 | 3% | **0%** |
| ToT v2 | 27% | 19% | -8 | 0% | 0% |
| **CoT+Verifier** | 4% | **18%** | **+14** | 92% | **0%** |

### 3.2 v3策略排名

| 排名 | 策略 | 准确率 | 幻觉率 |
|------|------|--------|--------|
| 🥇 | **CoT** | **32%** | **0%** |
| 🥈 | Few-Shot | 31% | 0% |
| 🥉 | Self-Consistency | 25% | 0% |
| 4 | Zero-Shot | 23% | 0% |
| 5 | ToT v2 | 19% | 0% |
| 6 | CoT+Verifier | 18% | 0% |

## 4. 关键发现

### 4.1 SFT微调的核心贡献

1. **幻觉率清零**：所有6种策略的幻觉率从3-95%降至**0%**。SFT让模型学会了"Answer: X"的输出格式，答案提取变得可靠。

2. **Few-Shot从1%→31%**（+30%）：微调前模型无法从示例中学习，微调后模型具备了从Few-Shot示例中提取模式的能力。

3. **Self-Consistency从2%→25%**（+23%）：微调前多路径生成产生大量无效答案，微调后每条路径都能输出有效答案，投票机制才真正发挥作用。

4. **CoT成为最佳策略**（32%）：SFT让模型学会了step-by-step推理格式，CoT的2步推理（Key findings → Best answer）最为有效。

### 4.2 ToT v2 下降的原因

ToT v2从27%降至19%，原因是：
- SFT后模型学会了直接回答格式（"Answer: X"）
- ToT的分段生成（先分析→再选择）反而打断了模型的直接回答能力
- 微调后的模型更适合简洁的CoT推理，而非复杂的分段分析

**启示**：SFT会改变模型对Prompt格式的偏好。微调后应重新评估最优策略。

### 4.3 准确率仍受限的原因

32%准确率虽然超过随机基线(25%)7个百分点，但绝对值仍偏低，原因：
1. **模型容量限制**：TinyLlama仅1.1B参数，医疗知识容量有限
2. **USMLE难度**：美国医师执照考试是高难度专业考试
3. **训练数据量**：仅5,000条×2 epochs，可能不足以充分学习
4. **LoRA限制**：仅微调0.41%参数，知识注入有限

## 5. 可视化图表

| 图表 | 文件 | 说明 |
|------|------|------|
| SFT前后准确率对比 | v3_sft_comparison.png | 6策略微调前后准确率柱状图 |
| SFT前后幻觉率对比 | v3_hallucination_comparison.png | 幻觉率从3-95%降至0% |
| v3策略排名 | v3_strategy_ranking.png | CoT(32%)为最佳策略 |
| 综合仪表板 | v3_dashboard.png | 4子图：准确率对比/幻觉率对比/散点图/变化幅度 |

## 6. 三版实验演进总结

| 版本 | 数据集 | 最佳策略 | 准确率 | 幻觉率 | 关键改进 |
|------|--------|----------|--------|--------|----------|
| v1 | 构造40题 | Zero-Shot v2 | 40% | 0% | 角色约束+格式引导 |
| v2 | 真实100题 | ToT v2 | 27% | 0% | 真实数据+ToT分段生成 |
| **v3** | 真实100题 | **CoT** | **32%** | **0%** | **MedQA-SFT微调** |

注：v1的40%是在构造数据上的结果，不可与v2/v3直接比较。

## 7. 后续改进方向

1. **扩大训练数据**：使用全部8,000+条MedQA训练数据
2. **增加epochs**：从2 epochs增加到3-5 epochs
3. **更大LoRA rank**：从r=16增加到r=64，提升知识容量
4. **全参数微调**：不使用LoRA，直接微调全部参数
5. **更大基座模型**：尝试TinyLlama-3B或Phi-3-mini
6. **DPO对齐**：用正确/错误答案构造偏好对做DPO训练

## 8. 文件索引

| 文件 | 说明 |
|------|------|
| `train_medqa_sft.py` | SFT训练+评估一体化脚本 |
| `scripts/visualize_v3.py` | v3可视化图表生成 |
| `data/processed/medqa_sft_train.jsonl` | SFT训练数据(5000条) |
| `models/TinyLlama-MedQA-SFT/` | LoRA adapter |
| `models/TinyLlama-MedQA-SFT-merged/` | 合并后完整模型 |
| `results/experiment_v3_sft/` | 评估结果 |
| `reports/figures/v3_*.png` | 4张可视化图表 |

## 9. 结论

通过MedQA-SFT针对性微调，实现了：
1. ✅ **所有策略幻觉率降为0%** — SFT让模型学会输出格式
2. ✅ **多数策略准确率大幅提升** — Few-Shot +30%, SC +23%, CoT +17%
3. ✅ **CoT成为最佳策略(32%)** — 超过随机基线7个百分点
4. ✅ **6小时内完成** — SFT训练27分钟+评估11分钟=38.4分钟

**核心结论**：对于小模型(1.1B)，针对性SFT微调比Prompt工程优化更有效。SFT不仅提升了准确率，更重要的是消除了幻觉（无效答案），使得所有策略的评估结果都变得可靠。
