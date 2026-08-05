#!/usr/bin/env python3
"""LLM优化实验 - 优化版（提升Zero-Shot和CoT准确率）

优化内容：
1. 更好的Prompt模板（角色+约束+格式引导）
2. 更严格的答案提取（从生成文本中精确提取A/B/C/D）
3. 生成参数优化（repetition_penalty, stop_sequence等）
4. Tree-of-Thought (ToT) 策略
5. Chain-of-Thought + Verifier 策略
"""
import os
import sys
import json
import time
import re
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def extract_answer_option(text):
    """
    从生成文本中精确提取答案选项（A/B/C/D）
    优先匹配明确的答案格式，多种模式尝试
    """
    if not text or len(text) == 0:
        return ''

    text_upper = text.upper()

    # 模式1：明确的"答案：X"或"最终答案：X"格式
    patterns = [
        r'最终答案[::\s]*([A-D])',
        r'答案[::\s]*([A-D])',
        r'正确选项[::\s]*([A-D])',
        r'选择[::\s]*([A-D])',
        r'选[项择]*[为是::\s]*([A-D])',
    ]
    for p in patterns:
        m = re.search(p, text_upper)
        if m:
            return m.group(1)

    # 模式2：行末单独的选项字母
    lines = text.strip().split('\n')
    for line in reversed(lines[-5:]):  # 检查最后5行
        line = line.strip().rstrip('.。')
        if len(line) == 1 and line in 'ABCD':
            return line
        m = re.match(r'^([A-D])[\.、\)]', line)
        if m:
            return m.group(1)

    # 模式3：A./A)/A： 格式
    for opt in ['D', 'C', 'B', 'A']:  # 从后往前，避免误匹配
        if f'{opt}.' in text_upper[:10] or f'{opt})' in text_upper[:10] or f'{opt}：' in text_upper[:10]:
            return opt

    # 模式4：文本中出现次数最多的选项
    counts = {}
    for opt in ['A', 'B', 'C', 'D']:
        # 计数，但要排除options列表中的
        count = len(re.findall(rf'(?<![A-Z]){opt}(?![A-Za-z])', text_upper))
        if count > 0:
            counts[opt] = count
    if counts:
        return max(counts, key=counts.get)

    return ''


def main():
    start = time.time()
    log("=" * 60)
    log("LLM优化实验 - 优化版")
    log("目标：提升Zero-Shot和CoT策略准确率")
    log("=" * 60)

    # 环境检查
    log("\n[0] 环境检查")
    import torch
    log(f"  CUDA={torch.cuda.is_available()}, Device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # 模型配置
    MODEL_PATH = r"c:/Users/17456/Documents/GitHub/Deep_learningPractice/Few-Shot  Meta-Learning/models/TinyLlama-SFT-merged"
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "medqa_processed.jsonl")

    log(f"\n[1] 加载模型: {MODEL_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    log(f"  模型加载成功")

    # 加载数据
    log(f"\n[2] 加载测试数据...")
    samples = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    log(f"  加载了 {len(samples)} 条样本，使用全部样本进行测试")
    test_samples = samples

    # ================== 优化的Prompt策略 ==================

    strategies = {
        "zero_shot_v2": {
            "description": "优化版Zero-Shot：明确角色+输出格式约束+停止引导",
            "template": """你是一名经验丰富的专业临床医生，拥有深厚的医学知识和临床诊断经验。现在你需要回答以下医学选择题。

请严格遵守以下规则：
1. 仔细阅读问题和每个选项
2. 运用你的医学知识分析每个选项的正确性
3. 从A、B、C、D四个选项中选择唯一正确答案
4. 回答的最后必须单独一行写明"答案：X"，其中X是A/B/C/D中的一个字母

问题：{question}
选项：
{options}

请开始作答：
""",
            "max_tokens": 300,
            "repetition_penalty": 1.1,
        },

        "few_shot_v2": {
            "description": "优化版Few-Shot：3个高质量医疗示例",
            "template": """你是一名经验丰富的专业临床医生。请参考以下示例的格式回答医学选择题。

示例1：
问题：以下哪种药物是治疗原发性高血压的首选钙通道阻滞剂？
选项：
A. 美托洛尔
B. 氨氯地平
C. 卡托普利
D. 氢氯噻嗪

分析：美托洛尔是β受体阻滞剂，氨氯地平是二氢吡啶类钙通道阻滞剂，卡托普利是ACEI，氢氯噻嗪是利尿剂。钙通道阻滞剂首选二氢吡啶类。
答案：B

示例2：
问题：糖尿病患者空腹血糖的正常参考范围是多少（mmol/L）？
选项：
A. 3.9-6.1
B. 7.0-8.4
C. 8.5-11.1
D. >11.1

分析：根据临床诊断标准，空腹血糖正常范围是3.9-6.1mmol/L，≥7.0可诊断糖尿病。
答案：A

示例3：
问题：急性心肌梗死最典型的症状是？
选项：
A. 阵发性头痛
B. 持续性胸骨后压榨性疼痛
C. 间歇性腹痛
D. 持续性咳嗽

分析：急性心梗典型表现为胸骨后压榨性、憋闷性疼痛，持续>30分钟，休息不能缓解。
答案：B

现在回答新的问题：

问题：{question}
选项：
{options}

请先给出分析，然后在最后单独一行写"答案：X"：
""",
            "max_tokens": 400,
            "repetition_penalty": 1.1,
        },

        "cot_v2": {
            "description": "优化版CoT：结构化3步推理+医学分析引导",
            "template": """你是一名经验丰富的专业临床医生。请按三步推理法回答医学选择题。

三步推理法：
步骤1【问题解析】：识别问题考察的医学领域和核心知识点
步骤2【选项分析】：逐一分析每个选项的正确性，说明为什么对或为什么错
步骤3【最终结论】：综合以上分析，选出正确答案

问题：{question}
选项：
{options}

现在开始按步骤推理：

步骤1【问题解析】：
""",
            "max_tokens": 500,
            "repetition_penalty": 1.1,
        },

        "self_consistency_v2": {
            "description": "优化版自洽性：5种推理路径+投票",
            "template": """你是一名经验丰富的专业临床医生。请从多种角度分析医学选择题。

请提供独立的分析路径（第{{n}}次推理）：
1. 明确问题所属医学分科
2. 分析关键病理/药理机制
3. 比较各选项差异
4. 选择最佳答案
5. 在最后一行用"答案：X"给出你的选择

问题：{question}
选项：
{options}

第{{n}}次推理分析：
""",
            "max_tokens": 400,
            "n_paths": 3,  # 减少路径数控制时间
            "repetition_penalty": 1.05,
        },

        "tot_v1": {
            "description": "Tree-of-Thought：思维树探索",
            "template": """你是一名经验丰富的专业临床医生。请采用思维树方法分析医学选择题。

思维树方法：
【分支1-直接排除法】：哪些选项明显错误？为什么？
【分支2-知识点匹配法】：该问题对应什么核心知识点？哪个选项最匹配？
【分支3-临床经验法】：如果在临床上遇到这个情况，你会怎么判断？

综合三个分支后，给出最终答案。最后一行写"答案：X"。

问题：{question}
选项：
{options}

思维树分析：
""",
            "max_tokens": 500,
            "repetition_penalty": 1.1,
        },
    }

    results = {}

    for strategy_name, strategy_config in strategies.items():
        log(f"\n{'='*50}")
        log(f"  运行 [{strategy_name}] 策略")
        log(f"  {strategy_config['description']}")
        log(f"{'='*50}")

        predictions = []
        references = []
        sample_details = []

        max_tokens = strategy_config['max_tokens']
        rep_pen = strategy_config.get('repetition_penalty', 1.0)

        n_paths = strategy_config.get('n_paths', 1)

        for idx, sample in enumerate(test_samples):
            question = sample.get('question', '')
            options = sample.get('options', {})
            answer = sample.get('answer', '')

            # 构造options文本
            options_text = "\n".join(f"{k}. {v}" for k, v in options.items())

            if n_paths == 1:
                # 单次推理
                prompt = strategy_config['template'].format(
                    question=question, options=options_text
                )
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=0.3 if strategy_name != 'zero_shot_v2' else 0.0,
                            do_sample=(0.3 > 0),
                            repetition_penalty=rep_pen,
                            top_k=50,
                            top_p=0.9,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    input_len = inputs["input_ids"].shape[1]
                    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                except Exception as e:
                    response = f"[错误: {e}]"

                pred = extract_answer_option(response)
            else:
                # 多次推理+投票
                path_predictions = []
                responses = []
                for p in range(1, n_paths + 1):
                    path_template = strategy_config['template'].replace('{{n}}', str(p))
                    prompt = path_template.format(question=question, options=options_text)
                    try:
                        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_tokens,
                                temperature=0.7 + (p * 0.1),
                                do_sample=True,
                                repetition_penalty=rep_pen,
                                top_k=50,
                                top_p=0.95,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )
                        input_len = inputs["input_ids"].shape[1]
                        path_response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
                        responses.append(path_response)
                        pred_p = extract_answer_option(path_response)
                        path_predictions.append(pred_p)
                    except Exception as e:
                        responses.append(f"[错误: {e}]")

                # 投票选出最多的选项
                valid_preds = [p for p in path_predictions if p in 'ABCD']
                if valid_preds:
                    from collections import Counter
                    vote_counts = Counter(valid_preds)
                    pred = vote_counts.most_common(1)[0][0]
                else:
                    pred = ''
                response = f"[3路径投票: {path_predictions}]\n" + "\n---路径1---\n".join(responses[:1])

            predictions.append(pred)
            references.append(answer)

            is_correct = pred == answer
            sample_details.append({
                'idx': idx,
                'question': question[:40],
                'ref': answer,
                'pred': pred,
                'correct': is_correct,
                'response_excerpt': response[:80],
            })

            if (idx + 1) % 10 == 0 or idx == len(test_samples) - 1:
                # 实时计算当前准确率
                cur_correct = sum(1 for p, r in zip(predictions, references) if p == r)
                cur_acc = cur_correct / len(predictions)
                log(f"    进度: {idx + 1}/{len(test_samples)}, 当前准确率: {cur_acc:.2%} (正确{cur_correct}题)")

        # 计算最终准确率
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        # 幻觉率：未提取到有效选项的比例
        hallucination_count = sum(1 for p in predictions if p == '')
        hallucination_rate = hallucination_count / total if total > 0 else 0.0

        results[strategy_name] = {
            "description": strategy_config['description'],
            "accuracy": float(accuracy),
            "hallucination_rate": float(hallucination_rate),
            "num_samples": total,
            "correct": correct,
            "sample_details": sample_details,
        }

        log(f"\n  ✅ 最终准确率: {accuracy:.2%} ({correct}/{total})")
        log(f"  📊 幻觉率(无效答案): {hallucination_rate:.2%}")

    # 保存完整结果
    log(f"\n[3] 保存实验结果...")
    results_dir = os.path.join(PROJECT_ROOT, "results", "experiment_optimized")
    os.makedirs(results_dir, exist_ok=True)

    # 保存精简版指标（不含details，便于查看）
    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'sample_details'} for k, v in results.items()}
    with open(os.path.join(results_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存完整版（含details）
    with open(os.path.join(results_dir, "metrics_full.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"  结果已保存: {results_dir}")

    # ================== 生成优化版报告 ==================
    log(f"\n[4] 生成优化实验报告...")
    report_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "optimized_experiment_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# LLM优化实验报告（优化版）\n\n")
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. 实验概述\n\n")
        f.write("本实验针对Zero-Shot和CoT策略准确率低的问题，采用以下优化方案：\n\n")
        f.write("1. **Prompt工程优化**\n")
        f.write("   - 明确的角色设定（资深临床医生）\n")
        f.write("   - 结构化输出格式约束\n")
        f.write("   - 输出格式引导：最后必须写\"答案：X\"\n\n")
        f.write("2. **答案提取优化**\n")
        f.write("   - 正则匹配多种答案格式模式\n")
        f.write("   - 优先匹配明确的\"答案：X\"标签\n")
        f.write("   - 多模式回退机制（行末、标点、频次统计）\n\n")
        f.write("3. **生成参数优化**\n")
        f.write("   - repetition_penalty=1.1 抑制重复\n")
        f.write("   - CoT策略温度=0.3 增强多样性\n")
        f.write("   - 自洽性多路径投票\n\n")
        f.write("4. **新增策略**\n")
        f.write("   - Tree-of-Thought (ToT) 多分支思维\n")
        f.write("   - Self-Consistency 多路径投票\n\n")

        f.write("## 2. 策略对比结果\n\n")
        f.write("| 策略 | 描述 | 准确率 | 正确/总数 | 幻觉率 |\n")
        f.write("|------|------|--------|-----------|--------|\n")
        for name, r in results.items():
            desc_short = r['description'][:30]
            f.write(f"| {name} | {desc_short} | {r['accuracy']:.2%} | {r['correct']}/{r['num_samples']} | {r['hallucination_rate']:.2%} |\n")

        f.write("\n## 3. 详细结果分析\n\n")
        sorted_strategies = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        best_name, best_r = sorted_strategies[0]
        f.write(f"### 最优策略：{best_name}\n\n")
        f.write(f"- **准确率**: {best_r['accuracy']:.2%}\n")
        f.write(f"- **正确题数**: {best_r['correct']}/{best_r['num_samples']}\n")
        f.write(f"- **特点**: {best_r['description']}\n\n")

        # 按策略写sample details
        f.write("## 4. 各策略答题详情\n\n")
        for name, r in sorted_strategies:
            f.write(f"### {name} ({r['description']})\n\n")
            f.write("| # | 问题(前40字) | 参考 | 预测 | 正确 |\n")
            f.write("|---|-------------|------|------|------|\n")
            for d in r['sample_details']:
                mark = "✅" if d['correct'] else "❌"
                f.write(f"| {d['idx']+1} | {d['question']}... | {d['ref']} | {d['pred'] if d['pred'] else '-'} | {mark} |\n")
            f.write("\n")

        f.write("## 5. 优化方案与改进方向\n\n")
        f.write("### 当前优化效果\n\n")
        # 与之前对比
        prev_best = 0.20  # 之前Few-Shot 20%
        curr_best = best_r['accuracy']
        improvement = ((curr_best - prev_best) / prev_best * 100) if prev_best > 0 else 0
        f.write(f"- 历史最佳准确率: {prev_best:.2%} (Few-Shot v1)\n")
        f.write(f"- 当前最佳准确率: {curr_best:.2%} ({best_name})\n")
        f.write(f"- 相对提升: {improvement:+.1f}%\n\n")

        f.write("### 后续可尝试的优化\n\n")
        f.write("1. **模型层面**\n")
        f.write("   - 继续SFT训练（使用更多高质量医疗数据）\n")
        f.write("   - DPO对齐训练（减少幻觉增强事实一致性）\n")
        f.write("   - 医疗领域继续预训练（MLM/CLM）\n\n")
        f.write("2. **Prompt层面**\n")
        f.write("   - 动态Few-Shot示例选择（按相似度检索）\n")
        f.write("   - Med-PaLM风格的医疗专家提示\n")
        f.write("   - 多轮对话式推理（Graph-of-Thought）\n\n")
        f.write("3. **答案校准**\n")
        f.write("   - 训练一个小分类器对LLM输出做选项校准\n")
        f.write("   - 集成学习：多个Prompt策略的软投票\n")

    log(f"  报告已生成: {report_path}")

    elapsed = time.time() - start
    log(f"\n{'='*60}")
    log(f"全部完成！总耗时: {elapsed/60:.1f}分钟 ({elapsed:.0f}秒)")
    log(f"{'='*60}")

    # 打印最终排名
    print()
    print("🏆 最终策略排名：")
    print("-" * 50)
    for rank, (name, r) in enumerate(sorted_strategies, 1):
        print(f"  第{rank}名: {name:20s} | 准确率: {r['accuracy']:>6.2%} | 正确: {r['correct']:2d}/{r['num_samples']}")


if __name__ == "__main__":
    main()
