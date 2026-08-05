#!/usr/bin/env python3
"""LLM优化实验 v2 - 基于真实MedQA数据集

优化内容（相比v1）：
1. 使用真实MedQA数据集（120条真实医疗选择题）
2. ToT策略重写：分段生成（分析→选择），避免小模型生成截断
3. Few-Shot优化：减少示例到2个，修改格式引导避免字面输出X
4. Self-Consistency优化：降温到0.4-0.5，增加路径数到5
5. 新增CoT+Verifier策略：CoT推理后用verifier验证
6. 英文Prompt（MedQA是英文数据集，TinyLlama基础能力也是英文）
7. 更鲁棒的答案提取
"""
import os
import sys
import json
import time
import re
from datetime import datetime
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def extract_answer_option(text):
    """从生成文本中精确提取答案选项（A/B/C/D）"""
    if not text or len(text) == 0:
        return ''

    text_upper = text.upper()

    # 模式1：明确的"Answer: X"或"The answer is X"格式
    patterns = [
        r'(?:FINAL\s+ANSWER|THE\s+ANSWER\s+IS|ANSWER)\s*[:\.\,]?\s*([A-D])\b',
        r'CORRECT\s+ANSWER\s*[:\.\,]?\s*([A-D])\b',
        r'OPTION\s*([A-D])\s+IS\s+CORRECT',
        r'答案\s*[：:]\s*([A-D])',
        r'最终答案\s*[：:]\s*([A-D])',
    ]
    for p in patterns:
        m = re.search(p, text_upper)
        if m:
            return m.group(1)

    # 模式2：行末单独的选项字母
    lines = text.strip().split('\n')
    for line in reversed(lines[-8:]):
        line = line.strip().rstrip('.。,:;')
        # 纯字母行
        if len(line) == 1 and line in 'ABCD':
            return line
        # "A." / "A)" / "A:" 开头
        m = re.match(r'^([A-D])[\.\)\:]\s*$', line)
        if m:
            return m.group(1)
        # "is A" / "choose A" / "select A"
        m = re.search(r'\b(?:IS|CHOOSE|SELECT|PICK)\s+([A-D])\s*\.?\s*$', line)
        if m:
            return m.group(1)

    # 模式3：开头就是选项字母
    m = re.match(r'^([A-D])[\.\)\:\s]', text_upper.strip())
    if m:
        return m.group(1)

    # 模式4：文本中出现次数最多的选项（排除选项内容中的字母）
    # 先移除已知选项文本
    counts = {}
    for opt in ['A', 'B', 'C', 'D']:
        count = len(re.findall(rf'(?<![A-Z]){opt}(?![A-Za-z])', text_upper))
        if count > 0:
            counts[opt] = count
    if counts:
        return max(counts, key=counts.get)

    return ''


def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.0,
             repetition_penalty=1.1, top_p=0.9, top_k=50):
    """统一的生成函数"""
    import torch
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return response
    except Exception as e:
        return f"[ERROR: {e}]"


def main():
    start = time.time()
    log("=" * 60)
    log("LLM优化实验 v2 - 真实MedQA数据集")
    log("目标：在真实数据上优化所有策略，特别是ToT")
    log("=" * 60)

    # 环境检查
    log("\n[0] 环境检查")
    import torch
    log(f"  CUDA={torch.cuda.is_available()}, Device={torch.cuda.get_device_name(0)}")

    # 配置
    MODEL_PATH = r"c:/Users/17456/Documents/GitHub/Deep_learningPractice/Few-Shot  Meta-Learning/models/TinyLlama-SFT-merged"
    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "medqa_real.jsonl")

    log(f"\n[1] 加载模型: {MODEL_PATH}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    log(f"  模型加载成功")

    # 加载真实数据
    log(f"\n[2] 加载真实MedQA测试数据...")
    samples = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # 前100条做测试，后20条作为few-shot示例池
    test_samples = samples[:100]
    fewshot_pool = samples[100:120]
    log(f"  测试样本: {len(test_samples)} 条 (真实MedQA)")
    log(f"  Few-Shot示例池: {len(fewshot_pool)} 条")

    # ================== 优化的Prompt策略 ==================

    strategies_results = {}

    # ---- 策略1: Zero-Shot v2 (保持优秀策略) ----
    log(f"\n{'='*50}")
    log(f"  [1/6] 运行 zero_shot_v2 策略")
    log(f"  优化版Zero-Shot：角色+格式约束+贪婪解码")
    log(f"{'='*50}")

    zs_template = """You are an experienced medical doctor answering a multiple-choice question from the US Medical Licensing Examination.

Read the question and all options carefully, then select the single best answer.

Question: {question}
Options:
{options}

Rules:
1. Analyze each option based on medical knowledge
2. Select the SINGLE best answer
3. End your response with "Answer: X" where X is A, B, C, or D

Your analysis and answer:
"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt = zs_template.format(question=sample['question'], options=options_text)
        response = generate(model, tokenizer, prompt, max_new_tokens=250,
                            temperature=0.0, repetition_penalty=1.1)
        pred = extract_answer_option(response)
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'],
                        'response_excerpt': response[:120]})
        if (idx + 1) % 20 == 0:
            cur_acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    进度: {idx+1}/100, 当前准确率: {cur_acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['zero_shot_v2'] = {
        'description': 'Zero-Shot v2: Role+format constraint+greedy decoding',
        'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': len(predictions), 'correct': int(acc * len(predictions)),
        'sample_details': details,
    }
    log(f"  ✅ 准确率: {acc:.2%}, 幻觉率: {hal:.2%}")

    # ---- 策略2: Few-Shot v2 (减少示例，修复X字面输出问题) ----
    log(f"\n{'='*50}")
    log(f"  [2/6] 运行 few_shot_v2 策略")
    log(f"  优化版Few-Shot：2个真实示例+格式修复")
    log(f"{'='*50}")

    # 从fewshot_pool取2条作为示例
    ex1, ex2 = fewshot_pool[0], fewshot_pool[1]
    fs_template = """You are an experienced medical doctor. Here are example questions with correct answers:

Example 1:
Question: {ex1_q}
Options:
{ex1_opts}
Analysis: Based on the clinical presentation and pathophysiology, the correct option is determined by the key diagnostic criteria.
Answer: {ex1_a}

Example 2:
Question: {ex2_q}
Options:
{ex2_opts}
Analysis: The key is to identify the most likely diagnosis based on the presenting symptoms and clinical guidelines.
Answer: {ex2_a}

Now answer this question:

Question: {question}
Options:
{options}

Provide a brief analysis, then end with "Answer: [letter]":
"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        ex1_opts = "\n".join(f"{k}. {v}" for k, v in ex1['options'].items())
        ex2_opts = "\n".join(f"{k}. {v}" for k, v in ex2['options'].items())
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt = fs_template.format(
            ex1_q=ex1['question'], ex1_opts=ex1_opts, ex1_a=ex1['answer'],
            ex2_q=ex2['question'], ex2_opts=ex2_opts, ex2_a=ex2['answer'],
            question=sample['question'], options=options_text,
        )
        response = generate(model, tokenizer, prompt, max_new_tokens=300,
                            temperature=0.1, repetition_penalty=1.1)
        pred = extract_answer_option(response)
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'],
                        'response_excerpt': response[:120]})
        if (idx + 1) % 20 == 0:
            cur_acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    进度: {idx+1}/100, 当前准确率: {cur_acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['few_shot_v2'] = {
        'description': 'Few-Shot v2: 2 real examples+format fix',
        'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': len(predictions), 'correct': int(acc * len(predictions)),
        'sample_details': details,
    }
    log(f"  ✅ 准确率: {acc:.2%}, 幻觉率: {hal:.2%}")

    # ---- 策略3: CoT v2 (简洁2步推理) ----
    log(f"\n{'='*50}")
    log(f"  [3/6] 运行 cot_v2 策略")
    log(f"  优化版CoT：简洁2步推理（分析→选择）")
    log(f"{'='*50}")

    cot_template = """You are an experienced medical doctor. Answer this USMLE question using step-by-step reasoning.

Question: {question}
Options:
{options}

Step 1 - Key findings: Identify the main clinical findings and what they point to.
Step 2 - Best answer: Based on your analysis, select the best option.

"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt = cot_template.format(question=sample['question'], options=options_text)
        response = generate(model, tokenizer, prompt, max_new_tokens=350,
                            temperature=0.2, repetition_penalty=1.15)
        pred = extract_answer_option(response)
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'],
                        'response_excerpt': response[:120]})
        if (idx + 1) % 20 == 0:
            cur_acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    进度: {idx+1}/100, 当前准确率: {cur_acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['cot_v2'] = {
        'description': 'CoT v2: Concise 2-step reasoning',
        'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': len(predictions), 'correct': int(acc * len(predictions)),
        'sample_details': details,
    }
    log(f"  ✅ 准确率: {acc:.2%}, 幻觉率: {hal:.2%}")

    # ---- 策略4: Self-Consistency v2 (降温+5路径) ----
    log(f"\n{'='*50}")
    log(f"  [4/6] 运行 self_consistency_v2 策略")
    log(f"  优化版SC：5路径+低温0.4+投票")
    log(f"{'='*50}")

    sc_template = """You are an experienced medical doctor. Analyze this question from a fresh perspective (attempt {n}).

Question: {question}
Options:
{options}

Analyze and end with "Answer: X":
"""
    predictions, references, details = [], [], []
    n_paths = 5
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        path_preds = []
        first_response = ""
        for p in range(1, n_paths + 1):
            prompt = sc_template.format(n=p, question=sample['question'], options=options_text)
            response = generate(model, tokenizer, prompt, max_new_tokens=250,
                                temperature=0.4 + p * 0.05, repetition_penalty=1.1,
                                top_p=0.9)
            if p == 1:
                first_response = response
            path_preds.append(extract_answer_option(response))

        valid = [p for p in path_preds if p in 'ABCD']
        if valid:
            pred = Counter(valid).most_common(1)[0][0]
        else:
            pred = ''
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'],
                        'response_excerpt': f"[votes: {path_preds}] {first_response[:80]}"})
        if (idx + 1) % 20 == 0:
            cur_acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    进度: {idx+1}/100, 当前准确率: {cur_acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['self_consistency_v2'] = {
        'description': 'Self-Consistency v2: 5 paths+low temp 0.4+majority vote',
        'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': len(predictions), 'correct': int(acc * len(predictions)),
        'sample_details': details,
    }
    log(f"  ✅ 准确率: {acc:.2%}, 幻觉率: {hal:.2%}")

    # ---- 策略5: ToT v2 (分段生成：分析→选择) ----
    log(f"\n{'='*50}")
    log(f"  [5/6] 运行 tot_v2 策略")
    log(f"  优化版ToT：分段生成（选项分析→最终选择）")
    log(f"{'='*50}")

    tot_stage1_template = """You are an experienced medical doctor. Analyze each option for this USMLE question.

Question: {question}
Options:
{options}

For each option, briefly state if it is correct or incorrect and why (1 sentence each):

Option A:"""
    tot_stage2_template = """Based on your analysis, select the single best answer.

Question: {question}
Options:
{options}

Your previous analysis:
{analysis}

The best answer is option (write only the letter):"""

    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())

        # Stage 1: 分析每个选项
        prompt1 = tot_stage1_template.format(
            question=sample['question'], options=options_text)
        analysis = generate(model, tokenizer, prompt1, max_new_tokens=300,
                            temperature=0.1, repetition_penalty=1.15)

        # Stage 2: 基于分析做最终选择
        prompt2 = tot_stage2_template.format(
            question=sample['question'], options=options_text,
            analysis=analysis[:500])
        response = generate(model, tokenizer, prompt2, max_new_tokens=50,
                            temperature=0.0, repetition_penalty=1.0)

        # 合并分析+选择
        full_response = analysis + "\n\nFinal: " + response
        pred = extract_answer_option(response) or extract_answer_option(full_response)

        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'],
                        'response_excerpt': full_response[:120]})
        if (idx + 1) % 20 == 0:
            cur_acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    进度: {idx+1}/100, 当前准确率: {cur_acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['tot_v2'] = {
        'description': 'ToT v2: Two-stage generation (analyze options -> select answer)',
        'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': len(predictions), 'correct': int(acc * len(predictions)),
        'sample_details': details,
    }
    log(f"  ✅ 准确率: {acc:.2%}, 幻觉率: {hal:.2%}")

    # ---- 策略6: CoT+Verifier (新增) ----
    log(f"\n{'='*50}")
    log(f"  [6/6] 运行 cot_verifier 策略")
    log(f"  CoT+Verifier：CoT推理后用verifier验证答案")
    log(f"{'='*50}")

    cot_gen_template = """You are an experienced medical doctor. Answer this USMLE question.

Question: {question}
Options:
{options}

Provide brief reasoning, then end with "Answer: X":
"""
    verifier_template = """You are a medical exam verifier. Check if the proposed answer is correct.

Question: {question}
Options:
{options}

Proposed answer: {proposed}

Is this correct? If not, what is the correct answer? End with "Verified answer: X":
"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())

        # Stage 1: CoT生成初始答案
        prompt1 = cot_gen_template.format(
            question=sample['question'], options=options_text)
        cot_response = generate(model, tokenizer, prompt1, max_new_tokens=250,
                                temperature=0.2, repetition_penalty=1.1)
        initial_pred = extract_answer_option(cot_response)

        # Stage 2: Verifier验证
        if initial_pred:
            prompt2 = verifier_template.format(
                question=sample['question'], options=options_text,
                proposed=initial_pred)
            verify_response = generate(model, tokenizer, prompt2, max_new_tokens=150,
                                       temperature=0.0, repetition_penalty=1.1)
            verified_pred = extract_answer_option(verify_response)
            pred = verified_pred if verified_pred else initial_pred
        else:
            verify_response = ""
            pred = initial_pred

        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'],
                        'response_excerpt': f"[init:{initial_pred}->ver:{pred}] {cot_response[:80]}"})
        if (idx + 1) % 20 == 0:
            cur_acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    进度: {idx+1}/100, 当前准确率: {cur_acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['cot_verifier'] = {
        'description': 'CoT+Verifier: CoT reasoning followed by answer verification',
        'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': len(predictions), 'correct': int(acc * len(predictions)),
        'sample_details': details,
    }
    log(f"  ✅ 准确率: {acc:.2%}, 幻觉率: {hal:.2%}")

    # ================== 保存结果 ==================
    log(f"\n[3] 保存实验结果...")
    results_dir = os.path.join(PROJECT_ROOT, "results", "experiment_v2_real")
    os.makedirs(results_dir, exist_ok=True)

    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'sample_details'}
               for k, v in strategies_results.items()}
    with open(os.path.join(results_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(results_dir, "metrics_full.json"), 'w', encoding='utf-8') as f:
        json.dump(strategies_results, f, ensure_ascii=False, indent=2)
    log(f"  结果已保存: {results_dir}")

    elapsed = time.time() - start
    log(f"\n{'='*60}")
    log(f"全部完成！总耗时: {elapsed/60:.1f}分钟 ({elapsed:.0f}秒)")
    log(f"{'='*60}")

    print("\n🏆 最终策略排名（真实MedQA 100题）：")
    print("-" * 60)
    for rank, (name, r) in enumerate(
        sorted(strategies_results.items(), key=lambda x: x[1]['accuracy'], reverse=True), 1):
        print(f"  第{rank}名: {name:25s} | 准确率: {r['accuracy']:>6.2%} | "
              f"正确: {r['correct']:2d}/{r['num_samples']} | 幻觉率: {r['hallucination_rate']:.2%}")


if __name__ == "__main__":
    main()
