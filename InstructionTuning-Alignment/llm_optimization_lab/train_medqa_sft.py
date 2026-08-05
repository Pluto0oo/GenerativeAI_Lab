#!/usr/bin/env python3
"""MedQA SFT微调训练 + 评估一体化脚本

流程：
1. 从真实MedQA数据中准备5000条SFT训练数据
2. 用LoRA对TinyLlama-1.1B-Chat做SFT微调（2 epochs）
3. 合并LoRA权重到完整模型
4. 用微调后模型重新评估6种Prompt策略
5. 保存结果

时间预算：
- 数据准备: 1分钟
- SFT训练: ~30-40分钟
- 权重合并: 1分钟
- 评估: ~40分钟
- 总计: ~1.5小时（远在6小时内）
"""
import os
import sys
import json
import time
import re
import random
from datetime import datetime
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['BITSANDBYTES_NOWELCOME'] = '1'

# 路径配置
BASE_MODEL_PATH = r"c:/Users/17456/Documents/GitHub/Deep_learningPractice/Few-Shot  Meta-Learning/models/TinyLlama-1.1B-Chat-v1.0"
RAW_MEDQA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "medqa", "medical_meadow_medqa.json")
SFT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "medqa_sft_train.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "TinyLlama-MedQA-SFT")
MERGED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "TinyLlama-MedQA-SFT-merged")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =====================================================================
# 第1步：准备SFT训练数据
# =====================================================================
def parse_medqa_sample(item):
    """解析单条MedQA样本"""
    input_text = item.get('input', '')
    output_text = item.get('output', '')

    q_match = re.match(r'Q:(.+?)\?\s*\n\{(.+?)\}', input_text, re.DOTALL)
    if not q_match:
        return None

    question = q_match.group(1).strip()
    options_str = '{' + q_match.group(2) + '}'

    try:
        options_str = options_str.replace("'", '"')
        options_dict = json.loads(options_str)
    except Exception:
        options_dict = {}
        for m in re.finditer(r"'([A-E])':\s*'([^']+)'", q_match.group(2)):
            options_dict[m.group(1)] = m.group(2)

    if not options_dict:
        return None

    ans_match = re.match(r'^([A-E])', output_text.strip())
    if not ans_match:
        return None
    answer = ans_match.group(1)

    if answer not in options_dict:
        return None

    return {'question': question, 'options': options_dict, 'answer': answer,
            'full_answer': output_text}


def prepare_sft_data():
    """准备SFT训练数据"""
    log("=" * 60)
    log("[Step 1] 准备MedQA SFT训练数据")
    log("=" * 60)

    with open(RAW_MEDQA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    log(f"  原始数据: {len(raw_data)} 条")

    parsed = []
    for item in raw_data:
        p = parse_medqa_sample(item)
        if p:
            parsed.append(p)
    log(f"  成功解析: {len(parsed)} 条")

    # 只保留4选项(A-D)的，或者从5选项中截取前4个
    processed = []
    for p in parsed:
        if set(p['options'].keys()) == {'A', 'B', 'C', 'D'}:
            processed.append(p)
        elif set(p['options'].keys()) == {'A', 'B', 'C', 'D', 'E'} and p['answer'] in ['A', 'B', 'C', 'D']:
            p['options'] = {k: v for k, v in p['options'].items() if k in ['A', 'B', 'C', 'D']}
            processed.append(p)

    log(f"  4选项样本: {len(processed)} 条")

    # 排除已用于测试的120条（medqa_real.jsonl中的）
    test_path = os.path.join(PROJECT_ROOT, "data", "processed", "medqa_real.jsonl")
    test_questions = set()
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                test_questions.add(d['question'][:80])

    train_data = [p for p in processed if p['question'][:80] not in test_questions]
    log(f"  排除测试集后: {len(train_data)} 条")

    # 取前5000条训练
    random.seed(42)
    random.shuffle(train_data)
    train_data = train_data[:5000]
    log(f"  最终训练集: {len(train_data)} 条")

    # 转为SFT格式
    os.makedirs(os.path.dirname(SFT_DATA_PATH), exist_ok=True)
    with open(SFT_DATA_PATH, 'w', encoding='utf-8') as f:
        for i, p in enumerate(train_data):
            options_text = "\n".join(f"{k}. {v}" for k, v in p['options'].items())
            instruction = f"You are a medical doctor answering a USMLE multiple-choice question.\n\nQuestion: {p['question']}\nOptions:\n{options_text}\n\nSelect the single best answer. End with 'Answer: X'."
            output = f"The correct answer is {p['answer']}.\nAnswer: {p['answer']}"

            f.write(json.dumps({
                'id': i,
                'instruction': instruction,
                'output': output,
            }, ensure_ascii=False) + '\n')

    log(f"  SFT数据保存: {SFT_DATA_PATH}")

    # 打印样例
    with open(SFT_DATA_PATH, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline())
    log(f"\n  样例instruction:\n    {sample['instruction'][:200]}...")
    log(f"  样例output:\n    {sample['output']}")

    return train_data


# =====================================================================
# 第2步：SFT训练
# =====================================================================
def train_sft():
    """用LoRA对TinyLlama做SFT微调"""
    log("\n" + "=" * 60)
    log("[Step 2] SFT训练 - LoRA微调TinyLlama")
    log("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # 加载tokenizer和模型
    log(f"  加载基础模型: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    log(f"  模型加载成功")

    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    def load_sft_data(path):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                d = json.loads(line.strip())
                data.append({'instruction': d['instruction'], 'output': d['output']})
        return data

    train_data = load_sft_data(SFT_DATA_PATH)
    log(f"  训练数据: {len(train_data)} 条")

    # 使用chat template格式化
    def formatting_func(example):
        messages = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['output']},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    dataset = Dataset.from_list(train_data)

    # SFT配置
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_strategy="steps",
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        packing=False,
        max_length=1024,
    )

    # 创建trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # 训练
    log(f"  开始训练 (2 epochs, lr=2e-4, LoRA r=16)...")
    train_start = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start

    log(f"  训练完成！耗时: {train_elapsed/60:.1f}分钟")
    log(f"  训练loss: {train_result.training_loss:.4f}")

    # 保存LoRA adapter
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"  LoRA adapter保存: {OUTPUT_DIR}")

    # 保存训练指标
    metrics = {
        'train_loss': train_result.training_loss,
        'train_runtime': train_result.metrics.get('train_runtime', 0),
        'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0),
        'epochs': 2,
        'learning_rate': 2e-4,
        'lora_r': 16,
        'lora_alpha': 32,
        'train_data_size': len(train_data),
    }
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    return train_elapsed


# =====================================================================
# 第3步：合并LoRA权重
# =====================================================================
def merge_lora():
    """合并LoRA权重到基础模型"""
    log("\n" + "=" * 60)
    log("[Step 3] 合并LoRA权重")
    log("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log(f"  加载基础模型: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"  加载LoRA adapter: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)

    log(f"  合并权重...")
    model = model.merge_and_unload()

    log(f"  保存合并后模型: {MERGED_MODEL_PATH}")
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_PATH, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    log(f"  合并完成！")

    # 清理GPU
    del model
    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# 第4步：评估
# =====================================================================
def extract_answer_option(text):
    """从生成文本中精确提取答案选项（A/B/C/D）"""
    if not text or len(text) == 0:
        return ''

    text_upper = text.upper()

    patterns = [
        r'(?:FINAL\s+ANSWER|THE\s+ANSWER\s+IS|ANSWER|CORRECT\s+ANSWER)\s*[:\.\,]?\s*([A-D])\b',
        r'OPTION\s*([A-D])\s+IS\s+CORRECT',
        r'答案\s*[：:]\s*([A-D])',
    ]
    for p in patterns:
        m = re.search(p, text_upper)
        if m:
            return m.group(1)

    lines = text.strip().split('\n')
    for line in reversed(lines[-8:]):
        line = line.strip().rstrip('.。,:;')
        if len(line) == 1 and line in 'ABCD':
            return line
        m = re.match(r'^([A-D])[\.\)\:]\s*$', line)
        if m:
            return m.group(1)

    m = re.match(r'^([A-D])[\.\)\:\s]', text_upper.strip())
    if m:
        return m.group(1)

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
        # 尝试用chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        else:
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


def evaluate():
    """用微调后模型评估6种策略"""
    log("\n" + "=" * 60)
    log("[Step 4] 评估 - 用MedQA-SFT模型测试6种策略")
    log("=" * 60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "medqa_real.jsonl")

    log(f"  加载微调后模型: {MERGED_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    log(f"  模型加载成功")

    # 加载测试数据
    samples = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    test_samples = samples[:100]
    fewshot_pool = samples[100:120]
    log(f"  测试样本: {len(test_samples)} 条")

    strategies_results = {}

    # ---- 策略1: Zero-Shot ----
    log(f"\n  [1/6] zero_shot")
    zs_template = """You are an experienced medical doctor answering a USMLE multiple-choice question.

Question: {question}
Options:
{options}

Select the single best answer. End with "Answer: X"."""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt = zs_template.format(question=sample['question'], options=options_text)
        response = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.0)
        pred = extract_answer_option(response)
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'], 'response_excerpt': response[:120]})
        if (idx + 1) % 25 == 0:
            acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    {idx+1}/100, acc={acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['zero_shot'] = {'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': 100, 'correct': int(acc*100), 'sample_details': details}
    log(f"  ✅ zero_shot: acc={acc:.2%}, hal={hal:.2%}")

    # ---- 策略2: Few-Shot ----
    log(f"\n  [2/6] few_shot")
    ex1, ex2 = fewshot_pool[0], fewshot_pool[1]
    fs_template = """Here are example USMLE questions with correct answers:

Example 1:
Question: {ex1_q}
Options:
{ex1_opts}
Answer: {ex1_a}

Example 2:
Question: {ex2_q}
Options:
{ex2_opts}
Answer: {ex2_a}

Now answer this:

Question: {question}
Options:
{options}

Answer:"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        ex1_opts = "\n".join(f"{k}. {v}" for k, v in ex1['options'].items())
        ex2_opts = "\n".join(f"{k}. {v}" for k, v in ex2['options'].items())
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt = fs_template.format(ex1_q=ex1['question'], ex1_opts=ex1_opts, ex1_a=ex1['answer'],
            ex2_q=ex2['question'], ex2_opts=ex2_opts, ex2_a=ex2['answer'],
            question=sample['question'], options=options_text)
        response = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.1)
        pred = extract_answer_option(response)
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'], 'response_excerpt': response[:120]})
        if (idx + 1) % 25 == 0:
            acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    {idx+1}/100, acc={acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['few_shot'] = {'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': 100, 'correct': int(acc*100), 'sample_details': details}
    log(f"  ✅ few_shot: acc={acc:.2%}, hal={hal:.2%}")

    # ---- 策略3: CoT ----
    log(f"\n  [3/6] cot")
    cot_template = """You are an experienced medical doctor. Answer this USMLE question step by step.

Question: {question}
Options:
{options}

Step 1 - Key findings:
Step 2 - Best answer:"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt = cot_template.format(question=sample['question'], options=options_text)
        response = generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.2)
        pred = extract_answer_option(response)
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'], 'response_excerpt': response[:120]})
        if (idx + 1) % 25 == 0:
            acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    {idx+1}/100, acc={acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['cot'] = {'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': 100, 'correct': int(acc*100), 'sample_details': details}
    log(f"  ✅ cot: acc={acc:.2%}, hal={hal:.2%}")

    # ---- 策略4: Self-Consistency ----
    log(f"\n  [4/6] self_consistency")
    sc_template = """You are a medical doctor. Answer this USMLE question (attempt {n}).

Question: {question}
Options:
{options}

Answer:"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        path_preds = []
        for p in range(1, 6):
            prompt = sc_template.format(n=p, question=sample['question'], options=options_text)
            response = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.4+p*0.05)
            path_preds.append(extract_answer_option(response))
        valid = [p for p in path_preds if p in 'ABCD']
        pred = Counter(valid).most_common(1)[0][0] if valid else ''
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'], 'response_excerpt': f"[votes:{path_preds}]"})
        if (idx + 1) % 25 == 0:
            acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    {idx+1}/100, acc={acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['self_consistency'] = {'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': 100, 'correct': int(acc*100), 'sample_details': details}
    log(f"  ✅ self_consistency: acc={acc:.2%}, hal={hal:.2%}")

    # ---- 策略5: ToT v2 (分段生成) ----
    log(f"\n  [5/6] tot_v2")
    tot_s1 = """You are a medical doctor. Analyze each option for this USMLE question.

Question: {question}
Options:
{options}

Analyze each option (1 sentence each):
Option A:"""
    tot_s2 = """Based on your analysis, select the best answer.

Question: {question}
Options:
{options}

Analysis:
{analysis}

The best answer is option (write only the letter):"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt1 = tot_s1.format(question=sample['question'], options=options_text)
        analysis = generate(model, tokenizer, prompt1, max_new_tokens=250, temperature=0.1)
        prompt2 = tot_s2.format(question=sample['question'], options=options_text, analysis=analysis[:400])
        response = generate(model, tokenizer, prompt2, max_new_tokens=50, temperature=0.0)
        pred = extract_answer_option(response) or extract_answer_option(analysis)
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'], 'response_excerpt': (analysis+response)[:120]})
        if (idx + 1) % 25 == 0:
            acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    {idx+1}/100, acc={acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['tot_v2'] = {'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': 100, 'correct': int(acc*100), 'sample_details': details}
    log(f"  ✅ tot_v2: acc={acc:.2%}, hal={hal:.2%}")

    # ---- 策略6: CoT+Verifier ----
    log(f"\n  [6/6] cot_verifier")
    cot_gen = """You are a medical doctor. Answer this USMLE question.

Question: {question}
Options:
{options}

Brief reasoning, then "Answer: X":"""
    ver_template = """Verify this answer.

Question: {question}
Options:
{options}

Proposed: {proposed}

Correct answer (write only the letter):"""
    predictions, references, details = [], [], []
    for idx, sample in enumerate(test_samples):
        options_text = "\n".join(f"{k}. {v}" for k, v in sample['options'].items())
        prompt1 = cot_gen.format(question=sample['question'], options=options_text)
        cot_resp = generate(model, tokenizer, prompt1, max_new_tokens=200, temperature=0.2)
        initial = extract_answer_option(cot_resp)
        if initial:
            prompt2 = ver_template.format(question=sample['question'], options=options_text, proposed=initial)
            ver_resp = generate(model, tokenizer, prompt2, max_new_tokens=50, temperature=0.0)
            verified = extract_answer_option(ver_resp)
            pred = verified if verified else initial
        else:
            pred = initial
        predictions.append(pred)
        references.append(sample['answer'])
        details.append({'idx': idx, 'ref': sample['answer'], 'pred': pred,
                        'correct': pred == sample['answer'], 'response_excerpt': f"[{initial}->{pred}]"})
        if (idx + 1) % 25 == 0:
            acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
            log(f"    {idx+1}/100, acc={acc:.2%}")

    acc = sum(1 for p, r in zip(predictions, references) if p == r) / len(predictions)
    hal = sum(1 for p in predictions if p == '') / len(predictions)
    strategies_results['cot_verifier'] = {'accuracy': float(acc), 'hallucination_rate': float(hal),
        'num_samples': 100, 'correct': int(acc*100), 'sample_details': details}
    log(f"  ✅ cot_verifier: acc={acc:.2%}, hal={hal:.2%}")

    # 保存结果
    results_dir = os.path.join(PROJECT_ROOT, "results", "experiment_v3_sft")
    os.makedirs(results_dir, exist_ok=True)
    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'sample_details'} for k, v in strategies_results.items()}
    with open(os.path.join(results_dir, "metrics_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(results_dir, "metrics_full.json"), 'w', encoding='utf-8') as f:
        json.dump(strategies_results, f, ensure_ascii=False, indent=2)
    log(f"\n  结果保存: {results_dir}")

    # 打印排名
    log(f"\n{'='*60}")
    log(f"最终排名（MedQA-SFT模型）:")
    log(f"{'='*60}")
    for rank, (name, r) in enumerate(
        sorted(strategies_results.items(), key=lambda x: x[1]['accuracy'], reverse=True), 1):
        log(f"  第{rank}名: {name:20s} | acc={r['accuracy']:>6.2%} | "
            f"correct={r['correct']:2d}/{r['num_samples']} | hal={r['hallucination_rate']:.2%}")

    return strategies_results


# =====================================================================
# 主函数
# =====================================================================
def main():
    total_start = time.time()
    log("=" * 60)
    log("MedQA SFT微调 + 评估 完整流程")
    log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    # Step 1: 准备数据
    prepare_sft_data()

    # Step 2: SFT训练
    train_time = train_sft()

    # Step 3: 合并权重
    merge_lora()

    # Step 4: 评估
    results = evaluate()

    total_elapsed = time.time() - total_start
    log(f"\n{'='*60}")
    log(f"全部完成！总耗时: {total_elapsed/60:.1f}分钟")
    log(f"  - SFT训练: {train_time/60:.1f}分钟")
    log(f"  - 评估: {(total_elapsed-train_time)/60:.1f}分钟")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
