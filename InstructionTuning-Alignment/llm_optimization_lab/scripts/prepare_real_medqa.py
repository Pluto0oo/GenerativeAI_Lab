#!/usr/bin/env python3
"""解析真实MedQA数据集，构造标准评估格式

从medical_meadow_medqa.json中提取问题、选项、答案，
生成标准的{question, options, answer}格式评估集。
"""
import os
import json
import re
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "medqa", "medical_meadow_medqa.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "medqa_real.jsonl")


def parse_medqa_sample(item):
    """解析单条MedQA样本"""
    input_text = item.get('input', '')
    output_text = item.get('output', '')

    # 提取问题部分（Q: 到 ?\n 之间）
    # 格式: Q:问题内容? \n{'A': '...', 'B': '...', ...},
    q_match = re.match(r'Q:(.+?)\?\s*\n\{(.+?)\}', input_text, re.DOTALL)
    if not q_match:
        return None

    question = q_match.group(1).strip()
    options_str = '{' + q_match.group(2) + '}'

    # 解析选项字典
    try:
        # 安全地解析字典字符串
        options_str = options_str.replace("'", '"')
        options_dict = json.loads(options_str)
    except Exception:
        # 用正则提取
        options_dict = {}
        for m in re.finditer(r"'([A-E])':\s*'([^']+)'", q_match.group(2)):
            options_dict[m.group(1)] = m.group(2)

    if not options_dict:
        return None

    # 提取答案字母
    ans_match = re.match(r'^([A-E])', output_text.strip())
    if not ans_match:
        return None
    answer = ans_match.group(1)

    # 只保留答案在选项中的
    if answer not in options_dict:
        return None

    return {
        'question': question,
        'options': options_dict,
        'answer': answer,
    }


def main():
    print(f"加载原始数据: {RAW_PATH}")
    with open(RAW_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"  共 {len(raw_data)} 条原始样本")

    # 解析所有样本
    parsed = []
    for item in raw_data:
        p = parse_medqa_sample(item)
        if p:
            parsed.append(p)

    print(f"  成功解析: {len(parsed)} 条")

    # 统计选项数量分布
    opt_counts = {}
    for p in parsed:
        n = len(p['options'])
        opt_counts[n] = opt_counts.get(n, 0) + 1
    print(f"  选项数量分布: {opt_counts}")

    # 筛选4选项(A-D)的题目，优先用这些
    four_opt = [p for p in parsed if set(p['options'].keys()) == {'A', 'B', 'C', 'D'}]
    five_opt = [p for p in parsed if set(p['options'].keys()) == {'A', 'B', 'C', 'D', 'E'}]
    print(f"  4选项题目: {len(four_opt)} 条")
    print(f"  5选项题目: {len(five_opt)} 条")

    # 如果4选项不够100条，从5选项中取前4个（且答案在前4个中）
    selected = list(four_opt)
    if len(selected) < 120:
        need = 120 - len(selected)
        for p in five_opt:
            if p['answer'] in ['A', 'B', 'C', 'D']:
                # 只保留前4个选项
                p['options'] = {k: v for k, v in p['options'].items() if k in ['A', 'B', 'C', 'D']}
                selected.append(p)
                if len(selected) >= 120:
                    break

    # 打乱并取120条（100条测试 + 20条few-shot示例池）
    random.seed(42)
    random.shuffle(selected)
    selected = selected[:120]

    print(f"  最终选取: {len(selected)} 条")

    # 答案分布
    ans_dist = {}
    for p in selected:
        ans_dist[p['answer']] = ans_dist.get(p['answer'], 0) + 1
    print(f"  答案分布: {ans_dist}")

    # 保存
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for i, p in enumerate(selected):
            p['id'] = i
            p['prompt'] = f"Question: {p['question']}\nOptions:\n" + \
                          "\n".join(f"{k}. {v}" for k, v in p['options'].items()) + \
                          f"\nAnswer:"
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\n保存到: {OUTPUT_PATH}")
    print(f"共 {len(selected)} 条真实医疗选择题")

    # 打印前3条样本
    print("\n--- 前3条样本 ---")
    for p in selected[:3]:
        print(f"\nQ: {p['question'][:100]}...")
        for k, v in p['options'].items():
            print(f"  {k}. {v[:60]}")
        print(f"  答案: {p['answer']}")


if __name__ == "__main__":
    main()
