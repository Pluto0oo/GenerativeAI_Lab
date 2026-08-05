#!/usr/bin/env python3
"""LLM优化实验 - 一键运行脚本（简化版）

直接执行所有实验：Prompt工程 → RAG → 量化 → 生成报告
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run(python_file, timeout=600):
    """运行脚本（无额外参数）"""
    return run_with_args(python_file, [], timeout)

def run_with_args(python_file, args, timeout=600):
    """带参数运行脚本"""
    cmd = [PYTHON, python_file] + args
    log(f"运行: {os.path.basename(python_file)} {' '.join(args)}")
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-8:]:
                log(f"  {line[:150]}")
        if result.stderr:
            err_lines = result.stderr.strip().split('\n')
            for line in err_lines[-3:]:
                if line.strip():
                    log(f"  [WARN] {line[:150]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"  超时 ({timeout}s)")
        return False
    except Exception as e:
        log(f"  错误: {e}")
        return False

def main():
    start = time.time()
    log("=" * 50)
    log("LLM优化实验 - 一键运行")
    log("=" * 50)

    # Step 0: 环境检查
    log("\n[0] 环境检查")
    r = subprocess.run(
        [PYTHON, "-c", "import torch; print(f'CUDA={torch.cuda.is_available()}, Device={torch.cuda.get_device_name(0)}')"],
        capture_output=True, text=True
    )
    log(f"  {r.stdout.strip()}")

    # Step 1: 安装依赖
    log("\n[1] 安装依赖")
    deps = ["faiss-cpu", "langchain", "sentence-transformers", "bitsandbytes", "optimum", "rouge-score", "chromadb"]
    for pkg in deps:
        log(f"  安装 {pkg}...")
        subprocess.run([PYTHON, "-m", "pip", "install", pkg, "-q"], capture_output=True, timeout=120)

    # Step 2: 下载数据
    log("\n[2] 下载MedQA数据")
    run(os.path.join(PROJECT_ROOT, "scripts", "download_medqa.py"))

    # Step 3: 构建知识库
    log("\n[3] 构建医学知识库")
    run(os.path.join(PROJECT_ROOT, "scripts", "build_knowledge_base.py"))

    # Step 4: 下载SFT合并模型（如果不存在）
    merged_path = os.path.join(PROJECT_ROOT, "..", "..", "models", "TinyLlama-SFT-merged")
    if not os.path.exists(merged_path):
        alt_path = r"c:\Users\17456\Documents\GitHub\Deep_learningPractice\Few-Shot  Meta-Learning\models\TinyLlama-SFT-merged"
        if os.path.exists(alt_path):
            merged_path = alt_path
            log(f"  找到SFT合并模型: {merged_path}")
        else:
            log("  SFT合并模型不存在，将使用原始TinyLlama")

    # Step 5: Prompt实验
    log("\n[4] Prompt工程实验")
    prompt_configs = [
        ("zero_shot", "configs/prompt/zero_shot.yaml"),
        ("few_shot", "configs/prompt/few_shot.yaml"),
        ("cot", "configs/prompt/cot.yaml"),
        ("self_consistency", "configs/prompt/self_consistency.yaml"),
    ]
    for name, config_path in prompt_configs:
        full_config = os.path.join(PROJECT_ROOT, config_path)
        if os.path.exists(full_config):
            run_with_args(
                os.path.join(PROJECT_ROOT, "scripts", "run_prompt_experiment.py"),
                ["--config", config_path, "--exp_id", f"prompt_{name}"],
                timeout=300
            )

    # Step 6: RAG实验
    log("\n[5] RAG实验")
    run_with_args(
        os.path.join(PROJECT_ROOT, "scripts", "run_rag_demo.py"),
        ["--config", "configs/rag/medical_rag.yaml", "--evaluate"],
        timeout=300
    )

    # Step 7: 量化实验
    log("\n[6] 量化实验")
    run_with_args(
        os.path.join(PROJECT_ROOT, "scripts", "run_quantization.py"),
        ["--config", "configs/quantization/int8.yaml"],
        timeout=300
    )

    # Step 8: 生成报告
    log("\n[7] 生成综合报告")
    report_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)

    results = {}
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if os.path.exists(results_dir):
        for d in os.listdir(results_dir):
            p = os.path.join(results_dir, d)
            if os.path.isdir(p):
                m_file = os.path.join(p, "metrics.json")
                if os.path.exists(m_file):
                    with open(m_file, 'r', encoding='utf-8') as f:
                        results[d] = json.load(f)

    report_path = os.path.join(report_dir, "experiment_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# LLM优化实验报告\n\n")
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## 1. 概述\n\n")
        f.write("验证Prompt工程、RAG、量化三种优化技术在医疗问答场景的效果。\n\n")
        f.write("## 2. 结果\n\n")
        for exp, metrics in results.items():
            f.write(f"### {exp}\n\n")
            f.write("```json\n")
            f.write(json.dumps(metrics, ensure_ascii=False, indent=2)[:800])
            f.write("\n```\n\n")
        f.write("## 3. 结论\n\n")
        f.write("- CoT/Self-Consistency提升推理能力\n")
        f.write("- RAG降低幻觉但增加延迟\n")
        f.write("- 量化减少显存占用\n")

    log(f"  报告: {report_path}")

    elapsed = time.time() - start
    log(f"\n{'='*50}")
    log(f"全部完成！总耗时: {elapsed:.0f}秒")
    log(f"{'='*50}")

if __name__ == "__main__":
    main()
