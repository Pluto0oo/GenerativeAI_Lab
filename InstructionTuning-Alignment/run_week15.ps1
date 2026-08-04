# ============================================================
# Week 15: 指令微调与对齐技术 (SFT + DPO)
# 一键运行脚本（使用 dl-gpu 环境）
# ============================================================
# 流程：
#  Step 1: 构造数据集（情感分类→指令格式 + 医疗好/坏回答偏好对）
#  Step 2: SFT-only 实验 (TinyLlama + LoRA)
#  Step 3: SFT+DPO 实验 (TinyLlama + LoRA + 偏好对齐)
#  Step 4: 对比 SFT-only vs SFT+DPO
#  Step 5: 汇总所有结果
#  Step 6: 生成最终 Markdown 报告
# ============================================================

$ErrorActionPreference = "Continue"

# ---- 环境设置 ----
$PYTHON  = "C:\Users\17456\anaconda3\envs\dl-gpu\python.exe"
$PROJECT = "c:\Users\17456\Documents\GitHub\Deep_learningPractice\Few-Shot  Meta-Learning"
Set-Location $PROJECT

# 国内网络环境如需镜像，取消下一行注释
# $env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Week 15: 指令微调与对齐技术 (SFT + DPO)" -ForegroundColor Cyan
Write-Host "  Python  : $PYTHON" -ForegroundColor DarkGray
Write-Host "  GPU Env : dl-gpu (CUDA 12.8)" -ForegroundColor DarkGray
Write-Host "============================================================`n" -ForegroundColor Cyan

# ============ Step 1: 构造数据集 ============
Write-Host "[Step 1/6] 构造数据集：情感分类→指令格式 & 医疗好/坏回答偏好对" -ForegroundColor Yellow
& $PYTHON scripts\build_week15_datasets.py
if ($LASTEXITCODE -ne 0) { Write-Host "    ! 数据集构造失败!" -ForegroundColor Red; exit 1 }

# ============ Step 2: SFT-only 实验 ============
Write-Host "`n[Step 2/6] SFT-only 实验: TinyLlama-1.1B + LoRA (情感+医疗指令微调)" -ForegroundColor Yellow
& $PYTHON scripts\run_experiment.py --config configs\experiment\week15_sft_tinyllama.yaml --exp_id week15_sft_only
if ($LASTEXITCODE -ne 0) {
    Write-Host "    ! SFT实验失败，尝试 tiny 配置兜底..." -ForegroundColor Red
    & $PYTHON scripts\run_experiment.py --config configs\experiment\sft_tiny.yaml --exp_id week15_sft_tiny_fallback
}

# ============ Step 3: SFT+DPO 实验 ============
Write-Host "`n[Step 3/6] SFT+DPO 实验: 基于SFT基座 + DPO偏好对齐" -ForegroundColor Yellow
& $PYTHON scripts\run_experiment.py --config configs\experiment\week15_dpo_tinyllama.yaml --exp_id week15_sft_dpo
if ($LASTEXITCODE -ne 0) {
    Write-Host "    ! DPO实验失败，尝试 tiny 配置兜底..." -ForegroundColor Red
    & $PYTHON scripts\run_experiment.py --config configs\experiment\dpo_tiny.yaml --exp_id week15_dpo_tiny_fallback
}

# ============ Step 4: 对比实验 SFT-only vs SFT+DPO ============
Write-Host "`n[Step 4/6] 对比实验：生成 SFT-only vs SFT+DPO 对比报告" -ForegroundColor Yellow
$all_cfgs = @(
    "configs\experiment\week15_sft_tinyllama.yaml",
    "configs\experiment\week15_dpo_tinyllama.yaml"
)
& $PYTHON scripts\run_comparison.py --configs $all_cfgs --output_dir results\week15_comparison
if ($LASTEXITCODE -ne 0) { Write-Host "    ! 对比脚本运行失败，继续下一步" -ForegroundColor Red }

# ============ Step 5: 汇总所有结果 ============
Write-Host "`n[Step 5/6] 汇总所有实验结果" -ForegroundColor Yellow
& $PYTHON scripts\aggregate_results.py --results_dir results --output reports
if ($LASTEXITCODE -ne 0) { Write-Host "    ! 汇总脚本运行失败，继续下一步" -ForegroundColor Red }

# ============ Step 6: 生成最终报告 ============
Write-Host "`n[Step 6/6] 生成最终实验报告 (Markdown)" -ForegroundColor Yellow
& $PYTHON scripts\generate_report.py --results_dir results --output reports\week15_final_report.md
if ($LASTEXITCODE -ne 0) { Write-Host "    ! 报告生成失败" -ForegroundColor Red }

# ============================================================
# 全部完成
# ============================================================
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  Week 15 完整实验流程结束！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green

Write-Host "`n主要产物：" -ForegroundColor Cyan
Write-Host "  [SFT结果]       results\week15_sft_only\"
Write-Host "  [SFT+DPO结果]   results\week15_sft_dpo\"
Write-Host "  [对比报告]     results\week15_comparison\comparison_report.md"
Write-Host "  [结果汇总]     reports\aggregate_stats_*.json"
Write-Host "  [最终报告]     reports\week15_final_report.md"
Write-Host "  [日志文件]     logs\ 目录下 (week15_*.log)"

Write-Host "`n打开报告：" -ForegroundColor Cyan
Write-Host "  notepad reports\week15_final_report.md"
Write-Host "  notepad results\week15_comparison\comparison_report.md"
Write-Host ""
