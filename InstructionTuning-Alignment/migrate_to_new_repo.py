"""迁移两个生成式AI项目到新仓库 GenerativeAI-Lab

排除规则：
- 目录: .git, __pycache__, .ipynb_checkpoints, models, logs, node_modules, .venv
- 扩展名: .safetensors, .pth, .bin, .ckpt, .npy, .pt (模型权重/大数组)
- 文件大小: > 50MB
- 保留: 代码、配置、报告、图表、小数据文件、metrics
"""
import os
import shutil

SRC_BASE = r"c:\Users\17456\Documents\GitHub\Deep_learningPractice"
DST_BASE = r"c:\Users\17456\Documents\GitHub\GenerativeAI-Lab"

# 源 -> 目标子目录名
PROJECTS = {
    "Few-Shot  Meta-Learning": "InstructionTuning-Alignment",
    "GenerativeAI": "GenerativeAI",
}

# 排除的目录名
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints", "models",
    "logs", "node_modules", ".venv", "wandb", ".trae",
    "data", ".pytest_cache", "tmp_dpo_test", "exercise_plots",
    "few_shot_plots", "few_shot_real_plots", "linear_attention_results",
    "transformer_outputs",
}

# 排除的文件扩展名（大文件/模型权重）
EXCLUDE_EXT = {".safetensors", ".pth", ".bin", ".ckpt", ".npy", ".pt"}

# 单文件大小上限 (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


def should_exclude_dir(dirname):
    return dirname in EXCLUDE_DIRS


def should_exclude_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in EXCLUDE_EXT


def copy_project(src, dst, project_label):
    """复制一个项目，返回 (复制数, 排除数, 排除的大文件列表)"""
    copied = 0
    excluded = 0
    big_files = []

    for root, dirs, files in os.walk(src):
        # 过滤排除目录
        dirs[:] = [d for d in dirs if not should_exclude_dir(d)]

        for f in files:
            src_file = os.path.join(root, f)

            if should_exclude_file(f):
                excluded += 1
                continue

            try:
                size = os.path.getsize(src_file)
            except OSError:
                continue

            if size > MAX_FILE_SIZE:
                rel = os.path.relpath(src_file, src)
                big_files.append(f"{size/1024/1024:.1f}MB  {rel}")
                excluded += 1
                continue

            # 计算目标路径
            rel_path = os.path.relpath(root, src)
            dst_dir = os.path.join(dst, rel_path) if rel_path != "." else dst
            dst_file = os.path.join(dst_dir, f)

            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied += 1

    print(f"\n[{project_label}]")
    print(f"  复制: {copied} 个文件")
    print(f"  排除: {excluded} 个文件")
    if big_files:
        print(f"  排除的大文件 (>50MB):")
        for bf in big_files:
            print(f"    - {bf}")
    return copied, excluded, big_files


def create_root_readme():
    """创建根目录 README"""
    readme = """# GenerativeAI-Lab

生成式人工智能实验项目集合，包含两个子项目：

## 项目结构

| 目录 | 项目 | 说明 |
|------|------|------|
| [InstructionTuning-Alignment](./InstructionTuning-Alignment/) | 指令微调与对齐（SFT/DPO） | 基于 TinyLlama 的医疗问答指令微调 + DPO 偏好对齐 |
| [GenerativeAI](./GenerativeAI/) | 生成式AI（GAN/Diffusion） | 基于 CIFAR-10 的 GAN 与 Diffusion 图像生成 |

## 环境

- Python 3.10 + PyTorch 2.x (CUDA 12.8)
- GPU: NVIDIA GeForce RTX 5060 Laptop
- 依赖: 见各子项目 requirements.txt

## 详细说明

各子项目均有独立的 README 和实验报告，请进入对应目录查看。
"""
    with open(os.path.join(DST_BASE, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def create_gitignore():
    """创建 .gitignore"""
    content = """# Python
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/

# 模型权重（大文件，不纳入版本控制）
*.safetensors
*.pth
*.bin
*.ckpt
*.pt
*.npy
models/

# 日志
logs/
*.log

# 数据（按需取消注释）
# data/processed/

# 环境
.venv/
.env
wandb/

# 系统
.DS_Store
Thumbs.db
"""
    with open(os.path.join(DST_BASE, ".gitignore"), "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("=" * 60)
    print("迁移生成式AI项目到新仓库 GenerativeAI-Lab")
    print("=" * 60)

    # 清理/创建目标目录
    if os.path.exists(DST_BASE):
        print(f"\n清理已有目标目录: {DST_BASE}")
        shutil.rmtree(DST_BASE)
    os.makedirs(DST_BASE)

    total_copied = 0
    total_excluded = 0
    all_big = []

    for src_sub, dst_sub in PROJECTS.items():
        src = os.path.join(SRC_BASE, src_sub)
        dst = os.path.join(DST_BASE, dst_sub)
        if not os.path.exists(src):
            print(f"\n[跳过] 源目录不存在: {src}")
            continue
        c, e, big = copy_project(src, dst, dst_sub)
        total_copied += c
        total_excluded += e
        all_big.extend(big)

    # 创建根目录文件
    create_root_readme()
    create_gitignore()
    print(f"\n[根目录] 创建 README.md + .gitignore")

    print(f"\n{'=' * 60}")
    print(f"迁移完成!")
    print(f"  总复制: {total_copied} 个文件")
    print(f"  总排除: {total_excluded} 个文件")
    print(f"  目标目录: {DST_BASE}")
    print(f"\n下一步: git init && git add . && git commit")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
