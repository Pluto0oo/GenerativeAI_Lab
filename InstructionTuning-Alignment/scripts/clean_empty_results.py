#!/usr/bin/env python3
"""清理 results/ 下的历史空目录（无任何文件的目录）

汇总报告之前把 results/ 下 76 个目录全当实验，其中 74 个是空目录显示 N/A。
本脚本只删除"完全没有任何文件"的目录（递归检查），保留有实际数据的目录。

使用方法:
    python scripts/clean_empty_results.py
"""
import os
import shutil


def main():
    results_dir = "results"
    if not os.path.isdir(results_dir):
        print("results 目录不存在")
        return

    removed = []
    kept = []

    for name in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, name)
        if not os.path.isdir(path):
            continue
        # 递归检查目录下是否有任何文件
        has_files = any(
            os.path.isfile(os.path.join(r, f))
            for r, d, fs in os.walk(path)
            for f in fs
        )
        if not has_files:
            shutil.rmtree(path)
            removed.append(name)
        else:
            kept.append(name)

    print(f"删除空目录: {len(removed)} 个")
    for n in removed:
        print(f"  - {n}")
    print(f"\n保留有效目录: {len(kept)} 个")
    for n in kept:
        print(f"  + {n}")


if __name__ == "__main__":
    main()
