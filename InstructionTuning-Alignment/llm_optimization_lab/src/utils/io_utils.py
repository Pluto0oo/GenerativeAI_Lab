"""
I/O 工具函数模块

提供文件读写、数据持久化等实用工具函数。
"""

import json
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional


def ensure_dir(path: str) -> str:
    """
    确保目录存在，若不存在则创建。

    Args:
        path: 目录路径

    Returns:
        确认存在的目录路径
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, filepath: str, indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    将数据保存为 JSON 文件。

    Args:
        data: 要保存的数据
        filepath: 输出文件路径
        indent: 缩进空格数
        ensure_ascii: 是否使用 ASCII 编码
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_json(filepath: str) -> Any:
    """
    从 JSON 文件加载数据。

    Args:
        filepath: JSON 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_yaml(data: Dict, filepath: str) -> None:
    """
    将字典保存为 YAML 文件。

    Args:
        data: 要保存的字典数据
        filepath: 输出文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_yaml(filepath: str) -> Dict:
    """
    从 YAML 文件加载数据。

    Args:
        filepath: YAML 文件路径

    Returns:
        加载的字典数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    将对象保存为 Pickle 文件。

    Args:
        obj: 要保存的对象
        filepath: 输出文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    从 Pickle 文件加载对象。

    Args:
        filepath: Pickle 文件路径

    Returns:
        加载的对象
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def read_lines(filepath: str, skip_empty: bool = True) -> List[str]:
    """
    按行读取文件内容。

    Args:
        filepath: 文件路径
        skip_empty: 是否跳过空行

    Returns:
        行列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if skip_empty:
        lines = [line.strip() for line in lines if line.strip()]
    return lines


def write_lines(lines: List[str], filepath: str) -> None:
    """
    将字符串列表按行写入文件。

    Args:
        lines: 字符串列表
        filepath: 输出文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))