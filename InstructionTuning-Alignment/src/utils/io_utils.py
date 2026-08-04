import os
import json
import yaml
import csv
import pandas as pd
from datetime import datetime

def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: dict, save_path: str):
    """保存配置到YAML文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def save_metrics_csv(metrics_history: list, save_path: str):
    """保存训练过程中的metrics到CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not metrics_history:
        return
    
    df = pd.DataFrame(metrics_history)
    df.to_csv(save_path, index=False)

def save_metrics_json(metrics: dict, save_path: str):
    """保存最终metrics到JSON"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 添加时间戳
    metrics['saved_at'] = datetime.now().isoformat()
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def load_metrics_json(load_path: str) -> dict:
    """加载metrics JSON文件"""
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_summary_md(metrics: dict, config: dict, save_path: str):
    """生成Markdown格式的实验摘要"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    summary = f"""# 实验摘要: {config['experiment']['name']}

## 基本信息
- **实验ID**: {config.get('exp_id', 'N/A')}
- **完成时间**: {metrics.get('saved_at', 'N/A')}
- **训练方法**: {config['training']['method']}
- **模型**: {config['model']['name']}

## 配置参数
| 参数 | 值 |
|------|-----|
| 学习率 | {config['training']['learning_rate']} |
| Batch Size | {config['training']['batch_size']} |
| Epochs | {config['training']['epochs']} |
| LoRA | {'启用' if config['model']['lora']['enabled'] else '禁用'} |

## 最终指标
| 指标 | 值 |
|------|-----|
"""
    
    # 添加metrics
    for key, value in metrics.get('metrics', {}).items():
        if isinstance(value, dict):
            summary += f"| {key} | {value.get('value', 'N/A')} |\n"
        else:
            summary += f"| {key} | {value} |\n"
    
    summary += f"""
## 硬件信息
- **GPU**: {metrics.get('hardware', {}).get('gpu_model', 'N/A')}
- **CUDA版本**: {metrics.get('hardware', {}).get('cuda_version', 'N/A')}

## 备注
- 本次实验由自动化Pipeline生成
- 详细日志请查看 logs/ 目录
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(summary)

def create_exp_dirs(base_dir: str, exp_id: str) -> dict:
    """创建实验结果目录结构"""
    paths = {
        'root': os.path.join(base_dir, exp_id),
        'plots': os.path.join(base_dir, exp_id, 'plots'),
        'checkpoints': os.path.join(base_dir, exp_id, 'checkpoints'),
        'repeats': os.path.join(base_dir, exp_id, 'repeats'),
        'stats': os.path.join(base_dir, exp_id, 'repeats', 'stats'),
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    return paths
