import yaml
import os
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if 'defaults' in config:
        for default_config in config['defaults']:
            default_path = os.path.join(os.path.dirname(config_path), f"{default_config}.yaml")
            if os.path.exists(default_path):
                default_config_dict = load_config(default_path)
                config = merge_configs(default_config_dict, config)
        config.pop('defaults')
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def generate_exp_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
