"""
项目配置加载工具
统一读取 config/default.json，并支持用自定义配置覆盖默认值。
"""

import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config' / 'default.json'

DEFAULT_CONFIG = {
    'data': {
        'source': 'simulated',
        'normal_path': 'data/normal.csv',
        'fault4_path': 'data/fault4.csv',
        'fault11_path': 'data/fault11.csv',
        'simulated_samples': 12000,
        'simulated_seed': 42,
    },
    'pipeline': {
        'variables': 22,
        'window_size': 150,
        'step_size': 1,
        'corr_threshold': 0.2,
        'lookback': 12,
        'wavelet': 'db4',
        'wavelet_level': 3,
        'train_size': 6000,
        'test_size': 3000,
        'risk_levels': [
            [0.0, 0.3, 'Low', '低风险'],
            [0.3, 0.6, 'Lower', '较低风险'],
            [0.6, 0.9, 'Higher', '较高风险'],
            [0.9, 1.0, 'High', '高风险'],
        ],
    },
    'training': {
        'seed': 42,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'hidden_size': 128,
        'num_layers': 2,
        'attention_dim': 64,
        'val_ratio': 0.1,
        'patience': 10,
        'device': 'auto',
    },
    'output': {
        'results_dir': 'results',
        'figures_dir': 'figures',
        'save_weights': True,
    },
}


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_project_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_config_path(config_path=None):
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    return resolve_project_path(config_path)


def load_project_config(config_path=None):
    resolved_path = resolve_config_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {resolved_path}")

    with resolved_path.open('r', encoding='utf-8') as f:
        user_config = json.load(f)

    merged_config = _deep_merge(DEFAULT_CONFIG, user_config)
    return merged_config, resolved_path
