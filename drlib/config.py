"""
Configuration management for DR classification experiments.

This module provides utilities to load and manage YAML configuration files
for reproducible experiments.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
    """
    Convert configuration dictionary to argparse Namespace.
    
    This allows using config files with existing argparse-based scripts.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        argparse.Namespace object
    """
    args = argparse.Namespace()
    
    # Data arguments
    if 'data' in config:
        args.fold_csv = config['data'].get('fold_csv')
        args.img_size = config['data'].get('img_size', 512)
        args.batch_size = config['data'].get('batch_size', 16)
        args.num_workers = config['data'].get('num_workers', 0)
        args.remove_borders = config['data'].get('remove_borders', True)
    
    # Model arguments
    if 'model' in config:
        args.model = config['model'].get('name', 'efficientnet_b3')
        args.freeze_backbone = config['model'].get('freeze_backbone', False)
        args.unfreeze_epoch = config['model'].get('unfreeze_epoch')
    
    # Training arguments
    if 'training' in config:
        args.epochs = config['training'].get('epochs', 30)
        args.lr = config['training'].get('lr', 1e-4)
        args.weight_decay = config['training'].get('weight_decay', 1e-4)
        args.lr_scheduler = config['training'].get('lr_scheduler', 'cosine')
        args.warmup_epochs = config['training'].get('warmup_epochs', 0)
        
        # Loss arguments
        if 'loss' in config['training']:
            loss_cfg = config['training']['loss']
            args.loss = loss_cfg.get('type', 'ce')
            args.use_class_weights = loss_cfg.get('use_class_weights', False)
            args.focal_gamma = loss_cfg.get('focal_gamma', 2.0)
            args.label_smoothing = loss_cfg.get('label_smoothing', 0.1)
        else:
            args.loss = 'ce'
            args.use_class_weights = False
    
    # Callbacks
    if 'callbacks' in config:
        args.monitor = config['callbacks'].get('monitor', 'val_qwk')
        args.patience = config['callbacks'].get('patience', 10)
    
    # Other
    args.seed = config.get('seed', 42)
    args.device = config.get('device', 'auto')
    
    return args


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base values
    
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

