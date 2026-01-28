#!/usr/bin/env python3
"""
Monitor training progress by reading Lightning logs.

This script shows real-time training progress by monitoring
Lightning log files and metrics.
"""

import argparse
import time
from pathlib import Path
import yaml
import pandas as pd
import json
import numpy as np


def find_latest_training():
    """Find the latest training run."""
    lightning_logs = Path('lightning_logs')
    if not lightning_logs.exists():
        return None
    
    versions = sorted(
        [d for d in lightning_logs.iterdir() if d.is_dir() and d.name.startswith('version')],
        key=lambda x: int(x.name.split('_')[1]) if '_' in x.name else 0,
        reverse=True
    )
    
    return versions[0] if versions else None


def read_metrics(version_dir):
    """Read metrics from CSV file."""
    metrics_file = version_dir / 'metrics.csv'
    if metrics_file.exists():
        try:
            df = pd.read_csv(metrics_file)
            return df
        except:
            return None
    return None


def read_hparams(version_dir):
    """Read hyperparameters."""
    hparams_file = version_dir / 'hparams.yaml'
    if hparams_file.exists():
        try:
            with open(hparams_file) as f:
                return yaml.safe_load(f)
        except:
            return None
    return None


def format_metrics(df):
    """Format metrics for display."""
    if df is None or len(df) == 0:
        return "No metrics available yet."
    
    # Combine information from all rows (Lightning logs LR and epoch separately)
    # Get the latest non-null values for each metric
    latest_info = {}
    
    # Find latest epoch
    epoch_rows = df[df['epoch'].notna()]
    if len(epoch_rows) > 0:
        latest_info['epoch'] = epoch_rows.iloc[-1]['epoch']
    
    # Find latest step
    step_rows = df[df['step'].notna()]
    if len(step_rows) > 0:
        latest_info['step'] = int(step_rows.iloc[-1]['step'])
    
    # Find latest LR
    lr_rows = df[df['lr-AdamW'].notna()]
    if len(lr_rows) > 0:
        latest_info['lr'] = lr_rows.iloc[-1]['lr-AdamW']
    
    # Find latest train loss
    loss_rows = df[df['train_loss_step'].notna()]
    if len(loss_rows) > 0:
        latest_info['train_loss'] = loss_rows.iloc[-1]['train_loss_step']
    
    # Check for epoch-level metrics (these appear after validation)
    epoch_level_cols = [col for col in df.columns if col.startswith('train_') or col.startswith('val_')]
    for col in epoch_level_cols:
        col_rows = df[df[col].notna()]
        if len(col_rows) > 0:
            latest_info[col] = col_rows.iloc[-1][col]
    
    lines = []
    
    # Epoch and step
    if 'epoch' in latest_info:
        epoch_val = latest_info['epoch']
        if pd.isna(epoch_val) or (isinstance(epoch_val, float) and np.isnan(epoch_val)):
            lines.append(f"Epoch: Training...")
        else:
            lines.append(f"Epoch: {int(epoch_val)}")
    else:
        lines.append(f"Epoch: Starting...")
    
    if 'step' in latest_info:
        lines.append(f"Step: {latest_info['step']}")
    
    # Learning rate
    if 'lr' in latest_info:
        lines.append(f"LR: {latest_info['lr']:.6f}")
    else:
        lines.append(f"LR: Initializing...")
    
    # Training metrics
    if 'train_loss' in latest_info:
        lines.append(f"Train Loss: {latest_info['train_loss']:.4f}")
    
    if 'train_qwk' in latest_info:
        lines.append(f"Train QWK: {latest_info['train_qwk']:.4f}")
    if 'train_acc' in latest_info:
        lines.append(f"Train Acc: {latest_info['train_acc']:.4f}")
    if 'train_f1' in latest_info:
        lines.append(f"Train F1: {latest_info['train_f1']:.4f}")
    
    # Validation metrics
    if 'val_loss' in latest_info:
        lines.append(f"Val Loss: {latest_info['val_loss']:.4f}")
    if 'val_qwk' in latest_info:
        lines.append(f"Val QWK: {latest_info['val_qwk']:.4f}")
    if 'val_acc' in latest_info:
        lines.append(f"Val Acc: {latest_info['val_acc']:.4f}")
    if 'val_f1' in latest_info:
        lines.append(f"Val F1: {latest_info['val_f1']:.4f}")
    
    return "\n".join(lines)


def monitor_training(version_dir=None, interval=10):
    """Monitor training progress."""
    if version_dir is None:
        version_dir = find_latest_training()
    
    if version_dir is None:
        print("No training runs found!")
        return
    
    print(f"Monitoring: {version_dir}")
    print("Press Ctrl+C to stop\n")
    
    hparams = read_hparams(version_dir)
    if hparams:
        print(f"Model: {hparams.get('model_name', 'Unknown')}")
        print(f"Image Size: {hparams.get('img_size', 'Unknown')}")
        print(f"Learning Rate: {hparams.get('lr', 'Unknown')}")
        print()
    
    try:
        while True:
            metrics = read_metrics(version_dir)
            if metrics is not None:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end='')
                print(f"Training Progress - {version_dir.name}")
                print("="*60)
                print(format_metrics(metrics))
                print("\n" + "="*60)
                print(f"Refreshing every {interval} seconds... (Ctrl+C to stop)")
            else:
                print("Waiting for metrics...")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    ap = argparse.ArgumentParser(description="Monitor training progress")
    ap.add_argument("--version", type=int, help="Version number to monitor (default: latest)")
    ap.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
    
    args = ap.parse_args()
    
    version_dir = None
    if args.version is not None:
        version_dir = Path(f'lightning_logs/version_{args.version}')
        if not version_dir.exists():
            print(f"Version {args.version} not found!")
            return
    
    monitor_training(version_dir, args.interval)


if __name__ == "__main__":
    main()

