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
    
    # Get latest epoch
    latest = df.iloc[-1]
    
    lines = []
    lines.append(f"Epoch: {latest.get('epoch', 'N/A')}")
    
    # Training metrics
    if 'train_loss' in latest:
        lines.append(f"Train Loss: {latest['train_loss']:.4f}")
    if 'train_qwk' in latest:
        lines.append(f"Train QWK: {latest['train_qwk']:.4f}")
    if 'train_acc' in latest:
        lines.append(f"Train Acc: {latest['train_acc']:.4f}")
    
    # Validation metrics
    if 'val_loss' in latest:
        lines.append(f"Val Loss: {latest['val_loss']:.4f}")
    if 'val_qwk' in latest:
        lines.append(f"Val QWK: {latest['val_qwk']:.4f}")
    if 'val_acc' in latest:
        lines.append(f"Val Acc: {latest['val_acc']:.4f}")
    if 'val_f1' in latest:
        lines.append(f"Val F1: {latest['val_f1']:.4f}")
    
    # Learning rate
    if 'lr-AdamW' in latest:
        lines.append(f"LR: {latest['lr-AdamW']:.6f}")
    
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

