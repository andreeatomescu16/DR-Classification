#!/usr/bin/env python3
"""
Benchmarking script for comparing different architectures.

Trains and evaluates multiple models with consistent settings:
- EfficientNet-B2 (384x384)
- EfficientNet-B4 (384x384)
- ViT-B/16 (384x384)

All models use:
- Weighted cross-entropy loss
- Macro F1 + Kappa evaluation
- Grad-CAM visualizations for 20 random validation samples
"""

import argparse
import sys
import subprocess
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from drlib.metrics import compute_all_metrics, get_confusion_matrix
from scripts.evaluate import load_model, evaluate_model
from drlib.datasets import DRDataset
from drlib.transforms import get_val_tf
from torch.utils.data import DataLoader


MODELS_TO_BENCHMARK = [
    {
        'name': 'efficientnet_b2.ra_in1k',
        'img_size': 384,
        'display_name': 'EfficientNet-B2'
    },
    {
        'name': 'efficientnet_b4.ra2_in1k',
        'img_size': 384,
        'display_name': 'EfficientNet-B4'
    },
    {
        'name': 'vit_base_patch16_224.augreg_in1k',
        'img_size': 224,  # ViT-B/16 is trained on 224x224
        'display_name': 'ViT-B/16'
    }
]


def train_model(model_config, fold_csv, epochs=30, batch_size=16, device='auto', log_file=None):
    """Train a single model with real-time progress output."""
    print(f"\n{'='*60}")
    print(f"Training {model_config['display_name']}")
    print(f"{'='*60}")
    
    model_name = model_config['name']
    img_size = model_config['img_size']
    
    cmd = [
        sys.executable, '-m', 'drlib.train',
        '--fold_csv', str(fold_csv),
        '--model', model_name,
        '--img_size', str(img_size),
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', '1e-4',
        '--loss', 'weighted_ce',
        '--use_class_weights',
        '--lr_scheduler', 'cosine',
        '--patience', '10',
        '--monitor', 'val_qwk',
        '--seed', '42'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"Progress will be shown in real-time. Checkpoints are saved automatically.\n")
    
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output in real-time
    if log_file:
        log_file = open(log_file, 'w')
    
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            if log_file:
                log_file.write(line)
                log_file.flush()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        process.terminate()
        if log_file:
            log_file.close()
        return None
    
    process.wait()
    
    if log_file:
        log_file.close()
    
    if process.returncode != 0:
        print(f"\nError training {model_config['display_name']} (return code: {process.returncode})")
        return None
    
    # Find the best checkpoint
    lightning_logs = Path('lightning_logs')
    versions = sorted([d for d in lightning_logs.iterdir() if d.is_dir() and d.name.startswith('version')], 
                      key=lambda x: int(x.name.split('_')[1]) if '_' in x.name else 0, reverse=True)
    
    if not versions:
        print(f"No checkpoints found for {model_config['display_name']}")
        return None
    
    latest_version = versions[0]
    checkpoints_dir = latest_version / 'checkpoints'
    
    if not checkpoints_dir.exists():
        print(f"No checkpoints directory found in {latest_version}")
        return None
    
    checkpoints = list(checkpoints_dir.glob('*.ckpt'))
    if not checkpoints:
        print(f"No checkpoint files found in {checkpoints_dir}")
        return None
    
    # Get the best checkpoint (usually named with metrics)
    best_ckpt = None
    for ckpt in checkpoints:
        if 'best' in ckpt.name.lower() or 'last' in ckpt.name.lower():
            best_ckpt = ckpt
            break
    
    if best_ckpt is None:
        best_ckpt = checkpoints[0]
    
    print(f"Best checkpoint: {best_ckpt}")
    return best_ckpt


def evaluate_model_for_benchmark(checkpoint_path, data_csv, split='val', img_size=384, 
                                  batch_size=32, device='auto', n_gradcam=20):
    """Evaluate a model and generate metrics."""
    print(f"\nEvaluating model: {checkpoint_path}")
    
    # Set device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Load model
    model, hparams = load_model(checkpoint_path, device=device)
    
    # Create dataset and dataloader
    dataset = DRDataset(
        data_csv,
        split=split,
        tfm=get_val_tf(img_size, remove_borders=True)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device != 'cpu' else False
    )
    
    # Evaluate
    results = evaluate_model(model, dataloader, device=device)
    
    # Compute metrics
    metrics = compute_all_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities'],
        n_classes=5
    )
    
    # Generate confusion matrix
    cm = get_confusion_matrix(
        results['labels'],
        results['predictions'],
        normalize=True
    )
    
    # Generate Grad-CAM visualizations
    gradcam_path = None
    if n_gradcam > 0:
        try:
            from drlib.explainability import visualize_predictions, find_target_layer
            
            # Get random samples
            n_samples = min(n_gradcam, len(dataset))
            sample_indices = np.random.choice(len(dataset), n_samples, replace=False)
            sample_images = []
            sample_labels = []
            
            for idx in sample_indices:
                img, label = dataset[idx]
                sample_images.append(img)
                sample_labels.append(label)
            
            sample_images = torch.stack(sample_images)
            sample_labels = torch.tensor(sample_labels)
            
            # Find target layer
            target_layer = find_target_layer(model, hparams.get('model_name', 'efficientnet'))
            
            # Generate visualizations
            gradcam_dir = Path('benchmark_results') / 'gradcam'
            gradcam_dir.mkdir(parents=True, exist_ok=True)
            gradcam_path = gradcam_dir / f"{Path(checkpoint_path).stem}_gradcam.png"
            
            visualize_predictions(
                model,
                sample_images,
                sample_labels,
                class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                target_layer=target_layer,
                device=device,
                save_path=str(gradcam_path)
            )
            print(f"Saved Grad-CAM visualizations to {gradcam_path}")
        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM: {e}")
            gradcam_path = None
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'checkpoint_path': str(checkpoint_path),
        'gradcam_path': str(gradcam_path) if gradcam_path else None
    }


def create_results_table(results_dict, save_path):
    """Create a results comparison table."""
    rows = []
    for model_name, result in results_dict.items():
        metrics = result['metrics']
        rows.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'QWK': f"{metrics['qwk']:.4f}",
            'Macro F1': f"{metrics['macro_f1']:.4f}",
            'ROC-AUC (OVR)': f"{metrics.get('roc_auc_ovr', 'N/A')}",
            'Precision (Class 0)': f"{metrics['precision_class_0']:.4f}",
            'Precision (Class 1)': f"{metrics['precision_class_1']:.4f}",
            'Precision (Class 2)': f"{metrics['precision_class_2']:.4f}",
            'Precision (Class 3)': f"{metrics['precision_class_3']:.4f}",
            'Precision (Class 4)': f"{metrics['precision_class_4']:.4f}",
            'Recall (Class 0)': f"{metrics['recall_class_0']:.4f}",
            'Recall (Class 1)': f"{metrics['recall_class_1']:.4f}",
            'Recall (Class 2)': f"{metrics['recall_class_2']:.4f}",
            'Recall (Class 3)': f"{metrics['recall_class_3']:.4f}",
            'Recall (Class 4)': f"{metrics['recall_class_4']:.4f}",
        })
    
    df = pd.DataFrame(rows)
    
    # Save as CSV
    df.to_csv(save_path / 'results_table.csv', index=False)
    print(f"\nSaved results table to {save_path / 'results_table.csv'}")
    
    # Save as formatted text
    with open(save_path / 'results_table.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("BENCHMARK RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
    
    print(f"Saved formatted results to {save_path / 'results_table.txt'}")
    
    return df


def plot_confusion_matrices(results_dict, save_path, class_names=None):
    """Plot confusion matrices for all models."""
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, result) in enumerate(results_dict.items()):
        cm = result['confusion_matrix']
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'},
            ax=axes[idx]
        )
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
        axes[idx].set_title(f'{model_name}\n(QWK: {result["metrics"]["qwk"]:.3f})')
    
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrices to {save_path / 'confusion_matrices.png'}")


def save_intermediate_results(results_dict, out_dir):
    """Save intermediate results as they become available."""
    if not results_dict:
        return
    
    # Save partial results
    create_results_table(results_dict, out_dir)
    
    # Save JSON
    json_results = {}
    for model_name, result in results_dict.items():
        json_results[model_name] = {
            'metrics': result['metrics'],
            'confusion_matrix': result['confusion_matrix'].tolist(),
            'checkpoint_path': result['checkpoint_path'],
            'gradcam_path': result['gradcam_path']
        }
    
    with open(out_dir / 'results_intermediate.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Intermediate results saved to {out_dir / 'results_intermediate.json'}")


def check_existing_checkpoint(model_config):
    """Check if a checkpoint already exists for this model."""
    lightning_logs = Path('lightning_logs')
    if not lightning_logs.exists():
        return None
    
    # Look for checkpoints matching this model
    model_name_short = model_config['name'].split('.')[0]  # e.g., 'efficientnet_b2'
    
    for version_dir in sorted(lightning_logs.iterdir(), reverse=True):
        if not version_dir.is_dir() or not version_dir.name.startswith('version'):
            continue
        
        checkpoints_dir = version_dir / 'checkpoints'
        if not checkpoints_dir.exists():
            continue
        
        # Check hparams.yaml to see if it matches
        hparams_file = version_dir / 'hparams.yaml'
        if hparams_file.exists():
            try:
                import yaml
                with open(hparams_file) as f:
                    hparams = yaml.safe_load(f)
                    if hparams.get('model_name', '').startswith(model_name_short):
                        checkpoints = list(checkpoints_dir.glob('*.ckpt'))
                        if checkpoints:
                            # Find best checkpoint
                            for ckpt in checkpoints:
                                if 'best' in ckpt.name.lower():
                                    return ckpt
                            return checkpoints[0]  # Return any checkpoint if no 'best' found
            except:
                continue
    
    return None


def main():
    ap = argparse.ArgumentParser(description="Benchmark multiple DR classification models")
    ap.add_argument("--fold_csv", required=True, help="Path to fold CSV file")
    ap.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    ap.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation")
    ap.add_argument("--n_gradcam", type=int, default=20, help="Number of samples for Grad-CAM")
    ap.add_argument("--skip_training", action='store_true', help="Skip training, only evaluate existing checkpoints")
    ap.add_argument("--checkpoints", nargs='+', help="Paths to existing checkpoints (if skipping training)")
    ap.add_argument("--out_dir", default="benchmark_results", help="Output directory")
    ap.add_argument("--resume", action='store_true', help="Resume from existing checkpoints if available")
    ap.add_argument("--log_dir", default=None, help="Directory to save training logs")
    
    args = ap.parse_args()
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {}
    
    if not args.skip_training:
        # Train all models
        for idx, model_config in enumerate(MODELS_TO_BENCHMARK):
            print(f"\n{'#'*60}")
            print(f"Model {idx+1}/{len(MODELS_TO_BENCHMARK)}: {model_config['display_name']}")
            print(f"{'#'*60}")
            
            # Check for existing checkpoint if resuming
            checkpoint = None
            if args.resume:
                checkpoint = check_existing_checkpoint(model_config)
                if checkpoint:
                    print(f"Found existing checkpoint: {checkpoint}")
                    print("Skipping training. Use --skip_training to evaluate only.")
                    response = input("Continue with this checkpoint? (y/n): ").strip().lower()
                    if response != 'y':
                        checkpoint = None
            
            # Train if no checkpoint found
            if not checkpoint:
                log_file = log_dir / f"{model_config['display_name'].replace('/', '_')}_training.log"
                checkpoint = train_model(
                    model_config,
                    args.fold_csv,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    log_file=str(log_file)
                )
            
            if checkpoint:
                print(f"\n✓ Training complete. Best checkpoint: {checkpoint}")
                
                # Evaluate immediately
                print(f"\nEvaluating {model_config['display_name']}...")
                result = evaluate_model_for_benchmark(
                    checkpoint,
                    args.fold_csv,
                    split='val',
                    img_size=model_config['img_size'],
                    batch_size=args.eval_batch_size,
                    n_gradcam=args.n_gradcam
                )
                
                if result:
                    results_dict[model_config['display_name']] = result
                    # Save intermediate results after each model
                    save_intermediate_results(results_dict, out_dir)
                    print(f"✓ {model_config['display_name']} evaluation complete!")
            else:
                print(f"✗ Failed to train/evaluate {model_config['display_name']}")
    else:
        # Use provided checkpoints
        if not args.checkpoints or len(args.checkpoints) != len(MODELS_TO_BENCHMARK):
            print("Error: Must provide checkpoint paths for all models when skipping training")
            print(f"Expected {len(MODELS_TO_BENCHMARK)} checkpoints")
            return
        
        for model_config, checkpoint_path in zip(MODELS_TO_BENCHMARK, args.checkpoints):
            result = evaluate_model_for_benchmark(
                checkpoint_path,
                args.fold_csv,
                split='val',
                img_size=model_config['img_size'],
                batch_size=args.eval_batch_size,
                n_gradcam=args.n_gradcam
            )
            
            if result:
                results_dict[model_config['display_name']] = result
    
    if not results_dict:
        print("No results to save!")
        return
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Results table
    create_results_table(results_dict, out_dir)
    
    # Confusion matrices
    plot_confusion_matrices(results_dict, out_dir)
    
    # Save full results as JSON
    json_results = {}
    for model_name, result in results_dict.items():
        json_results[model_name] = {
            'metrics': result['metrics'],
            'confusion_matrix': result['confusion_matrix'].tolist(),
            'checkpoint_path': result['checkpoint_path'],
            'gradcam_path': result['gradcam_path']
        }
    
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nSaved full results to {out_dir / 'results.json'}")
    print(f"\nBenchmark complete! Results saved to {out_dir}")


if __name__ == "__main__":
    main()

