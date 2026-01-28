#!/usr/bin/env python3
"""
Comprehensive evaluation script for Diabetic Retinopathy classification models.

This script evaluates trained models on test/validation sets and generates:
- Comprehensive metrics (accuracy, QWK, F1, ROC-AUC, etc.)
- Confusion matrices
- Per-class performance metrics
- ROC curves
- Prediction visualizations with Grad-CAM
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drlib.datasets import DRDataset
from drlib.transforms import get_val_tf
from drlib.models import create_model
from drlib.metrics import (
    compute_all_metrics, get_confusion_matrix,
    print_classification_report, compute_class_distribution
)
from drlib.explainability import visualize_predictions, find_target_layer


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    import lightning as L
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    model_name = hparams.get('model_name', 'efficientnet_b3')
    
    # Create model
    from drlib.train import DRModule
    model = DRModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    
    return model, hparams


def evaluate_model(model, dataloader, device='cpu', class_names=None):
    """
    Evaluate model on dataset.
    
    Returns:
        Dictionary with predictions, probabilities, and true labels
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_images = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store first few images for visualization
            if batch_idx == 0:
                all_images = images[:min(8, len(images))].cpu()
    
    return {
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels),
        'images': all_images,
        'class_names': class_names
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix."""
    cm = get_confusion_matrix(y_true, y_pred, normalize=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_roc_curves(y_true, y_proba, class_names, save_path=None):
    """Plot ROC curves for each class (one-vs-rest)."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_class_distribution(y_true, y_pred, class_names, save_path=None):
    """Plot class distribution comparison."""
    dist = compute_class_distribution(y_true, y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax1.bar(x - width/2, dist['true'], width, label='True', alpha=0.8)
    ax1.bar(x + width/2, dist['pred'], width, label='Predicted', alpha=0.8)
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_title('Class Distribution (Counts)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x - width/2, dist['true_pct'], width, label='True', alpha=0.8)
    ax2.bar(x + width/2, dist['pred_pct'], width, label='Predicted', alpha=0.8)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Class Distribution (Percentages)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_metrics_report(metrics, save_path):
    """Save metrics to CSV file."""
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)
    print(f"Saved metrics to {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate DR classification model")
    
    ap.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    ap.add_argument("--data_csv", required=True, help="Path to evaluation data CSV")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"], help="Data split to evaluate")
    ap.add_argument("--img_size", type=int, default=512, help="Image size")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    ap.add_argument("--remove_borders", action='store_true', default=True, help="Remove black borders")
    
    ap.add_argument("--out_dir", default="evaluation_results", help="Output directory for results")
    ap.add_argument("--device", default="auto", help="Device (auto, cpu, cuda, mps)")
    ap.add_argument("--visualize", action='store_true', help="Generate Grad-CAM visualizations")
    ap.add_argument("--n_samples", type=int, default=16, help="Number of samples for visualization")
    
    args = ap.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, hparams = load_model(args.checkpoint, device=device)
    print(f"Model: {hparams.get('model_name', 'unknown')}")
    
    # Create dataset and dataloader
    print(f"Loading data from {args.data_csv}...")
    dataset = DRDataset(
        args.data_csv,
        split=args.split,
        tfm=get_val_tf(args.img_size, remove_borders=args.remove_borders)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device != 'cpu' else False
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Evaluate
    results = evaluate_model(model, dataloader, device=device)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities'],
        n_classes=5
    )
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Quadratic Weighted Kappa: {metrics['qwk']:.4f}")
    print(f"Macro F1-Score: {metrics['macro_f1']:.4f}")
    if 'roc_auc_ovr' in metrics:
        print(f"ROC-AUC (OVR): {metrics['roc_auc_ovr']:.4f}")
    
    print("\nPer-Class Metrics:")
    class_names = results['class_names']
    for i in range(5):
        print(f"\n{class_names[i]}:")
        print(f"  Precision: {metrics[f'precision_class_{i}']:.4f}")
        print(f"  Recall: {metrics[f'recall_class_{i}']:.4f}")
        print(f"  F1-Score: {metrics[f'f1_class_{i}']:.4f}")
        if f'roc_auc_class_{i}' in metrics:
            print(f"  ROC-AUC: {metrics[f'roc_auc_class_{i}']:.4f}")
    
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print_classification_report(results['labels'], results['predictions'], class_names)
    
    # Save metrics
    save_metrics_report(metrics, out_dir / "metrics.csv")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        results['labels'],
        results['predictions'],
        class_names,
        save_path=cm_path
    )
    print(f"Saved confusion matrix to {cm_path}")
    
    # ROC curves
    roc_path = out_dir / "roc_curves.png"
    plot_roc_curves(
        results['labels'],
        results['probabilities'],
        class_names,
        save_path=roc_path
    )
    print(f"Saved ROC curves to {roc_path}")
    
    # Class distribution
    dist_path = out_dir / "class_distribution.png"
    plot_class_distribution(
        results['labels'],
        results['predictions'],
        class_names,
        save_path=dist_path
    )
    print(f"Saved class distribution to {dist_path}")
    
    # Grad-CAM visualizations
    if args.visualize:
        print("\nGenerating Grad-CAM visualizations...")
        try:
            target_layer = find_target_layer(model, hparams.get('model_name', 'efficientnet'))
            print(f"Using target layer: {target_layer}")
            
            # Get sample images and labels
            sample_indices = np.random.choice(len(results['labels']), args.n_samples, replace=False)
            sample_images = []
            sample_labels = []
            
            # Reload sample images
            for idx in sample_indices:
                img, label = dataset[idx]
                sample_images.append(img)
                sample_labels.append(label)
            
            sample_images = torch.stack(sample_images)
            sample_labels = torch.tensor(sample_labels)
            
            vis_path = out_dir / "gradcam_visualizations.png"
            visualize_predictions(
                model,
                sample_images,
                sample_labels,
                class_names=class_names,
                target_layer=target_layer,
                device=device,
                save_path=vis_path
            )
            print(f"Saved Grad-CAM visualizations to {vis_path}")
        except Exception as e:
            print(f"Warning: Could not generate Grad-CAM visualizations: {e}")
    
    print(f"\nEvaluation complete! Results saved to {out_dir}")


if __name__ == "__main__":
    main()

