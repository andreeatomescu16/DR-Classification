"""
Comprehensive evaluation metrics for Diabetic Retinopathy classification.

This module provides metrics suitable for ordinal classification tasks
with class imbalance, including Cohen's quadratic weighted kappa,
macro F1-score, per-class metrics, and ROC-AUC.
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
from typing import Dict, List, Tuple, Optional


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    """
    Compute Cohen's quadratic weighted kappa.
    
    This metric is particularly suitable for ordinal classification tasks
    like DR severity, as it penalizes larger misclassifications more heavily.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        n_classes: Number of classes (default: 5)
    
    Returns:
        Quadratic weighted kappa score (float)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # Ensure labels are in valid range
    y_true = np.clip(y_true, 0, n_classes - 1)
    y_pred = np.clip(y_pred, 0, n_classes - 1)
    
    O = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(float)
    N = O.sum()
    
    if N == 0:
        return 0.0
    
    # Weight matrix: quadratic weights
    w = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    
    # Observed and expected distributions
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / N
    
    # Compute kappa
    num = (w * O).sum()
    den = (w * E).sum()
    
    return 1.0 - num / den if den > 0 else 0.0


def compute_all_metrics(y_true, y_pred, y_proba=None, n_classes=5) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        y_proba: Predicted probabilities (array-like, shape: [n_samples, n_classes])
        n_classes: Number of classes
    
    Returns:
        Dictionary of metric names and values
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, 0, n_classes - 1)
    y_pred = np.clip(y_pred, 0, n_classes - 1)
    
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Quadratic weighted kappa
    metrics['qwk'] = quadratic_weighted_kappa(y_true, y_pred, n_classes)
    
    # Macro F1-score (average across classes)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics (with labels parameter to ensure all classes are included)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=np.arange(n_classes))
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=np.arange(n_classes))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=np.arange(n_classes))
    
    # Ensure we have metrics for all classes (pad with zeros if needed)
    if len(precision_per_class) < n_classes:
        precision_per_class = np.pad(precision_per_class, (0, n_classes - len(precision_per_class)), 'constant')
        recall_per_class = np.pad(recall_per_class, (0, n_classes - len(recall_per_class)), 'constant')
        f1_per_class = np.pad(f1_per_class, (0, n_classes - len(f1_per_class)), 'constant')
    
    for i in range(n_classes):
        metrics[f'precision_class_{i}'] = precision_per_class[i] if i < len(precision_per_class) else 0.0
        metrics[f'recall_class_{i}'] = recall_per_class[i] if i < len(recall_per_class) else 0.0
        metrics[f'f1_class_{i}'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
    
    # ROC-AUC (one-vs-rest) if probabilities are available
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.shape[1] == n_classes:
            try:
                # One-vs-rest ROC-AUC
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
                # Per-class ROC-AUC
                for i in range(n_classes):
                    y_binary = (y_true == i).astype(int)
                    if len(np.unique(y_binary)) > 1:  # Check if class exists
                        metrics[f'roc_auc_class_{i}'] = roc_auc_score(
                            y_binary, y_proba[:, i]
                        )
            except ValueError:
                # Some classes might not be present
                pass
    
    return metrics


def get_confusion_matrix(y_true, y_pred, n_classes=5, normalize=False):
    """
    Get confusion matrix with optional normalization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes
        normalize: If True, normalize to percentages
    
    Returns:
        Confusion matrix (numpy array)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_true = np.clip(y_true, 0, n_classes - 1)
    y_pred = np.clip(y_pred, 0, n_classes - 1)
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm / row_sums[:, np.newaxis]
        cm = cm * 100  # Convert to percentages
    
    return cm


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names (e.g., ['No DR', 'Mild', ...])
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(5)]
    
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    ))


def compute_class_distribution(y_true, y_pred=None, n_classes=5) -> Dict[str, np.ndarray]:
    """
    Compute class distribution statistics.
    
    Args:
        y_true: True labels
        y_pred: Optional predicted labels
        n_classes: Number of classes
    
    Returns:
        Dictionary with 'true' and optionally 'pred' distributions
    """
    y_true = np.asarray(y_true, dtype=int)
    y_true = np.clip(y_true, 0, n_classes - 1)
    
    result = {}
    
    # True distribution
    true_counts = np.bincount(y_true, minlength=n_classes)
    result['true'] = true_counts
    result['true_pct'] = true_counts / len(y_true) * 100
    
    # Predicted distribution
    if y_pred is not None:
        y_pred = np.asarray(y_pred, dtype=int)
        y_pred = np.clip(y_pred, 0, n_classes - 1)
        pred_counts = np.bincount(y_pred, minlength=n_classes)
        result['pred'] = pred_counts
        result['pred_pct'] = pred_counts / len(y_pred) * 100
    
    return result

