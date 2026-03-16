"""
Loss functions for Diabetic Retinopathy classification.

This module provides loss functions designed to handle class imbalance
common in medical imaging datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for handling class imbalance.
    
    Weights are computed inversely proportional to class frequencies,
    optionally with smoothing to prevent extreme weights.
    """
    def __init__(self, class_weights=None, reduction='mean'):
        """
        Args:
            class_weights: Tensor of shape (num_classes,) with weights for each class.
                          If None, weights will be computed from data.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,) with class indices
        """
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None
        
        return F.cross_entropy(logits, targets, weight=weights, reduction=self.reduction)
    
    @staticmethod
    def compute_weights_from_counts(class_counts, smoothing=1.0):
        """
        Compute class weights from class counts.
        
        Args:
            class_counts: Array or list of class frequencies
            smoothing: Smoothing factor to prevent extreme weights
        
        Returns:
            Tensor of normalized weights
        """
        class_counts = np.array(class_counts, dtype=np.float32)
        # Add smoothing to avoid division by zero
        class_counts = class_counts + smoothing
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        # Normalize so that weights sum to num_classes
        weights = weights * len(class_counts) / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss implemented via log_softmax + nll_loss (gather-based).

    p_t is always the true-class probability so the focal modulation is correct
    even when per-class weights are used.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        """
        Args:
            gamma: Focusing parameter — higher values push more focus onto
                   hard examples (default 2.0).
            weight: Optional Tensor of shape (num_classes,) with per-class
                    multiplicative weights.  Registered as a buffer so that
                    Lightning's .to(device) propagates it automatically.
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,) with class indices
        """
        log_p = F.log_softmax(logits, dim=1)
        # log-probability of the true class per sample
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(log_pt)

        focal_weight = (1.0 - p_t) ** self.gamma

        if self.weight is not None:
            class_w = self.weight.to(logits.device)[targets]
            focal_loss = class_w * focal_weight * (-log_pt)
        else:
            focal_loss = focal_weight * (-log_pt)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class OrdinalLoss(nn.Module):
    """
    Ordinal loss (CORAL-style) for ordinal classification.
    Treats each level as cumulative: P(Y>=k) via K-1 binary outputs.
    Penalizes distance-aware errors; QWK-oriented.
    """
    def __init__(self, num_classes=5, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, num_classes) - standard logits
        targets: (B,) - class indices 0..K-1
        """
        probs = F.softmax(logits, dim=1)
        # P(Y >= j) = 1 - P(Y < j) = 1 - sum_{k<j} P(Y=k)
        cum_sum = torch.cumsum(probs, dim=1)  # (B, K): [p0, p0+p1, ..., 1]
        cum_probs = torch.cat([torch.ones_like(probs[:, :1]), 1 - cum_sum[:, :-1]], dim=1)  # P(Y>=0)=1, P(Y>=1), ...
        # cum_probs[:, j] = P(Y >= j), j=0..K-1. Use j=1..K-1 for K-1 binary tasks.
        levels = torch.arange(1, self.num_classes, device=logits.device, dtype=torch.long)
        labels = (targets.unsqueeze(1) >= levels).float()  # (B, K-1)

        # Use BCE with logits (autocast-safe); logit(p) = log(p/(1-p))
        cum_p = torch.clamp(cum_probs[:, 1:], min=1e-7, max=1.0 - 1e-7)
        cum_logits = torch.logit(cum_p)
        bce = F.binary_cross_entropy_with_logits(cum_logits, labels, reduction='none')
        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        return bce


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross-Entropy Loss for regularization.
    
    Helps prevent overconfidence and improves generalization.
    """
    def __init__(self, num_classes, smoothing=0.1, reduction='mean'):
        """
        Args:
            num_classes: Number of classes
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform distribution)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,) with class indices
        """
        log_probs = F.log_softmax(logits, dim=1)
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -torch.sum(true_dist * log_probs, dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def build_class_weights(label_counts, mode='sqrt_inverse'):
    """
    Build a normalized per-class weight tensor from per-label sample counts.

    Args:
        label_counts: dict mapping label int -> sample count, e.g. {0: 1000, 1: 200, ...}
        mode: 'inverse'      — weight ∝ 1 / count  (aggressive up-weighting)
              'sqrt_inverse' — weight ∝ 1 / sqrt(count)  (gentler, default)

    Returns:
        Tensor of shape (num_classes,) normalized so weights sum to num_classes.
    """
    num_classes = max(label_counts.keys()) + 1
    counts = np.array(
        [label_counts.get(i, 1) for i in range(num_classes)], dtype=np.float32
    )
    counts = np.maximum(counts, 1)  # guard against zero counts

    if mode == 'inverse':
        weights = 1.0 / counts
    elif mode == 'sqrt_inverse':
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(
            f"Unknown weight mode '{mode}'. Choose 'inverse' or 'sqrt_inverse'."
        )

    # Normalize so the weights sum to num_classes (keeps the loss scale stable)
    weights = weights * num_classes / weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def create_loss(loss_type='ce', class_counts=None, **kwargs):
    """
    Factory function to create loss functions.

    Supported loss_type values
    --------------------------
    Existing (unchanged behaviour):
        'ce'              — standard CrossEntropyLoss
        'weighted_ce'     — inverse-frequency weighted CE (legacy helper)
        'label_smoothing' — LabelSmoothingLoss
        'ordinal'         — OrdinalLoss (CORAL-style)

    New:
        'focal'           — FocalLoss, no class weights (gamma from kwargs)
        'focal_weighted'  — FocalLoss with per-class weights via build_class_weights
        'ce_weighted'     — CrossEntropyLoss with per-class weights via build_class_weights

    Extra kwargs consumed by the factory (not forwarded to the loss module):
        gamma       (float, default 2.0)   — focal gamma
        weight_mode (str,  default 'sqrt_inverse') — mode for build_class_weights
        label_counts (dict {label: count}) — explicit counts; takes priority over
                                             class_counts array for weighted variants
        smoothing   (float) — used only by legacy 'weighted_ce' branch

    Args:
        loss_type:    One of the supported strings above.
        class_counts: Array-like of per-class sample counts (indices 0..K-1).
                      Accepted for backward compat; converted to dict internally.
        **kwargs:     See above.

    Returns:
        nn.Module loss function.
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)

    elif loss_type == 'weighted_ce':
        smoothing = kwargs.pop('smoothing', 1.0)
        if class_counts is not None:
            weights = WeightedCrossEntropyLoss.compute_weights_from_counts(
                class_counts, smoothing=smoothing
            )
        else:
            weights = kwargs.pop('class_weights', None)
        return WeightedCrossEntropyLoss(class_weights=weights, **kwargs)

    elif loss_type == 'focal':
        kwargs.pop('smoothing', None)
        gamma = kwargs.pop('gamma', 2.0)
        kwargs.pop('weight_mode', None)
        kwargs.pop('label_counts', None)
        return FocalLoss(gamma=gamma, **kwargs)

    elif loss_type == 'focal_weighted':
        kwargs.pop('smoothing', None)
        gamma = kwargs.pop('gamma', 2.0)
        weight_mode = kwargs.pop('weight_mode', 'sqrt_inverse')
        label_counts = kwargs.pop('label_counts', None)
        if label_counts is None and class_counts is not None:
            label_counts = {i: int(c) for i, c in enumerate(class_counts)}
        weight = build_class_weights(label_counts, mode=weight_mode) if label_counts is not None else None
        return FocalLoss(gamma=gamma, weight=weight, **kwargs)

    elif loss_type == 'ce_weighted':
        kwargs.pop('smoothing', None)
        weight_mode = kwargs.pop('weight_mode', 'sqrt_inverse')
        label_counts = kwargs.pop('label_counts', None)
        if label_counts is None and class_counts is not None:
            label_counts = {i: int(c) for i, c in enumerate(class_counts)}
        weight = build_class_weights(label_counts, mode=weight_mode) if label_counts is not None else None
        return WeightedCrossEntropyLoss(class_weights=weight, **kwargs)

    elif loss_type == 'label_smoothing':
        num_classes = kwargs.pop('num_classes', 5)
        smoothing = kwargs.pop('smoothing', 0.1)
        kwargs.pop('weight_mode', None)
        kwargs.pop('label_counts', None)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)

    elif loss_type == 'ordinal':
        num_classes = kwargs.pop('num_classes', 5)
        kwargs.pop('smoothing', None)
        kwargs.pop('weight_mode', None)
        kwargs.pop('label_counts', None)
        return OrdinalLoss(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

