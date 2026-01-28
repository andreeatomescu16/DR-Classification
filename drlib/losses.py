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
    Focal Loss for addressing class imbalance by down-weighting easy examples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for each class (Tensor of shape (num_classes,))
                   If None, uniform weighting is used.
            gamma: Focusing parameter (higher gamma = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,) with class indices
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets].to(logits.device)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    @staticmethod
    def compute_alpha_from_counts(class_counts, smoothing=1.0):
        """
        Compute alpha weights from class counts.
        
        Args:
            class_counts: Array or list of class frequencies
            smoothing: Smoothing factor
        
        Returns:
            Tensor of normalized alpha weights
        """
        class_counts = np.array(class_counts, dtype=np.float32)
        class_counts = class_counts + smoothing
        # Normalize to sum to 1
        alpha = class_counts / class_counts.sum()
        return torch.tensor(alpha, dtype=torch.float32)


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


def create_loss(loss_type='ce', class_counts=None, **kwargs):
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: 'ce' (cross-entropy), 'weighted_ce', 'focal', or 'label_smoothing'
        class_counts: Array of class frequencies (for weighted/focal losses)
        **kwargs: Additional arguments for specific loss functions
    
    Returns:
        Loss function module
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    
    elif loss_type == 'weighted_ce':
        smoothing = kwargs.pop('smoothing', 1.0)  # Remove smoothing from kwargs
        if class_counts is not None:
            weights = WeightedCrossEntropyLoss.compute_weights_from_counts(
                class_counts, smoothing=smoothing
            )
        else:
            weights = kwargs.get('class_weights', None)
        return WeightedCrossEntropyLoss(class_weights=weights, **kwargs)
    
    elif loss_type == 'focal':
        smoothing = kwargs.pop('smoothing', 1.0)  # Remove smoothing from kwargs
        gamma = kwargs.pop('gamma', 2.0)  # Remove gamma from kwargs
        if class_counts is not None:
            alpha = FocalLoss.compute_alpha_from_counts(
                class_counts, smoothing=smoothing
            )
        else:
            alpha = kwargs.get('alpha', None)
        return FocalLoss(alpha=alpha, gamma=gamma, **kwargs)
    
    elif loss_type == 'label_smoothing':
        num_classes = kwargs.get('num_classes', 5)
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing, **kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

