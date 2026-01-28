"""
Model architectures for Diabetic Retinopathy classification.

This module provides factory functions to create various CNN and Vision Transformer
architectures suitable for medical image classification.
"""

import timm
import torch.nn as nn
from typing import Optional


def create_model(
    name="efficientnet_b3",
    num_classes=5,
    pretrained=True,
    drop_rate=0.0,
    drop_path_rate=0.0
):
    """
    Create a classification model.
    
    Supported architectures:
    - EfficientNet: efficientnet_b0 through efficientnet_b7
    - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
    - Vision Transformers: vit_base_patch16_224, vit_large_patch16_224, etc.
    - ConvNeXt: convnext_tiny, convnext_small, convnext_base, convnext_large
    - RegNet: regnetx_002, regnetx_004, regnetx_006, etc.
    
    Args:
        name: Model architecture name (must be supported by timm)
        num_classes: Number of output classes (default: 5 for DR severity)
        pretrained: Whether to use pretrained weights
        drop_rate: Dropout rate for classifier head
        drop_path_rate: Stochastic depth rate (for some architectures)
    
    Returns:
        PyTorch model
    """
    # Check if model exists in timm (try with and without pretrained filter)
    all_models = timm.list_models(pretrained=pretrained)
    if name not in all_models:
        # Try to find similar model names
        similar = [m for m in all_models if name.split('.')[0] in m or name in m]
        if not similar:
            raise ValueError(
                f"Model '{name}' not found in timm. "
                f"Available models with similar names: {', '.join(similar[:10]) if similar else 'none'}..."
            )
        # Use the first similar match
        name = similar[0]
        print(f"Using model: {name}")
    
    # Create model with timm
    model = timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=3,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    
    return model


def get_model_info(name: str) -> dict:
    """
    Get information about a model architecture.
    
    Args:
        name: Model architecture name
    
    Returns:
        Dictionary with model information (parameters, FLOPs, etc.)
    """
    try:
        model = timm.create_model(name, pretrained=False, num_classes=5)
        info = timm.models.helpers.get_pretrained_cfg(name)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'name': name,
            'num_params': num_params,
            'trainable_params': trainable_params,
            'pretrained_cfg': info if info else None
        }
    except Exception as e:
        return {'name': name, 'error': str(e)}


# Recommended model configurations for different resource constraints
RECOMMENDED_MODELS = {
    'lightweight': 'efficientnet_b0',  # Fast, low memory
    'balanced': 'efficientnet_b3',     # Good accuracy/speed tradeoff
    'high_accuracy': 'efficientnet_b5', # Higher accuracy, more compute
    'transformer': 'vit_base_patch16_224',  # Vision Transformer
    'large': 'convnext_large',  # Large CNN
}


def create_recommended_model(
    profile='balanced',
    num_classes=5,
    pretrained=True,
    **kwargs
):
    """
    Create a model from recommended configurations.
    
    Args:
        profile: One of 'lightweight', 'balanced', 'high_accuracy', 'transformer', 'large'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments passed to create_model
    
    Returns:
        PyTorch model
    """
    if profile not in RECOMMENDED_MODELS:
        raise ValueError(
            f"Unknown profile: {profile}. "
            f"Choose from: {list(RECOMMENDED_MODELS.keys())}"
        )
    
    model_name = RECOMMENDED_MODELS[profile]
    return create_model(model_name, num_classes, pretrained, **kwargs)

