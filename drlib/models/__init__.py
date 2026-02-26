"""
Model architectures for Diabetic Retinopathy classification.
"""

from typing import Optional

from .hybrid_coatmini import build_hybrid_coatmini


def create_model(
    name="efficientnet_b3",
    num_classes=5,
    pretrained=True,
    drop_rate=0.0,
    drop_path_rate=0.0
):
    """Create a classification model. Supports timm models + hybrid_coatmini."""
    if name == "hybrid_coatmini":
        return build_hybrid_coatmini(num_classes=num_classes, drop_rate=drop_rate)

    import timm
    all_models = timm.list_models(pretrained=pretrained)
    if name not in all_models:
        similar = [m for m in all_models if name.split('.')[0] in m or name in m]
        if not similar:
            raise ValueError(
                f"Model '{name}' not found in timm. "
                f"Available: {', '.join(similar[:10]) if similar else 'none'}..."
            )
        name = similar[0]
        print(f"Using model: {name}")

    return timm.create_model(
        name,
        pretrained=pretrained,
        in_chans=3,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )


def get_model_info(name: str) -> dict:
    """Get model info (params, etc.)."""
    try:
        if name == "hybrid_coatmini":
            model = build_hybrid_coatmini(num_classes=5)
        else:
            import timm
            model = timm.create_model(name, pretrained=False, num_classes=5)
        num_params = sum(p.numel() for p in model.parameters())
        return {'name': name, 'num_params': num_params, 'trainable_params': num_params}
    except Exception as e:
        return {'name': name, 'error': str(e)}


RECOMMENDED_MODELS = {
    'lightweight': 'efficientnet_b0',
    'balanced': 'efficientnet_b3',
    'high_accuracy': 'efficientnet_b5',
    'transformer': 'vit_base_patch16_224',
    'large': 'convnext_large',
    'hybrid': 'hybrid_coatmini',
}


def create_recommended_model(profile='balanced', num_classes=5, pretrained=True, **kwargs):
    """Create model from recommended config."""
    if profile not in RECOMMENDED_MODELS:
        raise ValueError(f"Unknown profile: {profile}. Choose from: {list(RECOMMENDED_MODELS.keys())}")
    return create_model(RECOMMENDED_MODELS[profile], num_classes, pretrained, **kwargs)


__all__ = ["create_model", "get_model_info", "create_recommended_model", "RECOMMENDED_MODELS"]
