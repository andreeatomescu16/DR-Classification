"""
Explainability methods for Diabetic Retinopathy classification.

This module provides Grad-CAM and related visualization methods
to understand model predictions and highlight important image regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates heatmaps showing which image regions the model focuses on
    when making predictions. Useful for medical image interpretation.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (ICCV 2017)
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer name (e.g., 'features.7' for ResNet)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output is not None and len(grad_output) > 0:
                self.gradients = grad_output[0]
        
        # Find target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break
        
        if target_module is None:
            # Try partial match
            for name, module in self.model.named_modules():
                if self.target_layer in name:
                    target_module = module
                    self.target_layer = name  # Update to actual name
                    break
        
        if target_module is None:
            available_layers = [name for name, _ in self.model.named_modules() if len(name) > 0]
            raise ValueError(
                f"Target layer '{self.target_layer}' not found in model. "
                f"Available layers: {available_layers[:10]}..."
            )
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (None = use predicted class)
        
        Returns:
            CAM heatmap (numpy array, H x W)
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_heatmap(self, image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay CAM heatmap on original image.
        
        Args:
            image: Original image (H, W, 3) in RGB format, 0-255 range
            cam: CAM heatmap (H, W)
            alpha: Transparency factor
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image (H, W, 3)
        """
        # Resize CAM to image size if needed
        if cam.shape != image.shape[:2]:
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlaid


def find_target_layer(model, architecture='efficientnet'):
    """
    Find appropriate target layer for Grad-CAM based on architecture.
    
    Args:
        model: PyTorch model
        architecture: Model architecture name or type
    
    Returns:
        Layer name string
    """
    architecture = architecture.lower()
    
    # Try to infer from model name
    model_str = str(model).lower()
    
    if 'efficientnet' in architecture or 'efficientnet' in model_str:
        # EfficientNet: use last block before classifier
        for name, module in reversed(list(model.named_modules())):
            if 'blocks' in name and len(name.split('.')) == 2:
                return name
        return 'blocks.6'
    
    elif 'resnet' in architecture or 'resnet' in model_str:
        # ResNet: use last convolutional layer
        return 'layer4'
    
    elif 'vit' in architecture or 'vision_transformer' in model_str:
        # Vision Transformer: use attention blocks
        return 'blocks.11'
    
    elif 'convnext' in architecture or 'convnext' in model_str:
        # ConvNeXt: use stages
        return 'stages.3'
    
    else:
        # Default: try to find last convolutional layer
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = name
        if last_conv:
            return last_conv
        
        raise ValueError(f"Could not determine target layer for architecture: {architecture}")


def visualize_predictions(
    model,
    images,
    labels,
    class_names=None,
    target_layer=None,
    device='cpu',
    save_path=None
):
    """
    Visualize model predictions with Grad-CAM heatmaps.
    
    Args:
        model: Trained PyTorch model
        images: Batch of images (B, C, H, W) or list of images
        labels: True labels (B,)
        class_names: List of class names
        target_layer: Target layer for Grad-CAM (None = auto-detect)
        device: Device to run on
        save_path: Optional path to save visualization
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    model.eval()
    model.to(device)
    
    if isinstance(images, list):
        images = torch.stack(images)
    
    images = images.to(device)
    batch_size = images.shape[0]
    
    # Find target layer if not provided
    if target_layer is None:
        target_layer = find_target_layer(model)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Get predictions
    with torch.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)
    
    # Create visualization
    n_cols = 3  # Original, Heatmap, Overlay
    fig, axes = plt.subplots(batch_size, n_cols, figsize=(15, 5 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        img = images[i].cpu()
        label = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
        pred = preds[i].item()
        prob = probs[i, pred].item()
        
        # Denormalize image (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Generate CAM
        cam = gradcam.generate_cam(images[i:i+1], target_class=pred)
        
        # Overlay heatmap
        overlaid = gradcam.overlay_heatmap(img_np, cam)
        
        # Plot
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Original\nTrue: {class_names[label]}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(cam, cmap='jet')
        axes[i, 1].set_title(f'Grad-CAM Heatmap')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(overlaid)
        axes[i, 2].set_title(f'Overlay\nPred: {class_names[pred]} ({prob:.2f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

