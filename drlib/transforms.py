"""
Data preprocessing and augmentation transforms for Diabetic Retinopathy classification.

This module provides:
- Preprocessing functions for fundus image enhancement
- Training augmentation pipelines
- Validation/test preprocessing pipelines
"""

import cv2
import numpy as np
from albumentations import (
    Compose, Resize, HorizontalFlip, VerticalFlip, Affine,
    RandomBrightnessContrast, Blur, CLAHE, Normalize,
    OpticalDistortion, RandomRotate90
)
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

# ImageNet normalization constants for pretrained backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RemoveBlackBorders(ImageOnlyTransform):
    """
    Remove black borders from fundus images by detecting the circular fundus region.
    
    This transform crops the image to the fundus region, removing black borders
    that are common in retinal fundus photography.
    """
    def __init__(self, p=1.0):
        super().__init__(p=p)
    
    def apply(self, img, **params):
        """Crop image to remove black borders."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Threshold to find non-black regions
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Find largest contour (fundus region)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding (5% of dimensions)
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.05)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img.shape[1] - x, w + 2 * pad_x)
        h = min(img.shape[0] - y, h + 2 * pad_y)
        
        # Crop image
        cropped = img[y:y+h, x:x+w]
        
        # If crop is too small, return original
        if cropped.size < img.size * 0.1:
            return img
        
        return cropped
    
    def get_transform_init_args_names(self):
        return ()


def get_train_tf(size=512, remove_borders=True, strong_aug=True):
    """
    Get training augmentation pipeline.
    
    Args:
        size: Target image size (square)
        remove_borders: Whether to remove black borders
        strong_aug: Whether to use strong augmentation (for robustness)
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    # Preprocessing: remove black borders
    if remove_borders:
        transforms.append(RemoveBlackBorders(p=1.0))
    
    # Resize to target size
    transforms.append(Resize(size, size))
    
    # Contrast enhancement (CLAHE)
    transforms.append(CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.7))
    
    if strong_aug:
        # Geometric augmentations
        transforms.append(RandomRotate90(p=0.5))
        transforms.append(HorizontalFlip(p=0.5))
        transforms.append(VerticalFlip(p=0.3))  # Less common but useful
        
        # Affine transformations
        transforms.append(
            Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.05),
                rotate=(-15, 15),
                p=0.5
            )
        )
        
        # Color augmentations
        transforms.append(RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5))
        
        # Blur (simulating focus issues)
        transforms.append(Blur(blur_limit=3, p=0.2))
        
        # Optical distortions (simulating camera artifacts)
        transforms.append(OpticalDistortion(distort_limit=0.1, p=0.2))
        
        # Coarse dropout (regularization) - removed due to API changes
        # Note: CoarseDropout API varies by version, skipping for compatibility
    else:
        # Light augmentation only
        transforms.append(HorizontalFlip(p=0.5))
        transforms.append(
            Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.0, 0.03),
                rotate=(-15, 15),
                p=0.5
            )
        )
        transforms.append(RandomBrightnessContrast(0.1, 0.1, p=0.5))
    
    # Normalize to ImageNet statistics
    transforms.append(Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return Compose(transforms)


def get_val_tf(size=512, remove_borders=True):
    """
    Get validation/test preprocessing pipeline (no augmentation).
    
    Args:
        size: Target image size (square)
        remove_borders: Whether to remove black borders
    
    Returns:
        Albumentations Compose transform
    """
    transforms = []
    
    if remove_borders:
        transforms.append(RemoveBlackBorders(p=1.0))
    
    transforms.append(Resize(size, size))
    transforms.append(Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transforms.append(ToTensorV2())
    
    return Compose(transforms)