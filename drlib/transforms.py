from albumentations import (
    Compose, Resize, HorizontalFlip, Affine,
    RandomBrightnessContrast, Blur, CLAHE
)
from albumentations.pytorch import ToTensorV2

# basic normalization for ImageNet backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_tf(size=512):
    return Compose([
        Resize(size, size),
        CLAHE(clip_limit=3.0, tile_grid_size=(8,8), p=0.7),
        HorizontalFlip(p=0.5),
        Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.03), rotate=(-15, 15), p=0.5),
        RandomBrightnessContrast(0.1, 0.1, p=0.5),
        Blur(blur_limit=3, p=0.2),
        ToTensorV2(),
    ])

def get_val_tf(size=512):
    return Compose([Resize(size, size), ToTensorV2()])