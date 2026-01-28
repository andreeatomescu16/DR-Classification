# Diabetic Retinopathy Severity Classification

An end-to-end deep learning system for classifying Diabetic Retinopathy (DR) severity from retinal fundus images into five clinically defined categories (0-4).

## Project Overview

This project implements a comprehensive pipeline for DR classification, including:
- **Data preprocessing** with black border removal and fundus region cropping
- **Multiple model architectures** (EfficientNet, ResNet, Vision Transformers, ConvNeXt)
- **Advanced training strategies** (freeze/unfreeze, learning rate scheduling, class imbalance handling)
- **Comprehensive evaluation** (QWK, F1-score, ROC-AUC, per-class metrics)
- **Explainability** (Grad-CAM visualizations)
- **Reproducibility** (YAML configuration files, fixed random seeds)

## Dataset Structure

The project expects data organized as follows:

```
data/
├── aptos_master.csv          # APTOS dataset manifest
├── eyepacs_master.csv        # EyePACS dataset manifest
└── folds/
    ├── fold0.csv             # K-fold split files
    ├── fold1.csv
    └── ...
```

Each CSV file should contain columns:
- `image_path`: Full path to image file
- `label`: DR severity (0-4)
- `patient_id`: Patient identifier (for grouped splits)
- `is_valid`: Boolean flag for valid images
- `split`: train/val/test split

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd DR-Classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Datasets

#### APTOS Dataset:
```bash
python scripts/prepare_apots.py \
    --train_csv /path/to/aptos/train.csv \
    --train_imgdir /path/to/aptos/train_images \
    --val_csv /path/to/aptos/val.csv \
    --val_imgdir /path/to/aptos/val_images \
    --out_csv data/aptos_master.csv
```

#### EyePACS Dataset:
```bash
python scripts/prepare_eyepacs.py \
    --img_root /path/to/eyepacs/images \
    --labels_csv /path/to/eyepacs/labels.csv \
    --out_csv data/eyepacs_master.csv \
    --include_splits train val
```

### 2. Create K-Fold Splits

```bash
python scripts/kfold_split.py \
    --masters data/aptos_master.csv data/eyepacs_master.csv \
    --out_dir data/folds \
    --n_splits 5 \
    --seed 42
```

### 3. Train a Model

#### Basic Training:
```bash
python -m drlib.train \
    --fold_csv data/folds/fold0.csv \
    --model efficientnet_b3 \
    --img_size 512 \
    --batch_size 16 \
    --epochs 30 \
    --lr 1e-4 \
    --loss ce
```

#### Training with Class Weights (for imbalanced data):
```bash
python -m drlib.train \
    --fold_csv data/folds/fold0.csv \
    --model efficientnet_b3 \
    --loss weighted_ce \
    --use_class_weights
```

#### Training with Focal Loss:
```bash
python -m drlib.train \
    --fold_csv data/folds/fold0.csv \
    --model efficientnet_b3 \
    --loss focal \
    --focal_gamma 2.0 \
    --use_class_weights
```

#### Freeze/Unfreeze Strategy:
```bash
python -m drlib.train \
    --fold_csv data/folds/fold0.csv \
    --model efficientnet_b3 \
    --freeze_backbone \
    --unfreeze_epoch 10
```

### 4. Evaluate a Model

```bash
python scripts/evaluate.py \
    --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
    --data_csv data/folds/fold0.csv \
    --split val \
    --out_dir evaluation_results \
    --visualize
```

## Configuration Files

Use YAML configuration files for reproducible experiments:

```yaml
# configs/my_experiment.yaml
data:
  fold_csv: "data/folds/fold0.csv"
  img_size: 512
  batch_size: 16

model:
  name: "efficientnet_b3"
  freeze_backbone: true
  unfreeze_epoch: 10

training:
  epochs: 30
  lr: 0.0001
  loss:
    type: "focal"
    use_class_weights: true
```

Then load it in your training script (modify `train.py` to support `--config` argument).

## Model Architectures

Supported architectures (via `timm`):

- **EfficientNet**: `efficientnet_b0` through `efficientnet_b7`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **Vision Transformers**: `vit_base_patch16_224`, `vit_large_patch16_224`
- **ConvNeXt**: `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`
- **RegNet**: `regnetx_002`, `regnetx_004`, `regnetx_006`

Example:
```bash
python -m drlib.train --model vit_base_patch16_224 --img_size 224
```

## Loss Functions

1. **Cross-Entropy** (`ce`): Standard classification loss
2. **Weighted Cross-Entropy** (`weighted_ce`): Handles class imbalance
3. **Focal Loss** (`focal`): Focuses on hard examples
4. **Label Smoothing** (`label_smoothing`): Regularization technique

## Evaluation Metrics

The evaluation script computes:
- **Accuracy**: Overall classification accuracy
- **Quadratic Weighted Kappa (QWK)**: Ordinal classification metric
- **Macro F1-Score**: Average F1 across classes
- **ROC-AUC**: One-vs-rest area under ROC curve
- **Per-class metrics**: Precision, recall, F1-score for each severity level

## Explainability

Grad-CAM visualizations highlight image regions important for predictions:

```python
from drlib.explainability import visualize_predictions, GradCAM

# Generate visualizations
fig = visualize_predictions(
    model,
    images,
    labels,
    class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
    save_path='visualizations.png'
)
```

## Project Structure

```
DR-Classification/
├── drlib/                    # Main library
│   ├── datasets.py           # Dataset loading
│   ├── transforms.py         # Data preprocessing & augmentation
│   ├── models.py             # Model architectures
│   ├── train.py              # Training module (PyTorch Lightning)
│   ├── metrics.py            # Evaluation metrics
│   ├── losses.py             # Loss functions
│   ├── explainability.py     # Grad-CAM & visualization
│   └── config.py             # Configuration management
├── scripts/
│   ├── prepare_apots.py      # APTOS dataset preparation
│   ├── prepare_eyepacs.py    # EyePACS dataset preparation
│   ├── kfold_split.py        # K-fold cross-validation splits
│   └── evaluate.py           # Model evaluation
├── configs/                  # YAML configuration files
├── data/                     # Dataset manifests and splits
├── lightning_logs/           # Training logs and checkpoints
└── requirements.txt          # Python dependencies
```

## Key Features

### Data Preprocessing
- Automatic black border removal
- Fundus region cropping
- CLAHE contrast enhancement
- Comprehensive augmentation pipeline

### Training Features
- Freeze/unfreeze backbone training
- Learning rate scheduling (cosine, step, plateau)
- Class imbalance handling
- Early stopping
- Model checkpointing

### Evaluation Features
- Comprehensive metrics
- Confusion matrices
- ROC curves
- Per-class performance analysis
- Grad-CAM visualizations

## Reproducibility

The project emphasizes reproducibility:
- Fixed random seeds
- YAML configuration files
- Version-controlled code
- Detailed logging
- Checkpoint management

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (for small models like EfficientNet-B0)
- **Recommended**: GPU with 8GB+ VRAM (for EfficientNet-B3+ or ViT)
- **Mac M1/M2**: Uses MPS backend automatically

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dr-classification,
  title={Diabetic Retinopathy Severity Classification},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/DR-Classification}}
}
```

## License

[Specify your license here]

## Acknowledgments

- Datasets: APTOS 2019, EyePACS
- Libraries: PyTorch, PyTorch Lightning, timm, albumentations

## Contact

[Your contact information]
