# Benchmarking Guide

This guide explains how to run the benchmarking script to compare different architectures.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run benchmark (trains all models)
python scripts/benchmark.py \
    --fold_csv data/folds/fold0.csv \
    --epochs 30 \
    --batch_size 8 \
    --n_gradcam 20 \
    --out_dir benchmark_results
```

## Models Being Benchmarked

1. **EfficientNet-B2** (`efficientnet_b2.ra_in1k`)
   - Image size: 384×384
   - Balanced accuracy/speed tradeoff

2. **EfficientNet-B4** (`efficientnet_b4.ra2_in1k`)
   - Image size: 384×384
   - Higher accuracy, more parameters

3. **ViT-B/16** (`vit_base_patch16_224.augreg_in1k`)
   - Image size: 384×384
   - Vision Transformer architecture

## Training Configuration

All models use:
- **Loss**: Weighted Cross-Entropy (handles class imbalance)
- **Learning Rate**: 1e-4
- **Scheduler**: Cosine annealing
- **Early Stopping**: Patience of 10 epochs
- **Monitor Metric**: Validation QWK (Quadratic Weighted Kappa)
- **Class Weights**: Automatically computed from training data

## Evaluation Metrics

The benchmark computes:
- **Accuracy**: Overall classification accuracy
- **QWK**: Quadratic Weighted Kappa (ordinal metric)
- **Macro F1**: Average F1-score across classes
- **ROC-AUC**: One-vs-rest area under ROC curve
- **Per-class metrics**: Precision, recall, F1 for each severity level

## Output Files

After running the benchmark, you'll find in `benchmark_results/`:

1. **results_table.csv**: Comparison table with all metrics
2. **results_table.txt**: Formatted text version
3. **confusion_matrices.png**: Side-by-side confusion matrices
4. **results.json**: Complete results in JSON format
5. **gradcam/**: Directory with Grad-CAM visualizations for each model

## Example Results Table

```
Model            Accuracy  QWK     Macro F1  ROC-AUC (OVR)
EfficientNet-B2  0.xxxx   0.xxxx  0.xxxx    0.xxxx
EfficientNet-B4  0.xxxx   0.xxxx  0.xxxx    0.xxxx
ViT-B/16         0.xxxx   0.xxxx  0.xxxx    0.xxxx
```

## Grad-CAM Visualizations

Each model generates Grad-CAM visualizations for 20 random validation samples, showing:
- Original image
- Heatmap highlighting important regions
- Overlay visualization

These help understand what features each model focuses on.

## Skipping Training

If you already have trained checkpoints, you can skip training:

```bash
python scripts/benchmark.py \
    --skip_training \
    --checkpoints \
        lightning_logs/version_X/checkpoints/best.ckpt \
        lightning_logs/version_Y/checkpoints/best.ckpt \
        lightning_logs/version_Z/checkpoints/best.ckpt \
    --fold_csv data/folds/fold0.csv \
    --out_dir benchmark_results
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 4 or 8)
- Use smaller image size (e.g., 224 instead of 384)

### Model Not Found
- Check available models: `python -c "import timm; print(timm.list_models('*efficientnet*'))"`
- Update model names in `scripts/benchmark.py` if needed

### Training Takes Too Long
- Reduce `--epochs` for quick tests
- Use `--skip_training` if you have existing checkpoints

## Next Steps

After benchmarking:
1. Review results table to compare models
2. Examine confusion matrices for error patterns
3. Check Grad-CAM visualizations for interpretability
4. Select best model for your use case
5. Train final model on full dataset if needed

