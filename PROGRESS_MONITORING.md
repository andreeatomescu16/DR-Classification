# Progress Monitoring Guide

## Quick Progress Check

While training is running, you can check progress in several ways:

### 1. Check Current Status (Quick)

```bash
# Run the progress checker script
./scripts/check_progress.sh

# Or manually check the latest metrics
tail -1 lightning_logs/version_*/metrics.csv
```

### 2. Monitor in Real-Time

In a **new terminal**, run:

```bash
source venv/bin/activate
python scripts/monitor_training.py
```

This will show live updates every 10 seconds with:
- Current epoch
- Training/validation loss
- QWK (Quadratic Weighted Kappa)
- Accuracy and F1 scores
- Learning rate

Press `Ctrl+C` to stop monitoring.

### 3. Check Specific Version

```bash
# Monitor a specific version
python scripts/monitor_training.py --version 0

# Change refresh interval (default: 10 seconds)
python scripts/monitor_training.py --interval 5
```

### 4. View Training Logs

Training logs are saved to `benchmark_results/logs/` (or your specified log directory):

```bash
# View latest training log
tail -f benchmark_results/logs/EfficientNet-B2_training.log
```

### 5. Check Checkpoints

Checkpoints are automatically saved to:
```
lightning_logs/version_X/checkpoints/
```

Files:
- `best-*.ckpt` - Best model (highest validation QWK)
- `last.ckpt` - Latest checkpoint (can resume from here)

## Progress Preservation

### Automatic Checkpointing

✅ **Checkpoints are saved automatically** during training:
- Best model (based on validation QWK)
- Last checkpoint (every epoch)
- All checkpoints are preserved even if training is interrupted

### Resume Training

If training is interrupted, you can:

1. **Resume from checkpoint** (modify training script):
   ```python
   trainer.fit(model, dl_tr, dl_va, ckpt_path="lightning_logs/version_X/checkpoints/last.ckpt")
   ```

2. **Use existing checkpoint** in benchmark:
   ```bash
   python scripts/benchmark.py \
       --skip_training \
       --checkpoints \
           lightning_logs/version_X/checkpoints/best.ckpt \
       --fold_csv data/folds/fold0.csv \
       --out_dir benchmark_results
   ```

3. **Resume benchmark** (checks for existing checkpoints):
   ```bash
   python scripts/benchmark.py \
       --fold_csv data/folds/fold0.csv \
       --resume \
       --epochs 30 \
       --out_dir benchmark_results
   ```

### Intermediate Results

✅ **Results are saved after each model completes**:
- `benchmark_results/results_intermediate.json` - Updated after each model
- `benchmark_results/results_table.csv` - Updated after each model

So even if the benchmark is interrupted, you'll have results for completed models!

## What's Happening During Training

During training, you should see:
- Progress bars showing epoch/batch progress
- Metrics logged every 10 steps
- Validation metrics after each epoch
- Checkpoints saved automatically

If you don't see output, the training might be:
1. Still initializing (loading data, setting up model)
2. Running on GPU/MPS (may have less console output)
3. Writing to log files instead

## Troubleshooting

### No Output Visible

Check if training is actually running:
```bash
# Check Python processes
ps aux | grep python

# Check GPU/MPS usage (if applicable)
# On Mac: Activity Monitor -> GPU
```

### Check if Training Completed

```bash
# List all training runs
ls -lht lightning_logs/version_*/

# Check if metrics file exists
ls -lh lightning_logs/version_*/metrics.csv

# View final metrics
tail -20 lightning_logs/version_*/metrics.csv
```

### Training Seems Stuck

1. Check if it's actually training (monitor GPU/CPU usage)
2. Check log files for errors
3. Check disk space (checkpoints can be large)
4. If truly stuck, you can safely interrupt (Ctrl+C) - checkpoints are saved

## Expected Training Times

Approximate times per model (30 epochs, batch_size=8, 384x384):
- **EfficientNet-B2**: ~2-4 hours (depending on hardware)
- **EfficientNet-B4**: ~3-5 hours
- **ViT-B/16**: ~4-6 hours

Total benchmark time: ~9-15 hours

## Tips

1. **Run in screen/tmux** for long training sessions:
   ```bash
   screen -S benchmark
   # Run your command
   # Detach: Ctrl+A then D
   # Reattach: screen -r benchmark
   ```

2. **Monitor disk space** - checkpoints can be several GB each

3. **Check intermediate results** - don't wait for all models to finish!

4. **Use `--resume` flag** - automatically uses existing checkpoints if available

