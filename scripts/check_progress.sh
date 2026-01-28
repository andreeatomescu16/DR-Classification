#!/bin/bash
# Quick script to check training progress

echo "=== Current Training Progress ==="
echo ""

# Find latest version
LATEST=$(ls -td lightning_logs/version_* 2>/dev/null | head -1)

if [ -z "$LATEST" ]; then
    echo "No training runs found!"
    exit 1
fi

echo "Latest training: $LATEST"
echo ""

# Check for metrics
if [ -f "$LATEST/metrics.csv" ]; then
    echo "=== Latest Metrics ==="
    tail -1 "$LATEST/metrics.csv" | column -t -s,
    echo ""
fi

# Check for checkpoints
if [ -d "$LATEST/checkpoints" ]; then
    echo "=== Checkpoints ==="
    ls -lh "$LATEST/checkpoints/" | tail -5
    echo ""
fi

# Check hyperparameters
if [ -f "$LATEST/hparams.yaml" ]; then
    echo "=== Model Info ==="
    grep -E "model_name|img_size|lr|epochs" "$LATEST/hparams.yaml" 2>/dev/null || echo "Could not read hparams"
    echo ""
fi

echo "To monitor in real-time, run:"
echo "  python scripts/monitor_training.py"

