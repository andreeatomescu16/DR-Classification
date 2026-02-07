#!/bin/bash
# Fix PyTorch version conflict on Lambda Labs
# Uninstalls conflicting packages and reinstalls in venv

set -e

echo "=========================================="
echo "Fixing PyTorch Version Conflict"
echo "=========================================="

# Creează venv dacă nu există
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activează venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Uninstall conflicting packages din user space
echo "🧹 Cleaning up conflicting packages..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Instalează PyTorch cu CUDA (versiuni compatibile)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalează restul dependențelor
echo "📚 Installing other dependencies..."
pip install pytorch-lightning timm albumentations opencv-python-headless scikit-learn pillow pyyaml matplotlib seaborn pandas tqdm kaggle

# Verifică instalarea
echo ""
echo "✅ Verifying installation..."
python3 -c "
import torch
import torchvision
import timm
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Torchvision: {torchvision.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
print(f'✓ timm: {timm.__version__}')
"

echo ""
echo "=========================================="
echo "✅ PyTorch Conflict Fixed!"
echo "=========================================="
echo ""
echo "Now activate environment and run training:"
echo "  source venv/bin/activate"
echo "  python scripts/benchmark.py --fold_csv data/folds/fold0.csv --epochs 30 --batch_size 32 --num_workers 8"
echo ""
