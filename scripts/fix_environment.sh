#!/bin/bash
# Fix environment issues on Lambda Labs

set -e

echo "=========================================="
echo "Fixing Environment"
echo "=========================================="

# Verifică dacă suntem în directorul corect
if [ ! -f "scripts/benchmark.py" ]; then
    echo "⚠ ERROR: Rulează acest script din root-ul proiectului!"
    exit 1
fi

# Verifică dacă venv există
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment exists"
fi

# Activează venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Instalează dependențele
echo "📚 Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning timm albumentations opencv-python-headless scikit-learn pillow pyyaml matplotlib seaborn pandas tqdm kaggle

# Verifică instalarea
echo ""
echo "✅ Verifying installation..."
python3 -c "
import torch
import seaborn
import pytorch_lightning as pl
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ Seaborn: {seaborn.__version__}')
print(f'✓ Lightning: {pl.__version__}')
"

echo ""
echo "=========================================="
echo "✅ Environment Fixed!"
echo "=========================================="
echo ""
echo "Now activate environment and run training:"
echo "  source venv/bin/activate"
echo "  python scripts/benchmark.py --fold_csv data/folds/fold0.csv --epochs 30 --batch_size 32 --num_workers 8"
echo ""
