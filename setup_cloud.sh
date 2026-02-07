#!/bin/bash
# Setup script for Lambda Labs GPU instances
# Optimized for A10 GPU (24GB VRAM)
# This script installs all dependencies and prepares the environment

set -e

echo "=========================================="
echo "DR Classification - Lambda Labs Setup"
echo "=========================================="
echo "Optimized for: A10 GPU (24GB VRAM)"
echo "=========================================="

# Check Python version
echo ""
echo "📋 Checking Python version..."
python3 --version

# Check GPU availability
echo ""
echo "🎮 Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
    echo "✓ GPU detected: $GPU_NAME ($GPU_MEMORY)"
else
    echo "⚠ WARNING: nvidia-smi not found. GPU may not be available."
    exit 1
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git screen unzip

# Create virtual environment
echo ""
echo "🐍 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ Virtual environment already exists, skipping..."
else
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
echo "   (This may take a few minutes...)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
echo ""
echo "📚 Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠ requirements.txt not found, installing core dependencies..."
    pip install pytorch-lightning timm albumentations opencv-python-headless scikit-learn pillow pyyaml matplotlib seaborn pandas tqdm kaggle
fi

# Install Kaggle API (for dataset download)
echo ""
echo "📥 Installing Kaggle API..."
pip install kaggle

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠ WARNING: CUDA not available!')
    exit(1)
"

# Check if data exists
echo ""
echo "📊 Checking data files..."
if [ -f "data/folds/fold0.csv" ]; then
    echo "✓ Data file found: data/folds/fold0.csv"
    SAMPLE_COUNT=$(wc -l < data/folds/fold0.csv)
    echo "  Samples: $((SAMPLE_COUNT - 1))"
else
    echo "⚠ WARNING: data/folds/fold0.csv not found!"
    echo "  You need to:"
    echo "    1. Download dataset from Kaggle"
    echo "    2. Process dataset with: python scripts/prepare_combined_dataset.py"
    echo "    3. Create folds with: python scripts/kfold_split.py"
fi

# Make scripts executable
echo ""
echo "🔧 Making scripts executable..."
chmod +x scripts/*.sh scripts/*.py 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download dataset (if not done):"
echo "   # Configure Kaggle API first:"
echo "   mkdir -p ~/.kaggle"
echo "   nano ~/.kaggle/kaggle.json"
echo "   # Add your credentials"
echo "   chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "   # Download dataset:"
echo "   cd data/combined_dataset"
echo "   kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"
echo "   unzip *.zip && rm *.zip"
echo "   cd ../.."
echo ""
echo "3. Process dataset:"
echo "   python scripts/prepare_combined_dataset.py --dataset_dir data/combined_dataset"
echo "   python scripts/kfold_split.py --masters data/eyepacs_master.csv data/aptos_master.csv --out_dir data/folds"
echo ""
echo "4. Start training (use screen for persistence):"
echo "   screen -S training"
echo "   source venv/bin/activate"
echo "   python scripts/benchmark.py --fold_csv data/folds/fold0.csv --epochs 30 --batch_size 32 --num_workers 8"
echo "   # Press Ctrl+A then D to detach"
echo ""
echo "5. Monitor training:"
echo "   screen -r training  # Reattach"
echo "   # Or in another terminal:"
echo "   tail -f benchmark_results/logs/*.log"
echo ""
echo "6. Backup results when done:"
echo "   bash scripts/backup_results.sh"
echo "   # Then download:"
echo "   scp ubuntu@<ip>:~/DR-Classification/results_backup_*.tar.gz ./"
echo ""
echo "=========================================="
echo "🎉 Ready for training!"
echo "=========================================="
echo ""
