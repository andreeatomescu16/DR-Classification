#!/bin/bash
# Complete setup script for Lambda Labs
# Rulează totul automat: verificare, clone, setup, download dataset, procesare

set -e

echo "=========================================="
echo "DR Classification - Complete Lambda Setup"
echo "=========================================="

# 1. Verificare GPU
echo ""
echo "🎮 Step 1: Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "✓ GPU detected: $GPU_NAME"

# 2. Clone repository (dacă nu există deja)
echo ""
echo "📦 Step 2: Cloning repository..."
if [ -d "DR-Classification" ]; then
    echo "⚠ Repository already exists, skipping clone..."
    cd DR-Classification
else
    git clone https://github.com/andreeatomescu16/DR-Classification.git
    cd DR-Classification
fi

# 3. Setup environment
echo ""
echo "🔧 Step 3: Setting up environment..."
if [ -f "setup_cloud.sh" ]; then
    chmod +x setup_cloud.sh
    bash setup_cloud.sh
else
    echo "⚠ setup_cloud.sh not found, running manual setup..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning timm albumentations opencv-python-headless scikit-learn pillow pyyaml matplotlib seaborn pandas tqdm kaggle
fi

# 4. Activate environment
source venv/bin/activate

# 5. Verifică Kaggle API
echo ""
echo "📥 Step 4: Checking Kaggle API..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠ Kaggle API not configured!"
    echo "Please configure it now:"
    echo ""
    echo "Run these commands:"
    echo "  mkdir -p ~/.kaggle"
    echo "  nano ~/.kaggle/kaggle.json"
    echo ""
    echo "Add this content (replace with your credentials):"
    echo '  {'
    echo '    "username": "andreeatomescu",'
    echo '    "key": "KGAT_fa40c59d94f34c394164777195788046"'
    echo '  }'
    echo ""
    echo "Then: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Press Enter after configuring Kaggle API..."
else
    echo "✓ Kaggle API configured"
fi

# 6. Download dataset (dacă nu există)
echo ""
echo "📥 Step 5: Checking dataset..."
if [ ! -d "data/combined_dataset" ] || [ -z "$(ls -A data/combined_dataset 2>/dev/null)" ]; then
    echo "Dataset not found. Downloading from Kaggle..."
    echo "⚠ This will take 15-45 minutes..."
    mkdir -p data/combined_dataset
    cd data/combined_dataset
    kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy
    echo "Extracting files..."
    unzip -q *.zip
    rm *.zip
    cd ../..
    echo "✓ Dataset downloaded and extracted"
else
    echo "✓ Dataset already exists"
fi

# 7. Procesează dataset-ul
echo ""
echo "🔧 Step 6: Processing dataset..."
if [ ! -f "data/eyepacs_master.csv" ] || [ ! -f "data/aptos_master.csv" ]; then
    python scripts/prepare_combined_dataset.py --dataset_dir data/combined_dataset
else
    echo "✓ Master CSVs already exist"
fi

# 8. Creează K-fold splits
echo ""
echo "📊 Step 7: Creating K-fold splits..."
if [ ! -f "data/folds/fold0.csv" ]; then
    python scripts/kfold_split.py \
        --masters data/eyepacs_master.csv data/aptos_master.csv \
        --out_dir data/folds \
        --n_splits 5 \
        --seed 42
else
    echo "✓ K-fold splits already exist"
fi

# 9. Verificare finală
echo ""
echo "✅ Final verification..."
if [ -f "data/folds/fold0.csv" ]; then
    SAMPLE_COUNT=$(wc -l < data/folds/fold0.csv)
    echo "✓ Fold 0 ready with $((SAMPLE_COUNT - 1)) samples"
    
    # Verifică GPU
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU ready: {torch.cuda.get_device_name(0)}')
    print(f'✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠ GPU not available!')
    exit(1)
"
else
    echo "✗ ERROR: Fold 0 not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete! Ready for Training!"
echo "=========================================="
echo ""
echo "🚀 To start training:"
echo ""
echo "1. Create screen session (for persistence):"
echo "   screen -S training"
echo ""
echo "2. Activate environment and start training:"
echo "   source venv/bin/activate"
echo "   python scripts/benchmark.py \\"
echo "       --fold_csv data/folds/fold0.csv \\"
echo "       --epochs 30 \\"
echo "       --batch_size 32 \\"
echo "       --num_workers 8"
echo ""
echo "3. Detach screen: Ctrl+A then D"
echo ""
echo "4. Monitor training:"
echo "   screen -r training  # Reattach"
echo "   # Or: tail -f benchmark_results/logs/*.log"
echo ""
echo "=========================================="
