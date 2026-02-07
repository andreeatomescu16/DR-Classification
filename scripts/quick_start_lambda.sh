#!/bin/bash
# Quick start script pentru Lambda Labs
# Rulează toți pașii necesari pentru a începe training-ul

set -e

echo "=========================================="
echo "DR Classification - Quick Start Lambda Labs"
echo "=========================================="

# Verifică dacă suntem în directorul corect
if [ ! -f "setup_cloud.sh" ]; then
    echo "⚠ ERROR: Rulează acest script din root-ul proiectului!"
    exit 1
fi

# 1. Setup environment
echo ""
echo "📦 Step 1: Setting up environment..."
bash setup_cloud.sh

# 2. Verifică Kaggle API
echo ""
echo "📥 Step 2: Checking Kaggle API..."
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠ Kaggle API not configured!"
    echo "Please configure it:"
    echo "  mkdir -p ~/.kaggle"
    echo "  nano ~/.kaggle/kaggle.json"
    echo "  # Add: {\"username\": \"your_username\", \"key\": \"your_key\"}"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Press Enter after configuring Kaggle API..."
fi

# 3. Download dataset (dacă nu există)
echo ""
echo "📥 Step 3: Checking dataset..."
if [ ! -d "data/combined_dataset" ] || [ -z "$(ls -A data/combined_dataset 2>/dev/null)" ]; then
    echo "Dataset not found. Downloading..."
    mkdir -p data/combined_dataset
    cd data/combined_dataset
    kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy
    unzip *.zip
    rm *.zip
    cd ../..
    echo "✓ Dataset downloaded"
else
    echo "✓ Dataset already exists"
fi

# 4. Procesează dataset-ul
echo ""
echo "🔧 Step 4: Processing dataset..."
source venv/bin/activate

if [ ! -f "data/eyepacs_master.csv" ] || [ ! -f "data/aptos_master.csv" ]; then
    python scripts/prepare_combined_dataset.py --dataset_dir data/combined_dataset
else
    echo "✓ Master CSVs already exist"
fi

# 5. Creează K-fold splits
echo ""
echo "📊 Step 5: Creating K-fold splits..."
if [ ! -f "data/folds/fold0.csv" ]; then
    python scripts/kfold_split.py \
        --masters data/eyepacs_master.csv data/aptos_master.csv \
        --out_dir data/folds \
        --n_splits 5 \
        --seed 42
else
    echo "✓ K-fold splits already exist"
fi

# 6. Verificare finală
echo ""
echo "✅ Final verification..."
if [ -f "data/folds/fold0.csv" ]; then
    SAMPLE_COUNT=$(wc -l < data/folds/fold0.csv)
    echo "✓ Fold 0 ready with $((SAMPLE_COUNT - 1)) samples"
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
echo "1. Create screen session:"
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
