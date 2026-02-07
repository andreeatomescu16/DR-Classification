#!/bin/bash
# Backup script pentru rezultatele de training de pe Lambda Labs
# Creează un archive cu toate checkpoint-urile, logs și rezultatele

set -e

BACKUP_DIR="results_backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="results_backup_${TIMESTAMP}.tar.gz"

echo "=========================================="
echo "Backup Training Results"
echo "=========================================="

# Creează directorul de backup
mkdir -p "$BACKUP_DIR"

# Copiază checkpoint-uri
if [ -d "lightning_logs" ]; then
    echo "📦 Copying checkpoints..."
    cp -r lightning_logs "$BACKUP_DIR/"
    echo "✓ Checkpoints copied"
else
    echo "⚠ No lightning_logs directory found"
fi

# Copiază rezultate benchmark
if [ -d "benchmark_results" ]; then
    echo "📦 Copying benchmark results..."
    cp -r benchmark_results "$BACKUP_DIR/"
    echo "✓ Benchmark results copied"
else
    echo "⚠ No benchmark_results directory found"
fi

# Copiază master CSV-uri (dacă există)
if [ -d "data" ]; then
    echo "📦 Copying data files..."
    mkdir -p "$BACKUP_DIR/data"
    if [ -f "data/eyepacs_master.csv" ]; then
        cp data/eyepacs_master.csv "$BACKUP_DIR/data/"
    fi
    if [ -f "data/aptos_master.csv" ]; then
        cp data/aptos_master.csv "$BACKUP_DIR/data/"
    fi
    if [ -d "data/folds" ]; then
        cp -r data/folds "$BACKUP_DIR/data/"
    fi
    echo "✓ Data files copied"
fi

# Creează archive
echo "📦 Creating archive..."
tar -czf "$BACKUP_FILE" "$BACKUP_DIR"
echo "✓ Archive created: $BACKUP_FILE"

# Verifică dimensiunea
SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo ""
echo "=========================================="
echo "Backup Complete!"
echo "=========================================="
echo "Archive: $BACKUP_FILE"
echo "Size: $SIZE"
echo ""
echo "To download to your laptop, run:"
echo "  scp ubuntu@<ip-address>:~/DR-Classification/$BACKUP_FILE ./"
echo ""

# Șterge directorul temporar
rm -rf "$BACKUP_DIR"
