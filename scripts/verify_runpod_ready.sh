#!/usr/bin/env bash
# Quick verification before training — checks GPU, data, folds, one batch load.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FOLDS_DIR="${FOLDS_DIR:-/workspace/data/folds}"
if [[ ! -d "${FOLDS_DIR}" ]] && [[ -d "${REPO_ROOT}/data/folds" ]]; then
  FOLDS_DIR="${REPO_ROOT}/data/folds"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

echo "=========================================="
echo "RunPod readiness check"
echo "=========================================="
echo "FOLDS_DIR=${FOLDS_DIR}"
echo ""

# 1. GPU
echo "[1/4] GPU check"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  echo "[ok] GPU detected"
else
  echo "[warn] nvidia-smi not found — will use CPU"
fi
echo ""

# 2. Fold CSVs exist
echo "[2/4] Fold CSVs"
for i in 0 1 2 3 4; do
  fp="${FOLDS_DIR}/fold${i}.csv"
  if [[ -f "${fp}" ]]; then
    lines=$(wc -l < "${fp}")
    echo "  fold${i}.csv: ${lines} rows"
  else
    echo "  [fail] ${fp} not found"
    exit 1
  fi
done
echo "[ok] All fold CSVs present"
echo ""

# 3. Disk space
echo "[3/4] Disk space"
df -h . | tail -1
echo ""

# 4. One-batch load test
echo "[4/4] One-batch load test (fold0, split=train)"
cd "${REPO_ROOT}"
"${PYTHON_BIN}" - <<PY
from pathlib import Path
from torch.utils.data import DataLoader
from drlib.datasets import DRDataset
from drlib.transforms import get_train_tf

csv_path = Path("${FOLDS_DIR}") / "fold0.csv"
ds = DRDataset(str(csv_path), split="train", tfm=get_train_tf(224, remove_borders=True))
dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
xb, yb = next(iter(dl))
print(f"  batch: images={tuple(xb.shape)} labels={tuple(yb.shape)}")
print("[ok] Batch load succeeded")
PY

echo ""
echo "=========================================="
echo "[done] Ready for training."
echo "  bash scripts/train_fold0.sh"
echo "  bash scripts/train_kfold.sh"
echo "=========================================="
