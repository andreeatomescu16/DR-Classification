#!/usr/bin/env bash
set -euo pipefail

echo "==============================================="
echo "RunPod data setup (Kaggle -> manifest -> folds)"
echo "==============================================="

# Paths: override via env for RunPod; defaults work locally (project/data/raw, project/data/folds)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -d /workspace ]] && [[ -w /workspace ]] 2>/dev/null; then
  RAW_DIR="${RAW_DIR:-/workspace/data/raw}"
  FOLDS_DIR="${FOLDS_DIR:-/workspace/data/folds}"
else
  RAW_DIR="${RAW_DIR:-${REPO_ROOT}/data/raw}"
  FOLDS_DIR="${FOLDS_DIR:-${REPO_ROOT}/data/folds}"
fi

DATASET_SLUG="ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

echo "RAW_DIR=${RAW_DIR}"
echo "FOLDS_DIR=${FOLDS_DIR}"
echo "REPO_ROOT=${REPO_ROOT}"
mkdir -p "${RAW_DIR}" "${FOLDS_DIR}"

echo ""
echo "[1/7] Installing Python dependencies (if missing)..."
"${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null
if [[ -f "${REPO_ROOT}/requirements.txt" ]]; then
  "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements.txt" >/dev/null
fi
"${PYTHON_BIN}" -m pip install kaggle pandas scikit-learn >/dev/null

echo ""
echo "[2/7] Kaggle API credential setup..."
mkdir -p "${HOME}/.kaggle"
if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  chmod 600 "${HOME}/.kaggle/kaggle.json"
  echo "[ok] Found ${HOME}/.kaggle/kaggle.json"
else
  echo "[action required] Missing ${HOME}/.kaggle/kaggle.json"
  echo "Create it now with your Kaggle API token:"
  echo "  1) Kaggle -> Account -> Create New API Token"
  echo "  2) Upload token then run:"
  echo "     mkdir -p ~/.kaggle"
  echo "     cp /path/to/kaggle.json ~/.kaggle/kaggle.json"
  echo "     chmod 600 ~/.kaggle/kaggle.json"
  exit 1
fi

echo ""
echo "[3/7] Download dataset to ${RAW_DIR} ..."
cd "${RAW_DIR}"
kaggle datasets download -d "${DATASET_SLUG}" -p "${RAW_DIR}" --force

echo ""
echo "[4/7] Unzip dataset..."
ZIP_FILE="$(ls -1 *.zip | head -n 1 || true)"
if [[ -z "${ZIP_FILE}" ]]; then
  echo "[error] No zip file found in ${RAW_DIR} after Kaggle download."
  exit 1
fi
unzip -o "${ZIP_FILE}" -d "${RAW_DIR}" >/dev/null
echo "[ok] Unzipped ${ZIP_FILE}"

echo ""
echo "[5/7] Locate augmented_resized_V2 and verify file count..."
AUG_DIR="$("${PYTHON_BIN}" - <<PY
from pathlib import Path
root = Path("${RAW_DIR}")
matches = sorted([p for p in root.rglob("augmented_resized_V2") if p.is_dir()], key=lambda p: len(str(p)))
print(str(matches[0]) if matches else "")
PY
)"
if [[ -z "${AUG_DIR}" ]]; then
  echo "[error] Could not find augmented_resized_V2 under ${RAW_DIR}"
  echo "Fallback if Kaggle API is unavailable:"
  echo "  - Download dataset zip manually from browser"
  echo "  - Upload to ${RAW_DIR}"
  echo "  - unzip <file>.zip -d ${RAW_DIR}"
  exit 1
fi
echo "[ok] Found dataset root: ${AUG_DIR}"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
root = Path("${RAW_DIR}")
matches = sorted([p for p in root.rglob("augmented_resized_V2") if p.is_dir()], key=lambda p: len(str(p)))
aug = matches[0]
cnt = 0
for ext in ("*.jpg", "*.jpeg", "*.png"):
    cnt += len(list(aug.rglob(ext)))
print(f"[ok] discovered images under augmented_resized_V2: {cnt}")
PY

echo ""
echo "[6/7] Build master manifest + grouped 5-fold CSVs..."
cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/build_runpod_folds.py \
  --raw_root "${RAW_DIR}" \
  --folds_dir "${FOLDS_DIR}" \
  --n_splits 5 \
  --seed 42 \
  --img_size 224 \
  --batch_size 4

echo ""
echo "[7/7] Verification summary..."
"${PYTHON_BIN}" - <<PY
import pandas as pd
from pathlib import Path

folds = Path("${FOLDS_DIR}")
master = pd.read_csv(folds / "master.csv")
print(f"master rows: {len(master)}")
print("master class distribution:")
print(master["label"].value_counts().sort_index().to_string())

for i in range(5):
    f = pd.read_csv(folds / f"fold{i}.csv")
    print(f"\nfold{i} split counts:")
    print(f["split"].value_counts().to_string())
PY

echo ""
echo "==============================================="
echo "[done] Data setup complete."
echo "Files generated:"
echo "  - ${FOLDS_DIR}/master.csv"
echo "  - ${FOLDS_DIR}/fold0.csv ... fold4.csv"
echo "==============================================="
