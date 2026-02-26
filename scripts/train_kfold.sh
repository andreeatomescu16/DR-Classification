#!/usr/bin/env bash
# Train all 5 folds — RunPod-ready. Override MODEL, FOLDS_DIR, etc. via env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FOLDS_DIR="${FOLDS_DIR:-/workspace/data/folds}"
if [[ ! -d "${FOLDS_DIR}" ]] && [[ -d "${REPO_ROOT}/data/folds" ]]; then
  FOLDS_DIR="${REPO_ROOT}/data/folds"
fi

MODEL="${MODEL:-efficientnet_b4.ra2_in1k}"
IMG_SIZE="${IMG_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-16-mixed}"

for i in 0 1 2 3 4; do
  FOLD_CSV="${FOLDS_DIR}/fold${i}.csv"
  if [[ ! -f "${FOLD_CSV}" ]]; then
    echo "[error] Missing ${FOLD_CSV}. Run: bash scripts/runpod_setup_data.sh"
    exit 1
  fi
done

cd "${REPO_ROOT}"
echo "Training ${MODEL} on all 5 folds"
echo "  folds_dir=${FOLDS_DIR}"
echo "  img_size=${IMG_SIZE} batch_size=${BATCH_SIZE} epochs=${EPOCHS}"
echo ""

for i in 0 1 2 3 4; do
  FOLD_CSV="${FOLDS_DIR}/fold${i}.csv"
  echo "=========================================="
  echo "Fold ${i}/5"
  echo "=========================================="
  python -m drlib.train \
    --fold_csv "${FOLD_CSV}" \
    --model "${MODEL}" \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --loss weighted_ce \
    --use_class_weights \
    --lr 1e-4 \
    --lr_scheduler cosine \
    --patience 10 \
    --monitor val_qwk \
    --precision "${PRECISION}" \
    --seed 42
  echo ""
done

echo "[done] All folds complete. Checkpoints in lightning_logs/version_*/checkpoints/"
