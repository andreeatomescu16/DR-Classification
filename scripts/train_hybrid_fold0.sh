#!/usr/bin/env bash
# Train hybrid_coatmini on fold0 — RunPod budget preset.
# Sanity: 15 ep @224 | Main: 40 ep @224, optional 5 ep @384 fine-tune.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FOLDS_DIR="${FOLDS_DIR:-/workspace/data/folds}"
if [[ ! -d "${FOLDS_DIR}" ]] && [[ -d "${REPO_ROOT}/data/folds" ]]; then
  FOLDS_DIR="${REPO_ROOT}/data/folds"
fi

FOLD_CSV="${FOLDS_DIR}/fold0.csv"
MODEL="${MODEL:-hybrid_coatmini}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EPOCHS="${EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-}"
PRECISION="${PRECISION:-16-mixed}"
LOSS="${LOSS:-ordinal}"
SANITY="${SANITY:-0}"

if [[ ! -f "${FOLD_CSV}" ]]; then
  echo "[error] Fold CSV not found: ${FOLD_CSV}"
  echo "Run: bash scripts/runpod_setup_data.sh"
  exit 1
fi

if [[ "${SANITY}" == "1" ]]; then
  EPOCHS=15
  echo "[sanity] 15 epochs @224"
fi

cd "${REPO_ROOT}"
echo "Training ${MODEL} on fold0"
echo "  fold_csv=${FOLD_CSV}"
echo "  img_size=${IMG_SIZE} batch_size=${BATCH_SIZE} epochs=${EPOCHS}"
echo "  loss=${LOSS} precision=${PRECISION}"
[[ -n "${DATA_ROOT}" ]] && echo "  data_root=${DATA_ROOT} (local storage)"
echo ""

DATA_ROOT_ARGS=()
[[ -n "${DATA_ROOT}" ]] && DATA_ROOT_ARGS=(--data_root "${DATA_ROOT}")

python -m drlib.train \
  --fold_csv "${FOLD_CSV}" \
  "${DATA_ROOT_ARGS[@]}" \
  --model "${MODEL}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS}" \
  --loss "${LOSS}" \
  --lr 1e-4 \
  --lr_scheduler cosine \
  --patience 10 \
  --monitor val_qwk \
  --precision "${PRECISION}" \
  --seed 42

echo ""
echo "[done] Checkpoints in lightning_logs/version_*/checkpoints/"
