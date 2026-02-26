#!/usr/bin/env bash
# Train hybrid_coatmini on 3 folds (budget-safe). Override N_FOLDS=5 for full.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FOLDS_DIR="${FOLDS_DIR:-/workspace/data/folds}"
if [[ ! -d "${FOLDS_DIR}" ]] && [[ -d "${REPO_ROOT}/data/folds" ]]; then
  FOLDS_DIR="${REPO_ROOT}/data/folds"
fi

MODEL="${MODEL:-hybrid_coatmini}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EPOCHS="${EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-16-mixed}"
LOSS="${LOSS:-ordinal}"
N_FOLDS="${N_FOLDS:-3}"

for i in $(seq 0 $((N_FOLDS - 1))); do
  FOLD_CSV="${FOLDS_DIR}/fold${i}.csv"
  if [[ ! -f "${FOLD_CSV}" ]]; then
    echo "[error] Missing ${FOLD_CSV}. Run: bash scripts/runpod_setup_data.sh"
    exit 1
  fi
done

cd "${REPO_ROOT}"
echo "Training ${MODEL} on ${N_FOLDS} folds"

for i in $(seq 0 $((N_FOLDS - 1))); do
  FOLD_CSV="${FOLDS_DIR}/fold${i}.csv"
  echo "=========================================="
  echo "Fold ${i}/${N_FOLDS}"
  echo "=========================================="
  python -m drlib.train \
    --fold_csv "${FOLD_CSV}" \
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
done

echo "[done] Checkpoints in lightning_logs/version_*/checkpoints/"
