#!/usr/bin/env bash
# Train hybrid_coat_small on fold0 — same hyper-params as hybrid_coatmini baseline.
#
# Differences vs. baseline:
#   - model: hybrid_coat_small  (embed_dims=80/160/320/640, ~13.3M params)
#   - everything else identical for a clean scientific comparison
#
# Usage:
#   bash scripts/train_coat_small_fold0.sh            # full 40-epoch run
#   SANITY=1 bash scripts/train_coat_small_fold0.sh   # 5-epoch sanity check

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FOLDS_DIR="${FOLDS_DIR:-/workspace/data/folds}"
if [[ ! -d "${FOLDS_DIR}" ]] && [[ -d "${REPO_ROOT}/data/folds" ]]; then
  FOLDS_DIR="${REPO_ROOT}/data/folds"
fi

FOLD_CSV="${FOLDS_DIR}/fold0.csv"
MODEL="hybrid_coat_small"
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
  EPOCHS=5
  echo "[sanity] 5 epochs @224 — verifying model trains without error"
fi

cd "${REPO_ROOT}"
echo "Training ${MODEL} on fold0"
echo "  fold_csv  = ${FOLD_CSV}"
echo "  img_size  = ${IMG_SIZE}  batch_size = ${BATCH_SIZE}  epochs = ${EPOCHS}"
echo "  loss      = ${LOSS}  precision = ${PRECISION}"
[[ -n "${DATA_ROOT}" ]] && echo "  data_root = ${DATA_ROOT} (local NVMe)"
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
  --weight_decay 1e-4 \
  --lr_scheduler cosine \
  --patience 10 \
  --monitor val_qwk \
  --precision "${PRECISION}" \
  --seed 42

echo ""
echo "[done] Checkpoints in lightning_logs/version_*/checkpoints/"
echo "       Evaluate with:"
echo "       python scripts/evaluate.py \\"
echo "         --checkpoint <best_ckpt.ckpt> \\"
echo "         --data_csv ${FOLD_CSV} \\"
echo "         --split test --img_size 224 --batch_size 32 \\"
echo "         --num_workers 8 \\"
echo "         --out_dir evaluation_results/fold0_coat_small_test"
