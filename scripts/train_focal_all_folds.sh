#!/usr/bin/env bash
# 5-fold cross-validation runner for hybrid_coatmini + focal_weighted (controlled config).
# Trains folds 0–4 sequentially and evaluates the best checkpoint for each fold.
# Resumable: if fold{i}/eval/metrics.csv exists, that fold is skipped.

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
DATA_ROOT="${DATA_ROOT:-}"
PRECISION="${PRECISION:-16-mixed}"

# Focal-weighted knobs (keep identical to scripts/train_focal_fold0.sh)
LOSS="${LOSS:-focal_weighted}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
WEIGHT_MODE="${WEIGHT_MODE:-sqrt_inverse}"

SANITY="${SANITY:-0}"

if [[ "${SANITY}" == "1" ]]; then
  EPOCHS=5
  echo "[sanity] 5 epochs @${IMG_SIZE} — smoke test only"
fi

cd "${REPO_ROOT}"

DATA_ROOT_ARGS=()
[[ -n "${DATA_ROOT}" ]] && DATA_ROOT_ARGS=(--data_root "${DATA_ROOT}")

BASE_DIR="experiments/focal_crossval"
mkdir -p "${BASE_DIR}"

for FOLD in 0 1 2 3 4; do
  echo ""
  echo "======================================================================"
  echo "  FOLD ${FOLD} / 4  —  ${MODEL} + ${LOSS}"
  echo "======================================================================"

  FOLD_CSV="${FOLDS_DIR}/fold${FOLD}.csv"
  if [[ ! -f "${FOLD_CSV}" ]]; then
    echo "[error] Fold CSV not found: ${FOLD_CSV}"
    exit 1
  fi

  OUT_DIR="${BASE_DIR}/fold${FOLD}"
  EVAL_DIR="${OUT_DIR}/eval"
  METRICS_CSV="${EVAL_DIR}/metrics.csv"

  if [[ -f "${METRICS_CSV}" ]]; then
    echo "[skip] Found existing metrics: ${METRICS_CSV}"
    continue
  fi

  mkdir -p "${OUT_DIR}"

  echo "Training:"
  echo "  fold_csv=${FOLD_CSV}"
  echo "  out_dir=${OUT_DIR}"
  echo "  img_size=${IMG_SIZE} batch_size=${BATCH_SIZE} epochs=${EPOCHS}"
  echo "  loss=${LOSS} focal_gamma=${FOCAL_GAMMA} weight_mode=${WEIGHT_MODE}"
  echo "  lr=1e-4 scheduler=cosine patience=10 precision=${PRECISION} seed=42"
  [[ -n "${DATA_ROOT}" ]] && echo "  data_root=${DATA_ROOT}"
  echo ""

  python -m drlib.train \
    --fold_csv "${FOLD_CSV}" \
    "${DATA_ROOT_ARGS[@]}" \
    --model "${MODEL}" \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --loss "${LOSS}" \
    --focal_gamma "${FOCAL_GAMMA}" \
    --weight_mode "${WEIGHT_MODE}" \
    --lr 1e-4 \
    --lr_scheduler cosine \
    --patience 10 \
    --monitor val_qwk \
    --precision "${PRECISION}" \
    --seed 42 \
    --default_root_dir "${OUT_DIR}"

  echo ""
  echo "Selecting best checkpoint..."
  CKPT="$(ls -1 "${OUT_DIR}"/lightning_logs/version_*/checkpoints/*val_qwk*.ckpt 2>/dev/null | tail -n 1 || true)"
  if [[ -z "${CKPT}" ]]; then
    echo "[error] Could not find best checkpoint under: ${OUT_DIR}/lightning_logs/version_*/checkpoints/"
    exit 1
  fi
  echo "  ckpt=${CKPT}"

  mkdir -p "${EVAL_DIR}"
  echo ""
  echo "Evaluating fold ${FOLD}..."
  python scripts/evaluate.py \
    --checkpoint "${CKPT}" \
    --data_csv "${FOLD_CSV}" \
    --split val \
    --img_size "${IMG_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --out_dir "${EVAL_DIR}"

  if [[ -f "${METRICS_CSV}" ]]; then
    echo ""
    echo "[ok] Saved: ${METRICS_CSV}"
  else
    echo ""
    echo "[warn] Evaluation finished but metrics.csv not found at: ${METRICS_CSV}"
  fi
done

echo ""
echo "[done] Cross-val outputs in ${BASE_DIR}/"
