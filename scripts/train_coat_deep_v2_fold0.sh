#!/usr/bin/env bash
# Train hybrid_coat_deep on fold0 with warmup+cosine — v2 conservative optimization.
# Mirrors scripts/train_focal_fold0.sh; only optimization knobs + output dir differ.
# Sanity: SANITY=1 → 5 epochs (quick smoke test).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FOLDS_DIR="${FOLDS_DIR:-/workspace/data/folds}"
if [[ ! -d "${FOLDS_DIR}" ]] && [[ -d "${REPO_ROOT}/data/folds" ]]; then
  FOLDS_DIR="${REPO_ROOT}/data/folds"
fi

FOLD_CSV="${FOLDS_DIR}/fold0.csv"
MODEL="${MODEL:-hybrid_coat_deep}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EPOCHS="${EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-}"
PRECISION="${PRECISION:-16-mixed}"

# Loss knobs (kept identical to focal experiment)
LOSS="${LOSS:-focal_weighted}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
WEIGHT_MODE="${WEIGHT_MODE:-sqrt_inverse}"

# Optimization knobs (v2)
LR="${LR:-3e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
WARMUP_START_LR="${WARMUP_START_LR:-1e-6}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
PATIENCE="${PATIENCE:-10}"
SEED="${SEED:-42}"

SANITY="${SANITY:-0}"
RUN_DIR="${RUN_DIR:-coat_deep_v2_fold0}"

if [[ ! -f "${FOLD_CSV}" ]]; then
  echo "[error] Fold CSV not found: ${FOLD_CSV}"
  echo "Run: bash scripts/runpod_setup_data.sh"
  exit 1
fi

if [[ "${SANITY}" == "1" ]]; then
  EPOCHS=5
  echo "[sanity] 5 epochs @${IMG_SIZE} — smoke test only"
fi

cd "${REPO_ROOT}"
echo "Training ${MODEL} on fold0 (coat_deep v2)"
echo "  fold_csv=${FOLD_CSV}"
echo "  img_size=${IMG_SIZE} batch_size=${BATCH_SIZE} epochs=${EPOCHS}"
echo "  loss=${LOSS} focal_gamma=${FOCAL_GAMMA} weight_mode=${WEIGHT_MODE}"
echo "  lr=${LR} warmup_epochs=${WARMUP_EPOCHS} warmup_start_lr=${WARMUP_START_LR} scheduler=${LR_SCHEDULER}"
echo "  precision=${PRECISION}"
echo "  run_dir=${RUN_DIR}"
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
  --focal_gamma "${FOCAL_GAMMA}" \
  --weight_mode "${WEIGHT_MODE}" \
  --lr "${LR}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --warmup_start_lr "${WARMUP_START_LR}" \
  --patience "${PATIENCE}" \
  --monitor val_qwk \
  --precision "${PRECISION}" \
  --seed "${SEED}" \
  --default_root_dir "${RUN_DIR}"

echo ""
echo "[done] Checkpoints under ${RUN_DIR}/"
