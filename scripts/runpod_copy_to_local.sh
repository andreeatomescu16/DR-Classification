#!/usr/bin/env bash
# Copy dataset from /workspace (remote storage) to local NVMe for faster I/O.
# Run this ONCE after runpod_setup_data.sh, then train with DATA_ROOT=<LOCAL_DIR>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source: /workspace/data (remote, slow)
# Target: local disk (fast) - /tmp on RunPod is usually local NVMe
SOURCE_DATA="${SOURCE_DATA:-/workspace/data}"
LOCAL_DIR="${LOCAL_DIR:-/tmp/localdata}"

echo "==============================================="
echo "Copy dataset to local storage (faster I/O)"
echo "==============================================="
echo "Source: ${SOURCE_DATA}"
echo "Target: ${LOCAL_DIR}"
echo ""

if [[ ! -d "${SOURCE_DATA}/raw" ]]; then
  echo "[error] Source not found: ${SOURCE_DATA}/raw"
  echo "Run scripts/runpod_setup_data.sh first."
  exit 1
fi

# Check target has space (need ~20–30GB for augmented_resized_V2)
# Use parent dir for df when target doesn't exist yet
CHECK_DIR="${LOCAL_DIR}"
[[ -d "${CHECK_DIR}" ]] || CHECK_DIR="$(dirname "${LOCAL_DIR}")"
AVAIL=$(df -BG "${CHECK_DIR}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G' || true)
if [[ -z "${AVAIL}" ]] || ! [[ "${AVAIL}" =~ ^[0-9]+$ ]]; then
  AVAIL=$(df -h "${CHECK_DIR}" 2>/dev/null | tail -1 | awk '{gsub(/[GTM]/, "", $4); print $4+0}' || echo "0")
fi
[[ -z "${AVAIL}" ]] && AVAIL=0
if [[ "${AVAIL}" -lt 15 ]]; then
  echo "[warn] Low space on ${CHECK_DIR}: ${AVAIL}GB free. Need ~20GB+ for dataset."
  echo "Try: LOCAL_DIR=/dev/shm/localdata bash $0  (or another path with more space)"
  exit 1
fi

mkdir -p "${LOCAL_DIR}"
echo "[1/2] Copying raw data (this may take 5–15 min)..."
rsync -ah --info=progress2 "${SOURCE_DATA}/raw/" "${LOCAL_DIR}/raw/" || { echo "[error] rsync failed"; exit 1; }

echo ""
echo "[2/2] Verifying..."
AUG_COUNT=$(find "${LOCAL_DIR}" -path "*augmented_resized_V2*" -name "*.jpg" 2>/dev/null | wc -l || echo "0")
if [[ "${AUG_COUNT}" -lt 1000 ]]; then
  echo "[error] Too few images found (${AUG_COUNT}). Copy may have failed."
  exit 1
fi
echo "[ok] Found ${AUG_COUNT} images in augmented_resized_V2"

echo ""
echo "==============================================="
echo "[done] Dataset copied to ${LOCAL_DIR}"
echo ""
echo "Train with:"
echo "  DATA_ROOT=${LOCAL_DIR} NUM_WORKERS=16 bash scripts/train_hybrid_fold0.sh"
echo ""
echo "Or add to your command:"
echo "  --data_root ${LOCAL_DIR}"
echo "==============================================="
