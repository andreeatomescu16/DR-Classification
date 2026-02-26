# RunPod — Start Training (Exact Steps)

## Quality gates

**Do NOT proceed to training if:**
- `scripts/runpod_setup_data.sh` exits with non-zero
- `scripts/verify_runpod_ready.sh` fails
- Folds or dataset paths are missing

The setup script **must** finish successfully. It exits with non-zero if:
- Kaggle credentials missing
- Dataset not found
- Folds not created
- Leakage check fails
- Batch load fails

---

## 1. Create Pod

- **GPU**: RTX 4090 (24GB)
- **Volume Disk**: 80GB (persistent, mounted to `/workspace`)
- **Container Disk**: 20GB
- **Template**: PyTorch 2.x or Ubuntu + CUDA
- **SSH**: Enable, add your public key

---

## 2. SSH in

```bash
ssh <USER>@ssh.runpod.io -i ~/.ssh/your_key
```

Verify GPU:
```bash
nvidia-smi
```

---

## 3. Clone repo and install deps

```bash
cd /workspace
git clone https://github.com/andreeatomescu16/DR-Classification.git
cd DR-Classification
pip install -r requirements.txt kaggle
apt-get update && apt-get install -y unzip
```

---

## 4. Configure Kaggle

```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

---

## 5. Data setup (must succeed)

```bash
cd /workspace/DR-Classification
bash scripts/runpod_setup_data.sh
```

If it fails, **do not** start training. Fix the error first.

---

## 6. Verification

```bash
bash scripts/verify_runpod_ready.sh
```

---

## 6b. Copy data to local storage (recommended — much faster I/O)

`/workspace` is remote storage; reading many images from it is slow. Copy to local NVMe first:

```bash
bash scripts/runpod_copy_to_local.sh
```

This copies to `/tmp/localdata`. Then train with:

```bash
DATA_ROOT=/tmp/localdata NUM_WORKERS=16 bash scripts/train_hybrid_fold0.sh
```

---

## 7. Start training (tmux)

```bash
tmux new -s drtrain
cd /workspace/DR-Classification
# With local data (recommended):
DATA_ROOT=/tmp/localdata NUM_WORKERS=16 bash scripts/train_hybrid_fold0.sh
# Or without (slower):
bash scripts/train_hybrid_fold0.sh
```

Detach: `Ctrl+B`, then `D`  
Reattach: `tmux attach -t drtrain`

---

## 8. Where outputs/checkpoints are stored

| Path | Content |
|------|---------|
| `lightning_logs/version_*/checkpoints/` | Best (by QWK) + last checkpoint |
| `lightning_logs/version_*/checkpoints/*.ckpt` | `.ckpt` files |

---

## 9. Download outputs (scp/rsync)

From your Mac:

```bash
scp -r -i ~/.ssh/id_lambda_labs <USER>@ssh.runpod.io:/workspace/DR-Classification/lightning_logs ./lightning_logs_backup
```

---

## 10. Stop pod safely

1. Download checkpoints (step 9)
2. Stop pod from RunPod dashboard
3. Data on Volume Disk is erased when pod is terminated — download outputs before stopping

---

## Quick sanity check (15 epochs)

```bash
SANITY=1 bash scripts/train_hybrid_fold0.sh
```

---

## Alternative: EfficientNet or ViT

```bash
bash scripts/train_fold0.sh
# or
MODEL=vit_base_patch16_224.augreg_in1k IMG_SIZE=224 bash scripts/train_fold0.sh
```
