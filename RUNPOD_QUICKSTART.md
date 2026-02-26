# Cum să dai train pe RunPod from scratch

## Pași (copy-paste)

### 1. Creează Pod
- GPU: RTX 4090
- Atașează Persistent Volume (min 100GB)
- Template: PyTorch sau Ubuntu + CUDA

### 2. Conectare
```bash
ssh root@<IP> -i ~/.ssh/your_key
```

### 3. Setup
```bash
cd /workspace
git clone https://github.com/andreeatomescu16/DR-Classification.git
cd DR-Classification
pip install -r requirements.txt kaggle
```

### 4. Kaggle API
```bash
mkdir -p ~/.kaggle
# Copiază kaggle.json aici (din Kaggle → Account → Create API Token)
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Date
```bash
bash scripts/runpod_setup_data.sh
```

### 6. Verificare
```bash
bash scripts/verify_runpod_ready.sh
```

### 7. Training
```bash
tmux new -s drtrain
bash scripts/train_fold0.sh
# Ctrl+B, D pentru detach
```

### 8. Alt model (ex. ViT)
```bash
MODEL=vit_base_patch16_224.augreg_in1k IMG_SIZE=224 bash scripts/train_fold0.sh
```

### 9. Toate fold-urile
```bash
bash scripts/train_kfold.sh
```

### 10. Descarcă checkpoint-uri
```bash
# De pe Mac
scp -r root@<IP>:/workspace/DR-Classification/lightning_logs ./
```

---

**Ghid complet**: [RUNPOD_DEPLOYMENT.md](RUNPOD_DEPLOYMENT.md)
