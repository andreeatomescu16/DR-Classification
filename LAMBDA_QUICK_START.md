# Quick Start - Lambda Labs

Ghid rapid pentru a începe training-ul pe Lambda Labs în 5 minute.

## 🚀 Pași Rapizi

### 1. Launch Instance pe Lambda Labs

1. Mergi la [lambdalabs.com](https://lambdalabs.com)
2. **Launch instance:**
   - Instance: `1x A10 (24 GB PCIe)` - $0.75/hr
   - Region: `Virginia, USA`
   - Base image: `Lambda Stack 22.04`
   - Filesystem: `Don't attach`
   - Security: SSH key (deja configurat)

### 2. Conectare și Setup

```bash
# Conectează-te la instanță
ssh ubuntu@<ip-address>

# Clonează repository-ul
git clone https://github.com/andreeatomescu16/DR-Classification.git
cd DR-Classification

# Rulează setup complet (va face tot automat)
bash scripts/setup_lambda_complete.sh
```

Script-ul va:
- ✅ Instala toate dependențele
- ✅ Configura environment-ul
- ✅ Descărca dataset-ul de pe Kaggle
- ✅ Procesa dataset-urile
- ✅ Crea K-fold splits

### 3. Start Training

```bash
# Creează screen session pentru persistență
screen -S training

# Activează environment
source venv/bin/activate

# Start training
python scripts/benchmark.py \
    --fold_csv data/folds/fold0.csv \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 8

# Detach: Ctrl+A apoi D
```

### 4. Monitorizare

```bash
# Reattach screen
screen -r training

# Sau verifică logs
tail -f benchmark_results/logs/*.log
```

### 5. Backup Rezultate

```bash
# Creează backup
bash scripts/backup_results.sh

# Download pe laptop
scp ubuntu@<ip>:~/DR-Classification/results_backup_*.tar.gz ./
```

---

## ⚙️ Configurație Optimizată pentru A10

- **Batch size:** 32 (optim pentru 24GB VRAM)
- **num_workers:** 8 (CPU-uri multiple)
- **Image size:** 384×384 (EfficientNet), 224×224 (ViT)

---

## 💰 Cost Estimativ

- **EfficientNet-B2:** ~4-6 ore × $0.75 = $3-4.50
- **EfficientNet-B4:** ~6-8 ore × $0.75 = $4.50-6
- **ViT-B/16:** ~8-10 ore × $0.75 = $6-7.50
- **Total:** ~$13.50-18 EUR

---

## 📚 Documentație Completă

Pentru detalii complete, vezi [LAMBDA_LABS_SETUP.md](LAMBDA_LABS_SETUP.md)

---

**Succes cu training-ul! 🎉**
