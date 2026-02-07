# Ghid Complet - Setup Lambda Labs pentru Training

## 🎯 Overview

Acest ghid te va ajuta să configurezi complet proiectul pentru training pe Lambda Labs cu GPU A10 (24GB).

**Cost estimat:** ~$13.50-18 EUR pentru toate cele 3 modele (18-24 ore training)

---

## 📋 Pasul 1: Creare Cont și Launch Instance

### 1.1 Creare cont
1. Mergi la [lambdalabs.com](https://lambdalabs.com)
2. Creează cont (Sign Up)
3. Verifică email-ul

### 1.2 Launch Instance
1. Click pe **"Instances"** → **"Launch instance"**
2. Selectează:
   - **Instance type:** `1x A10 (24 GB PCIe)` - $0.75/hr
   - **Region:** `Virginia, USA (us-east-1)` (cel mai aproape de România)
   - **Base image:** `Lambda Stack 22.04` (sau `PyTorch 2.0`)
   - **Filesystem:** `Don't attach a filesystem` (pentru primul training)
   - **Security:** Adaugă SSH key (sau generează una nouă)
3. Click **"Launch instance"**

### 1.3 Conectare SSH
După ce instanța pornește, vei primi un SSH command de forma:
```bash
ssh ubuntu@<ip-address>
```

Copiază și rulează comanda în terminal-ul tău.

---

## 📦 Pasul 2: Setup Initial pe Lambda Labs

### 2.1 Conectare și verificare
```bash
# Conectează-te la instanță
ssh ubuntu@<ip-address>

# Verifică GPU
nvidia-smi

# Verifică Python
python3 --version
```

### 2.2 Clonează repository-ul
```bash
# Clonează repository-ul
git clone https://github.com/andreeatomescu16/DR-Classification.git
cd DR-Classification

# Verifică structura
ls -la
```

### 2.3 Rulează setup script
```bash
# Face script-ul executabil
chmod +x setup_cloud.sh

# Rulează setup (va instala toate dependențele)
bash setup_cloud.sh
```

Setup-ul va:
- Instala Python dependencies
- Crea virtual environment
- Instala PyTorch cu CUDA
- Instala toate pachetele necesare
- Verifica GPU availability

---

## 📥 Pasul 3: Download Dataset-uri

### Opțiunea A: Download direct de pe Kaggle (RECOMANDAT)

```bash
# Activează virtual environment
source venv/bin/activate

# Instalează Kaggle API (dacă nu e deja instalat)
pip install kaggle

# Configurează Kaggle API
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# Adaugă:
# {
#   "username": "andreeatomescu",
#   "key": "KGAT_fa40c59d94f34c394164777195788046"
# }
chmod 600 ~/.kaggle/kaggle.json

# Download dataset combinat
mkdir -p data/combined_dataset
cd data/combined_dataset

# Download (va dura 15-45 minute)
kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy

# Dezarhivează
unzip eyepacs-aptos-messidor-diabetic-retinopathy.zip
# Șterge zip-ul pentru a economisi spațiu
rm *.zip

cd ../..
```

### Opțiunea B: Transfer de pe laptop (dacă ai deja dataset-ul)

```bash
# Pe laptop-ul tău, rulează:
scp -r /path/to/dataset ubuntu@<ip-address>:~/DR-Classification/data/
```

---

## 🔧 Pasul 4: Procesare Dataset-uri

După download, procesează dataset-urile:

```bash
# Activează environment
source venv/bin/activate

# Procesează dataset-ul combinat
python scripts/prepare_combined_dataset.py \
    --dataset_dir data/combined_dataset \
    --output_dir data

# Creează K-fold splits
python scripts/kfold_split.py \
    --masters data/eyepacs_master.csv data/aptos_master.csv \
    --out_dir data/folds \
    --n_splits 5 \
    --seed 42
```

---

## 🚀 Pasul 5: Training Modele

### 5.1 Setup Screen pentru sesiune persistentă

```bash
# Instalează screen (dacă nu e instalat)
sudo apt-get install -y screen

# Creează sesiune screen
screen -S training

# Dacă te deconectezi, reattach cu:
# screen -r training
```

### 5.2 Rulează Training

```bash
# Activează environment
source venv/bin/activate

# Rulează benchmark (toate cele 3 modele)
python scripts/benchmark.py \
    --fold_csv data/folds/fold0.csv \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 8 \
    --log_dir benchmark_results/logs

# Sau antrenează un singur model:
python -m drlib.train \
    --fold_csv data/folds/fold0.csv \
    --model efficientnet_b2.ra_in1k \
    --img_size 384 \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --loss weighted_ce \
    --use_class_weights \
    --lr_scheduler cosine \
    --patience 10 \
    --monitor val_qwk \
    --seed 42 \
    --num_workers 8
```

### 5.3 Monitorizare Training

În alt terminal (pe laptop):
```bash
# Conectează-te la instanță
ssh ubuntu@<ip-address>

# Verifică progresul
tail -f benchmark_results/logs/EfficientNet-B2_training.log

# Sau folosește script-ul de monitoring
python scripts/monitor_training.py
```

---

## 💾 Pasul 6: Backup Rezultate

### 6.1 Download checkpoint-uri și rezultate

```bash
# Pe laptop-ul tău, rulează:
scp -r ubuntu@<ip-address>:~/DR-Classification/lightning_logs ./backup/
scp -r ubuntu@<ip-address>:~/DR-Classification/benchmark_results ./backup/
```

### 6.2 Sau folosește script-ul de backup

```bash
# Pe instanță, rulează:
bash scripts/backup_results.sh

# Va crea un tar.gz cu toate rezultatele
# Apoi download:
scp ubuntu@<ip-address>:~/DR-Classification/results_backup.tar.gz ./
```

---

## ⚙️ Configurare Optimizată pentru A10 GPU

### Batch Size Recommendations:
- **EfficientNet-B2 (384×384):** `batch_size=32` (optim pentru 24GB VRAM)
- **EfficientNet-B4 (384×384):** `batch_size=16-24` (depinde de memorie)
- **ViT-B/16 (224×224):** `batch_size=32-48` (optim pentru 24GB VRAM)

### num_workers:
- Setează `num_workers=8` pentru A10 (CPU-uri multiple disponibile)
- Va accelera semnificativ data loading

### Mixed Precision (opțional):
Pentru training mai rapid, poți activa mixed precision în `drlib/train.py`:
```python
trainer = L.Trainer(
    precision="16-mixed",  # În loc de "32-true"
    ...
)
```

---

## 🔍 Troubleshooting

### Problema: "Out of Memory (OOM)"
**Soluție:**
```bash
# Reduce batch_size
python scripts/benchmark.py --batch_size 16 --num_workers 8
```

### Problema: "Connection lost"
**Soluție:**
```bash
# Folosește screen pentru sesiuni persistente
screen -S training
# Rulează training-ul în screen
# Detach: Ctrl+A apoi D
# Reattach: screen -r training
```

### Problema: "Training prea lent"
**Soluție:**
```bash
# Verifică că GPU este folosit
nvidia-smi

# Verifică că num_workers > 0
python scripts/benchmark.py --num_workers 8
```

### Problema: "Dataset download failed"
**Soluție:**
```bash
# Verifică Kaggle credentials
cat ~/.kaggle/kaggle.json

# Verifică că ai acceptat termenii pentru dataset
# Mergi la: https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy
# Click "Download" pentru a accepta termenii
```

---

## 📊 Monitorizare Costuri

### Verifică costurile în timp real:
1. Mergi la [Lambda Labs Dashboard](https://lambdalabs.com/instances)
2. Vezi costul acumulat pentru instanța ta
3. Setează alertă pentru limită de cost (opțional)

### Estimare costuri:
- **EfficientNet-B2:** ~4-6 ore × $0.75 = $3-4.50
- **EfficientNet-B4:** ~6-8 ore × $0.75 = $4.50-6
- **ViT-B/16:** ~8-10 ore × $0.75 = $6-7.50
- **Total:** ~$13.50-18 EUR

---

## ✅ Checklist Final

Înainte de a începe training-ul, verifică:

- [ ] Instanța Lambda Labs este pornită și funcțională
- [ ] GPU este detectat (`nvidia-smi` funcționează)
- [ ] Repository-ul este clonat și setup-ul este complet
- [ ] Dataset-urile sunt descărcate și procesate
- [ ] K-fold splits sunt create (`data/folds/fold0.csv` există)
- [ ] Screen session este creat pentru persistență
- [ ] Backup plan este în loc (știi cum să descarci rezultatele)

---

## 🎓 Next Steps

După ce training-ul este complet:

1. **Download rezultatele** (checkpoint-uri, metrics, visualizări)
2. **Oprește instanța** pentru a economisi bani
3. **Analizează rezultatele** local pe laptop
4. **Scrie secțiunea de rezultate** pentru licență

---

## 📚 Resurse Suplimentare

- [Lambda Labs Documentation](https://lambdalabs.com/docs)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [Kaggle API Docs](https://www.kaggle.com/docs/api)

---

**Succes cu training-ul! 🚀**
