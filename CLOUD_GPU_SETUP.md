# Ghid Setup GPU Cloud pentru Training

## Opțiuni Recomandate (în ordinea eficienței cost/beneficiu)

### 🥇 1. Google Colab Pro/Pro+ (RECOMANDAT pentru început)
**Preț:** ~$10/lună (Pro) sau ~$20/lună (Pro+)
**GPU:** T4 (16GB) sau A100 (40GB) pentru Pro+
**Avantaje:**
- Setup foarte simplu (doar browser)
- Pre-instalat PyTorch, Lightning, etc.
- Integrare directă cu Google Drive
- Perfect pentru experimente și benchmark-uri

**Dezavantaje:**
- Limitări de timp (12h sesiune pentru Pro, 24h pentru Pro+)
- GPU-ul poate fi ocupat uneori
- Storage limitat (trebuie să folosești Google Drive)

**Setup:**
1. Creează cont pe [colab.research.google.com](https://colab.research.google.com)
2. Upgrade la Pro sau Pro+ ($10-20/lună)
3. Upload repository-ul pe Google Drive
4. Deschide notebook-ul `notebooks/train_on_colab.ipynb` (va fi creat)

---

### 🥈 2. Lambda Labs (BEST VALUE pentru research)
**Preț:** ~$0.50-1.10/ora pentru RTX 6000 Ada (48GB)
**GPU:** RTX 6000 Ada, A100, H100
**Avantaje:**
- Prețuri foarte competitive
- GPU-uri dedicate (nu sunt shared)
- Setup simplu (SSH în instanță pre-configurată)
- Perfect pentru training-uri lungi

**Dezavantaje:**
- Necesită setup SSH
- Trebuie să instalezi dependențele manual

**Setup:**
1. Creează cont pe [lambdalabs.com](https://lambdalabs.com)
2. Alege instanță GPU (recomand RTX 6000 Ada - 48GB)
3. Clonează repository-ul
4. Rulează `setup_cloud.sh` (va fi creat)

---

### 🥉 3. Paperspace Gradient
**Preț:** ~$0.51/ora pentru RTX 4000 (16GB) sau $1.10/ora pentru A4000 (16GB)
**GPU:** RTX 4000, A4000, A5000, A6000
**Avantaje:**
- Interfață web friendly
- Notebook-uri Jupyter integrate
- Storage persistent inclus
- Pay-as-you-go

**Dezavantaje:**
- Puțin mai scump decât Lambda Labs
- GPU-uri mai mici decât Lambda

**Setup:**
1. Creează cont pe [paperspace.com](https://paperspace.com)
2. Creează un Gradient Notebook
3. Selectează GPU și PyTorch template
4. Clonează repository-ul

---

### 4. RunPod / Vast.ai (CEL MAI IEFTIN)
**Preț:** ~$0.20-0.50/ora pentru RTX 3090 (24GB)
**GPU:** RTX 3090, A5000, A6000
**Avantaje:**
- Prețuri foarte mici
- Multe opțiuni de GPU
- Pay-per-use

**Dezavantaje:**
- Setup mai complex
- Calitatea instanțelor variază
- Suport limitat

---

### 5. AWS / GCP / Azure (Pentru proiecte enterprise)
**Preț:** ~$1-3/ora pentru p3.2xlarge (V100)
**GPU:** V100, A100, T4
**Avantaje:**
- Infrastructură robustă
- Integrare cu alte servicii cloud
- Scalare ușoară

**Dezavantaje:**
- Mai scump
- Setup mai complex
- Overkill pentru un proiect de licență

---

## Estimare Costuri pentru 3 Modele

**Presupuneri:**
- EfficientNet-B2: ~4-6 ore training
- EfficientNet-B4: ~6-8 ore training  
- ViT-B/16: ~8-10 ore training
- **Total: ~18-24 ore training**

### Costuri estimate:
- **Google Colab Pro:** $10-20/lună (unlimited GPU time în limită de sesiune)
- **Lambda Labs:** ~$12-24 (18-24 ore × $0.50-1.00/ora)
- **Paperspace:** ~$9-24 (18-24 ore × $0.51-1.10/ora)
- **RunPod:** ~$4-12 (18-24 ore × $0.20-0.50/ora)

---

## Recomandare Finală

**Pentru training from scratch, lung și stabil:**

### 🥇 Lambda Labs (RECOMANDAT)
- **GPU:** A10 (24GB) - $0.75/oră
- **Cost total:** ~$13.50-18 EUR (one-time)
- **Avantaje:**
  - GPU-uri dedicate (nu shared)
  - Fără limite de timp
  - Stabilitate maximă
  - Perfect pentru training from scratch
- **Ghid complet:** [LAMBDA_LABS_SETUP.md](LAMBDA_LABS_SETUP.md)
- **Quick start:** [LAMBDA_QUICK_START.md](LAMBDA_QUICK_START.md)

### 🥈 Google Colab Pro
- **Preț:** $10-20/lună
- **Perfect pentru:** Experimente rapide, testare
- **Limitări:** 12h/sesiune, GPU shared, RAM limitat

### 🥉 Paperspace Gradient
- **Preț:** ~$0.51-1.10/oră
- **Perfect pentru:** Interfață web friendly, fără SSH

---

## Setup Quick Start

### Opțiunea 1: Google Colab (Cel mai simplu)

1. Creează un notebook nou în Colab
2. Upload repository-ul pe Google Drive
3. Rulează:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/DR-Classification')

# Install dependencies
!pip install -r requirements.txt

# Run benchmark
!python scripts/benchmark.py --fold_csv data/folds/fold0.csv --epochs 30
```

### Opțiunea 2: Lambda Labs / Paperspace (SSH)

1. Clonează repository-ul:
```bash
git clone https://github.com/andreeatomescu16/DR-Classification.git
cd DR-Classification
```

2. Instalează dependențele:
```bash
pip install -r requirements.txt
```

3. Rulează benchmark-ul:
```bash
python scripts/benchmark.py --fold_csv data/folds/fold0.csv --epochs 30
```

---

## Configurare Optimizată pentru Cloud GPU

### Batch Size Recommendations:
- **EfficientNet-B2 (384x384):** batch_size=32-64 (pentru GPU 16GB+)
- **EfficientNet-B4 (384x384):** batch_size=16-32 (pentru GPU 16GB+)
- **ViT-B/16 (224x224):** batch_size=32-64 (pentru GPU 16GB+)

### num_workers:
- Setează `num_workers=4-8` pentru cloud GPU (nu 0!)
- Va accelera semnificativ data loading

### Mixed Precision:
- Poți activa `precision="16-mixed"` în Lightning pentru training mai rapid
- Economisește memorie și timp

---

## Monitorizare Training pe Cloud

Codul tău deja are:
- `scripts/monitor_training.py` - pentru monitoring local
- Lightning logs în `lightning_logs/`
- TensorBoard integration (poți activa)

Pentru cloud, poți:
1. Folosi `screen` sau `tmux` pentru sesiuni persistente
2. Redirect output la fișier: `python scripts/benchmark.py > training.log 2>&1`
3. Folosi `tail -f training.log` pentru monitoring

---

## Backup și Persistență

**IMPORTANT:** Cloud GPU-urile sunt efemere! Asigură-te că:
1. Checkpoint-urile sunt salvate automat (deja configurat în Lightning)
2. Upload rezultatele pe Google Drive / S3 / etc.
3. Commit codul pe GitHub înainte de training

---

## Troubleshooting

### Out of Memory (OOM):
- Reduce batch_size
- Reduce img_size temporar pentru test
- Folosește gradient accumulation

### Training prea lent:
- Verifică că `num_workers > 0`
- Verifică că GPU este folosit (nu CPU)
- Activează mixed precision

### Connection lost:
- Folosește `screen` sau `tmux`
- Configurează auto-restart pentru training
- Salvează checkpoint-uri frecvent
