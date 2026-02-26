# Ghid RunPod — Training DR Classification from Zero

Acest ghid te duce pas cu pas de la crearea Pod-ului până la training și descărcarea checkpoint-urilor.

---

## A) Crearea Pod-ului

1. Intră pe [runpod.io](https://runpod.io) → **Deploy** → **Pods**
2. **GPU**: RTX 4090 (24GB) — recomandat pentru buget <$50
3. **Template**: PyTorch 2.x sau Ubuntu + CUDA
4. **Persistent Storage**: atașează un volume (min 100GB pentru dataset + checkpoints)
5. **Disk**: 50–100GB system disk
6. **CPU/RAM**: 8 vCPU, 32GB RAM (suficient pentru dataloader)

**Cost estimat RTX 4090**: ~$0.44/h → ~$10–15 pentru 1–2 run-uri complete (data setup + training)

---

## B) Conectare și verificare GPU

```bash
# SSH în instanță (IP-ul din RunPod dashboard)
ssh root@<IP_RUNPOD> -i ~/.ssh/your_key

# Verifică GPU
nvidia-smi

# Verifică spațiu
df -h
```

---

## C) Setup mediul

```bash
# Clone repo
cd /workspace
git clone https://github.com/andreeatomescu16/DR-Classification.git
cd DR-Classification

# Crează venv (opțional, dar recomandat)
python3 -m venv venv
source venv/bin/activate

# Instalează dependențe
pip install --upgrade pip
pip install -r requirements.txt
pip install kaggle  # pentru download dataset
```

---

## D) Configurare Kaggle API

1. Pe [kaggle.com](https://kaggle.com) → Account → **Create New API Token** → descarcă `kaggle.json`
2. Pe RunPod: uploadezi `kaggle.json` sau înlocuiești:

```bash
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# Lipește conținutul (username + key)
chmod 600 ~/.kaggle/kaggle.json
```

---

## E) Setup date (obligatoriu înainte de training)

```bash
cd /workspace/DR-Classification
bash scripts/runpod_setup_data.sh
```

Acest script:
- descarcă dataset-ul Kaggle în `/workspace/data/raw`
- generează `master.csv` și `fold0.csv` … `fold4.csv` în `/workspace/data/folds`
- rulează verificări (class distribution, leakage check, batch load)

**Dacă rulezi fără Kaggle**: descarcă manual zip-ul de pe Kaggle, uploadează pe RunPod în `/workspace/data/raw`, apoi unzip și rulează doar `build_runpod_folds.py`:

```bash
cd /workspace/data/raw
unzip eyepacs-aptos-messidor-diabetic-retinopathy.zip -d .
cd /workspace/DR-Classification
python scripts/build_runpod_folds.py --raw_root /workspace/data/raw --folds_dir /workspace/data/folds
```

---

## F) Verificare înainte de training

```bash
bash scripts/verify_runpod_ready.sh
```

---

## G) Training (sessie persistentă cu tmux)

```bash
tmux new -s drtrain

# Activează venv dacă există
source venv/bin/activate
cd /workspace/DR-Classification

# Training fold 0 (EfficientNet-B4)
bash scripts/train_fold0.sh

# SAU training pe toate fold-urile
bash scripts/train_kfold.sh
```

**Detach**: `Ctrl+B` apoi `D`  
**Reattach**: `tmux attach -t drtrain`

---

## H) Unde se salvează output-urile

| Output | Locație |
|--------|----------|
| Checkpoints | `lightning_logs/version_*/checkpoints/*.ckpt` |
| Best by QWK | `*val_qwk=*.ckpt` |
| Last | `last.ckpt` |

---

## I) Descărcare checkpoint-uri (scp/rsync)

```bash
# De pe mașina ta locală
scp -r root@<IP_RUNPOD>:/workspace/DR-Classification/lightning_logs ./lightning_logs_backup
# sau doar best checkpoint
scp root@<IP_RUNPOD>:/workspace/DR-Classification/lightning_logs/version_0/checkpoints/*.ckpt ./
```

---

## J) Oprire Pod fără pierdere de date

1. **Dacă ai Persistent Volume**: datele sunt pe `/workspace`; poți opri Pod-ul și le găsești la următoarea pornire
2. **Descarcă checkpoint-urile** înainte de stop (vezi secțiunea I)
3. **Stop Pod** din RunPod dashboard

---

## Comenzi rapide (copy-paste)

```bash
# 1. După SSH
cd /workspace && git clone https://github.com/andreeatomescu16/DR-Classification.git
cd DR-Classification && pip install -r requirements.txt kaggle

# 2. Config Kaggle (după ce ai kaggle.json)
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 3. Setup date
bash scripts/runpod_setup_data.sh

# 4. Verificare
bash scripts/verify_runpod_ready.sh

# 5. Training
tmux new -s drtrain
bash scripts/train_fold0.sh
# Ctrl+B, D pentru detach
```

---

## Estimare costuri pentru buget <$50

| Activitate | Timp estimat | Cost (RTX 4090) |
|------------|--------------|-----------------|
| Data setup | ~30 min | ~$0.22 |
| Training 1 fold (40 ep) | ~4–6 h | ~$2–3 |
| Training 5 folds | ~20–30 h | ~$10–15 |
| Total (1 model, 5 folds) | ~25 h | ~$12–18 |
| 3 modele (EfficientNet, ViT, Hybrid) | ~75 h | ~$35–45 |
