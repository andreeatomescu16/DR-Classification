# Ghid Complet - Setup Google Colab Pro

## Pasul 1: Upload Repository pe Google Drive

### Opțiunea A: Upload manual (simplu)
1. Deschide [drive.google.com](https://drive.google.com)
2. Creează un folder numit `DR-Classification` în `My Drive`
3. Upload **toate** fișierele din repository (sau folosește Google Drive Desktop pentru sync)

### Opțiunea B: Folosind Git (recomandat)
1. Deschide Google Colab
2. Creează un notebook nou
3. Rulează:
```python
!pip install -q git+https://github.com/andreeatomescu16/DR-Classification.git
```

### Opțiunea C: Upload direct în Colab
1. Deschide Colab
2. Click pe iconița folder din stânga (Files)
3. Click "Upload" și selectează fișierele

---

## Pasul 2: Deschide Notebook-ul pe Colab

### Metoda 1: Upload notebook-ul
1. Deschide [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Selectează `notebooks/train_on_colab.ipynb` din repository-ul tău

### Metoda 2: Creează notebook nou și copiază conținutul
1. Deschide Colab
2. Creează notebook nou
3. Copiază celulele din `notebooks/train_on_colab.ipynb`

---

## Pasul 3: Verifică GPU

Înainte de a rula training-ul, asigură-te că GPU-ul este activat:

1. Click **Runtime → Change runtime type**
2. Selectează:
   - **Hardware accelerator:** GPU
   - **GPU type:** T4 (sau A100 dacă ai Pro+)
3. Click **Save**

---

## Pasul 4: Verifică Path-ul pe Google Drive

După ce rulezi prima celulă (mount Drive), verifică path-ul:

```python
# Verifică dacă repository-ul există
import os
drive_path = '/content/drive/MyDrive/DR-Classification'
if os.path.exists(drive_path):
    print(f"✓ Repository găsit la: {drive_path}")
    print(f"Fișiere găsite: {len(os.listdir(drive_path))}")
else:
    print(f"✗ Repository NU este la: {drive_path}")
    print("Verifică că ai upload-at repository-ul pe Google Drive!")
```

---

## Pasul 5: Verifică Datele

Înainte de training, verifică că datele există:

```python
import pandas as pd
from pathlib import Path

fold_csv = "data/folds/fold0.csv"
if Path(fold_csv).exists():
    df = pd.read_csv(fold_csv)
    print(f"✓ Data loaded: {len(df)} samples")
    print(f"Class distribution:")
    print(df['label'].value_counts().sort_index())
else:
    print(f"✗ ERROR: {fold_csv} not found!")
    print("Asigură-te că ai upload-at folderul 'data' pe Google Drive!")
```

---

## Pasul 6: Rulează Training-ul

După ce totul este configurat corect, rulează celula de training:

```python
# Run benchmark script
!python scripts/benchmark.py \
    --fold_csv data/folds/fold0.csv \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 4
```

---

## Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'google.colab'"
**Soluție:** Rulează notebook-ul pe Google Colab, nu local!

### Problema: "FileNotFoundError: data/folds/fold0.csv"
**Soluție:** 
1. Verifică că ai upload-at folderul `data/` pe Google Drive
2. Verifică path-ul în celula 2 (ar trebui să fie `/content/drive/MyDrive/DR-Classification`)

### Problema: "CUDA available: False"
**Soluție:**
1. Click **Runtime → Change runtime type**
2. Selectează **GPU** ca Hardware accelerator
3. Salvează și reîncarcă notebook-ul

### Problema: "Out of Memory"
**Soluție:**
- Reduce `batch_size` la 16 sau 8
- Reduce `num_workers` la 2
- Folosește mixed precision (va fi adăugat în viitor)

### Problema: Training se oprește după câteva ore
**Soluție:**
- Colab Pro are limită de 12h per sesiune (24h pentru Pro+)
- Checkpoint-urile se salvează automat, poți continua de unde ai rămas
- Folosește `--resume` flag pentru a continua training-ul

---

## Structura Recomandată pe Google Drive

```
My Drive/
└── DR-Classification/
    ├── drlib/
    ├── scripts/
    ├── data/
    │   └── folds/
    │       ├── fold0.csv
    │       ├── fold1.csv
    │       └── ...
    ├── configs/
    ├── notebooks/
    │   └── train_on_colab.ipynb
    ├── requirements.txt
    └── README.md
```

---

## Tips pentru Training Eficient

1. **Folosește Early Stopping:** Deja configurat (patience=10)
2. **Monitorizează GPU:** Click pe "RAM/Disk" în meniul din dreapta
3. **Salvează Checkpoint-uri:** Se salvează automat în `lightning_logs/`
4. **Download Rezultate:** După training, download folderul `benchmark_results/`

---

## Verificare Finală înainte de Training

Rulează această celulă pentru a verifica totul:

```python
import os
import torch
from pathlib import Path

print("="*60)
print("VERIFICARE SETUP")
print("="*60)

# 1. Verifică GPU
print(f"\n1. GPU Status:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("   ⚠ WARNING: GPU not available! Check Runtime settings.")

# 2. Verifică repository
print(f"\n2. Repository Status:")
repo_path = Path.cwd()
print(f"   Current directory: {repo_path}")
print(f"   Files: {len(list(repo_path.iterdir()))}")

# 3. Verifică date
print(f"\n3. Data Status:")
fold_csv = Path("data/folds/fold0.csv")
if fold_csv.exists():
    import pandas as pd
    df = pd.read_csv(fold_csv)
    print(f"   ✓ Found: {fold_csv}")
    print(f"   Samples: {len(df)}")
    print(f"   Classes: {df['label'].value_counts().sort_index().to_dict()}")
else:
    print(f"   ✗ NOT FOUND: {fold_csv}")

# 4. Verifică scripturi
print(f"\n4. Scripts Status:")
scripts = ["scripts/benchmark.py", "drlib/train.py"]
for script in scripts:
    if Path(script).exists():
        print(f"   ✓ {script}")
    else:
        print(f"   ✗ {script}")

print("\n" + "="*60)
print("Dacă toate verificările sunt OK, poți începe training-ul!")
print("="*60)
```
