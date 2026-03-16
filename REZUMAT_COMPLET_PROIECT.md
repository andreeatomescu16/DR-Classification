# Prezentare Completă: Clasificarea Severității Retinopatiei Diabetice (DR)

## Document de referință pentru înțelegerea integrală a proiectului

---

# 1. REZUMAT EXECUTIV

Acest proiect implementează un **sistem end-to-end de deep learning** pentru clasificarea severității Retinopatiei Diabetice (Diabetic Retinopathy — DR) din imagini fundus retiniene. Sistemul clasifică imaginile în **5 categorii clinice** (0–4), conform standardelor internaționale de gradare.

**Problema medicală**: Retinopatia diabetică este o complicație a diabetului care poate duce la orbire. Detecția timpurie și clasificarea corectă a severității sunt esențiale pentru tratament.

**Soluția tehnică**: Pipeline complet de clasificare bazat pe rețele neuronale, de la preprocesare la evaluare și vizualizare a explicațiilor (Grad-CAM).

**Rezultate principale**:
- **EfficientNet-B4**: Accuracy 89.45%, QWK 0.9060, ROC-AUC 0.9768
- **ViT-B/16**: Accuracy 83.32%, QWK 0.8456, ROC-AUC 0.9540
- Ambele modele performează excelent pe clasele severe (3, 4) — cele mai importante clinic

---

# 2. ARHITECTURA PROIECTULUI

## 2.1 Structura Fișierelor

```
DR-Classification/
├── drlib/                          # Biblioteca principală
│   ├── datasets.py                 # Încărcare date și DRDataset
│   ├── transforms.py               # Preprocesare și augmentare
│   ├── models/
│   │   ├── __init__.py             # Factory pentru modele
│   │   └── hybrid_coatmini.py     # Arhitectură hibridă CNN+Transformer
│   ├── train.py                    # Modul Lightning pentru antrenament
│   ├── metrics.py                  # Metrici (QWK, F1, ROC-AUC etc.)
│   ├── losses.py                   # Funcții de loss
│   ├── explainability.py           # Grad-CAM și vizualizări
│   └── config.py                   # Management configurare
├── scripts/
│   ├── prepare_apots.py            # Pregătire dataset APTOS
│   ├── prepare_eyepacs.py          # Pregătire dataset EyePACS
│   ├── prepare_combined_dataset.py # Combinare dataset-uri
│   ├── kfold_split.py              # Split K-fold stratificat
│   ├── evaluate.py                 # Evaluare model cu vizualizări
│   ├── benchmark.py                # Benchmark comparative
│   ├── train_fold0.sh / train_hybrid_fold0.sh
│   ├── train_kfold.sh / train_hybrid_kfold.sh
│   └── ... (script-uri pentru RunPod, Lambda Labs, Colab)
├── configs/
│   ├── default_config.yaml
│   └── runpod_budget.yaml
├── data/
│   ├── aptos_master.csv            # Manifest APTOS
│   ├── eyepacs_master.csv          # Manifest EyePACS
│   └── folds/
│       ├── fold0.csv ... fold4.csv # Split-uri K-fold
│       └── fold*_summary.csv       # Rezumaturi per fold
├── evaluation_results/             # Rezultate evaluare
├── results_for_github/benchmark_results/
├── lightning_logs/                 # Loguri și checkpoint-uri
└── requirements.txt
```

## 2.2 Stack Tehnologic

| Componentă | Tehnologie |
|------------|------------|
| Framework Deep Learning | PyTorch 2.x |
| High-level Training | PyTorch Lightning 2.x |
| Modele pre-antrenate | timm (PyTorch Image Models) |
| Augmentare imagini | Albumentations |
| Preprocesare | OpenCV |
| Metrici & ML utils | scikit-learn |
| Vizualizare | Matplotlib, Seaborn |

---

# 3. METODOLOGIE DETALIATĂ

## 3.1 Dataset-uri

### Sursa datelor
- **APTOS 2019**: Competition Kaggle, imagini fundus din India
- **EyePACS**: Dataset de screening DR, imagini din SUA
- **Combinare**: Ambii dataset-uri sunt unite într-un catalog master

### Structura etichetelor (5 clase)
| Clasă | Descriere | Interpretare clinică |
|-------|-----------|---------------------|
| 0 | No DR | Fără retinopatie diabetică |
| 1 | Mild | Leziuni minore (microanevrisme) |
| 2 | Moderate | Leziuni moderate |
| 3 | Severe | Leziuni severe non-proliferative |
| 4 | Proliferative | DR proliferativă (urgentă) |

### Distribuția datelor (exemplu fold 0)
- **Train**: ~172.775 imagini (100.588 clasa 0, 19.773 clasa 1, 33.461 clasa 2, 8.129 clasa 3, 10.824 clasa 4)
- **Val**: ~43.229 imagini (distribuție stratificată similară)
- **Dezvoltare**: Clase dezechilibrate — clasa 0 domină, clasele 3 și 4 sunt mai rare

### Format CSV așteptat
Coloane obligatorii: `image_path`, `label`, `patient_id`, `is_valid`, `split`

## 3.2 Split-uri K-Fold

- **Metodă**: `StratifiedGroupKFold` (sklearn)
- **Motive**: 
  - **Stratificat**: Păstrează proporția claselor în fiecare fold
  - **Grupat pe patient_id**: Previne data leakage — imagini din același pacient nu apar atât în train cât și în val
- **Configurare**: 5 fold-uri, seed=42 pentru reproductibilitate

## 3.3 Preprocesare și Augmentare

### Transformări de bază (`drlib/transforms.py`)

1. **RemoveBlackBorders** (custom)
   - Detectează regiunea circulară a fundus-ului
   - Elimină bordurile negre comune la fotografierea retinală
   - Utilizează threshold, contours OpenCV și bounding box

2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
   - Îmbunătățire contrast local
   - Aplicat cu probabilitate 0.7 în training

3. **Normalizare ImageNet**
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
   - Necesară pentru backbone-uri pre-antrenate

### Augmentare training (mod strong)
- Rotire 90° aleatoare, flip orizontal/vertical
- Affine: scale (0.9–1.1), translate, rotate (±15°)
- RandomBrightnessContrast
- Blur (simulare probleme de focalizare)
- OpticalDistortion (artefacte cameră)

### Pipeline val/test
- RemoveBlackBorders, Resize, Normalizare — fără augmentare

## 3.4 Arhitecturi de Model

### Modele prin timm (pre-antrenate ImageNet)
- **EfficientNet**: B0–B7 (lightweight → high accuracy)
- **ResNet**: 18, 34, 50, 101, 152
- **Vision Transformers**: ViT-B/16, ViT-L/16
- **ConvNeXt**: tiny, small, base, large
- **RegNet**: diverse dimensiuni

### Modele custom (antrenate de la zero)
- **hybrid_coatmini**: ~8.5M parametri
  - Stem CNN + Stage 1–2 CNN (DepthwiseConv) + Stage 3–4 Swin-like window attention
  - Embed dims: [64, 128, 256, 512], depths [2,2,2,2]
- **hybrid_coat_small**: ~13.3M parametri
  - Variantă 1.25× mai largă: [80, 160, 320, 640]

### Recomandări per use-case
| Profil | Model | Utilizare |
|--------|-------|-----------|
| lightweight | efficientnet_b0 | Resurse limitate |
| balanced | efficientnet_b3 | Echilibru performanță/viteză |
| high_accuracy | efficientnet_b5 | Maximă acuratețe |
| transformer | vit_base_patch16_224 | Arhitectură modernă |
| hybrid | hybrid_coatmini | Model custom, fără pretrain |

## 3.5 Funcții de Loss

| Loss | Descriere | Când se folosește |
|------|-----------|-------------------|
| **CE** | Cross-Entropy standard | Date echilibrate |
| **weighted_ce** | CE cu ponderi inverse frecvență | Dezechilibru clase |
| **focal** | Focal Loss (gamma=2) — atenție pe exemple grele | Clase rare, hard examples |
| **label_smoothing** | Regularizare anti-overconfidence | Generalizare |
| **ordinal** | CORAL-style: P(Y≥k) pe K-1 task-uri binare | Clasificare ordinală, QWK |

## 3.6 Antrenament (PyTorch Lightning)

### Parametri tipici
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing
- **Early stopping**: pe val_qwk, patience=10
- **Checkpoint**: salvare best după val_qwk, plus last
- **Precision**: 16-mixed (AMP) pentru GPU

### Strategii opționale
- **Freeze backbone**: Înghețare backbone până la epoch N, apoi unfreeze
- **Class weights**: Calcul automat din train split pentru weighted_ce / focal
- **Monitor**: val_qwk (Quadratic Weighted Kappa) — metrică principală

## 3.7 Metrici de Evaluare

| Metrică | Descriere | Interpretare |
|---------|-----------|--------------|
| **Accuracy** | % predicții corecte | General |
| **QWK** | Cohen's Quadratic Weighted Kappa | Ordinal, penalizează erorile mari |
| **Macro F1** | Media F1 pe clase | Robust la dezechilibru |
| **ROC-AUC (OVR)** | One-vs-Rest AUC | Separare clase |
| **Precision/Recall/F1** | Per clasă | Detaliu per severitate |

Scală QWK: >0.8 excelent, >0.6 bun, >0.4 acceptabil, <0.4 slab

## 3.8 Explainability

- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- Evidențiază regiunile din imagine importante pentru predicție
- Utilitar `find_target_layer` pentru detectare automată a layer-ului potrivit
- Vizualizări pentru interpretare medicală

---

# 4. REZULTATE EXPERIMENTALE

## 4.1 Configurație Benchmark (15 epoci)

- Dataset: EyePACS + APTOS
- Fold: 0
- Batch size: 32
- Loss: Weighted Cross-Entropy
- Optimizer: AdamW, lr 1e-4, Cosine scheduler

## 4.2 Comparație Modele Principale

| Model | Accuracy | QWK | Macro F1 | ROC-AUC (OVR) |
|-------|----------|-----|----------|---------------|
| **EfficientNet-B4** | **89.45%** | **0.9060** | **0.8923** | **0.9768** |
| ViT-B/16 | 83.32% | 0.8456 | 0.8325 | 0.9540 |

## 4.3 Performanță per Clasă — EfficientNet-B4

| Clasă | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| 0 - No DR | 0.917 | 0.933 | 0.925 | 0.962 |
| 1 - Mild | 0.779 | 0.804 | 0.791 | 0.964 |
| 2 - Moderate | 0.862 | 0.790 | 0.824 | 0.959 |
| 3 - Severe | **0.958** | **0.976** | **0.967** | **0.9996** |
| 4 - Proliferative | **0.941** | **0.968** | **0.954** | **0.9993** |

## 4.4 Performanță per Clasă — ViT-B/16

| Clasă | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| 0 - No DR | 0.884 | 0.880 | 0.882 | 0.930 |
| 1 - Mild | 0.670 | 0.724 | 0.696 | 0.929 |
| 2 - Moderate | 0.725 | 0.695 | 0.710 | 0.914 |
| 3 - Severe | **0.930** | **0.964** | **0.946** | **0.9995** |
| 4 - Proliferative | **0.928** | **0.928** | **0.928** | **0.9977** |

## 4.5 Rezultate Hybrid CoAtMini (Fold 0)

- Model: hybrid_coatmini (ordinal loss)
- Metrici: Accuracy 64.95%, QWK 0.768, Macro F1 0.490
- Observație: Performanță inferioară față de EfficientNet/ViT cu pretrain; model antrenat de la zero.

## 4.6 Analiza Rezultatelor

### Puncte forte EfficientNet-B4
- Performanță superioară pe toate metricile
- +6.13% accuracy față de ViT
- Foarte bun pe clase moderate (1, 2)
- ROC-AUC >0.99 pentru clase severe

### Puncte forte ViT-B/16
- Arhitectură state-of-the-art
- F1 >94% pe clase severe (3, 4)
- Separare bună între clase (ROC-AUC >0.99 pe severe)

### Limitări
- **Clasa 1 (Mild)**: Cea mai dificilă — similară cu 0 și 2
- **Clasa 2 (Moderate)**: Performanță mai mică decât pe severe
- **Model hybrid**: Necesită mai multă date/antrenament fără pretrain

### Concluzie clinică
- EfficientNet-B4 este **modelul recomandat** pentru utilizare
- Predicțiile pe clase 0, 3, 4 sunt foarte fiabile (F1 >0.92)
- Pentru clasele 1 și 2 se recomandă verificare suplimentară

---

# 5. FLUX DE LUCRU TIPIC

## 5.1 Pregătire Date

```bash
# APTOS
python scripts/prepare_apots.py \
  --train_csv /path/to/aptos/train.csv \
  --train_imgdir /path/to/aptos/train_images \
  --val_csv /path/to/aptos/val.csv \
  --val_imgdir /path/to/aptos/val_images \
  --out_csv data/aptos_master.csv

# EyePACS
python scripts/prepare_eyepacs.py \
  --img_root /path/to/eyepacs/images \
  --labels_csv /path/to/eyepacs/labels.csv \
  --out_csv data/eyepacs_master.csv

# K-fold
python scripts/kfold_split.py \
  --masters data/aptos_master.csv data/eyepacs_master.csv \
  --out_dir data/folds --n_splits 5 --seed 42
```

## 5.2 Antrenament

```bash
# EfficientNet
python -m drlib.train \
  --fold_csv data/folds/fold0.csv \
  --model efficientnet_b4 \
  --img_size 384 --batch_size 32 \
  --epochs 30 --loss weighted_ce --use_class_weights

# Hybrid (RunPod/Lambda)
bash scripts/train_hybrid_fold0.sh
```

## 5.3 Evaluare

```bash
python scripts/evaluate.py \
  --checkpoint lightning_logs/version_X/checkpoints/best.ckpt \
  --data_csv data/folds/fold0.csv \
  --split val --out_dir evaluation_results \
  --visualize
```

---

# 6. DEPLOYMENT ȘI INFRASTRUCTURĂ

## 6.1 Opțiuni Cloud

- **Lambda Labs**: Recomandat (~13–18 EUR total pentru 3 modele), GPU A10 24GB
- **RunPod**: RTX 4090, Volume Disk 80GB, setup automat prin script-uri
- **Google Colab**: Ghid disponibil în `COLAB_SETUP_GUIDE.md`

## 6.2 Cerințe Hardware

| Nivel | RAM | GPU VRAM | Utilizare |
|-------|-----|----------|-----------|
| Minim | 8GB | - | EfficientNet-B0, CPU |
| Recomandat | 16GB | 8GB+ | EfficientNet-B3/B4 |
| Performant | 32GB | 24GB | ViT, EfficientNet-B5+, multi-GPU |

## 6.3 Dependențe (requirements.txt)

```
torch>=2.0.0, torchvision, lightning>=2.0.0
timm>=0.9.0, albumentations>=1.3.0, opencv-python
scikit-learn, pillow, numpy, pyyaml, matplotlib, seaborn, pandas
```

---

# 7. FIȘIERE CHEIE ȘI ROLURI

| Fișier | Rol |
|--------|-----|
| `drlib/datasets.py` | DRDataset, rezolvare path-uri, suport data_root |
| `drlib/transforms.py` | RemoveBlackBorders, get_train_tf, get_val_tf |
| `drlib/train.py` | DRModule Lightning, make_loaders, main() |
| `drlib/models/__init__.py` | create_model, RECOMMENDED_MODELS |
| `drlib/models/hybrid_coatmini.py` | HybridCoAtMini CNN+Swin |
| `drlib/losses.py` | WeightedCE, FocalLoss, OrdinalLoss, create_loss |
| `drlib/metrics.py` | QWK, compute_all_metrics, confusion matrix |
| `drlib/explainability.py` | GradCAM, visualize_predictions |
| `scripts/kfold_split.py` | StratifiedGroupKFold pe patient_id |
| `scripts/evaluate.py` | Evaluare completă cu plot-uri |
| `scripts/benchmark.py` | Comparație modele, export rezultate |
| `configs/default_config.yaml` | Configurare implicită |

---

# 8. REPRODUCIBILITATE

- Seed fix: 42
- Configurare YAML versionată
- StratifiedGroupKFold cu seed
- Checkpoint-uri best + last
- Logging TensorBoard / Lightning

---

# 9. CONCLUZII FINALE

1. **Sistem complet**: De la date până la evaluare și explainability.
2. **EfficientNet-B4** oferă cele mai bune rezultate practice.
3. **QWK >0.9** indică acord foarte bun cu evaluări clinice.
4. **Clase severe (3, 4)** sunt detectate cu încredere ridicată — esențial clinic.
5. **Clase 1 și 2** beneficiază de îmbunătățiri viitoare (augmentare, fine-tuning, ensemble).
6. **Reproductibilitate** și documentație permit refacerea experimentelor.

---

# 10. REFERINȚE ȘI DOCUMENTAȚIE SUPLIMENTARĂ

- `README.md` — Ghid principal
- `RESULTS_SUMMARY.md` — Rezumat rezultate
- `IMPLEMENTATION_SUMMARY.md` — Detalii implementare
- `BENCHMARKING_GUIDE.md` — Cum se rulează benchmark-urile
- `LAMBDA_LABS_SETUP.md` — Setup Lambda Labs
- `README_RUNPOD.md` — Pași RunPod
- `PROJECT_STATUS.md` — Status curent proiect

---

*Document generat pentru înțelegerea integrală a proiectului DR-Classification.*
