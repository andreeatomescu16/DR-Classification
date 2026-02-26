# Rezultate Experimentale - Rezumat

## Rezumat Executiv

Acest document prezintă rezultatele obținute în urma antrenării și evaluării a două arhitecturi de deep learning pentru clasificarea severității Retinopatiei Diabetice (DR) pe o scară de 5 clase (0-4).

## Configurație Experimentală

- **Dataset**: EyePACS + APTOS (combinat)
- **Split**: K-fold cross-validation (fold 0 pentru benchmark)
- **Epochs**: 15
- **Batch Size**: 32
- **Image Size**: 
  - EfficientNet-B4: 384×384
  - ViT-B/16: 224×224
- **Loss Function**: Weighted Cross-Entropy
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 cu Cosine Annealing scheduler

## Rezultate Comparative

| Model | Accuracy | QWK | Macro F1 | ROC-AUC (OVR) |
|-------|----------|-----|----------|---------------|
| **EfficientNet-B4** | **89.45%** | **0.9060** | **0.8923** | **0.9768** |
| ViT-B/16 | 83.32% | 0.8456 | 0.8325 | 0.9540 |

**Concluzie**: EfficientNet-B4 obține performanțe superioare pe toate metricile.

## Analiză Detaliată per Model

### EfficientNet-B4

#### Metrici Generale
- **Accuracy**: 89.45%
- **Quadratic Weighted Kappa (QWK)**: 0.9060 (excelent - peste 0.8)
- **Macro F1-Score**: 0.8923
- **ROC-AUC (One-vs-Rest)**: 0.9768 (foarte bun)

#### Performanță per Clasă

| Clasă | Precision | Recall | F1-Score | ROC-AUC | Interpretare |
|-------|-----------|--------|----------|---------|-------------|
| 0 - No DR | 0.9173 | 0.9327 | 0.9249 | 0.9616 | Excelent |
| 1 - Mild | 0.7793 | 0.8037 | 0.7913 | 0.9643 | Bun |
| 2 - Moderate | 0.8622 | 0.7896 | 0.8243 | 0.9592 | Bun |
| 3 - Severe | **0.9576** | **0.9760** | **0.9667** | **0.9996** | Excelent |
| 4 - Proliferative | **0.9412** | **0.9678** | **0.9543** | **0.9993** | Excelent |

**Observații**:
- Modelul performează cel mai bine pe clasele severe (3 și 4), cu F1-score peste 95%
- Clasa 1 (Mild) are cea mai mică performanță, probabil din cauza similarității cu clasele adiacente
- Toate clasele au ROC-AUC peste 0.96, indicând o separare bună între clase

### ViT-B/16

#### Metrici Generale
- **Accuracy**: 83.32%
- **Quadratic Weighted Kappa (QWK)**: 0.8456 (foarte bun)
- **Macro F1-Score**: 0.8325
- **ROC-AUC (One-vs-Rest)**: 0.9540 (foarte bun)

#### Performanță per Clasă

| Clasă | Precision | Recall | F1-Score | ROC-AUC | Interpretare |
|-------|-----------|--------|----------|---------|-------------|
| 0 - No DR | 0.8845 | 0.8798 | 0.8821 | 0.9296 | Bun |
| 1 - Mild | 0.6704 | 0.7239 | 0.6961 | 0.9288 | Acceptabil |
| 2 - Moderate | 0.7255 | 0.6949 | 0.7099 | 0.9142 | Acceptabil |
| 3 - Severe | **0.9296** | **0.9640** | **0.9464** | **0.9995** | Excelent |
| 4 - Proliferative | **0.9279** | **0.9275** | **0.9277** | **0.9977** | Excelent |

**Observații**:
- Similar cu EfficientNet-B4, performează cel mai bine pe clasele severe (3 și 4)
- Clasele 1 și 2 au performanțe mai scăzute comparativ cu EfficientNet-B4
- ROC-AUC-urile sunt în general mai mici decât EfficientNet-B4, dar încă foarte bune

## Analiză Comparativă

### Puncte Forte EfficientNet-B4
1. **Performanță superioară**: +6.13% accuracy față de ViT-B/16
2. **QWK mai mare**: 0.9060 vs 0.8456 (diferență semnificativă)
3. **Mai bun pe clasele moderate**: F1-score pentru clasele 1 și 2 este cu ~10% mai mare
4. **ROC-AUC mai mare**: 0.9768 vs 0.9540

### Puncte Forte ViT-B/16
1. **Arhitectură modernă**: Vision Transformer este o arhitectură state-of-the-art
2. **Performanță bună pe clase severe**: F1-score peste 94% pentru clasele 3 și 4
3. **ROC-AUC excelent**: Peste 0.99 pentru clasele severe

### Limitări Identificate
1. **Clasa 1 (Mild)**: Ambele modele au dificultăți cu această clasă
   - Probabil din cauza similarității cu clasele 0 și 2
   - EfficientNet-B4: F1 = 0.79, ViT-B/16: F1 = 0.70

2. **Clasa 2 (Moderate)**: Performanță moderată
   - EfficientNet-B4: F1 = 0.82, ViT-B/16: F1 = 0.71

## Concluzii

1. **EfficientNet-B4 este modelul superior** pentru această sarcină:
   - Obține performanțe mai bune pe toate metricile
   - Este mai robust pe clasele moderate (1 și 2)
   - QWK de 0.9060 indică o acordare excelentă cu evaluările clinice

2. **Ambele modele performează excelent pe clasele severe**:
   - F1-score peste 94% pentru clasele 3 și 4
   - Acest lucru este crucial clinic, deoarece aceste clase necesită tratament urgent

3. **Clasele moderate (1 și 2) necesită îmbunătățiri**:
   - Poate fi necesară o strategie de augmentare mai agresivă
   - Sau o abordare de fine-tuning specifică pentru aceste clase

4. **QWK-urile obținute (>0.8) sunt excelente**:
   - Indică o acordare bună cu evaluările clinice
   - Sunt comparabile sau superioare altor lucrări din literatură

## Recomandări pentru Utilizare Clinică

1. **Model recomandat**: EfficientNet-B4
2. **Confidență ridicată**: Pentru clasele 0, 3, și 4 (F1 > 0.92)
3. **Atenție necesară**: Pentru clasele 1 și 2 - poate fi necesară o evaluare secundară

## Metrici de Referință

- **QWK > 0.8**: Excelent (ambele modele)
- **QWK > 0.6**: Bun
- **QWK > 0.4**: Acceptabil
- **QWK < 0.4**: Slab

Ambele modele obțin QWK > 0.8, indicând performanțe excelente.
