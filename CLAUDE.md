# EXPERIMENT SANDBOX: MSIN0025 Prediction Competition

## 1. PURPOSE
This folder is a **competition sandbox**, entirely separate from the graded coursework.
The one and only deliverable is `submission.csv`: a 10,000-row, one-column file of
predicted integer labels (0–9) for `product_images_for_prediction.csv`.
The goal is to maximise prediction accuracy on the unlabelled set.

## 2. HARD CONSTRAINTS
- **Algorithm**: k-Nearest Neighbours (k-NN) only. No SVM, random forests, neural nets.
- **Training set**: `../CSV/product_images.csv` (20,000 labeled images). Fixed. Cannot be augmented or extended.
- **Language**: Python / scikit-learn exclusively.
- **Data access**: Reference CSVs via `../CSV/`. Never copy them into this folder.

## 3. EVERYTHING ELSE IS FREE
| Parameter | Status |
|---|---|
| PCA (dimensionality reduction) | Optional — test with and without |
| Distance metric | Free — Euclidean, cosine, Manhattan, Minkowski, etc. |
| Feature engineering | Free — HOG, LBP, Gabor, pixel, combinations |
| Preprocessing | Free — L2 norm, MinMax, standardisation |
| k (number of neighbours) | Free |
| weights | Free — uniform, distance, custom |
| train/test split | Use 80/20 for internal accuracy estimates only |
| LOO cross-validation | Not required here |
| Dash app | Not required here |

## 4. TRAINING STRATEGY FOR FINAL SUBMISSION
When exporting `submission.csv`, **train on all 20,000 labeled images** (no held-out
test split). During experimentation, use 80/20 stratified split to estimate accuracy.
Final submission uses the full training set.

## 5. FILE STRUCTURE
```
EXPERIMENT/
├── CLAUDE.md          ← this file (context for Claude)
├── constraints.md     ← hard vs soft constraints reference
├── results_log.md     ← experiment tracking table
└── knn_experiment.py  ← self-contained baseline + experiment script
```

## 6. RELATIONSHIP TO MAIN PROJECT
- `PYTHON/` is the graded coursework — **do not modify any files there**.
- This folder is fully independent. knn_experiment.py does not import from PYTHON/.
- If a method here beats 88.3% significantly, consider reporting it in the main project's report as "future work" or as an alternative pipeline.

## 7. CURRENT BASELINE (from main project)
- Features: HOG cell=4 + HOG cell=7 + L2-normalised pixels → 2,404-dim
- PCA: n_components=300 (~88% variance retained)
- Metric: cosine, algorithm=brute
- k: 7, weights=distance
- 80/20 accuracy: **88.32%**
- This is already above the published k-NN ceiling (~85–86%) for this dataset with raw pixels.
- Realistic ceiling with k-NN: ~90–91%.
