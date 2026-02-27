# Constraints Reference — EXPERIMENT Sandbox

## FIXED (cannot change)
| Constraint | Detail |
|---|---|
| Algorithm | k-NN only (KNeighborsClassifier or equivalent from scikit-learn) |
| Training data | product_images.csv — 20,000 labeled images, 784 pixel columns |
| Unlabeled data | product_images_for_prediction.csv — 10,000 images |
| Language | Python / scikit-learn |
| Output | submission.csv — 10,000 rows, one column `label`, integer 0–9 |

## FREE (experiment with these)
| Parameter | Options to explore |
|---|---|
| Feature engineering | HOG (various scales), LBP, Gabor, raw pixels, combinations |
| Preprocessing | L2 normalisation, MinMax scaling, per-column standardisation |
| Dimensionality reduction | PCA (any n_components), no PCA, truncated SVD |
| Distance metric | cosine, euclidean, manhattan, minkowski (p=3,4) |
| k | Any integer — try 3, 5, 7, 9, 11 |
| weights | uniform, distance |
| algorithm | brute, ball_tree, kd_tree (note: cosine requires brute) |
| Training set for final export | Use ALL 20k (no held-out split) |

## EXPERIMENT PRIORITY ORDER
1. HOG scales: try cell=3, cell=5, cell=6 (in addition to current 4+7)
2. No PCA: does cosine on full 2404-dim beat PCA=300?
3. Higher PCA: try PCA=500, PCA=784
4. LBP features: combine with or replace HOG
5. MinMax scaling: replace L2 normalisation
6. k tuning on full 20k training set
7. Metric alternatives: try Manhattan on HOG features
