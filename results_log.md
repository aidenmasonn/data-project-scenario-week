# Results Log — EXPERIMENT Sandbox

All accuracy figures are from 80/20 stratified split unless marked as (full 20k).

| # | Script / variant | Features | PCA | k | Metric | Weights | 80/20 acc | Notes |
|---|---|---|---|---|---|---|---|---|
| 0 | knn_experiment.py (baseline) | HOG4+HOG7+L2px (2404-dim) | 300 | 7 | cosine | distance | **88.33%** | Parity confirmed with main project |
| 1a | Exp1: triple-scale HOG | HOG3+HOG4+HOG7+L2px (4708-dim) | 500 | 7 | cosine | distance | 88.45% | +0.12pp; var=87.5% |
| 1b | Exp1: triple-scale HOG | HOG3+HOG4+HOG7+L2px (4708-dim) | 600 | 7 | cosine | distance | **88.52%** | +0.19pp; var=89.7%; **NEW BEST** |
| 1c | Exp1: triple-scale HOG | HOG3+HOG4+HOG7+L2px (4708-dim) | 700 | 7 | cosine | distance | 88.35% | var=91.5%; plateau hit, noise |
| 2a | Exp2: +LBP 4x4 grid (P8) | HOG3+4+7+LBP4x4+L2px (4868-dim) | 600 | 7 | cosine | distance | 88.83% | 4x4 block LBP adds texture signal |
| 2b | Exp2: +LBP 7x7 grid (P8) | HOG3+4+7+LBP7x7+L2px (5198-dim) | 600 | 7 | cosine | distance | 88.95% | denser 7x7 grid better |
| 3a | Exp3: dual-scale LBP | HOG3+4+7+LBP(P8+P16)+L2px (6080-dim) | 600 | 7 | cosine | distance | 89.12% | P16,R2 adds broader texture |
| 3b | Exp3: PCA=700 | same | 700 | 7 | cosine | distance | **89.25%** | PCA=700 sweet spot |
| 4 | Exp4: 56x56 upscale | HOG(ppc=6,8,14 on 56x56)+dualLBP+L2px (6080-dim) | 700 | 7 | cosine | distance | **90.05%** | **BREAKTHROUGH** — upscale before HOG |
| 5 | k-sweep on Exp4 | same | 700 | 5,9,11 | cosine | distance | 89.83/89.95/89.65% | k=7 is optimal |
| 6 | HOG on 28+56 dual | same HOG × 2 resolutions (10004-dim) | 700 | 7 | cosine | distance | 89.78% | more dims → worse PCA coverage |
| 7a | Exp7: +PE features (PCA=750) | HOG+dualLBP+L2px+PE (6457-dim) | 750 | 7 | cosine | distance | 89.92% | PE hurts: dilutes HOG/LBP signal in cosine space |
| 7b | Exp7: +PE features (PCA=700) | same | 700 | 7 | cosine | distance | 89.75% | worse |
| 7c | Exp7: +PE features (PCA=800) | same | 800 | 7 | cosine | distance | 89.83% | all PE variants below 90.05% |
| — | Tested & rejected | 12 orientations, HOG ppc=2, cpb=(3,3), 5-scale HOG, Gabor, CLAHE, 84x84, 64x64, unsharp, no-PCA, MinMax px, spatial pyramid LBP, PE features (all PCA) | — | — | — | — | all worse than 90.05% |

## Best result so far
- Accuracy: **90.05%**
- Config: HOG(ppc=6,8,14 on 56×56) + dualLBP(P8+P16 on 28×28) + L2px(28×28) → 6080-dim, PCA=700, cosine, k=7, distance
- submission.csv generated: Yes (Exp 4)

## Notes
- Main project baseline: **88.32%** (HOG4+7+L2px, PCA=300, cosine, k=7, 80/20)
- Realistic k-NN ceiling on Fashion MNIST: ~90–91%
- Add rows above as experiments are run
