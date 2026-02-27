# Project Progress — Fashion MNIST k-NN Competition

---

## 1. What We Are Doing

We are competing in a university prediction competition. The task: given 20,000 labelled clothing images (Fashion MNIST), train a model to predict the clothing category of 10,000 unlabelled images.

**The answer must be delivered as `submission.csv`** — a file with 10,000 rows, each containing a single number from 0–9:

| Label | Category |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Each image is 28×28 pixels, greyscale (so 784 pixel values per image).

**Hard constraints from the professor:**
- Final classifier **must** be k-Nearest Neighbours (k-NN)
- Training set is exactly 20,000 images — cannot be added to
- Must use Python / scikit-learn
- Everything *before* k-NN is free: you can transform, extract features, reduce dimensions, learn a better distance metric, etc.

**Goal:** Maximise accuracy. We measure it using an 80/20 split (train on 80% = 16,000 images, test on 20% = 4,000 images). The best score so far is **90.05%**.

---

## 2. How the Current Best Pipeline Works (Plain English)

The script `knn_experiment.py` runs the full pipeline automatically. Here is what it does, step by step:

### Step 1 — Load the images
Read the CSV files. Each row is one image: 784 columns for pixels, 1 column for the label.

### Step 2 — Extract features (the main engineering work)
Raw pixels alone give ~85% accuracy. We compute richer descriptions of each image:

**HOG (Histogram of Oriented Gradients) — detects edges and shapes**
- The image is first upscaled from 28×28 → 56×56 pixels. This is a key trick: more pixels = more gradient information per HOG cell, which improves edge detection.
- HOG divides the image into small cells and measures which direction edges point in each cell.
- We run HOG at **3 different cell sizes** (fine, medium, coarse) to capture detail at multiple scales:
  - Fine scale (6px cells on 56×56): captures small details like collar edges, strap width → 2304 numbers
  - Medium scale (8px cells on 56×56): captures general shape outline → 1296 numbers
  - Coarse scale (14px cells on 56×56): captures overall silhouette → 324 numbers
- Total HOG: **3924 numbers**

**LBP (Local Binary Pattern) — detects texture**
- Computed on the original 28×28 image (not upscaled).
- LBP describes whether each pixel is brighter or darker than its neighbours, encoding local surface texture (smooth, bumpy, striped, etc.).
- We divide the image into a 7×7 grid of small blocks and compute an LBP histogram for each block to preserve spatial layout.
- We do this at **2 scales** to capture both fine and coarse texture:
  - Fine texture (P=8, R=1): uses 8 neighbours in a 1-pixel radius → 490 numbers
  - Coarse texture (P=16, R=2): uses 16 neighbours in a 2-pixel radius → 882 numbers
- Total LBP: **1372 numbers**

**Raw pixels — the image itself**
- The original 28×28 pixel values, normalised so their length = 1 (L2 normalisation).
- These capture the overall brightness pattern across the image.
- Total: **784 numbers**

**Concatenate everything:**
All three parts are joined into one long vector per image: 3924 + 1372 + 784 = **6080 numbers**.

### Step 3 — PCA (Principal Component Analysis)
6080 numbers is a lot. Many of them are correlated or noisy. PCA finds the 700 most important directions in the data and projects everything onto those — reducing 6080 → **700 numbers** while retaining 91.2% of the information. This also speeds up k-NN.

### Step 4 — k-NN Classification
For each test image, find the 7 most similar training images (nearest neighbours) in the 700-dim feature space, using **cosine distance** (measures angle between feature vectors, robust to scale differences). Classify by majority vote, weighted by how similar each neighbour is (closer = more votes). Output the winning class.

### Step 5 — Final submission
After the 80/20 accuracy test, retrain on **all 20,000 images** (no held-out test set), then predict all 10,000 unlabelled images. Save `submission.csv`.

---

## 3. All Experiments Run

| # | What we tried | Why | Result | Verdict |
|---|---|---|---|---|
| **Baseline** | HOG (2 scales, 28×28) + raw pixels, PCA=300, k=7 | Starting point from main coursework | 88.33% | Baseline confirmed |
| **Exp 1a** | Added a 3rd HOG scale (finer cells), PCA=500 | More HOG scales = more shape detail | 88.45% | +0.12pp — marginal |
| **Exp 1b** | Same, PCA=600 | More PCA components = retain more info | 88.52% | +0.19pp — small gain |
| **Exp 1c** | Same, PCA=700 | Test if even more PCA helps | 88.35% | Worse — PCA=700 too many for this feature set |
| **Exp 2a** | Added LBP texture features (4×4 grid), PCA=600 | HOG captures shape; LBP captures texture | 88.83% | +0.31pp — texture helps |
| **Exp 2b** | LBP on denser 7×7 grid | Finer spatial layout of texture blocks | 88.95% | +0.12pp more |
| **Exp 3a** | Added 2nd LBP scale (P=16, R=2), PCA=600 | Capture broader texture patterns | 89.12% | +0.17pp |
| **Exp 3b** | Same, PCA=700 | Optimal PCA for this feature size | 89.25% | +0.13pp — **new best at the time** |
| **Exp 4** | Upscale images 28×28 → 56×56 before HOG | More pixels per HOG cell → richer gradient info | **90.05%** | **+0.8pp — BREAKTHROUGH** |
| **Exp 5** | Tried k=5, 9, 11 (keeping best features) | Find optimal number of neighbours | 89.83 / 89.95 / 89.65% | k=7 is still best |
| **Exp 6** | Computed HOG on *both* 28×28 and 56×56 | More information = better? | 89.78% | Worse — too many dimensions hurt PCA |
| **Exp 7a–c** | Added Permutation Entropy (PE) features, PCA=700/750/800 | PE measures image complexity/disorder — helped SVM in a 2025 paper | 89.75–89.92% | All worse — PE dilutes the HOG signal in cosine k-NN |

**Also tested and rejected in earlier sessions (all worse than 90.05%):**
HOG with 12 orientations, HOG with very fine cells (ppc=2), cells_per_block=(3,3), 5-scale HOG, Gabor filters, CLAHE contrast enhancement, 64×64 and 84×84 upscale sizes, unsharp masking, dual-resolution HOG, spatial pyramid LBP, no PCA, MinMax pixel normalisation, Euclidean distance metric, Manhattan distance metric, k=3.

---

## 4. Current Active Parameters

### CONFIG block (top of knn_experiment.py)

| Parameter | Current Value | What it does | What happens if you change it |
|---|---|---|---|
| `USE_PCA` | `True` | Whether to apply PCA dimensionality reduction before k-NN | Set to `False` to skip PCA entirely (tried: 88.22% — worse) |
| `PCA_COMPONENTS` | `700` | How many PCA dimensions to keep (out of 6080) | Retains 91.2% of variance. Lower = loses information; higher = keeps noise. Sweet spot confirmed at 700. |
| `K` | `7` | Number of nearest neighbours to vote on each prediction | Tested 3, 5, 7, 9, 11. k=7 is optimal. |
| `METRIC` | `"cosine"` | Distance function used to compare feature vectors | cosine >> euclidean >> manhattan. Cosine is best for L2-normalised features. |
| `WEIGHTS` | `"distance"` | How neighbours vote: equal votes, or closer = more weight | `"distance"` is better: a very close neighbour should count more. |
| `ALGORITHM` | `"brute"` | How sklearn searches for nearest neighbours | Must be `"brute"` when using cosine metric. |
| `TEST_SIZE` | `0.20` | Fraction held out for accuracy estimation | 20% = 4,000 images for testing, 16,000 for training. Don't change. |

### Feature pipeline (inside extract_features())

| Feature block | Image used | What it captures | Dimensions |
|---|---|---|---|
| HOG fine (ppc=6) | 56×56 upscale | Fine edge details (collar, straps, buttons) | 2304 |
| HOG medium (ppc=8) | 56×56 upscale | General shape and contours | 1296 |
| HOG coarse (ppc=14) | 56×56 upscale | Overall silhouette | 324 |
| LBP fine (P=8, R=1) | 28×28 original | Micro-texture (surface smoothness, weave) | 490 |
| LBP coarse (P=16, R=2) | 28×28 original | Macro-texture (broader surface patterns) | 882 |
| Raw pixels (L2-norm) | 28×28 original | Overall brightness distribution | 784 |
| **Total** | | | **6080** → PCA → **700** |

---

## 5. Where We Stand and What's Left

### Current best: 90.05%
This beats the university baseline (88.32%) by **+1.73 percentage points**.

### Why 92% is hard
The published k-NN ceiling with hand-crafted features on Fashion MNIST is approximately **90–91%**. We are essentially at that ceiling. The gap to 92% cannot be closed by adding more hand-crafted features — we have tried everything that works. The remaining gap is a fundamental limitation of k-NN's distance metric.

### What could still work

| Approach | What it does | Status | Risk |
|---|---|---|---|
| **NCA (Neighborhood Components Analysis)** | sklearn built-in. Learns a linear transformation of the 700 PCA features that *directly maximises k-NN accuracy*. Changes what "distance" means for k-NN without changing the classifier. | **Not yet tried** | Computationally slow (~30–60 min); no guarantee it helps on this dataset |
| **LMNN (metric-learn library)** | Like NCA but uses a large-margin objective. Pulls same-class neighbours closer, pushes different-class neighbours away. | Not tried | Requires `pip install metric-learn`; slower than NCA |
| **k=6 (even number)** | Untested even k value — different tie-breaking behaviour | Not tried | Minimal expected gain |

### The honest ceiling
Based on research: **91–92% is the realistic maximum for k-NN with any feature engineering on this dataset using 20k training images**. Achieving 92%+ with k-NN would almost certainly require NCA/LMNN metric learning. Deep learning methods (CNNs) achieve 93–94% but are not allowed.

---

## 6. File Map

| File | Purpose |
|---|---|
| `knn_experiment.py` | The only script. Edit CONFIG block (lines 48–54) or `extract_features()` (lines 61–128) to experiment. |
| `results_log.md` | Raw experiment log table — every run recorded |
| `progress.md` | This file — human-readable summary |
| `submission.csv` | Current best predictions for the 10k unlabelled set (90.05% config) |
| `../CSV/product_images.csv` | 20k labelled training images (read-only) |
| `../CSV/product_images_for_prediction.csv` | 10k unlabelled images to predict (read-only) |
