# =============================================================================
# knn_experiment.py — Competition sandbox for Fashion MNIST k-NN
# =============================================================================
# Goal: maximise accuracy on 10k unlabelled prediction set.
# Deliverable: submission.csv (10k rows, one column 'label', integer 0-9)
#
# This script is self-contained — no imports from ../PYTHON/.
# Modify the CONFIG block and FEATURE EXTRACTION section to experiment.
#
# Workflow:
#   1. Edit CONFIG and/or extract_features() below
#   2. Run: python knn_experiment.py
#   3. Note 80/20 accuracy in results_log.md
#   4. If best so far, submission.csv is saved automatically
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize
from skimage.filters import unsharp_mask
import ordpy

# ---------------------------------------------------------------------------
# PATHS  (relative to EXPERIMENT/ — do not change)
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LABELED_CSV = os.path.join(BASE_DIR, "../CSV/product_images.csv")
PRED_CSV    = os.path.join(BASE_DIR, "../CSV/product_images_for_prediction.csv")
OUT_CSV     = os.path.join(BASE_DIR, "submission.csv")

LABEL_NAMES = {
    0: "T-shirt/top", 1: "Trouser",  2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal",      6: "Shirt",    7: "Sneaker",   8: "Bag",   9: "Ankle boot",
}

RANDOM_SEED = 42

# ===========================================================================
# CONFIG — change these to experiment
# ===========================================================================
USE_PCA        = True          # False = no PCA, pass raw features to kNN
PCA_COMPONENTS = 700           # ignored if USE_PCA=False
K              = 7
METRIC         = "cosine"      # "cosine", "euclidean", "manhattan"
WEIGHTS        = "distance"    # "distance" or "uniform"
ALGORITHM      = "brute"       # "brute" required for cosine; "ball_tree"/"kd_tree" for L2
TEST_SIZE      = 0.20          # fraction held out for accuracy estimation
# ===========================================================================


# ===========================================================================
# FEATURE EXTRACTION — modify this function to try new feature sets
# ===========================================================================
def extract_features(X_pixels: np.ndarray) -> np.ndarray:
    """
    Convert raw pixel matrix (n, 784) → feature matrix (n, D).

    Current pipeline (BEST — triple-scale HOG on 56x56 + dual LBP + L2px):
      - HOG at pixels_per_cell=6 on 56x56 → 2304-dim
      - HOG at pixels_per_cell=8 on 56x56 → 1296-dim
      - HOG at pixels_per_cell=14 on 56x56 →  324-dim
      - Block LBP P=8,R=1:  7x7 grid of 4x4px blocks → 490-dim
      - Block LBP P=16,R=2: 7x7 grid of 4x4px blocks → 882-dim
      - L2-normalised raw pixels  →  784-dim
      - Concatenate → 6080-dim → PCA(700) → cosine kNN → 90.05%
    """
    n = X_pixels.shape[0]
    hog3_list, hog4_list, hog7_list, lbp1_list, lbp2_list = [], [], [], [], []

    BLOCK_H    = 4
    BLOCK_W    = 4
    N_BLOCKS_Y = 28 // BLOCK_H   # 7
    N_BLOCKS_X = 28 // BLOCK_W   # 7

    for i in range(n):
        img28  = X_pixels[i].reshape(28, 28)
        # Upscale to 56x56 for richer HOG cell resolution
        img    = resize(img28, (56, 56), anti_aliasing=True).astype(np.float32) * 255
        img_u8 = img28.astype(np.uint8)   # LBP on original 28x28

        # HOG on 56x56 — ppc doubled to preserve same cell count as on 28x28
        h3 = hog(img, pixels_per_cell=(6, 6), cells_per_block=(2, 2),
                 orientations=9, feature_vector=True, channel_axis=None)
        h4 = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 orientations=9, feature_vector=True, channel_axis=None)
        h7 = hog(img, pixels_per_cell=(14, 14), cells_per_block=(2, 2),
                 orientations=9, feature_vector=True, channel_axis=None)

        hog3_list.append(h3)
        hog4_list.append(h4)
        hog7_list.append(h7)

        # LBP scale 1: P=8, R=1 → 10 uniform bins per block (on 28x28)
        lbp1 = local_binary_pattern(img_u8, 8, 1.0, method="uniform")
        bh1 = []
        for r in range(N_BLOCKS_Y):
            for c in range(N_BLOCKS_X):
                blk = lbp1[r*BLOCK_H:(r+1)*BLOCK_H, c*BLOCK_W:(c+1)*BLOCK_W]
                h, _ = np.histogram(blk, bins=10, range=(0, 10))
                bh1.append(h.astype(np.float32))
        lbp1_list.append(np.concatenate(bh1))   # 49*10 = 490 dims

        # LBP scale 2: P=16, R=2 → 18 uniform bins per block
        lbp2 = local_binary_pattern(img_u8, 16, 2.0, method="uniform")
        bh2 = []
        for r in range(N_BLOCKS_Y):
            for c in range(N_BLOCKS_X):
                blk = lbp2[r*BLOCK_H:(r+1)*BLOCK_H, c*BLOCK_W:(c+1)*BLOCK_W]
                h, _ = np.histogram(blk, bins=18, range=(0, 18))
                bh2.append(h.astype(np.float32))
        lbp2_list.append(np.concatenate(bh2))   # 49*18 = 882 dims

    X_hog3 = normalize(np.array(hog3_list, dtype=np.float32), norm="l2")  # (n, 2304)
    X_hog4 = normalize(np.array(hog4_list, dtype=np.float32), norm="l2")  # (n, 1296)
    X_hog7 = normalize(np.array(hog7_list, dtype=np.float32), norm="l2")  # (n,  324)
    X_lbp1 = normalize(np.array(lbp1_list, dtype=np.float32), norm="l2")  # (n,  490)
    X_lbp2 = normalize(np.array(lbp2_list, dtype=np.float32), norm="l2")  # (n,  882)
    X_l2   = normalize(X_pixels.astype(np.float32), norm="l2")             # (n,  784)

    return normalize(np.hstack([X_hog3, X_hog4, X_hog7, X_lbp1, X_lbp2, X_l2]), norm="l2")
    # Total: 2304 + 1296 + 324 + 490 + 882 + 784 = 6080 dims
# ===========================================================================


def load_pixels(csv_path: str, has_label: bool):
    """Load CSV, return (pixel_array, label_array_or_None)."""
    df = pd.read_csv(csv_path)
    pixel_cols = [c for c in df.columns if c.startswith("pixel_")]
    X = df[pixel_cols].values.astype(np.float32)
    y = df["label"].values.astype(int) if has_label else None
    return X, y


def run():
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("Loading labeled data...")
    X_raw, y = load_pixels(LABELED_CSV, has_label=True)
    print(f"  Labeled set: {X_raw.shape}")

    print("Loading unlabeled prediction data...")
    X_pred_raw, _ = load_pixels(PRED_CSV, has_label=False)
    print(f"  Unlabeled set: {X_pred_raw.shape}\n")

    # -----------------------------------------------------------------------
    # 2. Extract features
    # -----------------------------------------------------------------------
    print("Extracting features for labeled data...")
    X = extract_features(X_raw)
    print(f"  Feature shape: {X.shape}")

    print("Extracting features for unlabeled data...")
    X_pred = extract_features(X_pred_raw)
    print(f"  Feature shape: {X_pred.shape}\n")

    # -----------------------------------------------------------------------
    # 3. 80/20 split for accuracy estimation
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    # -----------------------------------------------------------------------
    # 4. Optional PCA
    # -----------------------------------------------------------------------
    if USE_PCA:
        print(f"Fitting PCA (n_components={PCA_COMPONENTS})...")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
        X_pred_pca = pca.transform(X_pred)
        var_retained = pca.explained_variance_ratio_.sum()
        print(f"  Variance retained: {var_retained:.3f}\n")
    else:
        X_pred_pca = X_pred
        print("PCA disabled — using raw features.\n")

    # -----------------------------------------------------------------------
    # 5. Train k-NN and evaluate on 80/20 split
    # -----------------------------------------------------------------------
    print(f"Training k-NN  (k={K}, metric={METRIC}, weights={WEIGHTS})...")
    knn = KNeighborsClassifier(
        n_neighbors=K,
        metric=METRIC,
        algorithm=ALGORITHM,
        weights=WEIGHTS,
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"80/20 Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'='*60}")
    class_names = [LABEL_NAMES[i] for i in range(10)]
    print(classification_report(y_test, y_pred, target_names=class_names))

    # -----------------------------------------------------------------------
    # 6. Retrain on FULL 20k, predict 10k unlabeled set
    # -----------------------------------------------------------------------
    print("Retraining on full 20k training set for final submission...")
    if USE_PCA:
        pca_full = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        X_full = pca_full.fit_transform(X)
        X_pred_final = pca_full.transform(X_pred)
    else:
        X_full = X
        X_pred_final = X_pred

    knn_full = KNeighborsClassifier(
        n_neighbors=K,
        metric=METRIC,
        algorithm=ALGORITHM,
        weights=WEIGHTS,
        n_jobs=-1,
    )
    knn_full.fit(X_full, y)
    y_submission = knn_full.predict(X_pred_final)

    # -----------------------------------------------------------------------
    # 7. Save submission.csv
    # -----------------------------------------------------------------------
    pd.DataFrame({"label": y_submission}).to_csv(OUT_CSV, index=False)
    print(f"\nSubmission saved: {OUT_CSV}")
    print(f"  Rows: {len(y_submission)}  |  Label distribution:")
    for lbl, name in LABEL_NAMES.items():
        count = (y_submission == lbl).sum()
        print(f"    {lbl} {name:<15} {count:>5}")

    print(f"\n--- Log this in results_log.md ---")
    pca_str = f"PCA={PCA_COMPONENTS}" if USE_PCA else "no PCA"
    print(f"  Features: [see extract_features docstring] | {pca_str} | k={K} | {METRIC} | {WEIGHTS}")
    print(f"  80/20 accuracy: {acc:.4f}")


if __name__ == "__main__":
    run()
