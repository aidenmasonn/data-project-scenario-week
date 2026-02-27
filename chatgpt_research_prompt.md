# ChatGPT Research Prompt: Push k-NN Fashion MNIST to 92%+

## Context for ChatGPT

I am competing in a university prediction competition on Fashion MNIST (28×28 greyscale images, 10 clothing classes, 20,000 labeled training images + 10,000 unlabelled test images). My deliverable is a `submission.csv` of predicted labels (0–9).

**Hard constraints:**
- Final classifier MUST be k-NN (`sklearn.neighbors.KNeighborsClassifier`)
- Training set is exactly 20,000 labeled images (fixed — cannot be augmented)
- Must use Python / scikit-learn
- Professor confirmed: everything BEFORE k-NN is free (any preprocessing, feature extraction, learned embeddings, metric learning, etc.) — only the final classification step must be k-NN

**Current best: 90.05%** (80/20 stratified split accuracy)

**Current pipeline:**
1. Upscale 28×28 → 56×56 (anti_aliasing=True)
2. Extract triple-scale HOG (pixels_per_cell=6, 8, 14 on 56×56; cells_per_block=(2,2); orientations=9)
3. Extract dual-scale block LBP (P=8,R=1 and P=16,R=2 on original 28×28; 7×7 spatial grid of 4×4px blocks; 10 and 18 uniform bins respectively)
4. L2-normalised raw 28×28 pixels
5. Concatenate + L2-renormalise → 6080-dim vector
6. PCA (n_components=700, ~91.2% variance retained)
7. Cosine k-NN (k=7, distance weights, brute algorithm)

**Already tested and rejected (all worse than 90.05%):**
- More HOG scales (4+5 scales), HOG orientations=12, cells_per_block=(3,3), transform_sqrt=True
- HOG ppc=2 (too fine), ppc=10/11 added (diminishing returns)
- 64×64 or 84×84 upscale (56×56 is optimal)
- Dual-resolution HOG (28×28 + 56×56 combined = too many dims, PCA suffers)
- Spatial pyramid LBP (global + 2×2 + 7×7)
- LBP computed on 56×56 image (worse than on original 28×28)
- Gabor filters, CLAHE, histogram equalisation, unsharp masking
- Euclidean metric (88.70%), Manhattan metric (85.58%)
- No PCA (88.22%), PCA=800+ or PCA<600 (all worse)
- k=3, 5, 9, 11 (k=7 is optimal)
- Autoencoder embeddings + k-NN: 87.14% (worse)
- Supervised UMAP + k-NN: 83% (much worse)
- CNN penultimate layer features + k-NN: ~90.58% (only marginally better, requires GPU training, constraint-borderline)

**Known published upper bound:** Best classical ML result on Fashion MNIST without neural networks is ~91.23–91.59% using HOG+LBP+SVM or PE/Corr+HOG+LBP+SVM. The classifier there is SVM, not k-NN.

---

## Research Tasks for ChatGPT

Please search the web (Kaggle, GitHub, arXiv, Papers With Code, Reddit r/MachineLearning, Stack Overflow, Medium, Towards Data Science) and answer the following:

### Question 1: NCA and LMNN — what accuracy gains are documented?
- Has `sklearn.neighbors.NeighborhoodComponentsAnalysis` (NCA) been applied before k-NN on Fashion MNIST? What accuracy did it achieve?
- Has `metric-learn` library's LMNN (Large Margin Nearest Neighbour) been tested on Fashion MNIST? What are the results?
- Is there any evidence these can push k-NN above 91% on Fashion MNIST with 20k training samples?
- Find any code examples showing NCA or LMNN applied to image features before k-NN

### Question 2: Permutation Entropy features — implementation details
- A 2025 arXiv paper (2507.13772) "Feature Engineering is Not Dead" reported 91.23% on Fashion MNIST using Permutation Entropy (PE) + HOG + LBP features with SVM.
- What exactly is the permutation entropy feature extraction step? How is it computed on image patches?
- Is there a Python/sklearn implementation? (The paper mentions `skimage.measure.shannon_entropy` over sliding windows + pixel ordering correlations)
- Find any GitHub repo or notebook implementing permutation entropy for image classification
- Could adding PE features to my current HOG+LBP pipeline push k-NN from 90.05% to 91%+?

### Question 3: What are the highest known k-NN accuracies on Fashion MNIST?
- Search Papers With Code (paperswithcode.com/sota/image-classification-on-fashion-mnist) for any k-NN entries
- Search Kaggle for Fashion MNIST notebooks using k-NN that report >90% accuracy — what preprocessing did they use?
- Is there any published result of k-NN exceeding 91% on Fashion MNIST (any training set size)?
- What is the theoretical maximum accuracy achievable with k-NN on this dataset?

### Question 4: Optimal transport / Wasserstein distance as a k-NN metric
- Has Earth Mover's Distance (Wasserstein distance) been used as the k-NN metric on image datasets?
- Would computing Wasserstein distance between 28×28 image histograms (or HOG feature maps) as the k-NN distance metric improve accuracy over cosine?
- Is there a Python implementation compatible with sklearn's k-NN? (sklearn doesn't natively support Wasserstein — would need custom metric)

### Question 5: Covariance / second-order features before k-NN
- Region covariance descriptors have been used for image recognition. Has this been tried on Fashion MNIST?
- Specifically: computing covariance matrices of HOG cell activations across spatial positions, then using k-NN with matrix distance metrics (log-Euclidean, Riemannian)?
- Is there a CPU-compatible Python implementation?

### Question 6: Any other creative approaches
- Are there any methods that:
  1. Keep k-NN as the final classifier
  2. Work with 20k training images
  3. Run on CPU (no GPU required)
  4. Have been shown to push accuracy above 90.05% on Fashion MNIST or similar image datasets
  5. Can be implemented in Python with sklearn-compatible libraries
- Consider: random kitchen sinks / Nystroem kernel approximation before k-NN, ITQ hashing, product quantization, or any other approximate metric learning

---

## Expected Output Format

Please return your findings as a structured Markdown file with:

1. A summary table: Method | Reported Accuracy (kNN final) | GPU Required? | Implementation Library | Priority (HIGH/MED/LOW)
2. For each HIGH/MED priority method:
   - What it does (2–3 sentences)
   - Evidence it can beat 90.05% on Fashion MNIST with k-NN
   - Relevant links (paper, GitHub repo, Kaggle notebook)
   - Python code sketch (5–15 lines showing how to slot it into the pipeline)
3. Honest assessment: given the constraints, what is the realistic maximum achievable accuracy with k-NN on Fashion MNIST, and what single change has the best chance of getting there?

Save your output as a `.md` or `.txt` file named `chatgpt_research_results.md` so I can paste it back into my project.
