# Lab 9: Support Vector Machine (SVM) & Principal Component Analysis (PCA)

## Objective
This lab implements and evaluates two fundamental machine learning techniques — **Support Vector Machine (SVM)** for supervised classification and **Principal Component Analysis (PCA)** for unsupervised dimensionality reduction — along with an extra-credit comparison against **Linear Discriminant Analysis (LDA)**.

## Notebook
- **File:** `Lab-9/2547237-svm-pca.ipynb`
- **Branch:** `Lab-9`

---

## Part A: Support Vector Machine (SVM)

- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569 | **Features:** 30 | **Classes:** Malignant / Benign

### Tasks Covered
1. **Data Loading & EDA** — Class distribution, correlation heatmap, feature box plots by diagnosis.
2. **Preprocessing** — 80:20 stratified train-test split, `StandardScaler` normalisation.
3. **Baseline SVM** — Linear kernel `SVC(C=1.0)` with 5-fold cross-validation.
4. **Hyper-parameter Tuning** — `GridSearchCV` over `C ∈ {0.001, 0.01, 0.1, 1, 10, 100, 1000}`.
5. **Evaluation** — Accuracy, Precision, Recall, F1 Score, Confusion Matrix, Classification Report.

### Key Results
| Metric | Score |
|--------|-------|
| Accuracy | ≥ 0.97 |
| Precision | ≥ 0.97 |
| Recall | ≥ 0.96 |
| F1 Score | ≥ 0.97 |

### Key Observations
- Feature standardisation is mandatory — SVM is scale-sensitive.
- A linear kernel is sufficient: the Breast Cancer dataset is approximately linearly separable in the 30-D standardised feature space.
- High recall for the malignant class is the priority metric in clinical screening.
- Small C (soft margin) values already yield excellent performance, indicating well-separated classes.

---

## Part B: Principal Component Analysis (PCA)

- **Dataset:** UCI Wine
- **Source:** `sklearn.datasets.load_wine()`
- **Samples:** 178 | **Features:** 13 | **Classes:** 3 cultivars

### Tasks Covered
1. **Data Loading & Standardisation** — `StandardScaler` applied before PCA.
2. **Full PCA Decomposition** — All 13 components; explained variance ratio table.
3. **Scree Plot & Cumulative Variance** — Identifies elbow and ≥ 95% variance threshold.
4. **2-Component Reduction** — 13 features → 2 principal components; 2D scatter plot.
5. **Component Loadings** — Bar charts showing feature contributions to PC1 & PC2.
6. **Dataset Comparison** — Original vs. PCA-reduced: features, variance, speed, interpretability.
7. **Advantages / Limitations / Applications** — Comprehensive discussion.

### Key Results
| Aspect | Original (13 features) | PCA (2 PCs) |
|--------|------------------------|-------------|
| Dimensionality | 13 | 2 |
| Variance Retained | 100% | ~55% |
| Visualisable | No | Yes (2D) |
| Training Speed | Baseline | Significantly faster |

---

## Extra Credit: Linear Discriminant Analysis (LDA)

- **Dataset:** UCI Wine (same as Part B)
- **Comparison:** LDA (supervised) vs. PCA (unsupervised) in 2D

### Tasks Covered
1. **LDA Reduction** — 13 features → 2 linear discriminants.
2. **Side-by-side Scatter** — PCA vs. LDA 2D projections.
3. **Separability Quantification** — kNN (k=5) 5-fold CV accuracy on original / PCA / LDA.
4. **Comprehensive Comparison Table** — Objective, type, assumptions, limitations.
5. **Conclusion** — When to prefer PCA vs. LDA.

### Key Finding
LDA achieves markedly better class separability than PCA in 2D because it explicitly optimises for the Fisher criterion (between-class / within-class scatter ratio), leveraging class labels during projection.

---

## How to Run
1. Clone the repository and switch to the `Lab-9` branch:
   ```bash
   git checkout Lab-9
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook "Lab-9/2547237-svm-pca.ipynb"
   ```

---
**Registration Number:** 2547237
