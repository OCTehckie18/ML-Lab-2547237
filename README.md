# Lab 4: Regression and Classification Evaluation Metrics

## Comprehensive Study of K-Nearest Neighbours (KNN) Classification using Breast Cancer Dataset and Comparison with Regression Evaluation Metrics

---

### 🎯 Aim

To implement KNN classification on the Breast Cancer dataset and analyze model performance using train-test split, heuristic K selection, cross-validation, ROC-AUC, and classification metrics. Also, to compare classification metrics with regression metrics studied in Linear Regression (Lab 3).

---

### 📊 Dataset

- **Breast Cancer Wisconsin (Diagnostic) Dataset** (`brca.csv`)
- **Samples:** 569
- **Features:** 30 numerical features (computed from digitized images of fine needle aspirate of breast mass)
- **Target (y):** B → Benign (encoded as 1), M → Malignant (encoded as 0)

---

### 🧪 Problem Statement

A healthcare analytics team is developing a predictive model for early cancer detection. We build a KNN classifier, optimize its performance using different validation techniques, and compare classification evaluation metrics with regression evaluation metrics from Lab 3.

---

### 📋 Tasks Overview

| Task | Description |
|------|-------------|
| **Task 1** | Data Preparation — Load, explore, clean, and scale the dataset |
| **Task 2** | Train-Test Split — Split data (80:20) with feature scaling via StandardScaler |
| **Task 3** | Heuristic K Selection — Select K using √n rule, experiment with K ± 5, and visualize decision boundaries |
| **Task 4** | Cross Validation — Apply 10-Fold CV to find the best K and compare with train-test split |
| **Task 5** | Classification Evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC |
| **Task 6** | Comparative Study — Classification Metrics vs Regression Metrics (Lab 3 Integration) |
| **Task 7** | Analytical Questions — Conceptual deep-dive into KNN and evaluation metrics |

---

### 🔑 Key Learnings

#### 1. KNN Algorithm & Distance-Based Learning
- KNN is a **lazy learner** — it stores all training data and defers computation to prediction time.
- It classifies new points based on the **majority vote** of the K nearest neighbours.
- **Feature scaling** (StandardScaler) is critical because KNN relies on distance calculations — features with larger magnitudes would dominate without normalization.

#### 2. Heuristic K Selection (√n Rule)
- The **√n heuristic** sets K = √n (number of training samples). For 455 training samples → K ≈ 21.
- **Small K** → overfitting (high variance, jagged decision boundaries).
- **Large K** → underfitting (high bias, overly smooth boundaries).
- **K = √n** provides a balanced starting point, refined further via cross-validation.

#### 3. Distance Metrics in KNN
- **Euclidean Distance:** Straight-line distance in n-dimensional space; suitable for continuous, scaled features.
- **Manhattan Distance:** Sum of absolute differences along each axis; more robust to outliers.
- Decision boundary visualization (via PCA reduction to 2D) shows how boundaries transition from **jagged (K=1)** to **smooth (large K)**.

#### 4. Cross-Validation Superiority
- **10-Fold Cross-Validation** is more reliable than a single train-test split because:
  - Every sample is used for both training and testing.
  - It reduces variance in performance estimates.
  - It provides a more robust and generalizable accuracy measure.
- The best K from cross-validation was used as the final model parameter.

#### 5. Classification Evaluation Metrics

| Metric | What It Measures | Relevance to Cancer Detection |
|--------|-----------------|-------------------------------|
| **Accuracy** | Overall correct predictions / total | General sense; can be misleading with imbalanced classes |
| **Precision** | TP / (TP + FP) | Of predicted benign, how many are actually benign? |
| **Recall** | TP / (TP + FN) | Of actual malignant, how many were detected? **(Critical in healthcare)** |
| **F1 Score** | Harmonic mean of Precision & Recall | Balanced metric when both matter |
| **Confusion Matrix** | Breakdown of TP, TN, FP, FN | Visualizes error patterns |
| **ROC-AUC** | Area under TPR vs FPR curve | Threshold-independent discriminative ability |

> **In cancer detection, Recall and ROC-AUC are more important than Accuracy** — missing a malignant case (false negative) can be fatal.

#### 6. Regression vs Classification Metrics (Lab 3 vs Lab 4)

| Aspect | Regression (Lab 3) | Classification (Lab 4) |
|--------|--------------------|------------------------|
| **Output type** | Continuous values | Discrete class labels |
| **What is measured** | Magnitude of prediction error | Correctness of class assignment |
| **Key metrics** | MAE, MSE, RMSE, R² | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Error interpretation** | How *far off* predictions are | How *correctly* classes are assigned |
| **Sensitivity to outliers** | High (especially MSE) | Low (binary outcome) |
| **Threshold dependency** | None | Classification depends on decision threshold |

**Key Insight:** Regression metrics answer *"How close are my predictions?"* while classification metrics answer *"How correct are my decisions?"*

#### 7. Bias-Variance Trade-off in KNN

| K Value | Bias | Variance | Behavior |
|---------|------|----------|----------|
| **Small K** (e.g., 1-3) | Low | High | Overfitting — jagged boundaries, sensitive to noise |
| **Optimal K** | Balanced | Balanced | Best generalization |
| **Large K** (e.g., n) | High | Low | Underfitting — overly smooth, predicts majority class |

---

### ✅ Expected Outcomes Achieved

- Understanding of the KNN algorithm and distance-based learning
- Ability to tune K using heuristic (√n) + cross-validation
- Knowledge of K-Fold cross-validation and its superiority over single splits
- Interpretation of ROC-AUC in medical classification contexts
- Strong conceptual link between regression evaluation metrics (Lab 3) and classification evaluation metrics (Lab 4)

---

### 📝 Conclusion

1. **Optimal K Value:** The heuristic rule K = √n gave K ≈ 21 as a baseline. Cross-validation refined this to the best-performing K. Both methods converged to a similar range, validating the heuristic as a solid starting point.

2. **Effect of Train-Test Split Variations:** Across 80:20, 70:30, and 90:10 splits, the model showed relatively stable performance. The 80:20 split offered the best balance between sufficient training data and reliable evaluation.

3. **Model Performance:** The final KNN model achieved strong performance on all metrics — high accuracy, precision, recall, F1 score, and ROC-AUC. The confusion matrix confirmed very few critical errors (false negatives for malignant cases). ROC-AUC close to 1.0 indicates excellent discriminative ability.

4. **Regression vs Classification Evaluation:** Regression metrics quantify **prediction error magnitude** for continuous outcomes. Classification metrics quantify **decision correctness** for discrete outcomes. In medical contexts, recall and ROC-AUC are more meaningful than accuracy.

5. **Lab 3 → Lab 4 Progression:** Lab 3 (regression) taught us to evaluate *how close* predictions are. Lab 4 (classification) taught us to evaluate *how correctly* the model assigns class labels. Regression measures **magnitude of error**; classification measures **pattern of decisions**.

---

### 🛠️ Technologies Used

- **Python 3.x**
- **NumPy** & **Pandas** — Data handling
- **Matplotlib** & **Seaborn** — Visualization
- **scikit-learn** — KNN classifier, preprocessing, model selection, evaluation metrics
- **PCA** — Dimensionality reduction for decision boundary visualization

---

### 📁 File Structure

```
Lab4/
├── Lab4.ipynb    # Complete Jupyter Notebook with code, outputs, and interpretations
└── README.md     # This file — summary of learnings and key takeaways
```

---

### 👤 Author

**Omkaar** — Register Number: 2547237  
4th Trimester, Machine Learning Lab  
Christ University
