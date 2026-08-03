# Lab 8: Implementation and Performance Evaluation of Categorical Naive Bayes Classifier

## Objective
This lab implements and evaluates a **Categorical Naive Bayes (CategoricalNB)** classifier on the Play Tennis dataset. It covers the complete machine learning pipeline — data preprocessing, EDA, model training, evaluation, single-sample inference, and multi-model comparison — with detailed interpretations at each step.

## Notebook
- **File:** `Lab 8/lab-8.ipynb`
- **Topic:** Categorical Naive Bayes Classification & Model Comparison
- **Dataset:** Play Tennis (`datasets/tennis.csv`)
- **Branch:** `Lab-8`

## Dataset
- **Name:** Play Tennis
- **Samples:** 50 instances (augmented classic benchmark)
- **Features:** 4 categorical input features
- **Target:** `Play Tennis` → `Yes` / `No`
- **Missing Values:** None

## Features Used
| Feature | Values |
|---------|--------|
| `Outlook` | Sunny, Overcast, Rain |
| `Temperature` | Hot, Mild, Cool |
| `Humidity` | High, Normal |
| `Wind` | Weak, Strong |

## Tasks Covered

1. **Data Loading & Exploration**
   - Loaded `tennis.csv`, dropped the row-index column, and renamed the target column to `Play`.
   - Displayed shape, data types, missing value counts, and class distribution.

2. **Exploratory Data Analysis (EDA)**
   - Grouped bar charts for all 4 features vs. the `Play` target.
   - Class distribution pie chart and stacked 100%-bar chart for Outlook.
   - Label-encoded correlation heatmap to identify feature–target relationships.

3. **Data Preprocessing — Label Encoding**
   - Applied `LabelEncoder` independently to each categorical feature column.
   - Stored fitted encoders in a dictionary for consistent query transformation at inference time.
   - Displayed a side-by-side comparison of original and encoded feature values.

4. **Dataset Partitioning**
   - 80:20 stratified train-test split (`random_state=42`) to preserve class proportions.

5. **Categorical Naive Bayes — Training & Evaluation**
   - Trained `CategoricalNB(alpha=1.0)` (Laplace smoothing) on the training data.
   - Reported overall **Model Accuracy**, **Confusion Matrix** (visual + numeric), and **Classification Report** (Precision, Recall, F1-Score per class).

6. **Single-Sample Inference**
   - Query: `{Outlook: Sunny, Temperature: Cool, Humidity: High, Wind: Strong}`
   - Displayed the **predicted class label** and **class probabilities** (ASCII bar + horizontal bar chart).

7. **Model Comparison**
   - Trained three additional classifiers on the same training data:
     - `DecisionTreeClassifier`
     - `LogisticRegression`
     - `SVC(kernel='rbf', probability=True)`
   - Compared all four models across:
     - Test set accuracy
     - Query prediction label
     - Query `P(No)` and `P(Yes)` probabilities
   - Rendered a 2-panel comparison dashboard (accuracy bars + probability grouped bars) and a 2×2 confusion matrix grid.

8. **Analysis Report**
   - 4-sentence explanation of why models produce different predictions and probability scores for the same test instance, covering the decision boundaries of NB, Decision Tree, Logistic Regression, and SVM.

## Models Compared
| Model | Algorithm Type |
|-------|---------------|
| Categorical Naive Bayes | Probabilistic (Bayes' theorem + conditional independence) |
| Decision Tree | Rule-based (recursive feature splitting) |
| Logistic Regression | Linear (log-odds with sigmoid) |
| SVM (RBF kernel) | Geometric (max-margin hyperplane + Platt scaling) |

## Key Learning Outcomes
- `CategoricalNB` is well-suited for purely categorical datasets; Laplace smoothing prevents zero-probability issues for unseen feature combinations.
- Label encoding must use the **same fitted encoder** for both training and inference to avoid label drift.
- Stratified splitting is critical on small datasets to maintain representative class proportions in both splits.
- Different classifiers optimise different objective functions, leading to varying probability estimates even when they agree on the final predicted class.
- The conditional independence assumption in Naive Bayes compounds evidence from individual features multiplicatively, which can produce more extreme probability scores than discriminative models.

## How to Run
1. Clone the repository and switch to the `Lab-8` branch:
   ```bash
   git checkout Lab-8
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook "Lab 8/lab-8.ipynb"
   ```

---
**Registration Number:** 2547237
