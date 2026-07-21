# Machine Learning Lab 6: Logistic Regression vs. K-Nearest Neighbors (KNN)

## Objective
The primary objective of this lab is to implement, evaluate, and compare the performance of **Logistic Regression** and **K-Nearest Neighbors (KNN)** classifiers. 

## Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository
- **Description:** The dataset contains 569 samples with 32 columns. Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
- **Target Variable:** `y` (Diagnosis: M = Malignant, B = Benign)

## Steps Performed
1. **Importing Libraries:** Utilized `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` for data manipulation, visualization, and modeling.
2. **Exploratory Data Analysis (EDA):** Analyzed the structure of the dataset and checked for basic statistics and data types to understand the distributions and scale of features.
3. **Data Preprocessing:**
   - Separated features (X) and the target variable (y).
   - Encoded the categorical target variable using `LabelEncoder`.
   - Addressed differences in feature scales by standardizing the data using `StandardScaler` (crucial for distance-based algorithms like KNN).
4. **Model Training:**
   - Split the dataset into training and testing sets.
   - Initialized and trained the **Logistic Regression** model.
   - Initialized and trained the **K-Nearest Neighbors** model.
5. **Model Evaluation:**
   - Evaluated both models using key classification metrics: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

## Key Findings & Conclusion
- The EDA revealed significant variance in the scales of different features (e.g., area vs. smoothness), underscoring the importance of feature scaling.
- By standardizing the data, the models—especially KNN, which relies on Euclidean distance—were able to perform optimally without being biased by features with larger magnitudes.
- Both models achieved high predictive performance on the test set. Specific metric values and visualizations are detailed within the notebook.

## How to Run
1. Clone the repository and navigate to the project root.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Run the notebook `Lab6.ipynb` inside the `Lab 6` directory using Jupyter Notebook or Jupyter Lab.

---
**Registration Number:** 2547237
