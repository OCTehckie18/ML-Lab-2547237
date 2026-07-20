# Lab 5: Linear Regression through Gradient Descent

## Comprehensive Study of Linear Regression using Gradient Descent Optimization on Student Performance Dataset

---

### 🎯 Aim

To implement Linear Regression using the Gradient Descent optimization algorithm from scratch and evaluate its performance on a real-world dataset.

---

### 📊 Dataset

- **Student Performance Dataset** (`student-mat.csv`)
- **Source:** UCI Machine Learning Repository
- **Samples:** 395
- **Features:** 30 demographic, social, and school-related attributes
- **Target (y):** `G3` — Final grade (numeric: 0 to 20)

---

### 🧪 Problem Statement

An educational analytics team aims to predict students' final Mathematics grades (`G3`) based on demographic, social, and academic attributes. We build a Linear Regression model from scratch using Batch Gradient Descent, experiment with different learning rates to analyze convergence, and evaluate the model using standard regression metrics (MAE, MSE, RMSE, and R² Score).

---

### 📋 Tasks Overview

| Task | Description |
|------|-------------|
| **Task 1** | Data Preparation — Load, explore, and check dataset characteristics |
| **Task 2** | Data Preprocessing — Encode categorical variables and scale features using StandardScaler |
| **Task 3** | Train-Test Split — Split data (80:20) for model training and evaluation |
| **Task 4** | Gradient Descent Implementation — Build Linear Regression from scratch with Batch Gradient Descent |
| **Task 5** | Learning Rate Experimentation — Train models with varying learning rates (0.001, 0.01, 0.05, 0.1) |
| **Task 6** | Convergence Analysis — Plot Loss (Cost) vs Iterations for each learning rate |
| **Task 7** | Model Evaluation — Evaluate using MAE, MSE, RMSE, and R² Score |

---

### 🔑 Key Learnings

#### 1. Batch Gradient Descent
- Gradient Descent is an iterative optimization algorithm used to find the parameters (weights and bias) that minimize the cost function (Mean Squared Error).
- **Batch Gradient Descent** computes the gradient using the entire training dataset at each step, ensuring a stable convergence but can be computationally expensive for very large datasets.
- Update rules involve subtracting a fraction (learning rate) of the gradient from the current parameters.

#### 2. Feature Scaling
- Feature scaling via `StandardScaler` is crucial for Gradient Descent.
- Features on different scales result in an elongated, elliptical cost function landscape, causing the algorithm to oscillate and converge slowly.
- Standardizing features to have mean ≈ 0 and standard deviation ≈ 1 creates a more spherical contour, allowing for faster and more direct convergence.

#### 3. Learning Rate Selection
- **Too small (e.g., 0.001):** The algorithm converges very slowly and may require a massive number of iterations to reach the minimum.
- **Optimal (e.g., 0.01 - 0.05):** The cost decreases steadily and reaches the minimum in a reasonable number of iterations.
- **Too large (e.g., 0.1 or higher):** The algorithm may overshoot the minimum, leading to divergence or oscillation.

#### 4. Regression Evaluation Metrics
- **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values. Easier to interpret in the target's original units.
- **Mean Squared Error (MSE):** Averages the squared differences. Heavily penalizes larger errors (outliers) due to squaring. Used as the cost function for Gradient Descent.
- **Root Mean Squared Error (RMSE):** The square root of MSE, bringing the error metric back to the target's original units while maintaining the penalization of large errors.
- **R² Score (Coefficient of Determination):** Measures the proportion of variance in the target variable explained by the model. Values closer to 1 indicate better fit.

---

### ✅ Expected Outcomes Achieved

- Understanding of the Gradient Descent algorithm for optimizing a Linear Regression model.
- Ability to preprocess data efficiently, including encoding and feature scaling.
- Implementation of Linear Regression from scratch using NumPy.
- Analysis of the effect of learning rates on loss convergence.
- Evaluation and interpretation of standard regression metrics.

---

### 🛠️ Technologies Used

- **Python 3.x**
- **NumPy** & **Pandas** — Data handling and numerical computations
- **Matplotlib** & **Seaborn** — Visualization of cost convergence
- **scikit-learn** — Data preprocessing (LabelEncoder, StandardScaler), splitting, and evaluation metrics

---

### 📁 File Structure

```
Lab5/
├── Lab5.ipynb    # Complete Jupyter Notebook with code, outputs, and interpretations
└── README.md     # This file — summary of learnings and key takeaways
```

---

### 👤 Author

**Omkaar** — Register Number: 2547237  
4th Trimester, Machine Learning Lab  
Christ University
