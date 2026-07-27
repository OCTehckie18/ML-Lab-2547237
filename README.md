# Machine Learning Lab Activity: Decision Trees

## Objective
This lab activity explores **Decision Tree classification** using the built-in Iris dataset from `sklearn.datasets.load_iris()`. The notebook is designed to be learner-centric, with individual code cells and short interpretations explaining why each step is used.

## Notebook
- **File:** `Lab_activity/decision_trees.ipynb`
- **Topic:** Decision Tree Classifier
- **Dataset:** Iris Dataset
- **Library Source:** `sklearn.datasets.load_iris()`

## Dataset
- **Name:** Iris Dataset
- **Samples:** 150 flower records
- **Features:** 4 numeric flower measurements
- **Target Classes:** `setosa`, `versicolor`, and `virginica`
- **Class Balance:** The dataset contains an equal number of samples from each class.

## Features Used
- `sepal length (cm)`
- `sepal width (cm)`
- `petal length (cm)`
- `petal width (cm)`

## Tasks Covered
1. **Dataset Exploration**
   - Loaded the Iris dataset.
   - Displayed the number of samples and features.
   - Listed feature names and target classes.
   - Displayed the first five records.
   - Checked class distribution.

2. **Data Preparation**
   - Split the dataset into 80% training data and 20% testing data using `random_state=42`.
   - Explained why separate training and testing sets are needed for model evaluation.

3. **Default Decision Tree Classifier**
   - Trained a default `DecisionTreeClassifier`.
   - Predicted class labels for the test dataset.
   - Evaluated the model using accuracy score, confusion matrix, and classification report.

4. **Decision Tree Visualization**
   - Visualized the trained tree using `plot_tree()`.
   - Identified the root node, internal nodes, leaf nodes, maximum tree depth, and first split feature.
   - Explained why the first split feature is selected based on impurity reduction.

5. **Gini Index vs. Entropy**
   - Trained separate models using `criterion="gini"` and `criterion="entropy"`.
   - Compared accuracy, tree depth, number of leaf nodes, and root feature selected.
   - Identified which criterion produced the simpler tree.

6. **Effect of Maximum Tree Depth**
   - Trained models with `max_depth` values of `1`, `2`, `3`, `4`, and `None`.
   - Compared training accuracy, testing accuracy, actual tree depth, and learner-friendly observations.
   - Discussed underfitting, best generalization, and overfitting risk.

7. **Effect of `min_samples_split`**
   - Trained models using values `2`, `5`, `10`, and `20`.
   - Compared accuracy, tree depth, number of leaf nodes, and model complexity.

8. **Effect of `min_samples_leaf`**
   - Trained models using values `1`, `2`, `5`, and `10`.
   - Explained how larger leaf-size requirements reduce overly specific rules and help control overfitting.

9. **Hyperparameter Tuning**
   - Used `GridSearchCV` to tune:
     - `criterion`: `gini`, `entropy`
     - `max_depth`: `2`, `3`, `4`, `None`
     - `min_samples_split`: `2`, `5`, `10`
     - `min_samples_leaf`: `1`, `2`, `4`
   - Reported the best hyperparameter combination, best cross-validation score, and optimized model test accuracy.

10. **Analysis**
    - Explained the role of the `criterion` parameter.
    - Discussed how `max_depth` affects underfitting and overfitting.
    - Explained why larger `min_samples_split` and `min_samples_leaf` values produce simpler trees.
    - Recommended a suitable Decision Tree model for the Iris dataset.

## Key Learning Outcomes
- Decision Trees are interpretable models because their predictions can be visualized as decision rules.
- `criterion` controls how split quality is measured.
- `max_depth` directly controls tree growth and has a strong effect on underfitting and overfitting.
- Larger `min_samples_split` and `min_samples_leaf` values usually produce simpler trees.
- Hyperparameter tuning helps select a model that balances accuracy and interpretability.

## How to Run
1. Clone the repository and navigate to the project root.
2. Install the required dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib jupyter
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook Lab_activity/decision_trees.ipynb
   ```

---
**Registration Number:** 2547237
