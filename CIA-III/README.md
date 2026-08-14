# 🛡️ Mission Crisis × Robin Hood Army
## Classifying Disaster Food-Aid Requests Using Ensemble Machine Learning

**Registration Number:** 2547237  
**Course:** MCA 521-4 — Machine Learning  
**Assessment:** CIA-III — ML for Social Good Ensemble Challenge (25 Marks)  
**Mission Domain:** Crisis / Disaster Response

---

## Problem Statement

During large-scale crises (floods, COVID lockdowns, earthquakes), humanitarian organisations like the **Robin Hood Army** receive hundreds of simultaneous messages requesting aid. Manually triaging which messages are genuine **food-aid requests** versus general chatter, news, or offers of help is a critical bottleneck that delays food delivery to those in need.

This project builds an **end-to-end ensemble ML pipeline** that classifies disaster messages as food-aid requests or not, using the **Multilingual Disaster Response Messages** dataset. The model acts as a triage assistant for RHA volunteers.

## Dataset

- **Name:** Multilingual Disaster Response Messages
- **Source:** [Kaggle — landlord/multilingual-disaster-response-messages](https://www.kaggle.com/datasets/landlord/multilingual-disaster-response-messages?resource=download)
- **Original Provider:** Figure Eight / Appen
- **Real-world origin:** Messages from the 2010 Haiti earthquake, 2010 Pakistan floods, 2012 Hurricane Sandy
- **Size:** ~26,000 messages × 36 binary category labels
- **License:** CC0 (Public Domain)

## Project Structure

```
CIA-III/
├── data/
│   └── disaster_messages.csv         # Dataset (download from Kaggle)
├── notebooks/
│   └── 2547237_mission_crisis.ipynb   # Main notebook (Q1–Q5)
├── models/
│   └── best_pipeline.joblib           # Saved best model pipeline
├── figures/
│   ├── eda_*.png                      # EDA visualisations
│   ├── model_comparison.png           # Metrics comparison
│   ├── confusion_matrices.png         # Confusion matrices
│   ├── roc_curves.png                 # ROC curves
│   └── shap_*.png                     # Explainability plots
├── README.md                          # This file
└── ethics_statement.md                # Ethics & limitations
```

## How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn joblib jupyter
```

### Steps

1. **Clone the repository** and navigate to the CIA-III directory:
   ```bash
   git clone <repo-url>
   cd ML-Lab-2547237/CIA-III
   ```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/landlord/multilingual-disaster-response-messages?resource=download) and place `disaster_messages.csv` in `data/`.

3. **Run the notebook:**
   ```bash
   jupyter notebook notebooks/2547237_mission_crisis.ipynb
   ```
   Execute all cells sequentially. The notebook handles:
   - Data loading, cleaning, and EDA
   - Feature engineering (TF-IDF + handcrafted domain features)
   - Model training: Decision Tree → Random Forest → XGBoost → LightGBM → Voting → Stacking
   - SHAP explainability (global + local)
   - Live prediction demo

4. **Verify model output:**
   ```python
   import joblib
   pipeline = joblib.load('models/best_pipeline.joblib')
   print(f"Model loaded: {pipeline['model_name']}")
   ```

## Models Implemented

| Model | Type | Description |
|---|---|---|
| Decision Tree | Baseline | Single tree with `max_depth=10`, `class_weight='balanced'` |
| Random Forest | Bagging | GridSearchCV-tuned; 200–300 trees |
| XGBoost | Boosting | Gradient boosting with `scale_pos_weight` for imbalance |
| LightGBM | Boosting | Leaf-wise gradient boosting with `is_unbalance=True` |
| Voting | Ensemble | Soft voting: RF + XGBoost + LightGBM |
| Stacking | Ensemble | Base: RF + XGBoost + LightGBM; Meta: Logistic Regression |

## Key Features Engineered

- **TF-IDF (300 features):** Unigrams and bigrams from message text
- **Food keyword count:** Matches against domain vocabulary (hungry, ration, starving, etc.)
- **Water/Medical/Shelter keyword counts:** Domain-informed features
- **Urgency keyword count:** Help, please, urgent, emergency, SOS, etc.
- **Message length & word count:** Structural features
- **Genre encoding:** Direct SMS vs. news vs. social media

## Responsible Use

- The model is a **triage assistant**, not an autonomous decision-maker.
- **Recall is prioritised** over precision — missing a food request is costlier than a false alarm.
- Human volunteers must review all flagged messages before action.
- See `ethics_statement.md` for the full ethics discussion.

---

**Registration Number:** 2547237
