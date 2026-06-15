# Credit Card Fraud Detection

## Overview

An end-to-end binary classification project to detect fraudulent credit card transactions using machine learning. The project covers data loading, exploratory data analysis, feature engineering, model training, evaluation, and model export.

---

## Project Details

| Item | Detail |
|------|--------|
| **Problem Type** | Binary Classification |
| **Target Variable** | is_fraud (0 = Legitimate, 1 = Fraud) |
| **Dataset** | credit_card_fraud_10k.csv (10,000 records) |
| **Best Model** | XGBoost Classifier |
| **Saved Model** | xgb_model.pkl (joblib) |
| **Train / Test Split** | 80% / 20% — random_state=42 |

---

## Model Comparison (Recorded Results)

| Model | Precision | Recall | F1 Score | AUC |
|-------|-----------|--------|----------|-----|
| Random Forest | 1.00 | 0.51 | 0.68 | 0.75 |
| Logistic Regression | 0.88 | 0.48 | 0.62 | 0.74 |
| XGBoost ✅ Best | ~1.00 | ~0.51 | ~0.68 | ~0.75 |

---

## XGBoost Configuration

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
```

---

## Pipeline

```
credit_card_fraud_10k.csv
  └── EDA (describe, isnull, duplicates, skew, groupby)
        └── Visualisation (boxplot, histplot, heatmap, countplot)
              └── Feature split: numerical (StandardScaler) + categorical (OneHotEncoder)
                    └── ColumnTransformer → Pipeline
                          └── Train/Test Split (80/20)
                                └── XGBClassifier → Evaluate → Export xgb_model.pkl
```

---

## Key EDA Findings

- Foreign transactions show higher fraud rates
- Transaction amount distribution is right-skewed
- Class imbalance present — fraud is minority class
- Correlation heatmap used to identify feature relationships

---

## Setup

```bash
pip install -r requirements.txt
```

Open the notebook in Google Colab or Jupyter and run cells top to bottom.

---

## File Structure

```app.py
├── xgb_model.pkl                        # Saved best model (joblib)
├── requirements.txt
└── README.md
```

---

## Notes

- Logistic Regression and Random Forest metrics were recorded from execution and documented in the notebook comparison table
- XGBoost achieved the best overall performance and was saved for future deployment
- High precision with lower recall indicates a conservative classifier — minimises false fraud alerts


- 
