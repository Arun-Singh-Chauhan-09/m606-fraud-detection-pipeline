# Credit Card Fraud Detection
**M606 Machine Learning — Gisma University of Applied Sciences**  
**Student:** Arun Singh Chauhan | **ID:** GH1052389 | **Professor:** Mohammad Mahdavi

---

## What this project is about
Credit card fraud is a real problem — globally, losses topped $33 billion in 2023. This project builds a full machine learning pipeline to automatically detect fraudulent transactions from a highly imbalanced dataset where only 0.17% of transactions are actually fraud. Because of that extreme imbalance, I focused on F1-Score and PR-AUC throughout rather than accuracy, which would be completely misleading here.

## Dataset
- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions over two days (September 2013, European cardholders)
- Features V1–V28 are PCA-transformed to protect cardholder privacy; Time and Amount are the only raw features

## Pipeline
1. **EDA** — checked for missing values, duplicates, class imbalance, feature distributions and correlations
2. **Preprocessing** — removed duplicates, scaled Time and Amount, stratified 80/20 train-test split
3. **Class imbalance** — random oversampling on training set only + `class_weight='balanced'`
4. **Model training** — Logistic Regression, Decision Tree, Random Forest
5. **Hyperparameter tuning** — `RandomizedSearchCV` with Stratified K-Fold, scored on F1
6. **Evaluation** — Confusion Matrix, ROC-AUC, PR-AUC, F1-Score for all three models
7. **Feature importance** — Random Forest importances (V17, V14, V12 were the strongest)
8. **Final model** — best model saved via `joblib`

## Results
Random Forest came out on top with the best F1-Score (0.83) and PR-AUC (0.80) on the held-out test set. Logistic Regression had a higher ROC-AUC but a much lower F1 — which makes sense, since ROC-AUC is less informative when one class is this rare.

## Files
| File | Description |
|------|-------------|
| `credit_card_fraud_detection.ipynb` | Main notebook |
| `credit_card_fraud_detection.html` | HTML export for submission |
| `best_fraud_detection_model.pkl` | Saved final model (generated on run) |

## How to Run
1. Clone the repo
```bash
git clone https://github.com/Arun-Singh-Chauhan-09/m606-fraud-detection-pipeline.git
cd m606-fraud-detection-pipeline
```
2. Set up Kaggle API credentials, then run the first cell — it will download and unzip the dataset automatically
3. Run all cells top to bottom

## Requirements
```
pandas, numpy, matplotlib, seaborn
scikit-learn, joblib, kaggle
Python 3.10+
```

## Note
No changes will be made to this repo after the submission deadline of **9 April 2026, 18:00 Berlin Time**, in line with assessment guidelines.
