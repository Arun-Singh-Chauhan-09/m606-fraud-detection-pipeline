# Credit Card Fraud Detection
M606 Machine Learning — Gisma University of Applied Sciences  
Arun Singh Chauhan | GH1052389 

---

I picked fraud detection because it's one of those problems that looks easy on paper but gets complicated fast — the dataset has 284,807 transactions and only 492 of them are actually fraud. That's 0.17%. So the first thing I had to figure out was what "good" even means for a model like this, because accuracy is basically useless here. A model that just predicts "genuine" every single time scores 99.83% accuracy and catches zero fraud — so I used F1-Score and PR-AUC throughout instead.

The data is from Kaggle (European cardholders, September 2013, two days of transactions). The features V1–V28 are anonymised via PCA, so there's no domain-specific feature engineering possible — but there's still enough signal in there to work with.

**What I did:**  
Started with EDA to understand the imbalance and spot which features looked useful — V17, V14, and V12 came out as the strongest correlates with fraud. Found 1,081 duplicate rows and dropped them. Scaled Time and Amount with StandardScaler (the PCA features were already scaled), did a stratified 80/20 split, then oversampled the fraud cases in the training set only. Doing oversampling before the split would've leaked data into the test set, so the order matters.

Trained three models with default hyperparameters — Logistic Regression as a sanity-check baseline, then Decision Tree and Random Forest. Random Forest came out on top with an F1 of 0.8242 and PR-AUC of 0.7926. Interestingly, Logistic Regression had a better ROC-AUC but much worse F1, which is exactly why I didn't lean on ROC-AUC as the main metric for an imbalanced problem like this.

**Files:**

| File | What it is |
|------|------------|
| `credit_card_fraud_detection.ipynb` | main notebook |
| `credit_card_fraud_detection.html` | HTML export of the same notebook — identical content, for submission |

> The `.ipynb` and `.html` are the same file in two formats — 44 cells, identical code and outputs. The HTML is just the submission-ready export.

**To run it:**  
Clone the repo, drop `creditcard.csv` (from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) into the project folder, then run all cells.

```bash
git clone https://github.com/Arun-Singh-Chauhan-09/m606-fraud-detection-pipeline.git
```

Needs: pandas, numpy, matplotlib, scikit-learn — Python 3.10+


