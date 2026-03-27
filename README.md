# Credit Card Fraud Detection
M606 Machine Learning — Gisma University of Applied Sciences  
Arun Singh Chauhan | GH1052389 | Prof. Mohammad Mahdavi

---

This was my M606 project. I picked fraud detection because it's one of those problems that looks easy on paper but gets complicated fast — the dataset has 284,807 transactions and only 492 of them are actually fraud. That's 0.17%. So the first thing I had to figure out was what "good" even means for a model like this, because accuracy is basically useless here.

I ended up using F1-Score and PR-AUC as my main metrics throughout. A model predicting "genuine" every single time scores 99.83% accuracy and catches zero fraud — so that tells you everything about why I ignored accuracy.

The data is from Kaggle (European cardholders, September 2013, two days of transactions). The features V1 to V28 have been anonymised via PCA so I couldn't do much domain-specific feature engineering, but there's still enough signal in there to work with.

**What I did:**  
Started with EDA to get a feel for the imbalance and which features looked like they'd be useful. Found 1,081 duplicate rows which I dropped. Scaled Time and Amount, did a stratified 80/20 split, then oversampled the fraud cases in the training set only — doing it before the split would've leaked data into the test set.

Trained three models — Logistic Regression as a sanity check baseline, then Decision Tree and Random Forest. Tuned all three with RandomizedSearchCV, scoring on F1 not accuracy. Random Forest won with F1 of 0.83 and PR-AUC of 0.80. Interestingly Logistic Regression had a better ROC-AUC but much worse F1, which is exactly why I didn't use ROC-AUC as the main metric. V17, V14 and V12 were the most important features, which matched what I saw in the correlation plots during EDA.

**Files:**

| File | What it is |
|------|------------|
| `credit_card_fraud_detection.ipynb` | main notebook |
| `credit_card_fraud_detection.html` | HTML export for submission |
| `best_fraud_detection_model.pkl` | saved model, created when you run the notebook |

**To run it:**  
Clone the repo, make sure you've got Kaggle API credentials set up, then just run all cells. The first cell downloads the dataset automatically.

```bash
git clone https://github.com/Arun-Singh-Chauhan-09/m606-fraud-detection-pipeline.git
```

Needs: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, kaggle — Python 3.10+

*Repo frozen after 9 April 2026, 18:00 Berlin time.*
