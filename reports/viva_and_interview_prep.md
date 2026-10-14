# FraudShield — Viva & Interview Preparation Guide

This document prepares you for project evaluations, technical interviews, and thesis viva defenses for **FraudShield: Intelligent Fraud Detection and Risk Analysis System**.

---

## 1. Resume-Ready Project Descriptions

### Option A: Bullet-Point Format (Recommended for Resume)
* **FraudShield — Intelligent Fraud Detection & Risk Analysis System**
  * Engineered an end-to-end Machine Learning pipeline to detect fraudulent financial transactions using **Python, Pandas, and Scikit-learn**.
  * Handled severe class imbalance (~0.17% fraud prevalence) using cost-sensitive learning (`class_weight='balanced'`) and stratified cross-validation.
  * Built and compared a baseline **Logistic Regression** model against a **Random Forest Classifier**, optimizing for **Precision-Recall AUC (PR-AUC)**, **Recall**, and **F1-Score**.
  * Developed an interactive **Streamlit dashboard** enabling real-time transaction scoring, dynamic risk threshold tuning, and model comparison heatmaps.
  * Built a **FastAPI** inference service exposing `/predict`, `/model-info`, and `/metrics` REST endpoints with schema validation.

### Option B: Concise 2-Sentence Summary
> *Developed a supervised machine learning decision-support system to classify credit card transactions as legitimate or potentially fraudulent using Scikit-learn. Compared Logistic Regression and Random Forest models across PR-AUC and Recall metrics, deploying a real-time risk assessment dashboard via Streamlit.*

---

## 2. Core Viva Questions & Expert Answers

### Q1: What is fraud detection in machine learning?
**Answer:** Fraud detection is a binary classification (or anomaly detection) task where an ML algorithm analyzes transaction attributes (e.g., amount, velocity, location, historical behavior) to determine whether a transaction has a high likelihood of being unauthorized or fraudulent, flagging it for immediate action or human review.

### Q2: Why is this framed as a classification problem rather than clustering?
**Answer:** Because we have historical, labeled transaction data where each transaction is verified as either legitimate (`Class = 0`) or fraudulent (`Class = 1`). Supervised classification allows us to learn explicit decision boundaries and evaluate quantifiable performance against ground truth labels.

### Q3: What is class imbalance and why does it occur here?
**Answer:** Class imbalance occurs when one class significantly outnumbers another. In credit card payments, legitimate transactions make up over 99.8% of all volume, while fraud represents only ~0.17%. If unaddressed, standard ML algorithms will simply learn to predict the majority class to minimize loss.

### Q4: Why is standard Accuracy a useless metric for fraud detection?
**Answer:** If 99.83% of transactions are legitimate and 0.17% are fraudulent, a "dummy" model that predicts every single transaction is legitimate will achieve **99.83% accuracy**. However, its recall on fraud is **0%**—it completely fails to catch any fraud. Hence, accuracy is dangerously misleading for imbalanced problems.

### Q5: What metrics should be used instead?
**Answer:**
1. **Recall (Sensitivity / Detection Rate)**: $\frac{TP}{TP + FN}$ — Of all actual fraud cases, how many did the system catch? In fraud prevention, maximizing recall reduces direct financial loss.
2. **Precision**: $\frac{TP}{TP + FP}$ — Of all transactions flagged as fraud, how many were actually fraud? High precision minimizes unnecessary customer friction and false alarms.
3. **F1-Score**: Harmonic mean of precision and recall ($\frac{2 \cdot P \cdot R}{P + R}$).
4. **PR-AUC (Precision-Recall Area Under Curve)**: Unlike ROC-AUC, PR-AUC evaluates performance specifically on the rare positive class across all decision thresholds, making it the gold standard for imbalanced classification.
5. **Confusion Matrix**: Visualizes True Positives, False Positives, True Negatives, and False Negatives.

### Q6: What is a False Positive vs. a False Negative in this context?
**Answer:**
* **False Positive (Type I Error)**: A legitimate customer's card is flagged/blocked. (Result: Customer annoyance, lost sale).
* **False Negative (Type II Error)**: A fraudulent charge goes undetected. (Result: Direct financial loss and chargeback fees).
* **Business Trade-off**: Financial institutions generally tolerate a certain number of False Positives to avoid catastrophic False Negatives.

### Q7: What is Data Leakage and how did you prevent it?
**Answer:** Data leakage happens when information from outside the training portion influences the training process, causing artificially inflated evaluation scores that fail in production.
We prevented leakage by:
1. Splitting the dataset into train and test sets **before** applying any feature transformations.
2. Using Scikit-learn's `Pipeline` and `ColumnTransformer` so that scalers (like `RobustScaler`) learn their parameters (`median`, `IQR`) strictly from the training folds.

### Q8: Why did you choose RobustScaler instead of StandardScaler?
**Answer:** The transaction `Amount` feature is heavily skewed with extreme outliers (e.g., $1 micro-transactions up to $25,000 corporate transfers). `StandardScaler` uses mean and variance, which are easily distorted by extreme outliers. `RobustScaler` uses the **median** and the **Interquartile Range (IQR)**, making the scaling resistant to outlier distortion.

### Q9: Why compare Logistic Regression and Random Forest?
**Answer:**
* **Logistic Regression (Baseline)**: Provides a fast, linear benchmark. It helps us understand whether the problem is linearly separable and proves whether a more complex model is justified.
* **Random Forest (Champion)**: An ensemble of decision trees using bootstrap aggregating (bagging). It captures complex non-linear combinations among the 28 PCA features without overfitting and natively handles skewed distributions.

### Q10: How does the decision threshold slider work?
**Answer:** By default, classification models assign class `1` if the predicted probability is $\ge 0.50$. However, depending on business risk appetite:
* Setting the threshold lower (e.g., `0.35`) increases **Recall** (catches more fraud) at the expense of more False Positives.
* Setting the threshold higher (e.g., `0.65`) increases **Precision** (minimizes customer friction) but risks missing subtle fraud.

### Q11: Is an ML prediction "proof" of fraud?
**Answer:** No. An ML model outputs a statistical probability score indicating how closely a transaction resembles historical fraud patterns. In responsible AI and real-world banking operations, it serves as an automated decision-support filter that triggers secondary verification (SMS OTP, biometric check) or routes to human analysts—it is never an automatic legal declaration.
