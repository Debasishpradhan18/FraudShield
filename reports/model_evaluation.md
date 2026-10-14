# FraudShield — Model Evaluation Report

## 1. Overview
This report documents the performance evaluation of the **FraudShield Intelligent Fraud Detection System** on unseen test data.

## 2. Model Comparison Summary
- **Baseline Model**: Logistic Regression (Balanced class weights)
- **Champion Model**: Random Forest Classifier (Balanced class weights, 100 trees)

### Precision-Recall & ROC Analysis
Evaluating models on imbalanced fraud datasets requires focusing on **Precision-Recall AUC (PR-AUC)** rather than standard ROC-AUC, because the majority class (non-fraud) can inflate ROC-AUC artificially.

## 3. Visual Artifacts
- **Confusion Matrix**: `reports/figures/confusion_matrix.png`
- **PR & ROC Curves**: `reports/figures/curves_comparison.png`
- **Threshold Trade-Off**: `reports/figures/threshold_tradeoff.png`

## 4. Operational Threshold Guidance
The default probability threshold (0.50) is not always ideal for banking operations:
- **Lower Threshold (0.30 - 0.40)**: Captures more fraud (higher Recall), reducing financial loss, with a small increase in false alerts.
- **Higher Threshold (0.60 - 0.70)**: Reduces false alerts to minimize customer inconvenience, but risks missing subtle fraudulent transactions.
