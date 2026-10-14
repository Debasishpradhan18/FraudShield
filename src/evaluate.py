"""
FraudShield - Model Evaluation & Threshold Analysis Module
Computes detailed performance metrics, plots Confusion Matrices and
Precision-Recall Curves, analyzes decision thresholds, and generates
the model evaluation report.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.preprocessing import load_and_split_data
from src.data_validation import FEATURE_NAMES

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "creditcard.csv")


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """Plots and saves an annotated confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_and_roc_curves(y_true, y_scores_baseline, y_scores_rf, save_path: str):
    """Plots comparison Precision-Recall and ROC curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Precision-Recall Curve (Crucial for class imbalance)
    p_base, r_base, _ = precision_recall_curve(y_true, y_scores_baseline)
    p_rf, r_rf, _ = precision_recall_curve(y_true, y_scores_rf)
    ap_base = average_precision_score(y_true, y_scores_baseline)
    ap_rf = average_precision_score(y_true, y_scores_rf)

    axes[0].plot(r_base, p_base, label=f"Logistic Regression (PR-AUC = {ap_base:.3f})", color="#1f77b4", lw=2)
    axes[0].plot(r_rf, p_rf, label=f"Random Forest (PR-AUC = {ap_rf:.3f})", color="#2ca02c", lw=2)
    axes[0].set_title("Precision-Recall Curve (Primary Imbalance Metric)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Recall (Detection Rate)")
    axes[0].set_ylabel("Precision (Positive Predictive Value)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left")

    # ROC Curve
    fpr_base, tpr_base, _ = roc_curve(y_true, y_scores_baseline)
    fpr_rf, tpr_rf, _ = roc_curve(y_true, y_scores_rf)
    roc_base = roc_auc_score(y_true, y_scores_baseline)
    roc_rf = roc_auc_score(y_true, y_scores_rf)

    axes[1].plot(fpr_base, tpr_base, label=f"Logistic Regression (ROC-AUC = {roc_base:.3f})", color="#1f77b4", lw=2)
    axes[1].plot(fpr_rf, tpr_rf, label=f"Random Forest (ROC-AUC = {roc_rf:.3f})", color="#2ca02c", lw=2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def generate_threshold_analysis(y_true, y_scores, save_path: str) -> pd.DataFrame:
    """Analyzes precision, recall, and false alarms across thresholds from 0.1 to 0.9."""
    thresholds = np.linspace(0.05, 0.95, 19)
    records = []

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        records.append({
            "threshold": round(t, 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        })

    df_t = pd.DataFrame(records)

    # Plot threshold trade-off
    plt.figure(figsize=(9, 4.5))
    plt.plot(df_t["threshold"], df_t["precision"], label="Precision", color="#1f77b4", lw=2)
    plt.plot(df_t["threshold"], df_t["recall"], label="Recall", color="#d62728", lw=2)
    plt.plot(df_t["threshold"], df_t["f1_score"], label="F1-Score", color="#2ca02c", lw=2, linestyle="--")
    plt.axvline(0.5, color="gray", linestyle=":", label="Default Threshold (0.50)")
    plt.title("Decision Threshold Trade-Off Analysis", fontsize=12, fontweight="bold")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return df_t


def run_full_evaluation():
    """Generates charts, saves metrics, and writes markdown evaluation report."""
    base_model_path = os.path.join(MODELS_DIR, "baseline_logistic_regression.joblib")
    rf_model_path = os.path.join(MODELS_DIR, "champion_random_forest.joblib")

    if not os.path.exists(base_model_path) or not os.path.exists(rf_model_path):
        print("[!] Models not found. Training first...")
        from src.train import train_models
        train_models()

    print("[*] Running full evaluation...")
    _, X_test, _, y_test = load_and_split_data(RAW_DATA_PATH)

    pipeline_base = joblib.load(base_model_path)
    pipeline_rf = joblib.load(rf_model_path)

    scores_base = pipeline_base.predict_proba(X_test)[:, 1]
    scores_rf = pipeline_rf.predict_proba(X_test)[:, 1]
    preds_rf = pipeline_rf.predict(X_test)

    # 1. Confusion Matrix
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_test, preds_rf, "Random Forest Confusion Matrix (Default Threshold 0.50)", cm_path)

    # 2. PR & ROC Curves
    curves_path = os.path.join(FIGURES_DIR, "curves_comparison.png")
    plot_pr_and_roc_curves(y_test, scores_base, scores_rf, curves_path)

    # 3. Threshold Analysis
    threshold_path = os.path.join(FIGURES_DIR, "threshold_tradeoff.png")
    df_thresholds = generate_threshold_analysis(y_test, scores_rf, threshold_path)

    # 4. Generate Markdown Report
    report_content = f"""# FraudShield — Model Evaluation Report

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
"""
    with open(os.path.join(REPORTS_DIR, "model_evaluation.md"), "w") as f:
        f.write(report_content)

    print(f"[+] Evaluation complete. Figures and report generated in: {REPORTS_DIR}")


if __name__ == "__main__":
    run_full_evaluation()
