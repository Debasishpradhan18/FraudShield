"""
FraudShield - Training Module
Trains both the Baseline (Logistic Regression) and Champion (Random Forest) models,
encapsulates preprocessing inside Scikit-learn Pipelines to avoid leakage,
evaluates on held-out test data, and persists the models using joblib.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
from typing import Dict, Any
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from src.preprocessing import build_preprocessor, load_and_split_data
from src.data_validation import FEATURE_NAMES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")


def evaluate_model_pipeline(pipeline: Pipeline, X_test, y_test) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics on unseen test data."""
    y_pred = pipeline.predict(X_test)
    y_scores = pipeline.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, y_scores)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    return metrics


def train_models(data_path: str = RAW_DATA_PATH, save_dir: str = MODELS_DIR) -> Dict[str, Any]:
    """
    Executes the training workflow:
    1. Loads and stratifies data.
    2. Trains Baseline: Logistic Regression (class_weight='balanced').
    3. Trains Champion: Random Forest Classifier (class_weight='balanced').
    4. Evaluates both models on held-out test set.
    5. Persists pipelines and training metadata.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Ensure dataset is available
    if not os.path.exists(data_path):
        from data.sample_data_generator import ensure_dataset_available
        data_path = ensure_dataset_available()

    print("[1/5] Loading and splitting dataset...")
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)

    # ----------------------------------------------------
    # Baseline Model: Logistic Regression
    # ----------------------------------------------------
    print("\n[2/5] Training Baseline Model: Logistic Regression...")
    start_time = time.time()
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_duration = time.time() - start_time
    baseline_metrics = evaluate_model_pipeline(baseline_pipeline, X_test, y_test)
    baseline_metrics["training_time_seconds"] = round(baseline_duration, 2)
    print(f"      Baseline Trained in {baseline_duration:.2f}s | Recall: {baseline_metrics['recall']:.4f} | PR-AUC: {baseline_metrics['pr_auc']:.4f}")

    # ----------------------------------------------------
    # Champion Model: Random Forest Classifier
    # ----------------------------------------------------
    print("\n[3/5] Training Champion Model: Random Forest Classifier...")
    start_time = time.time()
    champion_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    champion_pipeline.fit(X_train, y_train)
    champion_duration = time.time() - start_time
    champion_metrics = evaluate_model_pipeline(champion_pipeline, X_test, y_test)
    champion_metrics["training_time_seconds"] = round(champion_duration, 2)
    print(f"      Champion Trained in {champion_duration:.2f}s | Recall: {champion_metrics['recall']:.4f} | PR-AUC: {champion_metrics['pr_auc']:.4f}")

    # ----------------------------------------------------
    # Model Persistence
    # ----------------------------------------------------
    print("\n[4/5] Saving model pipelines to disk...")
    baseline_path = os.path.join(save_dir, "baseline_logistic_regression.joblib")
    champion_path = os.path.join(save_dir, "champion_random_forest.joblib")
    primary_pipeline_path = os.path.join(save_dir, "fraud_pipeline.joblib")

    joblib.dump(baseline_pipeline, baseline_path)
    joblib.dump(champion_pipeline, champion_path)
    # Default active pipeline points to the champion Random Forest
    joblib.dump(champion_pipeline, primary_pipeline_path)

    metadata = {
        "version": "1.0.0",
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": FEATURE_NAMES,
        "default_decision_threshold": 0.50,
        "recommended_operational_threshold": 0.40,
        "baseline_model": {
            "name": "Logistic Regression (Balanced)",
            "metrics": baseline_metrics,
            "artifact": "baseline_logistic_regression.joblib",
        },
        "champion_model": {
            "name": "Random Forest Classifier (Balanced)",
            "metrics": champion_metrics,
            "artifact": "champion_random_forest.joblib",
        },
    }

    metadata_path = os.path.join(save_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[5/5] Artifacts saved:")
    print(f"      -> {baseline_path}")
    print(f"      -> {champion_path}")
    print(f"      -> {primary_pipeline_path}")
    print(f"      -> {metadata_path}")
    print("\n[+] Training workflow finished successfully!")
    return metadata


if __name__ == "__main__":
    train_models()
