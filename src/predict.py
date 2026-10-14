"""
FraudShield - Inference Engine
Loads persisted pipeline, validates input transaction features, calculates
fraud probability score, applies dynamic decision threshold, and returns
structured risk assessment.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, Union, List, Optional
import joblib
import pandas as pd
from src.data_validation import validate_inference_input, FEATURE_NAMES

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fraud_pipeline.joblib")


class FraudPredictor:
    """Wrapper around trained Scikit-learn Pipeline for real-time risk scoring."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                f"Please run 'python src/train.py' first to train and save the model."
            )
        return joblib.load(self.model_path)

    @staticmethod
    def categorize_risk(probability: float, threshold: float = 0.50) -> Dict[str, str]:
        """Categorizes raw probability score into actionable operational risk tiers."""
        if probability >= threshold:
            return {
                "risk_tier": "HIGH RISK",
                "recommendation": "Flag for human investigation or trigger Multi-Factor Authentication.",
                "badge_color": "danger",
            }
        elif probability >= (threshold * 0.5):
            return {
                "risk_tier": "SUSPICIOUS / ELEVATED",
                "recommendation": "Monitor next transactions from account; secondary check suggested.",
                "badge_color": "warning",
            }
        else:
            return {
                "risk_tier": "LOW RISK",
                "recommendation": "Approve transaction automatically.",
                "badge_color": "success",
            }

    def predict_single(
        self,
        transaction: Dict[str, Any],
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Scores a single transaction dictionary.
        Returns prediction, probability score, risk tier, and recommendation.
        """
        validated_df = validate_inference_input(transaction)
        score = float(self.pipeline.predict_proba(validated_df)[0, 1])
        is_fraud = bool(score >= threshold)
        risk_info = self.categorize_risk(score, threshold=threshold)

        return {
            "prediction": 1 if is_fraud else 0,
            "status": "Potential Fraud" if is_fraud else "Legitimate",
            "fraud_probability": round(score, 4),
            "threshold_applied": threshold,
            "risk_tier": risk_info["risk_tier"],
            "recommendation": risk_info["recommendation"],
            "badge_color": risk_info["badge_color"],
            "disclaimer": "This risk score is an ML estimate for decision support and not legal proof of fraud.",
        }

    def predict_batch(
        self,
        transactions_df: pd.DataFrame,
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """Scores a DataFrame of transactions in bulk and adds prediction columns."""
        validated_df = validate_inference_input(transactions_df)
        scores = self.pipeline.predict_proba(validated_df)[:, 1]
        results_df = validated_df.copy()
        results_df["fraud_score"] = scores.round(4)
        results_df["predicted_fraud"] = (scores >= threshold).astype(int)
        results_df["risk_level"] = [
            self.categorize_risk(s, threshold=threshold)["risk_tier"] for s in scores
        ]
        return results_df


if __name__ == "__main__":
    # Smoke test if model exists
    if os.path.exists(DEFAULT_MODEL_PATH):
        predictor = FraudPredictor()
        dummy = {col: 0.0 for col in FEATURE_NAMES}
        dummy["Amount"] = 149.99
        dummy["Time"] = 3600.0
        result = predictor.predict_single(dummy)
        print("Sample Prediction Output:\n", result)
    else:
        print(f"Model not yet trained. Run src/train.py first.")
