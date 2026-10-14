"""
FraudShield - REST API Inference Service (FastAPI)
Implements Section 12 of the Project Guide:
- GET  /health     : Health check endpoint
- GET  /model-info : Returns model version, feature schema, and threshold metadata
- GET  /metrics    : Returns precomputed evaluation metrics
- POST /predict    : Real-time fraud scoring for single transaction payloads
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd

from src.data_validation import FEATURE_NAMES, validate_inference_input
from src.predict import FraudPredictor, DEFAULT_MODEL_PATH
from src.train import MODELS_DIR

app = FastAPI(
    title="FraudShield Inference API",
    description="Intelligent Fraud Detection and Risk Analysis Service",
    version="1.0.0",
)

# Global predictor instance
_predictor: Optional[FraudPredictor] = None


def get_predictor() -> FraudPredictor:
    global _predictor
    if _predictor is None:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model pipeline has not been trained yet. Please run 'python src/train.py' first.",
            )
        _predictor = FraudPredictor(DEFAULT_MODEL_PATH)
    return _predictor


class TransactionPayload(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping all 30 feature names (Time, V1-V28, Amount) to float values.",
        example={
            "Time": 1000.0,
            "Amount": 125.50,
            "V1": -0.25,
            "V2": 0.12,
            "V3": 0.45,
            "V4": -0.10,
            "V5": 0.05,
            "V6": -0.08,
            "V7": 0.22,
            "V8": -0.04,
            "V9": 0.15,
            "V10": -0.12,
            "V11": 0.08,
            "V12": -0.19,
            "V13": 0.03,
            "V14": -0.22,
            "V15": 0.06,
            "V16": -0.05,
            "V17": 0.11,
            "V18": -0.07,
            "V19": 0.04,
            "V20": -0.02,
            "V21": 0.01,
            "V22": -0.03,
            "V23": 0.02,
            "V24": -0.01,
            "V25": 0.05,
            "V26": -0.04,
            "V27": 0.02,
            "V28": -0.01,
        },
    )
    threshold: Optional[float] = Field(
        default=0.50,
        ge=0.01,
        le=0.99,
        description="Custom operational decision threshold (default: 0.50).",
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for legitimate-like, 1 for potentially fraudulent")
    label: str = Field(..., description="Classification category label")
    score: float = Field(..., description="Estimated fraud risk score (probability)")
    threshold_applied: float = Field(..., description="Threshold used for binary classification")
    risk_tier: str = Field(..., description="LOW RISK, SUSPICIOUS, or HIGH RISK")
    recommendation: str = Field(..., description="Operational policy recommendation")
    model_version: str = Field(..., description="Active model pipeline version")
    disclaimer: str = Field(..., description="Ethical & decision support disclaimer")


@app.get("/health", tags=["System"])
def health_check():
    """Confirms that the FraudShield inference service is alive and healthy."""
    model_ready = os.path.exists(DEFAULT_MODEL_PATH)
    return {
        "status": "healthy",
        "service": "FraudShield Inference API",
        "model_loaded": model_ready,
    }


@app.get("/model-info", tags=["Model Information"])
def get_model_info():
    """Returns active model version, required feature schema, and threshold metadata."""
    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            return {
                "version": meta.get("version", "1.0.0"),
                "trained_date": meta.get("trained_date"),
                "champion_model": meta.get("champion_model", {}).get("name"),
                "default_threshold": meta.get("default_decision_threshold", 0.50),
                "feature_count": len(FEATURE_NAMES),
                "required_features": FEATURE_NAMES,
            }
    return {
        "version": "1.0.0",
        "champion_model": "Random Forest Classifier",
        "default_threshold": 0.50,
        "feature_count": len(FEATURE_NAMES),
        "required_features": FEATURE_NAMES,
    }


@app.get("/metrics", tags=["Model Information"])
def get_metrics():
    """Returns precomputed evaluation metrics on the held-out test dataset."""
    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training metadata not found. Please train models first.",
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return {
        "baseline_logistic_regression": meta.get("baseline_model", {}).get("metrics"),
        "champion_random_forest": meta.get("champion_model", {}).get("metrics"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_transaction(payload: TransactionPayload):
    """
    Validates input transaction features and returns:
    - predicted class (0 = legitimate, 1 = potential fraud)
    - estimated fraud risk score
    - operational risk tier and recommendation
    """
    try:
        predictor = get_predictor()
        threshold = payload.threshold or 0.50
        res = predictor.predict_single(payload.features, threshold=threshold)

        return PredictionResponse(
            prediction=res["prediction"],
            label="potentially fraudulent" if res["prediction"] == 1 else "legitimate-like",
            score=res["fraud_probability"],
            threshold_applied=threshold,
            risk_tier=res["risk_tier"],
            recommendation=res["recommendation"],
            model_version="1.0.0",
            disclaimer=res["disclaimer"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {str(e)}")
