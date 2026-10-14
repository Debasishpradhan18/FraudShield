"""
FraudShield - Prediction Unit Tests
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from src.predict import FraudPredictor
from src.data_validation import FEATURE_NAMES


def test_categorize_risk():
    high = FraudPredictor.categorize_risk(0.85, threshold=0.50)
    assert high["risk_tier"] == "HIGH RISK"
    assert high["badge_color"] == "danger"

    suspicious = FraudPredictor.categorize_risk(0.35, threshold=0.50)
    assert suspicious["risk_tier"] == "SUSPICIOUS / ELEVATED"
    assert suspicious["badge_color"] == "warning"

    low = FraudPredictor.categorize_risk(0.05, threshold=0.50)
    assert low["risk_tier"] == "LOW RISK"
    assert low["badge_color"] == "success"
