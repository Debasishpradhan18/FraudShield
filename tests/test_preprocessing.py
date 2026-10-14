"""
FraudShield - Preprocessing & Validation Unit Tests
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd
from src.data_validation import (
    validate_raw_dataset,
    validate_inference_input,
    get_expected_features,
    FEATURE_NAMES,
)
from src.preprocessing import build_preprocessor


def test_expected_features_count():
    features = get_expected_features()
    assert len(features) == 30
    assert features[0] == "Time"
    assert features[-1] == "Amount"


def test_validate_inference_input_valid():
    dummy_dict = {col: 0.5 for col in FEATURE_NAMES}
    dummy_dict["Time"] = 100.0
    dummy_dict["Amount"] = 45.0
    validated_df = validate_inference_input(dummy_dict)
    assert isinstance(validated_df, pd.DataFrame)
    assert list(validated_df.columns) == FEATURE_NAMES
    assert len(validated_df) == 1


def test_validate_inference_input_missing_col_raises():
    incomplete_dict = {"Time": 100.0, "Amount": 50.0}
    with pytest.raises(ValueError, match="missing required features"):
        validate_inference_input(incomplete_dict)


def test_preprocessor_transformation_shape():
    preprocessor = build_preprocessor()
    dummy_df = pd.DataFrame(np.random.randn(10, 30), columns=FEATURE_NAMES)
    dummy_df["Time"] = np.linspace(10, 1000, 10)
    dummy_df["Amount"] = np.linspace(5, 500, 10)

    transformed = preprocessor.fit_transform(dummy_df)
    assert transformed.shape == (10, 30)
    assert not np.isnan(transformed).any()
