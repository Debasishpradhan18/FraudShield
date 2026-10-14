"""
FraudShield - Data Schema Validation Module
Validates schemas, data types, missing values, and column consistency
for both training pipelines and real-time inference payloads.
"""

from typing import List, Dict, Any, Tuple, Union
import numpy as np
import pandas as pd

# The 30 features expected by FraudShield models
FEATURE_NAMES: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_NAME: str = "Class"


def get_expected_features() -> List[str]:
    """Returns the ordered list of 30 feature names."""
    return list(FEATURE_NAMES)


def validate_raw_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates a training dataset:
    - Checks required feature columns and target 'Class'.
    - Checks for null or infinite values.
    - Checks data types.
    """
    errors: List[str] = []

    if not isinstance(df, pd.DataFrame):
        return False, ["Input is not a pandas DataFrame."]

    # Check required columns
    missing_features = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {missing_features}")

    if TARGET_NAME not in df.columns:
        errors.append(f"Missing target column: '{TARGET_NAME}'.")

    # Check null values
    null_counts = df[FEATURE_NAMES].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        errors.append(f"Columns contain null values: {cols_with_nulls}")

    # Check infinite values
    inf_counts = np.isinf(df[FEATURE_NAMES].select_dtypes(include=[np.number])).sum()
    cols_with_infs = inf_counts[inf_counts > 0].to_dict()
    if cols_with_infs:
        errors.append(f"Columns contain infinite values: {cols_with_infs}")

    # Check target values
    if TARGET_NAME in df.columns:
        unique_targets = set(df[TARGET_NAME].dropna().unique())
        if not unique_targets.issubset({0, 1}):
            errors.append(f"Target column '{TARGET_NAME}' must contain binary values (0 or 1). Found: {unique_targets}")

    return len(errors) == 0, errors


def validate_inference_input(
    input_data: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Validates and standardizes inference input for single or batch predictions.
    Ensures correct column order, numeric types, and absence of NaNs.
    Returns: Clean pandas DataFrame matching exact FEATURE_NAMES schema.
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}. Expected dict, list of dicts, or DataFrame.")

    # Drop target column if passed in inference payload
    if TARGET_NAME in df.columns:
        df = df.drop(columns=[TARGET_NAME])

    missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Inference input is missing required features: {missing_cols}")

    # Ensure exact column ordering
    df = df[FEATURE_NAMES]

    # Convert to numeric
    for col in FEATURE_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check for NaN / Inf
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"Inference data contains invalid/non-numeric/NaN values in columns: {nan_cols}")

    if np.isinf(df.values).any():
        raise ValueError("Inference data contains infinite values.")

    return df
