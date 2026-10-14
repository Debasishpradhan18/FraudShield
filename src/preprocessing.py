"""
FraudShield - Preprocessing & Feature Engineering Module
Builds leakage-free Scikit-learn ColumnTransformers and handles
stratified splitting to preserve the rare fraud class distribution.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from src.data_validation import FEATURE_NAMES, TARGET_NAME, validate_raw_dataset

SCALE_FEATURES: List[str] = ["Time", "Amount"]
PASSTHROUGH_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that:
    - Scales 'Time' and 'Amount' using RobustScaler (handles heavy-tailed transaction outliers).
    - Passes through V1-V28 (already PCA-transformed and zero-centered).
    Guarantees no data leakage when embedded into a Scikit-learn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", RobustScaler(), SCALE_FEATURES),
            ("pca_features", "passthrough", PASSTHROUGH_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def load_and_split_data(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw CSV data, validates integrity, and performs a Stratified train/test split.
    Stratification ensures both sets have the exact same rare fraud proportion.
    """
    df = pd.read_csv(csv_path)

    is_valid, errors = validate_raw_dataset(df)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {errors}")

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    print(f"[i] Dataset loaded: {len(df):,} total rows.")
    print(f"    Train set: {len(X_train):,} samples (Fraud count: {y_train.sum():,})")
    print(f"    Test set:  {len(X_test):,} samples (Fraud count: {y_test.sum():,})")

    return X_train, X_test, y_train, y_test
