"""
FraudShield - Dataset Manager and Synthetic Benchmark Generator
If real 'data/raw/creditcard.csv' exists, it inspects and prepares it.
If not found, it generates a statistically representative synthetic dataset
mirroring the 31 Kaggle credit card fraud features for offline development and testing.
"""

import os
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "raw", "creditcard.csv")
SAMPLE_TEST_PATH = os.path.join(os.path.dirname(__file__), "processed", "sample_transactions.csv")


def generate_benchmark_dataset(output_path: str = RAW_DATA_PATH, n_samples: int = 25000, fraud_ratio: float = 0.002, random_state: int = 42):
    """Generates a synthetic dataset following the Kaggle Credit Card schema (Time, V1-V28, Amount, Class)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.RandomState(random_state)
    
    n_fraud = max(10, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    # Legit transactions: Time uniform, Amount log-normal, V1-V28 standard normal
    time_legit = rng.uniform(0, 172800, n_legit)  # 2 days in seconds
    amount_legit = np.clip(rng.exponential(scale=88.0, size=n_legit), 0.5, 5000.0)
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))

    # Fraud transactions: shifted distributions in typical fraud-sensitive PCA features (e.g., V4, V11, V12, V14, V17)
    time_fraud = rng.uniform(0, 172800, n_fraud)
    amount_fraud = np.clip(rng.exponential(scale=120.0, size=n_fraud), 1.0, 3000.0)
    v_fraud = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))
    # Add fraud shifts to key PCA features
    v_fraud[:, 3] += rng.normal(3.5, 1.0, n_fraud)    # V4 tends to be higher in fraud
    v_fraud[:, 10] += rng.normal(3.0, 0.8, n_fraud)   # V11 tends to be higher in fraud
    v_fraud[:, 11] -= rng.normal(4.0, 1.0, n_fraud)   # V12 tends to be lower in fraud
    v_fraud[:, 13] -= rng.normal(5.0, 1.2, n_fraud)   # V14 tends to be lower in fraud
    v_fraud[:, 16] -= rng.normal(3.5, 1.0, n_fraud)   # V17 tends to be lower in fraud

    # Assemble legit & fraud
    time_all = np.concatenate([time_legit, time_fraud])
    amount_all = np.concatenate([amount_legit, amount_fraud])
    v_all = np.vstack([v_legit, v_fraud])
    class_all = np.array([0] * n_legit + [1] * n_fraud)

    # Shuffle
    shuffle_idx = rng.permutation(n_samples)
    time_all = time_all[shuffle_idx]
    amount_all = amount_all[shuffle_idx]
    v_all = v_all[shuffle_idx]
    class_all = class_all[shuffle_idx]

    feature_cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(v_all, columns=feature_cols)
    df.insert(0, "Time", np.round(time_all, 1))
    df["Amount"] = np.round(amount_all, 2)
    df["Class"] = class_all

    df.to_csv(output_path, index=False)
    print(f"[+] Benchmark dataset generated successfully at: {output_path}")
    print(f"    Total transactions: {len(df):,}")
    print(f"    Legitimate: {n_legit:,} ({(n_legit/n_samples)*100:.2f}%)")
    print(f"    Fraudulent: {n_fraud:,} ({(n_fraud/n_samples)*100:.2f}%)")
    return df


def ensure_dataset_available() -> str:
    """Verifies if dataset is present. If missing, prompts/generates the benchmark sample."""
    if os.path.exists(RAW_DATA_PATH) and os.path.getsize(RAW_DATA_PATH) > 1024:
        print(f"[i] Found existing dataset at: {RAW_DATA_PATH}")
        return RAW_DATA_PATH
    
    print("[!] No 'creditcard.csv' found in data/raw/. Generating benchmark sample dataset...")
    generate_benchmark_dataset(RAW_DATA_PATH)
    return RAW_DATA_PATH


def export_sample_test_records(csv_path: str = RAW_DATA_PATH, output_sample_path: str = SAMPLE_TEST_PATH, n_samples: int = 10):
    """Exports a lightweight batch of mixed legit and fraud samples for UI testing and demos."""
    if not os.path.exists(csv_path):
        ensure_dataset_available()
    df = pd.read_csv(csv_path)
    legit_samples = df[df["Class"] == 0].sample(n=min(n_samples, len(df[df["Class"] == 0])), random_state=42)
    fraud_samples = df[df["Class"] == 1].sample(n=min(n_samples, len(df[df["Class"] == 1])), random_state=42)
    samples = pd.concat([legit_samples, fraud_samples]).sample(frac=1.0, random_state=42)
    os.makedirs(os.path.dirname(output_sample_path), exist_ok=True)
    samples.to_csv(output_sample_path, index=False)
    print(f"[+] Demo sample records exported to: {output_sample_path}")


if __name__ == "__main__":
    ensure_dataset_available()
    export_sample_test_records()
