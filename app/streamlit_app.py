"""
FraudShield - Intelligent Fraud Detection & Risk Analysis Dashboard
Built with Streamlit.
Provides real-time transaction scoring, model comparison, threshold analysis,
and data exploration for risk analysts and evaluators.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_validation import FEATURE_NAMES
from src.predict import FraudPredictor, DEFAULT_MODEL_PATH
from src.train import train_models, MODELS_DIR
from data.sample_data_generator import ensure_dataset_available, RAW_DATA_PATH

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield | Risk Analysis System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-danger {
        color: #b91c1c;
        background-color: #fee2e2;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        color: #b45309;
        background-color: #fef3c7;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-success {
        color: #15803d;
        background-color: #dcfce7;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        return None
    return FraudPredictor(DEFAULT_MODEL_PATH)


@st.cache_data
def load_metadata():
    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_sample_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        ensure_dataset_available()
    df = pd.read_csv(RAW_DATA_PATH, nrows=5000)
    return df


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🛡️ FraudShield")
st.sidebar.caption("Intelligent Fraud Detection & Risk Analysis")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "💳 Real-Time Transaction Checker",
        "⚖️ Model Comparison & Metrics",
        "📈 Data & Imbalance Explorer",
        "📖 System Guide & Viva Q&A",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Decision Policy")
threshold = st.sidebar.slider(
    "Operational Risk Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Transactions with estimated fraud score ≥ this threshold trigger high-risk alerts.",
)

# ---------------------------------------------------------------------------
# PAGE 1: REAL-TIME TRANSACTION CHECKER
# ---------------------------------------------------------------------------
if nav_choice == "💳 Real-Time Transaction Checker":
    st.title("💳 Real-Time Transaction Risk Checker")
    st.write(
        "Evaluate incoming transactions in real time using the trained **Random Forest pipeline**. "
        "Select a pre-configured scenario, customize transaction features, or upload a CSV batch."
    )

    predictor = get_predictor()

    if predictor is None:
        st.warning("⚠️ No trained model pipeline found. Train the models to start scoring.")
        if st.button("🚀 Train Models Now (Baseline + Champion)"):
            with st.spinner("Training Logistic Regression and Random Forest models..."):
                train_models()
                st.success("Models trained successfully! Reloading...")
                st.rerun()
    else:
        tab_single, tab_batch = st.tabs(["Single Transaction Evaluation", "Batch CSV Scoring"])

        with tab_single:
            st.subheader("Select a Quick Scenario or Enter Custom Values")

            scenarios = {
                "☕ Typical Small Coffee Purchase ($4.50)": {
                    "Amount": 4.50, "Time": 12450.0,
                    "V1": 0.05, "V4": -0.12, "V11": -0.20, "V12": 0.15, "V14": 0.10, "V17": -0.05,
                },
                "💻 Normal Electronics Store Purchase ($489.00)": {
                    "Amount": 489.00, "Time": 45200.0,
                    "V1": -0.45, "V4": 0.22, "V11": 0.18, "V12": 0.05, "V14": -0.12, "V17": 0.15,
                },
                "🚨 Suspicious Velocity Surge ($1,299.00 - High Risk)": {
                    "Amount": 1299.00, "Time": 86300.0,
                    "V1": -3.20, "V4": 4.80, "V11": 3.90, "V12": -5.10, "V14": -6.40, "V17": -4.20,
                },
                "🕵️ Overnight Micro-Charge Probing ($0.99 - Anomalous)": {
                    "Amount": 0.99, "Time": 150000.0,
                    "V1": -2.10, "V4": 3.80, "V11": 2.70, "V12": -4.20, "V14": -5.50, "V17": -3.10,
                },
            }

            scenario_name = st.selectbox("Preset Simulation Scenario", list(scenarios.keys()))
            selected_preset = scenarios[scenario_name]

            col1, col2, col3 = st.columns(3)
            with col1:
                tx_amount = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=float(selected_preset["Amount"]), step=1.0)
            with col2:
                tx_time = st.number_input("Elapsed Seconds (Time)", min_value=0.0, max_value=200000.0, value=float(selected_preset["Time"]), step=100.0)
            with col3:
                st.info(f"Active Threshold: **{threshold:.2f}**")

            st.caption("PCA Features (Key Fraud-Sensitive Indicators V4, V11, V12, V14, V17)")
            pca_col1, pca_col2, pca_col3, pca_col4, pca_col5 = st.columns(5)
            with pca_col1:
                v4 = st.number_input("V4", value=float(selected_preset.get("V4", 0.0)), step=0.1)
            with pca_col2:
                v11 = st.number_input("V11", value=float(selected_preset.get("V11", 0.0)), step=0.1)
            with pca_col3:
                v12 = st.number_input("V12", value=float(selected_preset.get("V12", 0.0)), step=0.1)
            with pca_col4:
                v14 = st.number_input("V14", value=float(selected_preset.get("V14", 0.0)), step=0.1)
            with pca_col5:
                v17 = st.number_input("V17", value=float(selected_preset.get("V17", 0.0)), step=0.1)

            # Build full 30-feature vector
            tx_data = {col: 0.0 for col in FEATURE_NAMES}
            tx_data["Time"] = tx_time
            tx_data["Amount"] = tx_amount
            tx_data["V4"] = v4
            tx_data["V11"] = v11
            tx_data["V12"] = v12
            tx_data["V14"] = v14
            tx_data["V17"] = v17

            if st.button("🔍 Assess Transaction Risk", type="primary"):
                result = predictor.predict_single(tx_data, threshold=threshold)

                st.divider()
                r_col1, r_col2, r_col3 = st.columns([1, 1, 2])

                with r_col1:
                    score = result["fraud_probability"]
                    st.metric("Fraud Risk Score", f"{score * 100:.2f}%")
                    st.progress(min(1.0, score))

                with r_col2:
                    tier = result["risk_tier"]
                    if tier == "HIGH RISK":
                        st.markdown(f"### <span class='badge-danger'>{tier}</span>", unsafe_allow_html=True)
                    elif tier == "SUSPICIOUS / ELEVATED":
                        st.markdown(f"### <span class='badge-warning'>{tier}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### <span class='badge-success'>{tier}</span>", unsafe_allow_html=True)
                    st.caption(f"Decision: **{result['status']}**")

                with r_col3:
                    st.markdown("**Operational Recommendation:**")
                    st.write(result["recommendation"])
                    st.caption(f"ℹ️ {result['disclaimer']}")

        with tab_batch:
            st.subheader("Batch Transaction Risk Scoring")
            uploaded_file = st.file_uploader("Upload CSV containing transaction features", type=["csv"])
            if uploaded_file is not None:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(batch_df)} records.")
                try:
                    scored_df = predictor.predict_batch(batch_df, threshold=threshold)
                    st.dataframe(scored_df.head(20))
                    csv_export = scored_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Scored Transactions", csv_export, "fraudshield_batch_scored.csv", "text/csv")
                except Exception as e:
                    st.error(f"Error scoring batch: {e}")

# ---------------------------------------------------------------------------
# PAGE 2: MODEL COMPARISON & METRICS
# ---------------------------------------------------------------------------
elif nav_choice == "⚖️ Model Comparison & Metrics":
    st.title("⚖️ Model Comparison & Evaluation Metrics")
    st.write(
        "Direct performance comparison between the **Baseline (Logistic Regression)** and "
        "the **Champion (Random Forest)** on held-out test data."
    )

    metadata = load_metadata()
    if metadata is None:
        st.warning("⚠️ Training metadata not found. Train the models from the first page.")
    else:
        base_m = metadata["baseline_model"]["metrics"]
        champ_m = metadata["champion_model"]["metrics"]

        comp_data = {
            "Metric": ["Recall (Fraud Detection Rate)", "Precision", "F1-Score", "PR-AUC (Primary Metric)", "ROC-AUC", "Training Time"],
            "Logistic Regression (Baseline)": [
                f"{base_m['recall']:.4f}",
                f"{base_m['precision']:.4f}",
                f"{base_m['f1_score']:.4f}",
                f"{base_m['pr_auc']:.4f}",
                f"{base_m['roc_auc']:.4f}",
                f"{base_m.get('training_time_seconds', '-')}s",
            ],
            "Random Forest (Champion)": [
                f"{champ_m['recall']:.4f}",
                f"{champ_m['precision']:.4f}",
                f"{champ_m['f1_score']:.4f}",
                f"{champ_m['pr_auc']:.4f}",
                f"{champ_m['roc_auc']:.4f}",
                f"{champ_m.get('training_time_seconds', '-')}s",
            ],
        }
        st.table(pd.DataFrame(comp_data))

        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.subheader("Logistic Regression (Test Set)")
            st.write(f"• True Positives (Caught Fraud): **{base_m['true_positives']}**")
            st.write(f"• False Negatives (Missed Fraud): **{base_m['false_negatives']}**")
            st.write(f"• False Positives (False Alarms): **{base_m['false_positives']}**")
        with col_cm2:
            st.subheader("Random Forest (Test Set)")
            st.write(f"• True Positives (Caught Fraud): **{champ_m['true_positives']}**")
            st.write(f"• False Negatives (Missed Fraud): **{champ_m['false_negatives']}**")
            st.write(f"• False Positives (False Alarms): **{champ_m['false_positives']}**")

        st.divider()
        st.subheader("Evaluation Visualizations")
        fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
        curves_img = os.path.join(fig_dir, "curves_comparison.png")
        thresh_img = os.path.join(fig_dir, "threshold_tradeoff.png")
        cm_img = os.path.join(fig_dir, "confusion_matrix.png")

        if os.path.exists(curves_img):
            st.image(curves_img, caption="Precision-Recall and ROC Comparison Curves", use_container_width=True)
        if os.path.exists(thresh_img):
            st.image(thresh_img, caption="Operational Decision Threshold Trade-Off Analysis", use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3: DATA & IMBALANCE EXPLORER
# ---------------------------------------------------------------------------
elif nav_choice == "📈 Data & Imbalance Explorer":
    st.title("📈 Data Distribution & Class Imbalance Explorer")
    st.write("Understand the transaction dataset structure, extreme class imbalance, and feature properties.")

    df = load_sample_dataset()
    total_tx = len(df)
    fraud_tx = int(df["Class"].sum())
    legit_tx = total_tx - fraud_tx

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample Transactions", f"{total_tx:,}")
    c2.metric("Legitimate Transactions", f"{legit_tx:,}", f"{(legit_tx/total_tx)*100:.2f}%")
    c3.metric("Fraudulent Transactions", f"{fraud_tx:,}", f"{(fraud_tx/total_tx)*100:.2f}%")

    st.divider()
    st.subheader("Transaction Amount Summary ($)")
    st.dataframe(df.groupby("Class")["Amount"].describe())

    st.subheader("First 10 Dataset Records")
    st.dataframe(df.head(10))

# ---------------------------------------------------------------------------
# PAGE 4: SYSTEM GUIDE & VIVA QUESTIONS
# ---------------------------------------------------------------------------
elif nav_choice == "📖 System Guide & Viva Q&A":
    st.title("📖 System Guide & Viva Preparation")
    st.markdown(
        """
        ### Key Project Concept Summaries
        * **Why is accuracy insufficient for fraud detection?**
          In a dataset with 99.8% non-fraudulent transactions, a naive model that always predicts 'legitimate' achieves 99.8% accuracy while failing to detect any fraud whatsoever.
        * **Why PR-AUC instead of ROC-AUC?**
          ROC curves evaluate True Positive Rate against False Positive Rate. With a huge negative class, False Positive Rate stays small even with many false alarms. The Precision-Recall curve focuses directly on the minority positive class.
        * **Why two models?**
          1. **Logistic Regression**: Linear baseline benchmark to prove necessity of advanced algorithms.
          2. **Random Forest Classifier**: Ensemble of bagged decision trees that captures non-linear feature interactions without prone to outlier skew.
        """
    )
