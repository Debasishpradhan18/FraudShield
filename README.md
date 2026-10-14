# FraudShield: Intelligent Fraud Detection and Risk Analysis System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**FraudShield** is a machine learning system designed to identify fraudulent financial transactions in highly imbalanced payment environments. It establishes an interpretable **Logistic Regression baseline** and compares it against a **Random Forest ensemble**, evaluating both with precision-recall metrics and exposing real-time predictions via an interactive **Streamlit dashboard**.

---

## 📌 Architecture & ML Lifecycle

```text
Raw Transaction Data (Time, V1–V28, Amount)
          │
          ▼
   Stratified Train / Test Split (80/20)
          │
          ▼
   Leakage-Free Preprocessing Pipeline
   ├─ RobustScaler: 'Amount' and 'Time' (handles transaction outliers)
   └─ Passthrough:  'V1' to 'V28' (PCA components)
          │
    ┌─────┴─────────────────────────────────┐
    ▼                                       ▼
Baseline Model:                         Champion Model:
Logistic Regression                     Random Forest Classifier
(class_weight='balanced')               (class_weight='balanced', n_estimators=100)
    │                                       │
    ▼                                       ▼
Precision, Recall, PR-AUC               Precision, Recall, PR-AUC
    └───────────────────┬───────────────────┘
                        ▼
             Joblib Model Serialization
                        │
                        ▼
            Interactive Streamlit Dashboard
            ├─ Single & Batch Transaction Scoring
            ├─ Live Threshold Tuning Slider (0.10 - 0.90)
            ├─ Dynamic Risk Badges (Low, Elevated, High Risk)
            └─ Model Comparison & Confusion Matrix Heatmaps
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```powershell
# Navigate to project folder
cd d:\FraudShield

# Create and activate virtual environment
python -m venv .venv

# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.venv\Scripts\activate.bat

# Install all dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Generate / Prepare Data
Place the Kaggle `creditcard.csv` in `data/raw/creditcard.csv`, or run the benchmark data manager to automatically generate a representative sample:
```powershell
python data/sample_data_generator.py
```

### 3. Train the Models
Train both Logistic Regression and Random Forest models, evaluate on held-out test data, and persist the pipelines:
```powershell
python src/train.py
```

### 4. Run Evaluation Reports & Visualizations
Generate confusion matrix heatmaps, PR & ROC comparison curves, and threshold analysis:
```powershell
python src/evaluate.py
```

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run app/streamlit_app.py
```

---

## 📂 Project Structure

```text
FraudShield/
├── data/
│   ├── raw/                 # Raw dataset (creditcard.csv)
│   ├── processed/           # Processed samples
│   └── sample_data_generator.py # Data manager & benchmark generator
├── notebooks/
│   └── 01_exploratory_analysis.ipynb # EDA & data profiling
├── src/
│   ├── data_validation.py   # Schema integrity & input validation
│   ├── preprocessing.py     # Leakage-free ColumnTransformer & stratified splits
│   ├── train.py             # Model training (Baseline + Champion) & joblib export
│   ├── evaluate.py          # PR-AUC, ROC curves, confusion matrices
│   └── predict.py           # Real-time inference engine & risk categorizer
├── models/
│   ├── baseline_logistic_regression.joblib
│   ├── champion_random_forest.joblib
│   ├── fraud_pipeline.joblib
│   └── training_metadata.json
├── app/
│   └── streamlit_app.py     # Interactive Streamlit dashboard
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── reports/
│   ├── figures/             # Evaluation plots (curves, confusion matrix)
│   └── model_evaluation.md  # Detailed evaluation report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🎯 Model Comparison Summary

| Metric | Logistic Regression (Baseline) | Random Forest (Champion) | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Model Type** | Linear Model (Log-odds) | Non-linear Bagged Ensemble | Tests if complex patterns exist |
| **Class Weighting** | `balanced` | `balanced` | Counteracts the 0.17% fraud rarity |
| **Primary Metric** | **PR-AUC** | **PR-AUC** | Evaluates ranking on rare positive fraud class |
| **Detection Focus** | Fast probability estimate | Captures subtle multi-feature combinations | Maximizes Recall while controlling False Positives |

---

## 🛡️ Responsible Use & Limitations
- **Decision Support**: FraudShield risk scores are statistical estimates to assist human compliance analysts, not automated accusations of fraud.
- **Data Privacy**: Anonymized PCA features preserve customer privacy.
- **Threshold Sensitivity**: The operational threshold must be calibrated according to business cost trade-offs (loss from missed fraud vs. cost of false alerts).
