# 💳 Credit Card Fraud Detection

A machine learning web application that detects fraudulent credit card transactions in real time using a Random Forest classifier trained with SMOTE oversampling.

**Live demo:** [credit-card-fraud-detection-v2.streamlit.app](https://credit-card-fraud-detection-v2-glmy8ax4xn228wcmjhccsy.streamlit.app/)

---

## Features

- Real-time single transaction analysis with fraud probability score
- Batch CSV upload and analysis with downloadable results
- Adjustable fraud threshold slider
- Sample fraud / normal transactions to test instantly
- Dark-themed responsive UI

---

## Model Performance

Trained on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate).

| Metric | Value |
|---|---|
| Overall Accuracy | **99.95%** |
| ROC-AUC | **~0.97** |
| Fraud Precision | **89%** |
| Fraud Recall | **83%** |
| Fraud F1-Score | **86%** |

**Confusion Matrix (test set — 56,962 samples):**

|  | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 56,854 (TN) | 10 (FP) |
| **Actual Fraud** | 17 (FN) | 81 (TP) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | Scikit-learn |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Web App | Streamlit |
| Model Serialization | Joblib |
| Data Processing | Pandas, NumPy |

---

## Project Structure

```
credit-card-fraud-detection/
│
├── app/
│   └── app.py              # Streamlit web application
│
├── Data/
│   └── creditcard.csv      # Raw dataset (not committed — see below)
│
├── models/
│   ├── fraud_model.pkl     # Trained Random Forest model
│   └── scaler.pkl          # Fitted StandardScaler
│
├── Notebook/
│   └── fraud_detection.ipynb  # Full training pipeline
│
├── .streamlit/
│   └── config.toml         # Streamlit theme and server config
│
├── requirements.txt
└── README.md
```

---

## ML Pipeline

```
Raw Data (284,807 rows)
    ↓
Stratified Train/Test Split (80% / 20%)
    ↓
StandardScaler  (fit on train set only)
    ↓
SMOTE oversampling (train only: 394 → 227,451 fraud samples)
    ↓
RandomForestClassifier(n_estimators=200, class_weight='balanced')
    ↓
Evaluate on original unbalanced test set
    ↓
Export model + scaler via joblib
```

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset (optional — only needed to retrain)

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `Data/` folder.  
The pre-trained `fraud_model.pkl` and `scaler.pkl` are already included in `models/`, so you can run the app without retraining.

### 5. Run the app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub (make sure `models/fraud_model.pkl` and `models/scaler.pkl` are committed — they're needed at runtime).
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Select your repo, set the branch to `main`, and set the main file path to `app/app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` automatically.

> **Note:** The `Data/creditcard.csv` file (~150 MB) should be added to `.gitignore` — it's only needed for retraining, not for running the app.

---

## Dataset

- **Source:** [ULB Machine Learning Group — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions over 2 days in September 2013
- **Fraud rate:** 492 frauds out of 284,807 transactions (0.172%)
- **Features:** `Time`, `V1–V28` (PCA-anonymized), `Amount`, `Class`
