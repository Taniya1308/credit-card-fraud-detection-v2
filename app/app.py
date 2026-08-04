import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL & SCALER
# =========================
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "fraud_model.pkl")
    scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# Feature names matching the training dataset (Time, V1-V28, Amount)
FEATURE_NAMES = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)

# =========================
# REAL FRAUD SAMPLE (from dataset)
# =========================
SAMPLE_FRAUD = [
    406.0, -2.3122265423263, 1.95199201064158, -1.60985073229769,
    3.9979055875468, -0.522187864667764, -1.42654531920595, -2.53738730624579,
    1.39165724829804, -2.77008927719433, -2.77227214465915, 3.20203320709635,
    -2.89990738849473, -0.595221881324605, -4.28925378244217, 0.389724120274487,
    -1.14074717980657, -2.83005567450437, -0.0168224681808257, 0.416955705037907,
    0.126910559061474, 0.517232370861764, -0.0350493686052974, -0.465211076182388,
    0.320198198514526, 0.0445191674731724, 0.177839798284401, 0.261145002567677,
    -0.143275874698919, 0.0
]

SAMPLE_NORMAL = [
    149.0, -1.359807134, -0.072781173, 2.536346738, 1.378155224,
    -0.338320769, 0.462387778, 0.239598554, 0.098697901, 0.363786970,
    0.090794172, -0.551599533, -0.617800856, -0.991389847, -0.311169354,
    1.468176972, -0.470400525, 0.207971242, 0.025790986, 0.403992960,
    0.251412098, -0.018306778, 0.277837576, -0.110473910, 0.066928075,
    0.128539358, -0.189114844, 0.133558377, -0.021053053, 149.62
]

# =========================
# STYLING
# =========================
st.markdown("""
<style>
.main { background-color: #0B1220; color: white; }
h1, h2, h3 { color: #E5E7EB; }
.block-container { padding-top: 2rem; max-width: 1200px; }

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 10px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    color: white;
    font-weight: 600;
    font-size: 15px;
    border: none;
    transition: transform 0.15s ease;
}
.stButton > button:hover { transform: scale(1.01); }

.result-card {
    padding: 22px 26px;
    border-radius: 14px;
    background: #111827;
    border: 1px solid #1F2937;
    margin-bottom: 12px;
}

div[data-testid="metric-container"] {
    background: #0F172A;
    border: 1px solid #1F2937;
    padding: 14px;
    border-radius: 12px;
}

.feature-hint {
    font-size: 12px;
    color: #6B7280;
    margin-top: 4px;
    line-height: 1.6;
}

.badge-fraud {
    display: inline-block;
    background: #7F1D1D;
    color: #FCA5A5;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
}
.badge-normal {
    display: inline-block;
    background: #14532D;
    color: #86EFAC;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center; margin-bottom:4px;'>💳 Fraud Detection AI</h1>
<p style='text-align:center; color:#9CA3AF; font-size:15px; margin-bottom:0;'>
    Real-Time Credit Card Fraud Detection &nbsp;•&nbsp; Random Forest + SMOTE
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Probability above this value is classified as FRAUD"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
| Property | Value |
|---|---|
| Algorithm | Random Forest |
| Trees | 200 |
| Balancing | SMOTE |
| Features | 30 |
| Dataset | 284,807 txns |
    """)

    st.markdown("---")
    st.markdown("### 🎯 Test Performance")
    st.markdown("""
| Metric | Score |
|---|---|
| Accuracy | 99.95% |
| ROC-AUC | ~0.97 |
| Fraud Recall | 83% |
| Fraud Precision | 89% |
    """)

    st.markdown("---")
    st.info("Threshold controls the decision boundary. Lower = more sensitive to fraud.", icon="ℹ️")

# =========================
# METRICS DASHBOARD
# =========================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "99.95%", "on 56,962 test samples")
col2.metric("ROC-AUC", "~0.97", "strong fraud discrimination")
col3.metric("Fraud Recall", "83%", "of 98 fraud cases caught")
col4.metric("Fraud Precision", "89%", "of fraud flags are real")

st.markdown("---")

# =========================
# INPUT MODE
# =========================
st.subheader("🔍 Transaction Analysis")

mode = st.radio(
    "Select Input Mode",
    ["Manual Input", "Batch CSV"],
    horizontal=True,
    help="Manual: enter one transaction. Batch: upload multiple as CSV."
)

# =========================
# MANUAL INPUT
# =========================
if mode == "Manual Input":

    # Quick-load sample buttons
    st.markdown("**Load a sample transaction:**")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    load_fraud = btn_col1.button("🚨 Load Fraud Sample")
    load_normal = btn_col2.button("✅ Load Normal Sample")

    # Initialize session state for the text area
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""

    if load_fraud:
        st.session_state["input_text"] = ", ".join(map(str, SAMPLE_FRAUD))
    if load_normal:
        st.session_state["input_text"] = ", ".join(map(str, SAMPLE_NORMAL))

    input_text = st.text_area(
        "Enter 30 comma-separated feature values",
        value=st.session_state["input_text"],
        height=110,
        placeholder="Time, V1, V2, ..., V28, Amount",
        help="Paste values in the order: Time, V1–V28, Amount"
    )

    # Feature order hint
    st.markdown(
        f"<div class='feature-hint'>Expected order: "
        f"{', '.join(FEATURE_NAMES)}</div>",
        unsafe_allow_html=True
    )

    if st.button("🔍 Analyze Transaction"):
        if not input_text.strip():
            st.error("Please enter feature values or load a sample above.")
            st.stop()

        try:
            values = [float(x.strip()) for x in input_text.split(",") if x.strip()]
        except ValueError:
            st.error("Invalid input — all values must be numbers separated by commas.")
            st.stop()

        expected = model.n_features_in_
        if len(values) != expected:
            st.error(f"Expected **{expected}** features, got **{len(values)}**. "
                     f"Check the order: {', '.join(FEATURE_NAMES)}")
            st.stop()

        # Scale and predict
        X = pd.DataFrame([values], columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X)

        with st.spinner("Analyzing transaction..."):
            time.sleep(0.4)
            prob = model.predict_proba(X_scaled)[0][1]
            is_fraud = prob >= threshold

        # ---- Results ----
        st.markdown("---")
        st.subheader("📊 Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{prob:.2%}")
        r2.metric("Risk Score", f"{prob * 100:.1f} / 100")
        r3.metric("Decision Threshold", f"{threshold:.0%}")

        st.progress(min(int(prob * 100), 100))

        if prob >= 0.80:
            st.error("🚨 HIGH RISK — Likely Fraudulent Transaction")
        elif prob >= threshold:
            st.warning("⚠️  MEDIUM RISK — Flagged as Potentially Fraudulent")
        else:
            st.success("✅ LOW RISK — Transaction Appears Legitimate")

        verdict = "FRAUD" if is_fraud else "LEGITIMATE"
        badge_class = "badge-fraud" if is_fraud else "badge-normal"
        st.markdown(
            f"<br>Verdict at threshold {threshold:.0%}: "
            f"<span class='{badge_class}'>{verdict}</span>",
            unsafe_allow_html=True
        )

# =========================
# BATCH CSV INPUT
# =========================
else:
    st.markdown("""
    **Expected CSV format:**  
    Columns must be exactly: `Time, V1, V2, ..., V28, Amount`  
    Each row is one transaction. No `Class` column needed.
    """)

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        # Validate columns
        missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in FEATURE_NAMES and c != "Class"]

        if missing_cols:
            st.error(
                f"Missing columns: **{', '.join(missing_cols)}**  \n"
                f"Expected: {', '.join(FEATURE_NAMES)}"
            )
            st.stop()

        # Drop Class column if present, keep only feature columns
        df_features = df[FEATURE_NAMES].copy()

        if extra_cols:
            st.warning(f"Ignoring unrecognized columns: {', '.join(extra_cols)}")

        st.success(f"Loaded **{len(df_features):,}** transactions")
        st.dataframe(df_features.head(5), use_container_width=True)

        if st.button("🔍 Run Batch Fraud Detection"):
            with st.spinner(f"Analyzing {len(df_features):,} transactions..."):
                X_scaled = scaler.transform(df_features)
                probs = model.predict_proba(X_scaled)[:, 1]

            results = df_features.copy()
            results["fraud_probability"] = probs.round(4)
            results["risk_score"] = (probs * 100).round(1)
            results["prediction"] = (probs >= threshold).astype(int)
            results["verdict"] = results["prediction"].map({1: "FRAUD", 0: "LEGITIMATE"})

            n_fraud = results["prediction"].sum()
            n_total = len(results)
            fraud_pct = n_fraud / n_total * 100

            st.markdown("---")
            st.subheader("📊 Batch Results")

            b1, b2, b3 = st.columns(3)
            b1.metric("Total Transactions", f"{n_total:,}")
            b2.metric("Flagged as Fraud", f"{n_fraud:,}", f"{fraud_pct:.2f}%")
            b3.metric("Threshold Used", f"{threshold:.0%}")

            # Show flagged ones first
            st.markdown("#### Flagged Transactions (Fraud First)")
            st.dataframe(
                results.sort_values("fraud_probability", ascending=False),
                use_container_width=True
            )

            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv_out,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

# =========================
# MODEL DETAILS EXPANDER
# =========================
st.markdown("---")
with st.expander("📘 Model & Dataset Details"):
    st.markdown("""
### Algorithm
- **Random Forest Classifier** — 200 trees, `class_weight='balanced'`, `n_jobs=-1`
- Preprocessing: **StandardScaler** (all 30 features)
- Imbalance handling: **SMOTE** oversampling on training set

### Dataset
- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **284,807** transactions, **492** fraud cases (0.17%)
- Features: `Time`, `V1–V28` (PCA-transformed), `Amount`

### Training Pipeline
```
Raw Data → Stratified Train/Test Split (80/20)
        → StandardScaler (fit on train)
        → SMOTE (train only: 394 → 227,451 fraud)
        → RandomForestClassifier
        → Evaluate on original test set
```

### Test Set Performance
| Metric | Class 0 (Normal) | Class 1 (Fraud) |
|---|---|---|
| Precision | 1.00 | 0.89 |
| Recall | 1.00 | 0.83 |
| F1-Score | 1.00 | 0.86 |
| Support | 56,864 | 98 |

**Overall Accuracy:** 99.95%  
**Confusion Matrix:** TN=56,854 · FP=10 · FN=17 · TP=81
    """)

# =========================
# FOOTER
# =========================
st.markdown("""
<div style='text-align:center; color:#6B7280; margin-top:30px; font-size:13px;'>
    Built with Python · Streamlit · Scikit-learn · imbalanced-learn
</div>
""", unsafe_allow_html=True)
