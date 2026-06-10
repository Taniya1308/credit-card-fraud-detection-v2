# Credit Card Fraud Detection

## Overview

Machine learning project for detecting fraudulent credit card transactions using classification algorithms.

The model was trained on an imbalanced dataset and improved using SMOTE oversampling.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SMOTE
- Streamlit
- Joblib

---

## Features

- Handles imbalanced transaction data
- Fraud prediction using Random Forest Classifier
- Model evaluation using:
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Streamlit web application interface

---

## Dataset

Kaggle Credit Card Fraud Detection Dataset

---

## Project Structure

```text
credit-card-fraud-detection/
│
├── app/
├── data/
├── models/
├── notebook/
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app/app.py
```

---

## Model Evaluation

The project focuses on handling class imbalance effectively using SMOTE and evaluating fraud detection performance using recall and F1-score instead of relying only on accuracy.

---

## Future Improvements

- XGBoost integration
- SHAP explainability
- Better Streamlit UI
- Cloud deployment
- Real-time API integration
