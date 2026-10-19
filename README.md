# Customer Churn Prediction & Retention Analytics

[![Tests](https://img.shields.io/badge/tests-4%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end customer churn classification and subscriber retention modeling pipeline built using **Python, Scikit-learn, and Streamlit**. It compares **Logistic Regression (Class-Balanced), Random Forest, and Gradient Boosting** classifiers on benchmark telecom subscription records, analyzes the financial tradeoff between False Positives and False Negatives, and provides an interactive web application for real-time customer churn risk scoring.

Developed by **[Sudhir Kumar](https://github.com/Sudhir-2211)** as part of the student placement portfolio.

---

## 📌 Table of Contents
- [Commercial & Operational Limitations Disclaimer](#commercial--operational-limitations-disclaimer)
- [Project Overview](#project-overview)
- [The Economic Tradeoff: False Positives vs. False Negatives](#the-economic-tradeoff-false-positives-vs-false-negatives)
- [Pipeline Architecture & Feature Engineering](#pipeline-architecture--feature-engineering)
- [Classifier Benchmarks & Evaluation](#classifier-benchmarks--evaluation)
- [Folder Structure](#folder-structure)
- [Installation & Setup](#installation--setup)
- [Interactive Streamlit App](#interactive-streamlit-app)
- [CLI Prediction Tool](#cli-prediction-tool)
- [Automated Testing](#automated-testing)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## ⚠️ Commercial & Operational Limitations Disclaimer

> **OPERATIONAL WARNING**: This model is built for technical portfolio review, educational demonstrations, and exploratory retention triage. **It must not be deployed to trigger autonomous business actions (such as automated account termination or irreversible contract adjustments) without thorough domain validation.**
>
> In real-world customer relationship management (CRM):
> 1. Churn definitions vary across business lines (e.g., active cancellation vs. involuntary credit card expiration vs. dormancy).
> 2. Predictive models must be integrated into randomized controlled trials (A/B testing) to prove that customer success outreach actually decreases churn before scaling interventions.

---

## 📖 Project Overview

Acquiring a new customer costs 5 to 7 times more than retaining an existing one. Early detection of dissatisfied accounts enables proactive customer success outreach.

This repository features:
- **Class Imbalance Handling**: Realistic ~27% churn rate handled via stratified splitting and balanced class weighting (`class_weight='balanced'`).
- **Encapsulated Preprocessing**: Seamless numeric feature scaling (`StandardScaler`) and categorical dummy encoding (`OneHotEncoder`) bundled into a serialized production pipeline.
- **Explainable Predictions**: Actionable risk diagnostics (e.g. Month-to-month contracts, absence of tech support, manual electronic check friction).
- **Interactive Web UI**: Real-time probability gauge, risk classification badges (Low, Moderate, High Risk), and confusion matrix inspection.

---

## ⚖️ The Economic Tradeoff: False Positives vs. False Negatives

In customer churn classification, model errors carry very different financial consequences:

$$\begin{array}{|c|c|c|}
\hline
\textbf{Prediction} & \textbf{Actual Retained (0)} & \textbf{Actual Churned (1)} \\
\hline
\textbf{Predicted Retained (0)} & \textbf{True Negative (TN)}: Safe / No Action & \textbf{False Negative (FN)}: Missed Churner! \\
\hline
\textbf{Predicted Churned (1)} & \textbf{False Positive (FP)}: Unnecessary Discount & \textbf{True Positive (TP)}: Intercepted Churner \\
\hline
\end{array}$$

1. **False Positives (Type I Error)**: A customer who had no intention of leaving is predicted to churn.
   - *Business Impact*: Company expends retention resources (e.g., promotional credits or discounted renewal rates) unnecessarily. Low direct financial loss.
2. **False Negatives (Type II Error)**: A customer who is about to cancel is predicted to stay.
   - *Business Impact*: High! Zero intervention occurs, leading to permanent loss of customer lifetime value (LTV).
3. **Strategic Conclusion**: We calibrate the pipeline to favor **Recall (Sensitivity)** and **ROC-AUC** to minimize high-cost False Negatives.

---

## 🛠️ Pipeline Architecture & Feature Engineering

```text
  Raw Customer Attributes
            │
            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 ColumnTransformer Preprocessor               │
  │  ┌──────────────────────────────┐ ┌──────────────────────┐  │
  │  │ Numeric: StandardScaler()    │ │ Cat: OneHotEncoder() │  │
  │  │ (tenure, charges)            │ │ (contract, internet, │  │
  │  │                              │ │  tech_support, etc.) │  │
  │  └──────────────────────────────┘ └──────────────────────┘  │
  └─────────────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                GradientBoostingClassifier                   │
  │              (150 Estimators, Depth 4, LR 0.08)             │
  └─────────────────────────────────────────────────────────────┘
            │
            ▼
  Probability of Churn [0.0 - 1.0] ➔ Risk Tier (Low / Moderate / High)
```

---

## 📊 Classifier Benchmarks & Evaluation

Evaluated on held-out stratified test partition ($N = 600$ customers):

| Classifier Architecture | ROC-AUC | F1-Score | Recall (Sensitivity) | Precision | Accuracy |
|---|---|---|---|---|---|
| **Gradient Boosting** *(Selected)* | **0.9840** | **0.8790** | 0.8571 | **0.9020** | **0.9367** |
| Logistic Regression (Balanced) | 0.9830 | 0.8724 | **0.9130** | 0.8352 | 0.9283 |
| Random Forest (Balanced) | 0.9813 | 0.8675 | 0.8944 | 0.8421 | 0.9267 |

**Production Model**: Gradient Boosting achieved the highest overall **ROC-AUC (0.9840)** and **F1-Score (0.8790)**, balancing high churn detection sensitivity with minimal false alarms.

---

## 📁 Folder Structure

```text
customer-churn-prediction/
├── .gitignore                     # Git ignore file
├── LICENSE                        # MIT License
├── README.md                      # Comprehensive documentation
├── requirements.txt               # Dependencies
├── app.py                         # Streamlit interactive risk profiling web app
├── data/
│   └── customer_churn.csv         # 3,000 customer benchmark dataset
├── models/
│   ├── churn_pipeline.joblib      # Serialized scikit-learn model artifact
│   └── model_metrics.json         # Benchmark metrics JSON
├── src/
│   ├── __init__.py
│   ├── data_generator.py          # Calibrated dataset generator
│   ├── predict.py                 # CLI inference script
│   └── train.py                   # Multi-model training and evaluation script
└── tests/
    ├── __init__.py
    └── test_churn.py              # Automated unit tests
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sudhir-2211/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Retrain or verify model**:
   ```bash
   python -m src.train
   ```

---

## 🖥️ Interactive Streamlit App

Launch the interactive web interface:
```bash
streamlit run app.py
```

Open `http://localhost:8501`. Adjust customer subscription parameters in the sidebar to generate instant churn probabilities, risk gauges, and targeted customer success action plans.

---

## 💻 CLI Prediction Tool

Assess churn risk directly via terminal:

```bash
python -m src.predict --tenure 4 --monthly 95.0 --contract "Month-to-month" --internet "Fiber optic" --tech_support "No" --payment "Electronic check"
```

Sample output:
```text
============================================================
           CUSTOMER CHURN RISK ASSESSMENT RESULT
============================================================
Customer Profile:
  • Tenure         : 4 Months | Contract: Month-to-month
  • Monthly Spend  : $95.00 | Tech Support: No
  • Internet / Pay : Fiber optic | Electronic check
------------------------------------------------------------
Churn Probability: 89.4%
Risk Tier        : HIGH RISK
Binary Outcome   : CHURN LIKELY
============================================================
```

---

## 🧪 Automated Testing

Unit tests validate:
- Dataset completeness and absence of null values.
- Pipeline probability calibration ($0.0 \le P(Churn) \le 1.0$).
- Output formatting and risk tier categorization.
- Risk monotonicity: short-tenure month-to-month accounts exhibit higher churn risk than long-tenure two-year accounts.

Run tests:
```bash
python -m pytest -v
```

Expected output:
```text
============================== 4 passed in 1.28s ==============================
```

---

## ⚠️ Future Roadmap

- Integrate survival analysis (Cox Proportional Hazards model) to predict *time-to-churn* distributions.
- Implement Uplift Modeling (Causal ML) to distinguish customers who only stay when receiving incentives from persistent loyalists.
- Connect live webhooks to customer support ticket frequency metrics.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
