"""Customer Churn Classification Pipeline Training and Benchmarking.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "tech_support",
    "online_security",
    "paperless_billing",
    "payment_method",
]


def build_preprocessor() -> ColumnTransformer:
    """Builds scikit-learn ColumnTransformer for churn features."""
    num_transformer = Pipeline([("scaler", StandardScaler())])
    cat_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([
        ("num", num_transformer, NUMERIC_FEATURES),
        ("cat", cat_transformer, CATEGORICAL_FEATURES),
    ])


def train_and_evaluate(data_path: str, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
    }

    results = {}
    best_model_name = None
    best_pipeline = None
    best_roc_auc = -float("inf")

    print("==================================================================")
    print("      TRAINING & EVALUATING CUSTOMER CHURN CLASSIFIERS            ")
    print("==================================================================")

    for name, clf in candidate_models.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(roc), 4),
            "Confusion_Matrix": cm,
        }

        print(f"\n{name}:")
        print(f"  • ROC-AUC   : {roc:.4f}")
        print(f"  • F1-Score  : {f1:.4f}")
        print(f"  • Recall    : {rec:.4f} (Churn detection sensitivity)")
        print(f"  • Precision : {prec:.4f}")
        print(f"  • Accuracy  : {acc:.4f}")

        if roc > best_roc_auc:
            best_roc_auc = roc
            best_model_name = name
            best_pipeline = pipeline

    print("\n------------------------------------------------------------------")
    print(f"Best Classifier: {best_model_name} (ROC-AUC = {best_roc_auc:.4f})")
    print("------------------------------------------------------------------")

    # Serialize best pipeline
    model_path = os.path.join(models_dir, "churn_pipeline.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model pipeline to: {model_path}")

    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_model": best_model_name, "comparison": results}, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_path}")

    return best_pipeline, results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d_path = os.path.join(base_dir, "data", "customer_churn.csv")
    m_dir = os.path.join(base_dir, "models")
    train_and_evaluate(d_path, m_dir)
