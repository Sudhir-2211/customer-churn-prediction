"""Automated unit tests for Customer Churn Classification pipeline."""

import pytest
import os
import joblib
import pandas as pd
from src.predict import predict_churn
from src.train import NUMERIC_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture(scope="module")
def churn_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "churn_pipeline.joblib")
    assert os.path.exists(model_path), "Model pipeline artifact must exist"
    return joblib.load(model_path)


@pytest.fixture(scope="module")
def churn_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "customer_churn.csv")
    assert os.path.exists(data_path), "Dataset must exist"
    return pd.read_csv(data_path)


def test_dataset_integrity(churn_dataset):
    assert len(churn_dataset) >= 2000
    assert not churn_dataset.isnull().any().any()
    assert 0.15 < churn_dataset["churn"].mean() < 0.45
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["churn"]:
        assert col in churn_dataset.columns


def test_model_inference(churn_pipeline):
    sample = pd.DataFrame([{
        "tenure_months": 12,
        "monthly_charges": 75.0,
        "total_charges": 900.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No",
        "online_security": "No",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
    }])
    pred = churn_pipeline.predict(sample)
    prob = churn_pipeline.predict_proba(sample)
    assert len(pred) == 1
    assert pred[0] in [0, 1]
    assert 0.0 <= prob[0][1] <= 1.0


def test_predict_churn_wrapper():
    res = predict_churn(
        tenure_months=24,
        monthly_charges=50.0,
        total_charges=1200.0,
        contract_type="One year",
        internet_service="DSL",
        tech_support="Yes",
        online_security="Yes",
        paperless_billing="No",
        payment_method="Credit card",
    )
    assert "churn_prediction" in res
    assert "churn_probability" in res
    assert "risk_tier" in res
    assert res["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert 0.0 <= res["churn_probability"] <= 1.0


def test_risk_monotonicity():
    # High risk customer: 1 month tenure, month-to-month contract, no tech support, fiber optic, electronic check
    high_risk = predict_churn(
        tenure_months=1,
        monthly_charges=105.0,
        total_charges=105.0,
        contract_type="Month-to-month",
        internet_service="Fiber optic",
        tech_support="No",
        online_security="No",
        paperless_billing="Yes",
        payment_method="Electronic check",
    )
    # Low risk customer: 60 months tenure, two year contract, full security, bank transfer
    low_risk = predict_churn(
        tenure_months=60,
        monthly_charges=30.0,
        total_charges=1800.0,
        contract_type="Two year",
        internet_service="No",
        tech_support="No internet service",
        online_security="No internet service",
        paperless_billing="No",
        payment_method="Bank transfer",
    )

    assert high_risk["churn_probability"] > low_risk["churn_probability"]
    assert high_risk["risk_tier"] == "High Risk"
    assert low_risk["risk_tier"] == "Low Risk"
