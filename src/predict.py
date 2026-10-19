"""Command-Line Interface for Customer Churn Inference.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
import argparse
import joblib
import pandas as pd


def predict_churn(
    tenure_months: int,
    monthly_charges: float,
    total_charges: float,
    contract_type: str,
    internet_service: str,
    tech_support: str,
    online_security: str,
    paperless_billing: str,
    payment_method: str,
    model_path: str = None,
) -> dict:
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "churn_pipeline.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run 'python -m src.train' first.")

    pipeline = joblib.load(model_path)

    sample = pd.DataFrame([{
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
    }])

    prob = float(pipeline.predict_proba(sample)[0][1])
    is_churn = int(prob >= 0.50)

    if prob < 0.30:
        risk_tier = "Low Risk"
    elif prob < 0.65:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "High Risk"

    return {
        "churn_prediction": is_churn,
        "churn_probability": round(prob, 4),
        "churn_risk_pct": round(prob * 100.0, 1),
        "risk_tier": risk_tier,
    }


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Risk Classifier")
    parser.add_argument("--tenure", type=int, default=6, help="Tenure in months (1-72)")
    parser.add_argument("--monthly", type=float, default=85.50, help="Monthly subscription charges ($)")
    parser.add_argument("--total", type=float, default=513.0, help="Total historical charges ($)")
    parser.add_argument("--contract", default="Month-to-month", choices=["Month-to-month", "One year", "Two year"])
    parser.add_argument("--internet", default="Fiber optic", choices=["Fiber optic", "DSL", "No"])
    parser.add_argument("--tech_support", default="No", choices=["Yes", "No", "No internet service"])
    parser.add_argument("--security", default="No", choices=["Yes", "No", "No internet service"])
    parser.add_argument("--paperless", default="Yes", choices=["Yes", "No"])
    parser.add_argument("--payment", default="Electronic check", choices=["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    args = parser.parse_args()

    result = predict_churn(
        tenure_months=args.tenure,
        monthly_charges=args.monthly,
        total_charges=args.total,
        contract_type=args.contract,
        internet_service=args.internet,
        tech_support=args.tech_support,
        online_security=args.security,
        paperless_billing=args.paperless,
        payment_method=args.payment,
    )

    print("\n" + "=" * 60)
    print("           CUSTOMER CHURN RISK ASSESSMENT RESULT")
    print("=" * 60)
    print("Customer Profile:")
    print(f"  • Tenure         : {args.tenure} Months | Contract: {args.contract}")
    print(f"  • Monthly Spend  : ${args.monthly:,.2f} | Tech Support: {args.tech_support}")
    print(f"  • Internet / Pay : {args.internet} | {args.payment}")
    print("-" * 60)
    print(f"Churn Probability: {result['churn_risk_pct']}%")
    print(f"Risk Tier        : {result['risk_tier'].upper()}")
    print(f"Binary Outcome   : {'CHURN LIKELY' if result['churn_prediction'] == 1 else 'RETAINED'}")
    print("=" * 60)
    print("DISCLAIMER: Statistical risk score for retention triage only,")
    print("not an automated customer cancellation trigger.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
