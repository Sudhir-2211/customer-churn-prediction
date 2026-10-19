"""Generates realistic customer churn classification dataset.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    tenure = np.random.randint(1, 73, size=n_samples)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.25, 0.20], size=n_samples)
    internet = np.random.choice(["Fiber optic", "DSL", "No"], p=[0.45, 0.35, 0.20], size=n_samples)

    tech_support = []
    online_security = []
    monthly_charges = []

    for i in range(n_samples):
        net = internet[i]
        if net == "No":
            tech_support.append("No internet service")
            online_security.append("No internet service")
            monthly_charges.append(np.random.normal(24, 4))
        elif net == "DSL":
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            online_security.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            monthly_charges.append(np.random.normal(55, 10))
        else:  # Fiber optic
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.28, 0.72]))
            online_security.append(np.random.choice(["Yes", "No"], p=[0.30, 0.70]))
            monthly_charges.append(np.random.normal(88, 12))

    monthly_charges = np.array(monthly_charges).clip(18.5, 120.0)
    total_charges = np.round(tenure * monthly_charges * np.random.normal(1.0, 0.03, n_samples), 2)

    paperless = np.random.choice(["Yes", "No"], p=[0.60, 0.40], size=n_samples)
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        p=[0.34, 0.22, 0.22, 0.22],
        size=n_samples,
    )

    # Calibrated baseline log-odds for ~27% churn rate
    log_odds = (
        -1.95
        - 0.045 * tenure
        + 0.015 * monthly_charges
        + (contract == "Month-to-month") * 1.35
        - (contract == "Two year") * 1.50
        + (internet == "Fiber optic") * 0.75
        + (np.array(tech_support) == "No") * 0.60
        + (payment == "Electronic check") * 0.50
        + (paperless == "Yes") * 0.25
        + np.random.normal(0, 0.35, n_samples)
    )

    prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (prob > 0.50).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{1000 + i}" for i in range(n_samples)],
        "tenure_months": tenure,
        "contract_type": contract,
        "internet_service": internet,
        "tech_support": tech_support,
        "online_security": online_security,
        "paperless_billing": paperless,
        "payment_method": payment,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": total_charges,
        "churn": churn,
    })

    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    df = generate_churn_dataset()
    out_path = os.path.join(data_dir, "customer_churn.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} customer records (Churn Rate: {df['churn'].mean()*100:.1f}%) saved to {out_path}.")
