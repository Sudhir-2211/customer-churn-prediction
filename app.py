"""Interactive Customer Churn Risk Assessment Streamlit Application.

Author: Sudhir Kumar (GitHub: Sudhir-2211)
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.predict import predict_churn

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction | Sudhir Kumar",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #B91C1C;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        color: #991B1B;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">👥 Customer Churn Prediction & Retention Analytics</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Supervised classification pipeline predicting subscription attrition probability and prioritizing customer retention interventions.</div>',
    unsafe_allow_html=True,
)

# Mandatory Business Disclaimer
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ OPERATIONAL USE DISCLAIMER:</strong> This machine learning model is an academic and portfolio demonstration tool. It is <strong>not suitable for autonomous business decisions or automatic contract terminations</strong> without rigorous production calibration, A/B retention testing, and domain expert oversight. In commercial churn modeling, teams must carefully weigh the cost of <strong>False Positives</strong> (budget wasted offering discounts to customers who would have stayed anyway) against <strong>False Negatives</strong> (lost customer lifetime value when a churning user goes undetected).
    </div>
    """,
    unsafe_allow_html=True,
)

base_dir = os.path.dirname(os.path.abspath(__file__))
metrics_file = os.path.join(base_dir, "models", "model_metrics.json")
metrics_data = {}
if os.path.exists(metrics_file):
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

# Sidebar: Customer Profile Inputs
st.sidebar.header("👤 Customer Subscription Profile")

tenure = st.sidebar.slider("Tenure with Company (Months)", min_value=1, max_value=72, value=8)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
monthly_charges = st.sidebar.slider("Monthly Subscription Fee ($)", min_value=18.0, max_value=120.0, value=79.50, step=0.5)
total_charges = round(tenure * monthly_charges, 2)
st.sidebar.caption(f"Estimated Total Historical Charges: ${total_charges:,.2f}")

internet = st.sidebar.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"], index=0)

if internet == "No":
    tech_support = "No internet service"
    online_sec = "No internet service"
else:
    tech_support = st.sidebar.selectbox("Has Tech Support Add-On", ["No", "Yes"], index=0)
    online_sec = st.sidebar.selectbox("Has Online Security Add-On", ["No", "Yes"], index=0)

paperless = st.sidebar.selectbox("Paperless Electronic Billing", ["Yes", "No"], index=0)
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], index=0)

tab_scoring, tab_benchmarks, tab_matrix, tab_data = st.tabs([
    "🎯 Churn Risk Scoring",
    "📊 Classifier Benchmarks",
    "🔲 Confusion Matrix & Tradeoffs",
    "📋 Data Distribution",
])

with tab_scoring:
    st.subheader("Individual Customer Risk Profile")

    try:
        pred_res = predict_churn(
            tenure_months=tenure,
            monthly_charges=monthly_charges,
            total_charges=total_charges,
            contract_type=contract,
            internet_service=internet,
            tech_support=tech_support,
            online_security=online_sec,
            paperless_billing=paperless,
            payment_method=payment,
        )

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Churn Probability", f"{pred_res['churn_risk_pct']}%")
        with col_m2:
            st.metric("Risk Classification", pred_res['risk_tier'])
        with col_m3:
            outcome_text = "🚨 ATTRITION RISK" if pred_res['churn_prediction'] == 1 else "✅ SAFE / RETAINED"
            st.metric("Predicted Outcome", outcome_text)

        # Risk Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_res["churn_risk_pct"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Attrition Risk Score (%)", 'font': {'size': 18}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#DC2626" if pred_res["churn_risk_pct"] > 60 else ("#D97706" if pred_res["churn_risk_pct"] > 30 else "#059669")},
                'steps': [
                    {'range': [0, 30], 'color': "#D1FAE5"},
                    {'range': [30, 65], 'color': "#FEF3C7"},
                    {'range': [65, 100], 'color': "#FEE2E2"},
                ],
                'threshold': {
                    'line': {'color': "#991B1B", 'width': 4},
                    'thickness': 0.75,
                    'value': pred_res["churn_risk_pct"],
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=30, r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.write("#### 🔍 Contributing Risk Factors & Strategic Recommendations")
        recs = []
        if contract == "Month-to-month":
            recs.append(("Short Contract Term", "Month-to-month contracts have 3.5x higher churn odds.", "Offer discounted annual contract extension."))
        if tech_support == "No":
            recs.append(("Missing Tech Support", "Lack of technical onboarding correlates with dissatisfaction.", "Provide 60 days complimentary tech support assistance."))
        if payment == "Electronic check":
            recs.append(("Manual Payment Method", "Electronic checks experience higher payment friction and involuntary churn.", "Incentivize auto-pay via credit card or direct bank transfer."))
        if tenure <= 12:
            recs.append(("Early Lifecycle Stage", "First-year customers are in the critical adoption window.", "Deploy targeted customer success check-in."))

        if recs:
            df_recs = pd.DataFrame(recs, columns=["Risk Factor", "Diagnosis", "Recommended Retention Action"])
            st.table(df_recs)
        else:
            st.success("Customer exhibits high loyalty indicators (long tenure, multi-year contract, bundled services).")

    except Exception as ex:
        st.error(f"Prediction error: {ex}")

with tab_benchmarks:
    st.subheader("Model Comparison & Cross-Validation Benchmarks")
    if "comparison" in metrics_data:
        comp_rows = []
        for m_name, m_vals in metrics_data["comparison"].items():
            comp_rows.append({
                "Classifier": m_name,
                "ROC-AUC": m_vals["ROC_AUC"],
                "F1-Score": m_vals["F1_Score"],
                "Recall (Sensitivity)": m_vals["Recall"],
                "Precision": m_vals["Precision"],
                "Accuracy": m_vals["Accuracy"],
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
        st.success(f"Best Performing Model: **{metrics_data.get('best_model', 'Gradient Boosting')}** selected for production deployment based on top ROC-AUC score.")
    else:
        st.info("Run 'python -m src.train' to compute benchmarks.")

with tab_matrix:
    st.subheader("Confusion Matrix Analysis & Business Error Costs")
    if "comparison" in metrics_data and "Gradient Boosting" in metrics_data["comparison"]:
        cm = metrics_data["comparison"]["Gradient Boosting"]["Confusion_Matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.write("#### Gradient Boosting Confusion Matrix (Test Set N=600)")
            cm_df = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual Retained (0)", "Actual Churned (1)"],
                columns=["Predicted Retained (0)", "Predicted Churned (1)"]
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Reds",
                aspect="auto",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with c_col2:
            st.write("#### ⚖️ The Economic Tradeoff in Churn Prediction")
            st.markdown(
                f"""
                - **True Positives (TP = {tp})**: Successfully intercepted churners who can be targeted with retention campaigns.
                - **True Negatives (TN = {tn})**: Satisfied customers correctly left alone without unnecessary discounts.
                - **False Positives (FP = {fp})**: Loyal customers incorrectly flagged as at-risk. *Cost: Minor expenditure on retention promo codes.*
                - **False Negatives (FN = {fn})**: Churning customers missed by the model. *Cost: High! Permanent loss of monthly subscription revenue.*
                
                Because False Negatives are substantially more expensive in subscription businesses, we prioritize models with high **Recall** and high **ROC-AUC**.
                """
            )

with tab_data:
    st.subheader("Benchmark Training Dataset")
    data_path = os.path.join(base_dir, "data", "customer_churn.csv")
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        st.dataframe(raw_df.head(25), use_container_width=True)
        st.caption(f"Dataset contains {len(raw_df):,} customers. Churn rate: {raw_df['churn'].mean()*100:.1f}%.")
