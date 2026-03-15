# ============================================================
# TELECOM CUSTOMER CHURN ANALYTICS PLATFORM
# Author: Vishnu Durgam
# Description: End-to-end ML analytics dashboard
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics Platform",
    layout="wide"
)

# ------------------------------------------------------------
# Load trained ML model
# ------------------------------------------------------------

model = joblib.load("model.pkl")

# ------------------------------------------------------------
# Application Header
# ------------------------------------------------------------

st.title("Telecom Customer Churn Analytics Platform")

st.markdown("""
This platform provides **predictive analytics and decision support**
for telecom customer churn management.

The system enables:

• Customer churn risk prediction  
• Model explainability and insights  
• Customer risk segmentation  
• Strategic business recommendations
""")

# ------------------------------------------------------------
# Create Navigation Tabs
# ------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
"Executive Overview",
"Customer Prediction",
"Model Insights",
"Business Strategy"
])

# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================

with tab1:

    st.header("Executive Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "Random Forest")

    col2.metric("Model Accuracy", "~80%")

    col3.metric("Deployment Status", "Active")

    st.write("""
    This machine learning system identifies telecom customers at risk of churn.
    
    It helps customer retention teams prioritize interventions
    and reduce revenue loss.
    """)

# ============================================================
# TAB 2 — CUSTOMER PREDICTION
# ============================================================

with tab2:

    st.header("Customer Churn Prediction Tool")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox("Gender", ["Male","Female"])

        SeniorCitizen = st.selectbox("Senior Citizen",[0,1])

        Partner = st.selectbox("Partner",["Yes","No"])

        Dependents = st.selectbox("Dependents",["Yes","No"])

        tenure = st.slider("Tenure (months)",0,72,12)

        PhoneService = st.selectbox("Phone Service",["Yes","No"])

        MultipleLines = st.selectbox(
            "Multiple Lines",
            ["Yes","No","No phone service"]
        )

    with col2:

        InternetService = st.selectbox(
            "Internet Service",
            ["DSL","Fiber optic","No"]
        )

        Contract = st.selectbox(
            "Contract",
            ["Month-to-month","One year","Two year"]
        )

        PaymentMethod = st.selectbox(
            "Payment Method",
            [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
            ]
        )

        MonthlyCharges = st.number_input(
            "Monthly Charges",
            0.0,200.0,70.0
        )

        TotalCharges = st.number_input(
            "Total Charges",
            0.0,10000.0,800.0
        )

    # Create input dataframe
    input_df = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[SeniorCitizen],
        "Partner":[Partner],
        "Dependents":[Dependents],
        "tenure":[tenure],
        "PhoneService":[PhoneService],
        "MultipleLines":[MultipleLines],
        "InternetService":[InternetService],
        "OnlineSecurity":["No"],
        "OnlineBackup":["No"],
        "DeviceProtection":["No"],
        "TechSupport":["No"],
        "StreamingTV":["No"],
        "StreamingMovies":["No"],
        "Contract":[Contract],
        "PaperlessBilling":["Yes"],
        "PaymentMethod":[PaymentMethod],
        "MonthlyCharges":[MonthlyCharges],
        "TotalCharges":[TotalCharges]
    })

    if st.button("Predict Churn Risk"):

        prediction = model.predict(input_df)

        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        st.metric("Churn Probability", round(probability,2))

        if probability > 0.75:
            st.error("High Risk Customer")

        elif probability > 0.45:
            st.warning("Medium Risk Customer")

        else:
            st.success("Low Risk Customer")

# ============================================================
# TAB 3 — MODEL INSIGHTS
# ============================================================

with tab3:

    st.header("Model Explainability")

    feature_importance = model.named_steps["model"].feature_importances_

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature":feature_names,
        "Importance":feature_importance
    }).sort_values("Importance", ascending=False)

    top_features = importance_df.head(10)

    fig, ax = plt.subplots(figsize=(8,6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=top_features,
        ax=ax
    )

    ax.set_title("Top Drivers of Customer Churn")

    st.pyplot(fig)

# ============================================================
# TAB 4 — BUSINESS STRATEGY
# ============================================================

with tab4:

    st.header("Strategic Business Insights")

    st.write("""
    The model identifies key churn drivers including:

    • Short customer tenure  
    • High monthly charges  
    • Month-to-month contracts  
    • Premium internet services

    **Recommended Actions**

    1. Target high-risk customers with loyalty discounts  
    2. Encourage migration to annual contracts  
    3. Offer bundled service packages
    """)
