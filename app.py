# Import libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained ML pipeline
model = joblib.load("model.pkl")

# App title
st.title("Telecom Customer Churn Prediction System")

st.write("Predict the probability that a telecom customer will churn.")

# Customer input features

gender = st.selectbox("Gender", ["Male","Female"])

SeniorCitizen = st.selectbox("Senior Citizen",[0,1])

Partner = st.selectbox("Partner",["Yes","No"])

Dependents = st.selectbox("Dependents",["Yes","No"])

tenure = st.slider("Tenure (months)",0,72,12)

PhoneService = st.selectbox("Phone Service",["Yes","No"])

MultipleLines = st.selectbox("Multiple Lines",["Yes","No","No phone service"])

InternetService = st.selectbox("Internet Service",["DSL","Fiber optic","No"])

OnlineSecurity = st.selectbox("Online Security",["Yes","No","No internet service"])

OnlineBackup = st.selectbox("Online Backup",["Yes","No","No internet service"])

DeviceProtection = st.selectbox("Device Protection",["Yes","No","No internet service"])

TechSupport = st.selectbox("Tech Support",["Yes","No","No internet service"])

StreamingTV = st.selectbox("Streaming TV",["Yes","No","No internet service"])

StreamingMovies = st.selectbox("Streaming Movies",["Yes","No","No internet service"])

Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])

PaperlessBilling = st.selectbox("Paperless Billing",["Yes","No"])

PaymentMethod = st.selectbox(
"Payment Method",
[
"Electronic check",
"Mailed check",
"Bank transfer (automatic)",
"Credit card (automatic)"
]
)

MonthlyCharges = st.number_input("Monthly Charges",0.0,200.0,70.0)

TotalCharges = st.number_input("Total Charges",0.0,10000.0,800.0)


# Run prediction
if st.button("Predict Churn Risk"):

    # Create dataframe matching training features
    input_df = pd.DataFrame({
        "gender":[gender],
        "SeniorCitizen":[SeniorCitizen],
        "Partner":[Partner],
        "Dependents":[Dependents],
        "tenure":[tenure],
        "PhoneService":[PhoneService],
        "MultipleLines":[MultipleLines],
        "InternetService":[InternetService],
        "OnlineSecurity":[OnlineSecurity],
        "OnlineBackup":[OnlineBackup],
        "DeviceProtection":[DeviceProtection],
        "TechSupport":[TechSupport],
        "StreamingTV":[StreamingTV],
        "StreamingMovies":[StreamingMovies],
        "Contract":[Contract],
        "PaperlessBilling":[PaperlessBilling],
        "PaymentMethod":[PaymentMethod],
        "MonthlyCharges":[MonthlyCharges],
        "TotalCharges":[TotalCharges]
    })

    # Predict churn
    prediction = model.predict(input_df)

    # Predict churn probability
    probability = model.predict_proba(input_df)[0][1]

    st.write("Churn Probability:", round(probability,2))

    # Risk classification
    if probability > 0.75:
        st.error("High Risk Customer")

    elif probability > 0.45:
        st.warning("Medium Risk Customer")

    else:
        st.success("Low Risk Customer")


# ---------------------------------------------------
# Feature Importance Section
# Shows which features influence churn predictions
# ---------------------------------------------------

st.subheader("Key Drivers of Customer Churn")

# Extract feature importance from RandomForest model
feature_importance = model.named_steps["model"].feature_importances_

# Get feature names after preprocessing (important)
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# Create dataframe for visualization
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# Show top 10 most important features
top_features = importance_df.head(10)

# Plot horizontal bar chart
fig, ax = plt.subplots()

ax.barh(top_features["Feature"], top_features["Importance"])

ax.set_title("Top Factors Influencing Customer Churn")

ax.invert_yaxis()

# Display chart in Streamlit
st.pyplot(fig)
