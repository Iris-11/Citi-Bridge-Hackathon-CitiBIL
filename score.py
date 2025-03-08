import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from autogluon.tabular import TabularPredictor


# Load models
with open("random_forest_model.pkl", "rb") as file:
    rf_model = pickle.load(file)


with open("xgboost_model.pkl", "rb") as file:
    xgb_model = pickle.load(file)


# Load AutoGluon model
autogluon_model_path = "ag_model"
predictor = TabularPredictor.load(autogluon_model_path)


# Define expected features
expected_features = [
    "Business_ID", "Annual_Revenue (₹)", "Loan_Amount (₹)", "GST_Compliance (%)",
    "Past_Defaults", "Bank_Transactions", "Market_Trend"
]


# Mapping for Label Encoding
bank_transactions_mapping = {"Low Volume": 0, "Stable": 1, "Unstable": 2}
market_trend_mapping = {"Growth": 0, "Stable": 1, "Decline": 2}


def preprocess_input(data):
    """Preprocesses user input to match the training feature set."""
    df = pd.DataFrame([data])
    df["Business_ID"] = 0  # Dummy value


    # Ensure all features exist
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # Default for missing columns


    return df[expected_features]  # Ensure correct column order


with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path
    #st.image("logo.png", use_column_width=True)
    st.markdown("### 📚 CitiBil")
    st.markdown("---")
    

# Streamlit UI
st.title("CitiBIL")


# User inputs
annual_revenue = st.number_input("Annual Revenue (₹)", min_value=0.0, format="%f")
loan_amount = st.number_input("Loan Amount (₹)", min_value=0.0, format="%f")
gst_compliance = st.slider("GST Compliance (%)", 0, 100, 50)
past_defaults = st.number_input("Past Defaults", min_value=0, step=1)
bank_transactions = st.selectbox("Bank Transactions", list(bank_transactions_mapping.keys()))
market_trend = st.selectbox("Market Trend", list(market_trend_mapping.keys()))


selected_model = st.radio("Choose Model", ("Random Forest", "XGBoost"))


# Convert categorical inputs using Label Encoding
input_data = {
    "Annual_Revenue (₹)": annual_revenue,
    "Loan_Amount (₹)": loan_amount,
    "GST_Compliance (%)": gst_compliance,
    "Past_Defaults": past_defaults,
    "Bank_Transactions": bank_transactions_mapping[bank_transactions],
    "Market_Trend": market_trend_mapping[market_trend]
}


# Preprocess input
input_data_encoded = preprocess_input(input_data)


if st.button("Predict"):
    if selected_model == "Random Forest":
        prediction = rf_model.predict(input_data_encoded)[0]
    else:
        prediction = xgb_model.predict(input_data_encoded)[0]


    # Scale prediction to CIBIL score range (300-900)
    cibil_score = prediction


    # Define risk level based on CIBIL score
    if cibil_score < 500:
        risk_level = "High Risk"
        risk_color = "red"
    elif 500 <= cibil_score < 700:
        risk_level = "Medium Risk"
        risk_color = "orange"
    else:
        risk_level = "Low Risk"
        risk_color = "green"


    # **CIBIL Score Gauge Chart**
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cibil_score,
        title={"text": "CIBIL Score"},
        gauge={
            "axis": {"range": [300, 900]},
            "bar": {"color": risk_color},
            "steps": [
                {"range": [300, 500], "color": "red"},
                {"range": [500, 700], "color": "orange"},
                {"range": [700, 900], "color": "green"}
            ],
        }
    ))
   
    # Display prediction & gauge chart
    st.write("### Prediction:")
    st.write(f"**{selected_model}:** {cibil_score} ({risk_level})")
    st.plotly_chart(fig)


# SHAP Explanation
if st.button("Explain (XAI)"):
    if selected_model == "Random Forest":
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_data_encoded)
    else:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_data_encoded)


    # Exclude 'Business_ID' from SHAP values
    feature_names = list(input_data_encoded.columns)
    if "Business_ID" in feature_names:
        feature_names.remove("Business_ID")


    # SHAP Summary Plot
    st.write("### Feature Importance (SHAP Summary)")
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values[:,:-1], input_data_encoded[feature_names], feature_names=feature_names, show=False)
    st.pyplot(fig)
