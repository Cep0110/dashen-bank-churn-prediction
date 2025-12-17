  import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD PIPELINE ----------------
pipeline = joblib.load("churn_model.pkl")  # this IS your pipeline

st.set_page_config(page_title="Dashen Bank Churn Predictor", layout="centered")

st.title("🏦 Dashen Bank Customer Churn Predictor")
st.write("Predict whether a customer is likely to leave the bank.")

# ---------------- USER INPUTS (RAW) ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 80, 35)
region = st.selectbox("Region", ["Addis Ababa", "Oromia", "Amhara", "Tigray", "SNNPR"])
has_partner = st.selectbox("Has Partner?", ["Yes", "No"])
tenure = st.slider("Tenure (Years)", 0.0, 20.0, 3.0)
account_type = st.selectbox("Account Type", ["Savings", "Current"])
mobile_banking = st.selectbox("Uses Mobile Banking?", ["Yes", "No"])
credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
balance = st.number_input("Account Balance (ETB)", min_value=0.0, value=5000.0)
monthly_fee = st.number_input("Monthly Service Fee (ETB)", min_value=0.0, value=50.0)

# ---------------- INPUT DATAFRAME (MUST MATCH TRAINING) ----------------
input_df = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Region": [region],
    "HasPartner": [has_partner],
    "TenureYears": [tenure],
    "AccountType": [account_type],
    "HasMobileBanking": [mobile_banking],
    "HasCreditCard": [credit_card],
    "IsActiveMember": [active_member],
    "Balance_ETB": [balance],
    "MonthlyServiceFee_ETB": [monthly_fee]
})

# ---------------- PREDICTION ----------------
if st.button("Predict Churn Risk"):
    prob = pipeline.predict_proba(input_df)[0][1]

    st.subheader("🔍 Prediction Result")
    st.metric("Churn Probability", f"{prob:.2%}")

    if prob >= 0.6:
        st.error("⚠️ High Risk of Churn")
        st.write("👉 Immediate retention action recommended")
    else:
        st.success("✅ Low Risk of Churn")
        st.write("👉 Maintain customer engagement")

