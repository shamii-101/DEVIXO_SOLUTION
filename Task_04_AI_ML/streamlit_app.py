import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/best_model.joblib")

# ==========================================
# TITLE
# ==========================================

st.title("📊 Customer Churn Prediction System")
st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to churn or stay."
)

st.divider()

# ==========================================
# USER INPUT
# ==========================================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=4,
        value=1
    )

with col2:
    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    has_card = st.selectbox(
        "Has Credit Card",
        ["Yes", "No"]
    )

    active_member = st.selectbox(
        "Is Active Member",
        ["Yes", "No"]
    )

    salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0
    )

# ==========================================
# FEATURE ENGINEERING
# ==========================================

balance_salary_ratio = balance / (salary + 1)

# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [1 if has_card == "Yes" else 0],
    "IsActiveMember": [1 if active_member == "Yes" else 0],
    "EstimatedSalary": [salary],
    "Geography_Germany": [1 if geography == "Germany" else 0],
    "Geography_Spain": [1 if geography == "Spain" else 0],
    "Gender_Male": [1 if gender == "Male" else 0],
    "BalanceSalaryRatio": [balance_salary_ratio]
})

# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

if st.button("🔮 Predict Customer Churn", use_container_width=True):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")

    else:
        st.success("✅ Customer is likely to STAY")

    st.metric(
        "Churn Probability",
        f"{probability * 100:.2f}%"
    )

    st.write("### Customer Input")
    st.dataframe(input_data, use_container_width=True)