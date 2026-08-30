import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.joblib"
)

model = joblib.load(MODEL_PATH)

# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Prediction System")

st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to churn or stay."
)

st.divider()

# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")

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
        value=40
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
        value=50000.0,
        step=1000.0
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

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0,
        step=1000.0
    )

# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict Customer Churn", use_container_width=True):

    # Convert categorical values
    geography_germany = 1 if geography == "Germany" else 0
    geography_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    has_credit_card = 1 if has_card == "Yes" else 0
    is_active_member = 1 if active_member == "Yes" else 0

    # Feature Engineering
    balance_salary_ratio = (
        balance / estimated_salary
        if estimated_salary != 0
        else 0
    )

    # Create input dataframe
    input_data = pd.DataFrame([{

        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_credit_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography_Germany": geography_germany,
        "Geography_Spain": geography_spain,
        "Gender_Male": gender_male,
        "BalanceSalaryRatio": balance_salary_ratio

    }])

    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(input_data)[0]

    # Probability
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0][1]

    else:

        probability = None

    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.header("📈 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Customer is likely to CHURN")

    else:

        st.success("✅ Customer is likely to STAY")

    # Probability
    if probability is not None:

        st.subheader("Churn Probability")

        st.progress(float(probability))

        st.write(
            f"**{probability * 100:.2f}%**"
        )

    # =====================================================
    # CUSTOMER INPUT SUMMARY
    # =====================================================

    st.subheader("### Customer Input")

    st.dataframe(
        input_data,
        use_container_width=True
    )