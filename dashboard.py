import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load('models/churn_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

st.title("Customer Churn Predictor")

# Input form
tenure = st.slider("Tenure (months)", 0, 100, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer"])

if st.button("Predict Churn"):
    data = {
        "gender": "Female",  # Example static value (update as needed)
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": tenure * monthly_charges,  # Simulate total charges
        "InternetService": internet_service,
        "PaymentMethod": payment_method
    }
    df = pd.DataFrame([data])
    processed_data = preprocessor.transform(df)
    proba = model.predict_proba(processed_data)[0][1]
    st.success(f"Churn Probability: {proba:.2f}")
    