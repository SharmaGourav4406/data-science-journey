import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, and columns
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))  # VERY IMPORTANT

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# USER INPUTS
age = st.number_input("Age", min_value=0)
tenure = st.number_input("Tenure", min_value=0)
usage = st.number_input("Usage Frequency", min_value=0)
payment_delay = st.number_input("Payment Delay", min_value=0)
total_spend = st.number_input("Total Spend", min_value=0.0)

gender = st.selectbox("Gender", ["Male", "Female"])
subscription = st.selectbox("Subscription Type", ["Basic", "Premium", "Standard"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly"])
# PREDICTION
if st.button("Predict"):

    # Create empty input with all columns = 0
    input_dict = {col: 0 for col in columns}

    # Fill numeric values
    input_dict['Age'] = age
    input_dict['Tenure'] = tenure
    input_dict['Usage Frequency'] = usage
    input_dict['Payment Delay'] = payment_delay
    input_dict['Total Spend'] = total_spend

    # Fill categorical (one-hot encoding)
    input_dict[f'Gender_{gender}'] = 1
    input_dict[f'Subscription Type_{subscription}'] = 1
    input_dict[f'Contract Length_{contract}'] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])
    #Prediction 
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Customer will churn ")
    else:
        st.success("Customer will NOT churn ")