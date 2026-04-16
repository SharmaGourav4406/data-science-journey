import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('house_price_model.pkl', 'rb'))

st.title("🏠 House Price Prediction App")

st.write("Enter house details:")

area = st.number_input("Area (sq ft)", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
stories = st.number_input("Stories", min_value=0)
parking = st.number_input("Parking spaces", min_value=0)

if st.button("Predict Price"):
    # Feature engineering
    area_per_bedroom = area / (bedrooms + 1)

    # Input for model
    input_data = np.array([[area, bedrooms, bathrooms, stories, parking, area_per_bedroom]])

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")