import streamlit as st
import pandas as pd
import joblib

st.title("🏠 Real Estate Price Prediction")

# Load model
model = joblib.load("model.joblib")

st.write("Enter details to predict house price:")

area = st.number_input("Area (sqft)", min_value=500, max_value=10000, step=100)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms}])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {prediction:,.2f}")
