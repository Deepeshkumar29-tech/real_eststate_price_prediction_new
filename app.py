iimport streamlit as st
import pandas as pd
import joblib

st.title("🏠 Real Estate Price Prediction")

# Load model
model = joblib.load("model.joblib")

st.write("Enter details to predict house price:")

# Property details
area = st.number_input("Area (sqft)", min_value=500, max_value=10000, step=100)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

# Location options
location = st.selectbox(
    "Location",
    options=["City Center", "Suburb", "Outskirts"],
    help="Select the area where the property is located"
)

# Additional features
st.subheader("Additional Features")
has_parking = st.checkbox("Parking Available")
has_garden = st.checkbox("Garden")
near_metro = st.checkbox("Near Metro Station")
property_age = st.slider("Property Age (years)", min_value=0, max_value=50, value=5)

if st.button("Predict Price"):
    # Create input dataframe with all features
    input_df = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms, 
        "bathrooms": bathrooms,
        "location_city_center": 1 if location == "City Center" else 0,
        "location_suburb": 1 if location == "Suburb" else 0,
        "location_outskirts": 1 if location == "Outskirts" else 0,
        "has_parking": 1 if has_parking else 0,
        "has_garden": 1 if has_garden else 0,
        "near_metro": 1 if near_metro else 0,
        "property_age": property_age
    }])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {prediction:,.2f}")
    
    # Show location impact
    st.info(f"📍 Location: {location}")
    
    # Additional insights
    st.subheader("💡 Insights")
    if location == "City Center":
        st.write("• Prime location with higher value")
        st.write("• Better accessibility to amenities")
    elif location == "Suburb":
        st.write("• Balanced between convenience and affordability")
        st.write("• Good community facilities")
    else:
        st.write("• More affordable option")
        st.write("• Peaceful environment with more space")
