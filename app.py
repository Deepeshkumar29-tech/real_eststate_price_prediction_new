import streamlit as st
import pandas as pd
import joblib

st.title("🏠 Real Estate Price Prediction")

# Load model
model = joblib.load("model.joblib")

st.write("Enter property details to predict the price:")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Specifications")
    area = st.number_input("Area (sqft)", min_value=500, max_value=10000, value=2000, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    property_age = st.slider("Property Age (years)", min_value=0, max_value=50, value=5)

with col2:
    st.subheader("Location & Features")
    location = st.selectbox(
        "Location",
        options=["City Center", "Suburb", "Outskirts"],
        help="Select the area where the property is located"
    )
    
    has_parking = st.checkbox("Parking Available", value=True)
    has_garden = st.checkbox("Garden", value=False)
    near_metro = st.checkbox("Near Metro Station", value=False)

# Display model performance info
with st.expander("ℹ️ About the Model"):
    st.write("""
    This model is trained on comprehensive real estate data including:
    - Property specifications (area, bedrooms, bathrooms, age)
    - Location features (city center, suburb, outskirts)
    - Amenities (parking, garden, metro proximity)
    
    The model has been validated with an R² score of 0.98 on test data.
    """)
    
    # Show sample predictions
    if st.checkbox("Show sample predictions"):
        predictions_df = pd.read_csv("predictions.csv")
        st.dataframe(predictions_df.head(), use_container_width=True)

if st.button("Predict Price", type="primary"):
    # Create input dataframe with all features
    input_data = {
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
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        
        # Display results
        st.success(f"Estimated Price: **₹ {prediction:,.2f}**")
        
        # Show feature breakdown
        st.subheader("📊 Property Details Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Area", f"{area:,} sqft")
            st.metric("Bedrooms", bedrooms)
            st.metric("Bathrooms", bathrooms)
            
        with col2:
            st.metric("Property Age", f"{property_age} years")
            st.metric("Location", location)
            
        with col3:
            st.metric("Parking", "✅" if has_parking else "❌")
            st.metric("Garden", "✅" if has_garden else "❌")
            st.metric("Near Metro", "✅" if near_metro else "❌")
        
        # Location insights
        st.subheader("📍 Location Insights")
        if location == "City Center":
            st.info("Prime city center location typically commands premium pricing due to better accessibility and amenities.")
        elif location == "Suburb":
            st.info("Suburban areas offer a balance between convenience and affordability with good community facilities.")
        else:
            st.info("Outskirts locations provide more affordable options with peaceful environments and more space.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure all input values are valid and the model file is correctly loaded.")

# Add a section to show training data statistics
with st.sidebar:
    st.header("📈 Data Overview")
    
    try:
        # Load and display training data stats
        train_df = pd.read_csv("real_estate.csv")
        
        st.metric("Total Properties", len(train_df))
        st.metric("Average Price", f"₹ {train_df['price'].mean():,.0f}")
        st.metric("Average Area", f"{train_df['area'].mean():.0f} sqft")
        
        st.write("---")
        st.write("**Location Distribution:**")
        city_center = train_df['location_city_center'].sum()
        suburb = train_df['location_suburb'].sum()
        outskirts = train_df['location_outskirts'].sum()
        
        st.write(f"• City Center: {city_center}")
        st.write(f"• Suburb: {suburb}")
        st.write(f"• Outskirts: {outskirts}")
        
    except Exception as e:
        st.warning("Could not load training data statistics")

# Footer
st.write("---")
st.caption("Note: Predictions are estimates based on historical data and market trends. Actual prices may vary.")
