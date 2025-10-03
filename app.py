import streamlit as st
import pandas as pd
import joblib

st.title("🏠 Real Estate Price Prediction")

# Load model
try:
    model = joblib.load("model.joblib")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Debug: Check model features if available
try:
    if hasattr(model, 'feature_names_in_'):
        st.sidebar.write("**Model Features:**", list(model.feature_names_in_))
except:
    pass

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
    """)

if st.button("Predict Price", type="primary"):
    # Try different feature name combinations based on common patterns
    feature_combinations = [
        # Option 1: Original feature names from your CSV
        {
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
        },
        # Option 2: Without underscores
        {
            "area": area,
            "bedrooms": bedrooms, 
            "bathrooms": bathrooms,
            "locationcitycenter": 1 if location == "City Center" else 0,
            "locationsuburb": 1 if location == "Suburb" else 0,
            "locationoutskirts": 1 if location == "Outskirts" else 0,
            "hasparking": 1 if has_parking else 0,
            "hasgarden": 1 if has_garden else 0,
            "nearmetro": 1 if near_metro else 0,
            "propertyage": property_age
        },
        # Option 3: Basic features only (fallback)
        {
            "area": area,
            "bedrooms": bedrooms, 
            "bathrooms": bathrooms
        }
    ]
    
    prediction_made = False
    
    for i, input_data in enumerate(feature_combinations):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            st.success(f"Estimated Price: **₹ {prediction:,.2f}**")
            
            # Show which feature set worked
            if i == 0:
                st.info("✅ Used full feature set with underscores")
            elif i == 1:
                st.info("✅ Used feature set without underscores")
            else:
                st.info("✅ Used basic features only (area, bedrooms, bathrooms)")
            
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
            
            prediction_made = True
            break
            
        except Exception as e:
            if i == len(feature_combinations) - 1:  # Last attempt
                st.error(f"❌ Prediction failed with all feature combinations")
                st.error(f"Error: {e}")
                
                # Debug information
                with st.expander("🔧 Debug Information"):
                    st.write("Try these solutions:")
                    st.write("1. Check if your model.joblib file matches your CSV data")
                    st.write("2. Retrain your model with the current feature names")
                    st.write("3. Verify the model was saved correctly")
                    
                    # Show what features we tried
                    st.write("Features tried:")
                    for j, features in enumerate(feature_combinations):
                        st.write(f"Option {j+1}: {list(features.keys())}")

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
        
        # Show actual feature names from CSV
        st.write("---")
        st.write("**CSV Features:**")
        st.write(list(train_df.columns))
        
    except Exception as e:
        st.warning("Could not load training data statistics")

# Footer
st.write("---")
st.caption("Note: Predictions are estimates based on historical data and market trends. Actual prices may vary.")
