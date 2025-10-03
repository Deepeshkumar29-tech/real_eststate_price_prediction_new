import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

st.title("🏠 Real Estate Price Prediction")

# Option 1: Retrain model with all features
@st.cache_resource
def train_model_with_all_features():
    try:
        # Load the dataset
        df = pd.read_csv("real_estate.csv")
        
        # Prepare features and target
        feature_columns = [
            'area', 'bedrooms', 'bathrooms', 'location_city_center', 
            'location_suburb', 'location_outskirts', 'has_parking', 
            'has_garden', 'near_metro', 'property_age'
        ]
        
        X = df[feature_columns]
        y = df['price']
        
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate R² score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        return model, r2, feature_columns
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

# Option 2: Manual adjustment factors
def calculate_manual_adjustment(base_price, location, has_parking, has_garden, near_metro, property_age):
    """Apply manual adjustments based on real estate factors"""
    
    # Location multipliers
    location_multipliers = {
        "City Center": 1.3,
        "Suburb": 1.1, 
        "Outskirts": 0.9
    }
    
    # Feature value adjustments
    feature_values = {
        "parking": 500000,  # ₹5,00,000 for parking
        "garden": 300000,   # ₹3,00,000 for garden
        "metro": 400000,    # ₹4,00,000 for metro proximity
    }
    
    # Property age depreciation (0.5% per year)
    age_depreciation = 0.005 * property_age
    
    # Calculate adjusted price
    adjusted_price = base_price * location_multipliers[location]
    
    # Add feature values
    if has_parking:
        adjusted_price += feature_values["parking"]
    if has_garden:
        adjusted_price += feature_values["garden"]
    if near_metro:
        adjusted_price += feature_values["metro"]
    
    # Apply age depreciation
    adjusted_price *= (1 - age_depreciation)
    
    return adjusted_price

# Main app
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

# Model selection
st.subheader("🎯 Prediction Method")
prediction_method = st.radio(
    "Choose prediction method:",
    ["Manual Adjustments (Recommended)", "Retrain Model with All Features"],
    help="Manual adjustments work with your current model but add realistic adjustments for extra features"
)

if st.button("Predict Price", type="primary"):
    if prediction_method == "Manual Adjustments (Recommended)":
        # Use current model with manual adjustments
        try:
            model = joblib.load("model.joblib")
            
            # Get base prediction from current model
            input_df = pd.DataFrame([{
                "area": area,
                "bedrooms": bedrooms, 
                "bathrooms": bathrooms
            }])
            
            base_prediction = model.predict(input_df)[0]
            
            # Apply manual adjustments for other features
            final_prediction = calculate_manual_adjustment(
                base_prediction, location, has_parking, has_garden, near_metro, property_age
            )
            
            st.success(f"Estimated Price: **₹ {final_prediction:,.2f}**")
            
            # Show breakdown
            with st.expander("📊 Price Breakdown"):
                st.write(f"Base Price (area, bedrooms, bathrooms): ₹ {base_prediction:,.2f}")
                st.write(f"Location Adjustment ({location}): {['+30%', '+10%', '-10%'][['City Center', 'Suburb', 'Outskirts'].index(location)]}")
                if has_parking:
                    st.write("Parking Available: +₹ 5,00,000")
                if has_garden:
                    st.write("Garden: +₹ 3,00,000")
                if near_metro:
                    st.write("Near Metro: +₹ 4,00,000")
                st.write(f"Property Age ({property_age} years): -{property_age * 0.5}%")
                
        except Exception as e:
            st.error(f"Error with manual adjustment method: {e}")
            
    else:  # Retrain Model with All Features
        model, r2_score, feature_columns = train_model_with_all_features()
        
        if model is not None:
            try:
                # Prepare input data with all features
                input_data = {
                    'area': area,
                    'bedrooms': bedrooms, 
                    'bathrooms': bathrooms,
                    'location_city_center': 1 if location == "City Center" else 0,
                    'location_suburb': 1 if location == "Suburb" else 0,
                    'location_outskirts': 1 if location == "Outskirts" else 0,
                    'has_parking': 1 if has_parking else 0,
                    'has_garden': 1 if has_garden else 0,
                    'near_metro': 1 if near_metro else 0,
                    'property_age': property_age
                }
                
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                
                st.success(f"Estimated Price: **₹ {prediction:,.2f}**")
                st.info(f"Model R² Score: {r2_score:.3f}")
                
            except Exception as e:
                st.error(f"Error with retrained model: {e}")

# Show feature impact explanation
with st.expander("💡 How Features Affect Price"):
    st.write("""
    **Location Impact:**
    - 🏙️ City Center: +30% (Premium for accessibility & amenities)
    - 🏡 Suburb: +10% (Balance of convenience & affordability)  
    - 🌳 Outskirts: -10% (More affordable, peaceful environment)
    
    **Feature Values:**
    - 🅿️ Parking: +₹5,00,000
    - 🌿 Garden: +₹3,00,000
    - 🚇 Metro Proximity: +₹4,00,000
    
    **Age Depreciation:**
    - 0.5% per year (older properties lose value)
    """)

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
