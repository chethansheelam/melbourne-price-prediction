import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Load Model and Data Assets ---
# Use st.cache_resource to load these only once and improve app performance
@st.cache_resource
def load_assets():
    """Loads the trained model, model columns, and app data."""
    try:
        model = joblib.load('melbourne_house_price_model.joblib')
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        with open('app_data.json', 'r') as f:
            app_data = json.load(f)
    except FileNotFoundError:
        st.error("Model assets not found. Please ensure the required .joblib and .json files are in the same directory as app.py.")
        st.stop() # Stop the app if assets aren't available
    return model, model_columns, app_data

model, model_columns, app_data = load_assets()

# --- Web App Interface ---
st.set_page_config(page_title="Melbourne House Price Predictor", layout="wide")
st.title('🏡 Melbourne House Price Predictor')
st.write("Enter the details of a property to get an estimated market value based on historical data.")
st.markdown("---")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Features")
    rooms = st.slider('Rooms', 1, 16, 3, help="Total number of rooms.")
    bedroom2 = st.slider('Bedrooms', 1, 30, 3, help="Number of bedrooms.")
    bathroom = st.slider('Bathrooms', 1, 12, 1, help="Number of bathrooms.")
    car = st.slider('Car Spots', 0, 26, 1, help="Number of car parking spots.")
    age = st.number_input('Property Age (Years)', min_value=0, max_value=200, value=25, help="Calculated as 2024 - YearBuilt.")
    landsize_log = np.log1p(st.number_input('Land Size (sqm)', min_value=0, max_value=50000, value=500, help="The land area in square meters."))

with col2:
    st.subheader("Location & Sale Details")
    # Use sorted lists for dropdowns for a better user experience
    suburb = st.selectbox('Suburb', sorted(app_data['categorical_cols']['Suburb']))
    regionname = st.selectbox('Region', sorted(app_data['categorical_cols']['Regionname']))
    property_type = st.selectbox('Type', sorted(app_data['categorical_cols']['Type']), help="h: house, u: unit, t: townhouse")
    method = st.selectbox('Sale Method', sorted(app_data['categorical_cols']['Method']), help="S: sold, SP: sold prior, etc.")
    distance = st.slider('Distance from CBD (km)', 0.0, 50.0, 10.0, help="Distance from Melbourne's Central Business District.")
    propertycount = st.number_input('Property Count in Suburb', min_value=0, value=7500, help="Number of properties in the suburb.")

st.markdown("---")

# Prediction Button
if st.button('Predict Price', type="primary", use_container_width=True):
    
    # --- Create Input DataFrame for Prediction ---
    input_dict = {
        'Rooms': rooms, 'Distance': distance, 'Bedroom2': bedroom2, 'Bathroom': bathroom,
        'Car': car, 'Propertycount': propertycount, 'Age': age, 'Landsize_log': landsize_log,
        'Suburb': suburb, 'Type': property_type, 'Method': method, 'Regionname': regionname
    }
    input_df = pd.DataFrame([input_dict])
    
    # --- Preprocess the Input Data to Match Model's Training Format ---
    input_encoded = pd.get_dummies(input_df)
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # --- Make Prediction ---
    prediction_log = model.predict(final_input)
    predicted_price = np.expm1(prediction_log[0])
    
    # --- Display the Result ---
    st.success(f"**Estimated Property Price: ${predicted_price:,.0f} AUD**")
