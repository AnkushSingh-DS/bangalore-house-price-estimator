import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
from PIL import Image

# Load the pre-trained model
pickle_in = open('banglore_home_prices_model.pickle', 'rb')
model = pickle.load(pickle_in)

# List of available locations (Reduced list)
locations = ["Indira Nagar", "Whitefield", "Marathahalli", "Koramangala", "Jayanagar"]

# Mapping of locations to numerical values
location_to_index = {loc: i for i, loc in enumerate(locations)}

def predict_price(location, sqft, bath, bhk):
    x = np.zeros(len(locations) + 3)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    loc_index = location_to_index.get(location, -1)
    if loc_index != -1:
        x[loc_index + 3] = 1
    prediction = model.predict([x])[0]
    unit = "Cr" if prediction >= 100 else "Lakhs"
    return f"{round(prediction, 2)} {unit}"

# Streamlit UI
st.title("Bangalore House Price Prediction")
st.image(Image.open("bangalore.jpg"), use_column_width=True)

# User inputs
location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Total Square Feet Area", min_value=100.0, step=10.0)
bath = st.slider("Select Number of Bathrooms", 1, 10, 2)
bhk = st.slider("Select BHK", 1, 10, 3)

if st.button("Predict Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"Predicted Price: {price}")
