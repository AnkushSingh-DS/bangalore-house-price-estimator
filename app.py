
import streamlit as st
import numpy as np
import pickle

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('banglore_home_prices_model.pickle', 'rb') as file:
        return pickle.load(file)

model = load_model()

# List of available locations
locations = [
    "1st block jayanagar", "1st phase jp nagar", "2nd phase judicial layout",
    "2nd stage nagarbhavi", "5th block hbr layout", "5th phase jp nagar",
    "6th phase jp nagar", "7th phase jp nagar", "8th phase jp nagar",
    "9th phase jp nagar", "aecs layout", "abbigere"
]  # Trimmed for readability; use full list in actual code

# Mapping locations to numerical indices
location_to_index = {loc: i for i, loc in enumerate(locations)}

def predict_price(location, sqft, bath, bhk):
    """Predict house price based on input parameters."""
    x = np.zeros(len(locations) + 3)  # Adjust feature vector size dynamically
    x[0], x[1], x[2] = sqft, bath, bhk

    # Encode the location
    loc_index = location_to_index.get(location)
    if loc_index is not None:
        x[loc_index + 3] = 1

    return round(model.predict([x])[0], 2)

def main():
    st.title("🏠 Bangalore House Price Prediction")
    st.write("Enter the details below to get an estimated house price.")

    selected_location = st.selectbox("📍 Select a location", locations)
    sqft = st.text_input("📏 Square Feet Area", "")
    bath = st.text_input("🚿 Number of Bathrooms", "")
    bhk = st.text_input("🛏 Number of BHK", "")

    if st.button("💰 Predict Price"):
        try:
            sqft, bath, bhk = float(sqft), float(bath), float(bhk)
            price = predict_price(selected_location, sqft, bath, bhk)
            st.success(f"🏡 The estimated house price is **₹{price} lakhs**.")
        except ValueError:
            st.error("⚠️ Please enter valid numeric values for Sq-ft, Bathrooms, and BHK.")

    st.sidebar.markdown("### About")
    st.sidebar.info("This app predicts Bangalore house prices using a trained model.")

if __name__ == "__main__":
    main()
