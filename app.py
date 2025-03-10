import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from PIL import Image

# Load the pre-trained model and necessary data
pickle_in = open('banglore_home_prices_model.pickle', 'rb')
classifier = pickle.load(pickle_in)

# List of available locations
locations = ["1st block jayanagar", "1st phase jp nagar", "2nd phase judicial layout", "2nd stage nagarbhavi", 
             "5th block hbr layout", "5th phase jp nagar", "6th phase jp nagar", "7th phase jp nagar", 
             "8th phase jp nagar", "9th phase jp nagar", "aecs layout", "abbigere", "akshaya nagar", "ambalipura",
             "ambedkar nagar", "amruthahalli", "anandapura", "ananth nagar", "anekal", "anjanapura", "ardendale", 
             "arekere", "attibele", "beml layout", "btm 2nd stage", "btm layout", "babusapalaya", "badavala nagar", 
             "balagere", "banashankari", "banashankari stage ii", "banashankari stage iii", "banashankari stage v", 
             "banashankari stage vi", "banaswadi", "banjara layout", "bannerghatta", "bannerghatta road", 
             "basavangudi", "basaveshwara nagar", "battarahalli", "begur", "begur road", "bellandur", "benson town", 
             "bharathi nagar", "bhoganhalli", "billekahalli", "binny pete", "bisuvanahalli", "bommanahalli", 
             "bommasandra", "bommasandra industrial area", "bommenahalli", "brookefield", "budigere", "cv raman nagar", 
             "chamrajpet", "chandapura", "channasandra", "chikka tirupathi", "chikkabanavar", "chikkalasandra", 
             "choodasandra", "cooke town", "cox town", "cunningham road", "dasanapura", "dasarahalli", "devanahalli", 
             "devarachikkanahalli", "dodda nekkundi", "doddaballapur", "doddakallasandra", "doddathoguru", "domlur", 
             "dommasandra", "epip zone", "electronic city", "electronic city phase ii", "electronics city phase 1", 
             "frazer town", "gm palaya", "garudachar palya", "giri nagar", "gollarapalya hosahalli", "gottigere", 
             "green glen layout", "gubbalala", "gunjur", "hal 2nd stage", "hbr layout", "hrbr layout", "hsr layout", 
             "haralur road", "harlur", "hebbal", "hebbal kempapura", "hegde nagar", "hennur", "hennur road", "hoodi", 
             "horamavu agara", "horamavu banaswadi", "hormavu", "hosa road", "hosakerehalli", "hoskote", "hosur road", 
             "hulimavu", "isro layout", "itpl", "iblur village", "indira nagar", "jp nagar", "jakkur", "jalahalli", 
             "jalahalli east", "jigani", "judicial layout", "kr puram", "kadubeesanahalli", "kadugodi", "kaggadasapura", 
             "kaggalipura", "kaikondrahalli", "kalena agrahara", "kalyan nagar", "kambipura", "kammanahalli", "kammasandra", 
             "kanakapura", "kanakpura road", "kannamangala", "karuna nagar", "kasavanhalli", "kasturi nagar", "kathriguppe", 
             "kaval byrasandra", "kenchenahalli", "kengeri", "kengeri satellite town", "kereguddadahalli", "kodichikkanahalli", 
             "kodigehaali", "kodigehalli", "kodihalli", "kogilu", "konanakunte", "koramangala", "kothannur", "kothanur", 
             "kudlu", "kudlu gate", "kumaraswami layout", "kundalahalli", "lb shastri nagar", "laggere", "lakshminarayana pura", 
             "lingadheeranahalli", "magadi road", "mahadevpura", "mahalakshmi layout", "mallasandra", "malleshpalya", 
             "malleshwaram", "marathahalli", "margondanahalli", "marsur", "mico layout", "munnekollal", "murugeshpalya", 
             "mysore road", "ngr layout", "nri layout", "nagarbhavi", "nagasandra", "nagavara", "nagavarapalya", 
             "narayanapura", "neeladri nagar", "nehru nagar", "ombr layout", "old airport road", "old madras road", 
             "padmanabhanagar", "pai layout", "panathur", "parappana agrahara", "pattandur agrahara", "poorna pragna layout", 
             "prithvi layout", "r.t. nagar", "rachenahalli", "raja rajeshwari nagar", "rajaji nagar", "rajiv nagar", 
             "ramagondanahalli", "ramamurthy nagar", "rayasandra", "sahakara nagar", "sanjay nagar", "sarakki nagar", 
             "sarjapur", "sarjapur  road", "sarjapura - attibele road", "sector 2 hsr layout", "sector 7 hsr layout", 
             "seegehalli", "shampura", "shivaji nagar", "singasandra", "somasundara palya", "sompura", "sonnenahalli", 
             "subramanyapura", "sultan palaya", "tc palaya", "talaghattapura", "thanisandra", "thigalarapalya", 
             "thubarahalli", "thyagaraja nagar", "tindlu", "tumkur road", "ulsoor", "uttarahalli", "varthur", 
             "varthur road", "vasanthapura", "vidyaranyapura", "vijayanagar", "vishveshwarya layout", "vishwapriya layout", 
             "vittasandra", "whitefield", "yelachenahalli", "yelahanka", "yelahanka new town", "yelenahalli", "yeshwanthpur"] 

# Mapping of locations to numerical values
location_to_index = {loc: i for i, loc in enumerate(locations)}

def Welcome():
    return 'WELCOME ALL!'

def predict_price(location, sqft, bath, bhk):
    """Predict house price based on input parameters."""
    x = np.zeros(244)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # Encode the selected location
    loc_index = location_to_index.get(location, -1)
    if loc_index != -1:
        x[loc_index + 3] = 1  # Starting from index 3 because of sqft, bath, and bhk

    return round(classifier.predict([x])[0], 2)

def main():
    st.title("Bangalore House Rate Prediction")
    st.markdown("<h3 style='text-align: center;'>Enter the required details to get the predicted house price.</h3>", unsafe_allow_html=True)

    selected_location = st.selectbox("**Select a location**", locations)
    
    # Form input for sqft, bath, and bhk
    col1, col2, col3 = st.columns(3)
    with col1:
        sqft = st.text_input("Sq-ft Area", "")
    with col2:
        bath = st.text_input("Number of Bathrooms", "")
    with col3:
        bhk = st.text_input("Number of BHK", "")

    result = ""
    
    # Handle user input validation
    if st.button("Predict House Price"):
        try:
            sqft = float(sqft)
            bath = float(bath)
            bhk = float(bhk)
            
            if sqft <= 0 or bath <= 0 or bhk <= 0:
                st.warning("Please enter positive values for Sq-ft, Bathrooms, and BHK.")
            else:
                result = predict_price(selected_location, sqft, bath, bhk)
                if result >= 100:
                    result_in_cr = result / 100
                    st.success(f'The predicted house price is {result_in_cr:.2f} crores')
                else:
                    st.success(f'The predicted house price is {result:.2f} lakhs')
        except ValueError:
            st.error("Please enter valid numeric values for Sq-ft, Bathrooms, and BHK.")

    # About section
    if st.button("About"):
        st.text("Please find the code at")
        st.text("https://github.com/AnkushSingh-DS/bangalore-house-price-estimator")

if __name__ == '__main__':
    main()
