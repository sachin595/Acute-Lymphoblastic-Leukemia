import streamlit as st
import pandas as pd
import numpy as np

import os
import gdown
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from keras import backend as K

def download_model():
    try:
        if not os.path.exists('komati_rf.joblib'):
            url = 'https://drive.google.com/uc?id=1AoTJzqwAiF1yM4MXDklHRUHGw-68-ZMN'
            gdown.download(url, 'komati_rf.joblib', quiet=False)
        if not os.path.exists('myscale_rf.joblib'):
            url = 'https://drive.google.com/uc?id=1Ix83VnEu8l5gGsND_HygxLhCoMfcP2Fe'
            gdown.download(url, 'myscale_rf.joblib', quiet=False)
    except Exception as e:
        st.write(f"Failed to download the model or scaler: {e}")

download_model()
try:
    model = load('komati_rf.joblib')
except Exception as e:
    st.write('Model not loaded:', e)

try:
    scaler = load('myscale_rf.joblib')
except Exception as e:
    st.write('Scaler not loaded:', e)



def predict(inputs):
    
    # Mapping dictionaries
    sex_map = {'Male': 0, 'Female': 1}
    ethnicity_map = {'Hispanic': 1, 'Non-Hispanic': 0}
    race_map = {'White': 1, 'Black': 0}
    agegroup_map = {
                    '1-4': 0,
                    '5-9': 1,
                    '10-14': 2,
                    '15-19': 3,
                    '20-24': 4,
                    '25-29': 5,
                    '30-34': 6,
                    '35-39': 7,
                    '40-44': 8,
                    '45-49': 9,
                    '50-54': 10,
                    '55-59': 11,
                    '60-64': 12,
                    '65-69': 13,
                    '70-74': 14,
                    '75-79': 15,
                    '80-84': 16,
                    '85+': 17
                    }
    
    
    # Apply mappings
    inputs['Sex'] = sex_map[inputs['Sex']]
    inputs['Ethnicity'] = ethnicity_map[inputs['Ethnicity']]
    inputs['Race'] = race_map[inputs['Race']]
    inputs['AgeGroup'] = agegroup_map[inputs['AgeGroup']]
    
      # Assuming the model expects a DataFrame with the same structure as during training
    input_df = pd.DataFrame([inputs])

    # Standardize the input data
    input_df_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_df_scaled)
    return prediction



# Streamlit user interface
st.title('Predictive Analytics for Acute Lymphoblastic Leukemia')

# Creating form for input
with st.form(key='prediction_form'):
    sex = st.selectbox('Sex', options=['Male', 'Female'])
    year = st.number_input('Year', min_value=1900, max_value=2999)
    agegroup = st.selectbox('Age Group', options=[
        '1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', 
        '60-64', '65-69', '70-74', '75-79', '80-84', '85+'
    ])
    ethnicity = st.selectbox('Ethnicity', options=['Hispanic', 'Non-Hispanic'])
    race =  st.selectbox('Race', options=['White','Black'])
    
    submit_button = st.form_submit_button(label='Predict')

# Processing prediction
if submit_button:
    input_data = {
        'Sex': sex,
        'Year':year,
        'AgeGroup': agegroup, 
        'Ethnicity': ethnicity,
        'Race': race,
               
    }
    # Predict and decode
    y_pred = predict(input_data)  # Ensure predict() returns the appropriate numeric predictions
    #decoded_predictions = decode_predictions(y_pred.flatten())
    # Create a new array for the rounded predictions
    y_predicted_rounded = np.zeros_like(y_pred)

    # Round each column of y_pred separately to different decimal places
    y_predicted_rounded[:, 0] = np.round(y_pred[:, 0], 1)  # Round "Crude Rate" to 1 decimal place
    y_predicted_rounded[:, 1] = np.round(y_pred[:, 1], 6)  # Round "Survival Rate" to 9 decimal places
    st.write('Crude Mortality Rate:', y_pred[:,0])
    st.write('Survival Rate:', y_pred[:,1])
  