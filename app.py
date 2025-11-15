# ---
# app.py (Your Streamlit Web App)
# ---

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD THE TRAINED MODEL AND SCALER ---
# Load the model
try:
    model = joblib.load('diabetes_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'diabetes_model.joblib' is in the same directory.")
    st.stop()

# Load the scaler
try:
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Scaler file not found. Please ensure 'scaler.joblib' is in the same directory.")
    st.stop()


# --- 2. DEFINE THE APP INTERFACE ---
st.set_page_config(page_title="DiaPredict", layout="wide")
st.title("🤖 AI-Powered Diabetes Prediction")
st.write("This application serves as a Clinical Decision Support System (CDSS) for the early detection of Type 2 Diabetes. It leverages an AI model to generate a risk score based on 8 key health indicators, aiding in the timely identification of high-risk patients.")

# --- 3. CREATE THE INPUT FORM (in the sidebar) ---
st.sidebar.header("Patient Data Input")

# Use a function to get user input for cleaner code
def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3) # (label, min, max, default)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 117)
    BloodPressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    SkinThickness = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 30)
    BMI = st.sidebar.slider('BMI (weight in kg/(height in m)^2)', 0.0, 67.1, 32.0)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age (years)', 21, 81, 29)

    # Store in a dictionary
    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- 4. PREPROCESS INPUT, PREDICT, AND DISPLAY RESULTS ---

# Display the user's input data
st.subheader("Patient Input Features")
st.write(input_df)

# Create a "Predict" button
if st.sidebar.button("Predict Diabetes Risk"):
    
    # Preprocess the input data using the loaded scaler
    # Note: We must scale the data just as we did in training
    try:
        input_scaled = scaler.transform(input_df)
    
        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
    
        st.subheader("Prediction Result")
    
        # Display the result
        if prediction[0] == 1:
            st.error(f"**Result: High Risk of Diabetes** (Confidence: {prediction_proba[0][1]*100:.2f}%)")
            st.warning("Please consult a medical professional for a formal diagnosis and advice.")
        else:
            st.success(f"**Result: Low Risk of Diabetes** (Confidence: {prediction_proba[0][0]*100:.2f}%)")
            st.info("It's always a good practice to maintain a healthy lifestyle and have regular check-ups.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project by:** [Anant Singh E22CSEU0884]")
st.sidebar.markdown("**Guide:** [Dr. Neeraj Jain]")