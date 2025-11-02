import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load('final_model_pipeline.pkl')

st.title("Diabetes Prediction Web App")
st.write("Enter your health details to predict diabetes risk.")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=100.0,step=0.01)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0,step=0.01)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0,step=0.01)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5,step=0.01, format="%.3f")
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    user_input = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]],
                              columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                       'Insulin','BMI','DiabetesPedigreeFunction','Age'])
    
    prediction = pipeline.predict(user_input)[0]
    probability = pipeline.predict_proba(user_input)[0][1]
    
    threshold = 0.55
    if probability >= threshold:
        st.error(f"The model predicts DIABETES with probability {probability:.2f}")
    else:
        st.success(f"The model predicts NO DIABETES with probability {probability:.2f}")
