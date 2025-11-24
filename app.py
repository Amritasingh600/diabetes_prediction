from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

try:
    pipeline = joblib.load("final_model_pipelines.pkl")
except FileNotFoundError:
    pipeline = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({'error': 'Model not found'}), 500
    
    try:
        data = request.json
        user_input = pd.DataFrame([[
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['dpf']),
            float(data['age'])
        ]], columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                     'Insulin','BMI','DiabetesPedigreeFunction','Age'])
        
        prediction = pipeline.predict(user_input)[0]
        probability = pipeline.predict_proba(user_input)[0][1]
        
        risk_level = "High" if probability >= 0.55 else "Low"
        
        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability * 100, 1),
            'risk_level': risk_level
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
