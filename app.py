import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ===============================
# 🎨 PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="🩺 Diabetes Risk Predictor",
    page_icon="💉",
    layout="centered",
)

# ===============================
# 🧠 LOAD TRAINED PIPELINE
# ===============================
pipeline = joblib.load(r"C:\Users\kanak\Desktop\ML_Model\diabetes_pred\final_model_pipelines.pkl")

# ===============================
# 🌈 SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Insights", "📚 About"])

st.sidebar.markdown("---")
st.sidebar.write("""
**About the App:**  
Predict your diabetes risk instantly using a trained ML model.  
Developed by **Amrita Singh** 👩‍💻  

**Model:** Gradient Boosting  
**Data:** Pima Indians Diabetes Dataset
""")


# HOME PAGE (PREDICTION)
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction Web App")
    st.subheader("Enter your health details below to check your diabetes risk.")

    with st.expander("ℹ️ Feature Explanations"):
        st.markdown("""
        - **Pregnancies:** Number of times pregnant  
        - **Glucose:** Blood sugar level (mg/dL)  
        - **Blood Pressure:** Diastolic pressure (mm Hg)  
        - **Skin Thickness:** Triceps skin fold thickness (mm)  
        - **Insulin:** Serum insulin level (μU/mL)  
        - **BMI:** Body Mass Index  
        - **DPF:** Family history impact factor  
        - **Age:** Age of the person  
        """)

    #  INPUT SECTION
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input("🤰 Pregnancies", 0, 20, 0)
            glucose = st.number_input("🍬 Glucose Level", 0.0, 300.0, 100.0, step=0.01)
            blood_pressure = st.number_input("💓 Blood Pressure", 0.0, 200.0, 70.0, step=0.01)
        with col2:
            skin_thickness = st.number_input("🧍 Skin Thickness", 0.0, 100.0, 20.0)
            insulin = st.number_input("💉 Insulin Level", 0.0, 900.0, 80.0)
            bmi = st.number_input("⚖️ BMI", 0.0, 70.0, 25.0, step=0.01)
        with col3:
            dpf = st.number_input("👨‍👩‍👧‍👦 DPF", 0.0, 2.5, 0.5, step=0.01, format="%.3f")
            age = st.number_input("🎂 Age", 0, 120, 30)

    #  PREDICT BUTTON
    st.markdown("---")
    if st.button("🔍 Predict My Risk"):
        user_input = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, dpf, age]],
                                  columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                           'Insulin','BMI','DiabetesPedigreeFunction','Age'])
        
        prediction = pipeline.predict(user_input)[0]
        probability = pipeline.predict_proba(user_input)[0][1]
        threshold = 0.55

        # 📊 Gauge Chart for Probability
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Diabetes Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if probability >= threshold else "green"},
                'steps': [
                    {'range': [0, 55], 'color': "#b3ffb3"},
                    {'range': [55, 80], 'color': "#fff2b3"},
                    {'range': [80, 100], 'color': "#ffb3b3"},
                ],
            }
        ))
        st.plotly_chart(fig)

        if probability >= threshold:
            st.error(f"⚠️ **High Risk**: The model predicts **DIABETES** with probability **{probability:.2f}**")
        else:
            st.success(f"✅ **Low Risk**: The model predicts **NO DIABETES** with probability **{probability:.2f}**")

        st.markdown("### 💡 Health Suggestions:")
        if probability >= 0.7:
            st.warning("""
            - 🚨 Very High Risk — Please consult a doctor soon  
            - 🍎 Adopt a low-sugar diet  
            - 🏃 Exercise daily  
            - 💧 Stay hydrated  
            - 🩸 Regular glucose monitoring  
            """)
        elif probability >= threshold:
            st.info("""
            - ⚠️ Moderate Risk — Watch your lifestyle  
            - 🍏 Eat fiber-rich foods  
            - 🏃 Walk 30 mins a day  
            - ❌ Avoid processed sugar  
            """)
        else:
            st.success("""
            - 🎉 Low Risk — Keep up the good habits!  
            - Maintain healthy BMI  
            - Stay active and do regular checkups  
            """)

        # 📥 Download Report
        result = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DPF": dpf,
            "Age": age,
            "Prediction": "Diabetes" if probability >= threshold else "No Diabetes",
            "Probability": round(probability, 3)
        }
        result_df = pd.DataFrame([result])
        st.download_button("📄 Download Report", result_df.to_csv(index=False), "diabetes_report.csv")


# 📈 INSIGHTS PAGE

elif page == "📈 Insights":
    st.title("📊 Model Insights")

    st.markdown("""
    ### 🔍 Model Evaluation Metrics
    | Metric | Value |
    |---------|-------|
    | Accuracy | **85%** |
    | ROC-AUC | **0.91** |
    | F1-Score | **0.78** |
    | Precision | **0.80** |
    | Recall | **0.76** |
    """)

    st.info("These metrics are based on the test dataset used during model evaluation.")

    st.markdown("---")
    st.markdown("### 📘 Feature Importance")
    st.write("""
    Typically, **Glucose**, **BMI**, and **Age** have the most impact  
    on diabetes prediction according to model analysis.
    """)

# ABOUT PAGE

elif page == "📚 About":
    st.title("👩‍💻 About the Developer")
    st.write("""
    **Amrita Singh**  
    B.Tech CSE (AIML) • 2nd Year • CPI 8.26  
    - 💡 Microsoft Azure Certified  
    - 👩‍💻 AI & ML Enthusiast | Data Team @ CSED Club  
    - 💭 Passionate about building intelligent, human-centered AI tools  
    """)
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & Scikit-learn.")
