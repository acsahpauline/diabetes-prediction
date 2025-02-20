import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the trained model
model = pickle.load(open('diabetes_model.sav', 'rb'))

def predict_diabetes(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

# Streamlit UI
st.set_page_config(page_title='Diabetes Prediction', layout='centered')
st.title('🔍 Diabetes Prediction App')

st.markdown("""
    ### Enter Patient Details Below
    **Fill in the required values to check for diabetes prediction.**
""")

# Input fields with better UI
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1, step=1)
    glucose = st.slider('Glucose Level', min_value=0, max_value=200, value=100)
    blood_pressure = st.slider('Blood Pressure', min_value=0, max_value=150, value=80)
    skin_thickness = st.slider('Skin Thickness', min_value=0, max_value=100, value=20)

with col2:
    insulin = st.slider('Insulin Level', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.01)
    age = st.slider('Age', min_value=1, max_value=120, value=30)

# Prediction Button
if st.button('Predict 🔮'):
    result = predict_diabetes([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    if result == 'Diabetic':
        st.error(f'⚠️ The patient is **{result}**')
    else:
        st.success(f'✅ The patient is **{result}**')

# Data Visualization
st.markdown("---")
st.subheader("📊 Data Insights")

if st.checkbox("Show Dataset Summary"):
    df = pd.read_csv("diabetes.csv")
    st.write(df.describe())

if st.checkbox("Show Correlation Heatmap"):
    df = pd.read_csv("diabetes.csv")
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
