import streamlit as st
from PIL import Image
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the trained model and scaler
try:
    with open("dtreeModel.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure they are in the correct location.")
    st.stop()

# Streamlit UI
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1563291074-2bf8677ac0e5?q=80&w=2014&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center;
}
div.stButton > button {
    background-color: orange !important;  /* Adjust button color if needed */
    border-radius: 8px;
    font-weight: bold;
}

/* Make all text black */
.stApp label, .stApp p, .stApp .stSelectbox, .stApp .stNumberInput {
    color: black !important;
}

/* Make headers black */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: black !important;
}

</style>
'''


st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("Heart Disease Prediction Model")
st.write("This app predicts the risk of heart disease based on various health factors.")

# User Inputs
age = st.number_input("Enter Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Select Gender", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL?", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
slope = st.selectbox("ST Segment Slope", ["Upsloping", "Flat", "Downsloping"])
ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia Type", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert Inputs to Numerical Values
sex = 1 if sex == "Male" else 0
cp = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}[cp]
fbs = 1 if fbs == "Yes" else 0
restecg = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
exang = 1 if exang == "Yes" else 0
slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
thal = {"Normal": 2, "Fixed Defect": 1, "Reversible Defect": 3}[thal]

# Create a numpy array for the model
user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Prediction Button
if st.button("Predict"):
    try:
        # Scale the input
        scaled_input = scaler.transform(user_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)
        
        result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
        st.write(f"### Prediction: {result}")
        
        # Add interpretation
        st.write("### Interpretation:")
        st.write("This prediction is based on the provided health factors. A 'High Risk' result suggests a higher likelihood of heart disease, while a 'Low Risk' result indicates a lower likelihood. However, this is not a definitive diagnosis. Please consult with a healthcare professional for proper medical advice and interpretation of these results.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add disclaimer
st.write("### Disclaimer:")
st.write("This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
