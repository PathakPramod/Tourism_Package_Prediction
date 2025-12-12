import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="PPathakH18/tourism-prediction-model", filename="best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of purchasing the Wellness Tourism Package by a customer based on several parameters.
Please enter the data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=20)
number_of_person = st.number_input("Number of Person Visiting", min_value=1, max_value=5, value=1)
number_of_trip = st.number_input("Number of Trips", min_value=1, max_value=30, value=1)
contact_type = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
Tier = st.selectbox("City Tier", ["1", "2", "3"])
Occupation = st.selectbox("Customer Occupation", ["Salaried", "Free Lancer", "Large Business", "Small Business"])
Gender = st.selectbox("Customer Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Customer Martial Status", ["Married", "Single", "Divorced", "Unmarried"])
property_rating = st.selectbox("Preferred Hotel Rating", ["3", "4", "5"])
car = st.selectbox("Car Owner", ["0", "1"])
income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=1000)
designation = st.selectbox("Designation", ["AVP", "Manager", "Executive", "Senior Manager", "VP"])
passport = st.selectbox("Passport", ["0", "1"])
childvisit = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': contact_type,
    'CityTier': Tier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': number_of_person,
    'PreferredPropertyStar': property_rating,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': number_of_trip,
    'Passport': passport,
    'OwnCar': car,
    'NumberOfChildrenVisiting': childvisit,
    'Designation': designation,
    'MonthlyIncome': income
}])


if st.button("Predict Tourism Package"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Purchased" if prediction == 1 else "Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
