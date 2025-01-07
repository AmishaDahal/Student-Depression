import streamlit as st
import requests

st.title("Student Depression Prediction")

st.sidebar.header("Enter Student Features")

features = {
    "Gender_Label": st.sidebar.selectbox("Gender (0=Male, 1=Female)", [0, 1]),
    "Age": st.sidebar.number_input("Age"),
    "Financial_Stress": st.sidebar.slider("Financial Stress (1-10)", 1, 10),
    "Academic_Pressure": st.sidebar.slider("Academic Pressure (1-10)", 1, 10),
    "CGPA": st.sidebar.number_input("CGPA"),
    "Study_Satisfaction": st.sidebar.slider("Study Satisfaction (1-10)", 1, 10),
    "Work_Study_Hours": st.sidebar.number_input("Work/Study Hours"),
    "Degree_Label": st.sidebar.selectbox("Degree (0=Undergrad, 1=Postgrad)", [0, 1]),
    "Dietary_Habits_label": st.sidebar.selectbox("Dietary Habits (0=Unhealthy, 1=Healthy)", [0, 1]),
    "Family_History_of_Mental_Illness": st.sidebar.selectbox("Family History of Mental Illness (0=No, 1=Yes)", [0, 1]),
    "Suicidal_Thoughts": st.sidebar.selectbox("Suicidal Thoughts (0=No, 1=Yes)", [0, 1]),
    "Sleep_Duration_Label": st.sidebar.slider("Sleep Duration (Hours)", 0, 24)
}

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict/", json=features)
    if response.status_code == 200:
        result = response.json()
        st.subheader("Prediction Result")
        st.write("**Depressed:**" if result["depressed"] else "**Not Depressed:**")
        st.write(f"**Probability:** {result['probability']:.2f}")
    else:
        st.error("Failed to connect to the prediction API.")
