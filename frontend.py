import streamlit as st
import requests

st.title("Student Depression Prediction")

st.sidebar.header("Enter Student Features")

features = {
    "Gender_Label": st.sidebar.selectbox("Gender (0=Female, 1=Male)", [0, 1]),
    "Age": st.sidebar.number_input("Age"),
    "Financial_Stress": st.sidebar.number_input("Financial Stress (1-5)", min_value=1, max_value=5, step=1),
    "Academic_Pressure": st.sidebar.number_input("Academic Pressure (1-5)", min_value=1, max_value=5, step=1),
    "CGPA": st.sidebar.number_input("CGPA (1-10)", min_value=1.0, max_value=10.0, step=0.01),
    "Study_Satisfaction": st.sidebar.slider("Study Satisfaction (1-10)", 1, 10),
    "Work_Study_Hours": st.sidebar.number_input("Work/Study Hours (1-12)", min_value=1, max_value=12, step=1),
    "Degree_Label": st.sidebar.selectbox(
        "Degree",
        [
            "B.Arch",
            "B.Com",
            "B.Ed",
            "B.Pharm",
            "B.Tech",
            "BA",
            "BBA",
            "BCA",
            "BE",
            "BHM",
            "BSc",
            "Class 12",
            "LLB",
            "LLM",
            "M.Com",
            "M.Ed",
            "M.Pharm",
            "M.Tech",
            "MA",
            "MBA",
            "MBBS",
            "MCA",
            "MD",
            "ME",
            "MHM",
            "MSc",
            "Others",
            "PhD",
        ],
    ),
    "Dietary_Habits_label": st.sidebar.selectbox(
        "Dietary Habits (0=Unhealthy, 1=Moderate, 2=Healthy)", [0, 1, 2]
    ),
    "Family_History_of_Mental_Illness": st.sidebar.selectbox(
        "Family History of Mental Illness (0=No, 1=Yes)", [0, 1]
    ),
    "Suicidal_Thoughts": st.sidebar.selectbox("Suicidal Thoughts (0=No, 1=Yes)", [0, 1]),
    "Sleep_Duration_Label": st.sidebar.selectbox(
        "Sleep Duration",
        [
            "Less than 5 hours",
            "5-6 hours",
            "7-8 hours",
            "More than 8 hours",
            "Others",
        ],
    ),
}

# Map the Degree and Sleep Duration inputs to their corresponding labels/values
features["Degree_Label"] = {
    "B.Arch": 0,
    "B.Com": 1,
    "B.Ed": 2,
    "B.Pharm": 3,
    "B.Tech": 4,
    "BA": 5,
    "BBA": 6,
    "BCA": 7,
    "BE": 8,
    "BHM": 9,
    "BSc": 10,
    "Class 12": 11,
    "LLB": 12,
    "LLM": 13,
    "M.Com": 14,
    "M.Ed": 15,
    "M.Pharm": 16,
    "M.Tech": 17,
    "MA": 18,
    "MBA": 19,
    "MBBS": 20,
    "MCA": 21,
    "MD": 22,
    "ME": 23,
    "MHM": 24,
    "MSc": 25,
    "Others": 26,
    "PhD": 27,
}[features["Degree_Label"]]

features["Sleep_Duration_Label"] = {
    "Less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "More than 8 hours": 4,
    "Others": 5,
}[features["Sleep_Duration_Label"]]

if st.button("Predict"):
    response = requests.post("http://127.0.0.1:8000/predict/", json=features)
    if response.status_code == 200:
        result = response.json()
        st.subheader("Prediction Result")
        st.write("**Depressed:**" if result["depressed"] else "**Not Depressed:**")
        st.write(f"**Probability:** {result['probability']:.2f}")
    else:
        st.error("Failed to connect to the prediction API.")
