from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("student_depression_model.h5")

# Load the scaler used during training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class StudentFeatures(BaseModel):
    Gender_Label: int
    Age: int
    Financial_Stress: float
    Academic_Pressure: float
    CGPA: float
    Study_Satisfaction: float
    Work_Study_Hours: float
    Degree_Label: int
    Dietary_Habits_label: int
    Family_History_of_Mental_Illness: int
    Suicidal_Thoughts: int
    Sleep_Duration_Label: int

@app.post("/predict/")
def predict(features: StudentFeatures):
    data = np.array([[
        features.Gender_Label, features.Age, features.Financial_Stress, 
        features.Academic_Pressure, features.CGPA, features.Study_Satisfaction, 
        features.Work_Study_Hours, features.Degree_Label, 
        features.Dietary_Habits_label, features.Family_History_of_Mental_Illness, 
        features.Suicidal_Thoughts, features.Sleep_Duration_Label
    ]])
    data_scaled = scaler.transform(data)
    data_cnn = data_scaled.reshape(1, 3, 4, 1)
    prediction = model.predict(data_cnn)
    probability = float(prediction[0][0])
    depressed = probability > 0.5
    return {"depressed": depressed, "probability": probability}

