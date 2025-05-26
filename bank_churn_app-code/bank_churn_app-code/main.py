from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

class CustomerData(BaseModel):
    CreditScore: float
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.post("/predict")
def predict_churn(data: CustomerData):
    input_data = np.array([[data.CreditScore, data.Gender, data.Age, data.Tenure,
                            data.Balance, data.NumOfProducts, data.HasCrCard,
                            data.IsActiveMember, data.EstimatedSalary]])
    input_scaled = scaler.transform(input_data)
    prob = model.predict(input_scaled)[0][0]
    return {"churn_probability": float(prob), "churn": int(prob > 0.5)}
