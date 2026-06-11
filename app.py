from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained pipeline model
model = joblib.load("xgb_model.pkl")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predict fraudulent credit card transactions",
    version="1.0"
)

class TransactionData(BaseModel):
    transaction_id: int
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int

@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API Running Successfully"
    }

@app.post("/predict")
def predict(data: TransactionData):
    try:

        input_data = pd.DataFrame([{
            "transaction_id": data.transaction_id,
            "amount": data.amount,
            "transaction_hour": data.transaction_hour,
            "merchant_category": data.merchant_category,
            "foreign_transaction": data.foreign_transaction,
            "location_mismatch": data.location_mismatch,
            "device_trust_score": data.device_trust_score,
            "velocity_last_24h": data.velocity_last_24h,
            "cardholder_age": data.cardholder_age
        }])

        prediction = model.predict(input_data)[0]

        result = "Fraud" if prediction == 1 else "Not Fraud"

        return {
            "prediction": int(prediction),
            "result": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }