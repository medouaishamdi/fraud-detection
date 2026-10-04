from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

from config import MODEL_PATH


# ============================================================
# AI FRAUD DETECTION API
# STEP 7 — FASTAPI
# ============================================================


# ============================================================
# 1. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Fraud Detection API",
    description=(
        "Machine Learning API for detecting "
        "potentially fraudulent transactions."
    ),
    version="1.0.0"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================



if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. Run `python src/train.py` first."
    )


model = joblib.load(MODEL_PATH)


# ============================================================
# 3. TRANSACTION DATA MODEL
# ============================================================

class Transaction(BaseModel):

    amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount"
    )

    transaction_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Transaction hour (0-23)"
    )

    distance_from_home: float = Field(
        ...,
        ge=0,
        description="Distance from home in kilometers"
    )

    merchant_risk: float = Field(
        ...,
        ge=0,
        le=1,
        description="Merchant risk score from 0 to 1"
    )

    device_trusted: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 = trusted device, 0 = unknown device"
    )

    previous_transactions: int = Field(
        ...,
        ge=0,
        description="Number of previous transactions"
    )


# ============================================================
# 4. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Fraud Detection API is running",
        "version": "1.0.0",
        "status": "online"
    }


# ============================================================
# 5. HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# 6. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_fraud(
    transaction: Transaction
):

    # Convert request into DataFrame
    data = pd.DataFrame([
        {
            "amount": transaction.amount,
            "transaction_hour":
                transaction.transaction_hour,
            "distance_from_home":
                transaction.distance_from_home,
            "merchant_risk":
                transaction.merchant_risk,
            "device_trusted":
                transaction.device_trusted,
            "previous_transactions":
                transaction.previous_transactions
        }
    ])


    # Make prediction
    prediction = model.predict(
        data
    )[0]


    # Calculate probability
    probability = model.predict_proba(
        data
    )[0][1]


    # Determine risk level
    if probability >= 0.75:

        risk_level = "HIGH"

    elif probability >= 0.40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # Return JSON response
    return {

        "is_fraud": bool(prediction),

        "fraud_probability":
            round(
                float(probability),
                4
            ),

        "fraud_probability_percent":
            round(
                float(probability * 100),
                2
            ),

        "risk_level": risk_level

    }


# ============================================================
# 7. INFORMATION ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "model": "Random Forest",

        "purpose":
            "Credit card transaction fraud detection",

        "features": [

            "amount",

            "transaction_hour",

            "distance_from_home",

            "merchant_risk",

            "device_trusted",

            "previous_transactions"

        ],

        "model_file":
            MODEL_PATH.name

    }


# ============================================================
# APPLICATION STARTUP
# ============================================================

print("=" * 60)
print("AI FRAUD DETECTION API")
print("=" * 60)

print("\nModel loaded successfully!")

print(
    f"Model path: {os.path.abspath(MODEL_PATH)}"
)

print("\nAvailable endpoints:")

print("GET  /")
print("GET  /health")
print("GET  /model-info")
print("POST /predict")

print("\nAPI ready!")