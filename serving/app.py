import sys
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Add src folder path
sys.path.append(os.path.abspath("src"))

from features import add_features

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="FastAPI backend for real-time and batch fraud detection",
    version="1.0"
)

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/fraud_model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
threshold = bundle["threshold"]
model_name = bundle["model_name"]


class Transaction(BaseModel):
    amount: float
    merchant_cat: str
    merchant_id_hash: str = "M_0000"
    card_id_hash: str = "C_00000"
    city: str
    country: str
    device_type: str
    channel: str
    hour: int
    dayofweek: int
    prev_24h_tx_count_card: float
    prev_24h_amt_card: float
    prev_1h_tx_count_card: float
    velocity_amt_1h: float
    is_international: bool
    is_night: bool


def prepare_features(df):
    df = df.copy()

    string_columns = [
        "merchant_cat",
        "city",
        "country",
        "device_type",
        "channel"
    ]

    for col in string_columns:
        df[col] = df[col].fillna("Unknown").astype(str)

    bool_columns = [
        "is_international",
        "is_night"
    ]

    for col in bool_columns:
        df[col] = df[col].fillna(False).astype(int)

    integer_columns = [
        "is_weekend",
        "merchant_cat_rare"
    ]

    for col in integer_columns:
        df[col] = df[col].fillna(0).astype(int)

    numeric_columns = [
        "amount",
        "log_amount",
        "prev_24h_tx_count_card",
        "prev_24h_amt_card",
        "prev_1h_tx_count_card",
        "velocity_amt_1h",
        "avg_tx_amt_24h",
        "velocity_ratio",
        "hour",
        "dayofweek"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "model": model_name,
        "threshold": round(float(threshold), 3)
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": model_name
    }


@app.post("/score")
def score_transactions(transactions: List[Transaction]):
    input_data = [tx.model_dump() for tx in transactions]

    df = pd.DataFrame(input_data)
    df = add_features(df)
    df = prepare_features(df)

    probabilities = model.predict_proba(df)[:, 1]

    results = []

    for i, probability in enumerate(probabilities):
        probability = float(probability)
        prediction = int(probability >= threshold)

        if probability >= 0.80:
            risk_level = "HIGH"
        elif probability >= 0.50:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        results.append({
            "transaction_index": i,
            "fraud_probability": round(probability, 4),
            "fraud_probability_percent": round(probability * 100, 2),
            "prediction": "Fraud" if prediction == 1 else "Non-Fraud",
            "decision": "REVIEW" if prediction == 1 else "ALLOW",
            "risk_level": risk_level,
            "threshold": round(float(threshold), 3)
        })

    return {
        "model": model_name,
        "total_transactions": len(results),
        "results": results
    }


@app.post("/stream")
def stream_transaction(transaction: Transaction):
    df = pd.DataFrame([transaction.model_dump()])
    df = add_features(df)
    df = prepare_features(df)

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= threshold)

    if probability >= 0.80:
        risk_level = "HIGH"
    elif probability >= 0.50:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(probability, 4),
        "fraud_probability_percent": round(probability * 100, 2),
        "prediction": "Fraud" if prediction == 1 else "Non-Fraud",
        "decision": "REVIEW" if prediction == 1 else "ALLOW",
        "risk_level": risk_level,
        "message": "Streaming hook simulated successfully"
    }