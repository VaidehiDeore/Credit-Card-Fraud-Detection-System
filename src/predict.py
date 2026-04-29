import joblib
import pandas as pd
from features import add_features

bundle = joblib.load("models/fraud_model.joblib")

model = bundle["model"]
threshold = bundle["threshold"]

def predict_transaction(transaction_data: dict):
    df = pd.DataFrame([transaction_data])
    df = add_features(df)

    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= threshold)

    result = {
        "fraud_probability": round(float(probability), 4),
        "prediction": "Fraud" if prediction == 1 else "Non-Fraud",
        "decision": "REVIEW" if prediction == 1 else "ALLOW",
        "threshold": round(float(threshold), 3)
    }

    return result

if __name__ == "__main__":
    sample_transaction = {
        "amount": 25000,
        "merchant_cat": "electronics",
        "merchant_id_hash": "M_1234",
        "card_id_hash": "C_56789",
        "city": "Unknown",
        "country": "Foreign",
        "device_type": "unknown",
        "channel": "online",
        "hour": 2,
        "dayofweek": 6,
        "prev_24h_tx_count_card": 8,
        "prev_24h_amt_card": 50000,
        "prev_1h_tx_count_card": 5,
        "velocity_amt_1h": 25000,
        "is_international": True,
        "is_night": True
    }

    print(predict_transaction(sample_transaction))