import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

os.makedirs("data", exist_ok=True)

N = 50000

start_date = datetime(2025, 1, 1)

transaction_ids = [f"TXN_{i:06d}" for i in range(N)]
timestamps = [start_date + timedelta(minutes=i * 5) for i in range(N)]

amount = np.random.exponential(scale=2500, size=N)
amount = np.round(amount, 2)

merchant_categories = np.random.choice(
    ["grocery", "fuel", "shopping", "travel", "electronics", "restaurant", "gaming"],
    size=N,
    p=[0.25, 0.15, 0.20, 0.08, 0.10, 0.17, 0.05]
)

cities = np.random.choice(
    ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Unknown"],
    size=N,
    p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
)

countries = np.random.choice(
    ["India", "India", "India", "India", "Foreign"],
    size=N,
    p=[0.35, 0.30, 0.20, 0.10, 0.05]
)

device_types = np.random.choice(
    ["mobile", "desktop", "pos", "unknown"],
    size=N,
    p=[0.55, 0.20, 0.20, 0.05]
)

channels = np.random.choice(
    ["online", "offline", "atm", "wallet"],
    size=N,
    p=[0.45, 0.30, 0.10, 0.15]
)

hour = np.array([t.hour for t in timestamps])
dayofweek = np.array([t.weekday() for t in timestamps])

prev_24h_tx_count_card = np.random.poisson(lam=3, size=N)
prev_24h_amt_card = prev_24h_tx_count_card * np.random.exponential(scale=2000, size=N)
prev_1h_tx_count_card = np.random.poisson(lam=1, size=N)
velocity_amt_1h = prev_1h_tx_count_card * np.random.exponential(scale=1500, size=N)

is_international = countries == "Foreign"
is_night = (hour >= 0) & (hour <= 5)

fraud_score = (
    (amount > 12000).astype(int) * 0.25
    + (is_international.astype(int)) * 0.25
    + (is_night.astype(int)) * 0.20
    + (prev_1h_tx_count_card > 3).astype(int) * 0.15
    + (velocity_amt_1h > 10000).astype(int) * 0.20
    + (cities == "Unknown").astype(int) * 0.15
    + (device_types == "unknown").astype(int) * 0.15
)

fraud_probability = fraud_score + np.random.normal(0, 0.08, N)
is_fraud = (fraud_probability > 0.55).astype(int)

# Force imbalance if fraud is too high
fraud_indices = np.where(is_fraud == 1)[0]
if len(fraud_indices) > int(0.03 * N):
    keep_fraud = np.random.choice(fraud_indices, size=int(0.03 * N), replace=False)
    is_fraud[:] = 0
    is_fraud[keep_fraud] = 1

df = pd.DataFrame({
    "tx_id": transaction_ids,
    "ts": timestamps,
    "amount": amount,
    "merchant_cat": merchant_categories,
    "merchant_id_hash": [f"M_{np.random.randint(1000,9999)}" for _ in range(N)],
    "card_id_hash": [f"C_{np.random.randint(10000,99999)}" for _ in range(N)],
    "city": cities,
    "country": countries,
    "device_type": device_types,
    "channel": channels,
    "hour": hour,
    "dayofweek": dayofweek,
    "prev_24h_tx_count_card": prev_24h_tx_count_card,
    "prev_24h_amt_card": np.round(prev_24h_amt_card, 2),
    "prev_1h_tx_count_card": prev_1h_tx_count_card,
    "velocity_amt_1h": np.round(velocity_amt_1h, 2),
    "is_international": is_international,
    "is_night": is_night,
    "is_fraud": is_fraud
})

df.to_csv("data/transactions.csv", index=False)

print("Synthetic dataset created successfully!")
print("Shape:", df.shape)
print("Fraud rate:", round(df["is_fraud"].mean() * 100, 2), "%")
print("Saved at: data/transactions.csv")