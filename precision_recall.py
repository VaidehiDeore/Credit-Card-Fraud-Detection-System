import os
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Import your feature pipeline
sys.path.append(os.path.abspath("src"))
from features import add_features

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("data/transactions.csv")

# Apply same feature engineering
df = add_features(df)

# Split data
X = df.drop(columns=["is_fraud", "ts", "tx_id", "merchant_id_hash", "card_id_hash"])
y = df["is_fraud"]

# Load model
bundle = joblib.load("models/fraud_model.joblib")
model = bundle["model"]

# Get probabilities (IMPORTANT)
y_scores = model.predict_proba(X)[:, 1]

# Compute Precision-Recall
precision, recall, _ = precision_recall_curve(y, y_scores)
ap_score = average_precision_score(y, y_scores)

# Plot
plt.figure(figsize=(8,6))
plt.plot(recall, precision, color="purple", linewidth=2)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (AP = {ap_score:.2f})")

plt.grid(True)
plt.savefig("images/precision_recall_curve.png", bbox_inches="tight")
plt.show()

print("Saved at images/precision_recall_curve.png")