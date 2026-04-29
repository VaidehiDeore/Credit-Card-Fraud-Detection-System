import os
import sys
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.abspath("src"))
from features import add_features

os.makedirs("images", exist_ok=True)

DROP_COLUMNS = [
    "is_fraud",
    "ts",
    "tx_id",
    "merchant_id_hash",
    "card_id_hash"
]

df = pd.read_csv("data/transactions.csv")

# Add same features used during model training
df = add_features(df)

X = df.drop(columns=DROP_COLUMNS)
y = df["is_fraud"]

bundle = joblib.load("models/fraud_model.joblib")
model = bundle["model"]
threshold = bundle["threshold"]

probabilities = model.predict_proba(X)[:, 1]
y_pred = (probabilities >= threshold).astype(int)

cm = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Fraud", "Fraud"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Credit Card Fraud Detection")
plt.savefig("images/confusion_matrix.png", bbox_inches="tight")
plt.show()

print("Confusion matrix saved at images/confusion_matrix.png")