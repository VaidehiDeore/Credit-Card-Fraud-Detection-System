import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, precision_recall_curve
from features import add_features

os.makedirs("images", exist_ok=True)

DROP_COLUMNS = [
    "is_fraud",
    "ts",
    "tx_id",
    "merchant_id_hash",
    "card_id_hash"
]


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


def create_visuals():
    print("Loading dataset...")

    df = pd.read_csv(
        "data/transactions.csv",
        engine="python",
        on_bad_lines="skip"
    )

    # Use sample to avoid memory issues
    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    print("Adding features...")
    df = add_features(df)
    df = prepare_features(df)

    split_index = int(len(df) * 0.8)
    valid_df = df.iloc[split_index:]

    X_valid = valid_df.drop(columns=DROP_COLUMNS)
    y_valid = valid_df["is_fraud"]

    print("Loading model...")
    bundle = joblib.load("models/fraud_model.joblib")

    model = bundle["model"]
    threshold = bundle["threshold"]

    print("Making predictions...")
    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("Creating charts...")

    plt.figure(figsize=(6, 4))
    sns.countplot(x="is_fraud", data=df)
    plt.title("Fraud vs Non-Fraud Transactions")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig("images/class_distribution.png", bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_valid, predictions)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("images/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_valid, probabilities)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig("images/precision_recall_curve.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(df["amount"], bins=40)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.savefig("images/amount_distribution.png", bbox_inches="tight")
    plt.close()

    print("Visualizations saved successfully in images/ folder")


if __name__ == "__main__":
    create_visuals()