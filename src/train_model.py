import os
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

from xgboost import XGBClassifier

from features import add_features
from pipeline import preprocessor

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/transactions.csv"

DROP_COLUMNS = [
    "is_fraud",
    "ts",
    "tx_id",
    "merchant_id_hash",
    "card_id_hash"
]


def choose_best_threshold(y_true, probabilities, fn_cost=5000, fp_cost=50):
    best_threshold = 0.5
    best_cost = float("inf")

    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        cost = (fn * fn_cost) + (fp * fp_cost)

        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    return best_threshold, best_cost


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


def train():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    print("Adding features...")
    df = add_features(df)
    df = prepare_features(df)

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    valid_df = df.iloc[split_index:]

    X_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df["is_fraud"]

    X_valid = valid_df.drop(columns=DROP_COLUMNS)
    y_valid = valid_df["is_fraud"]

    print("\nTraining columns:")
    print(X_train.columns.tolist())

    print("\nTraining data types:")
    print(X_train.dtypes)

    print("\nClass distribution:")
    print(y_train.value_counts())

    pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=500,
                class_weight="balanced"
            ))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    best_model_name = None
    best_model = None
    best_pr_auc = -1
    best_probabilities = None

    report_lines = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_valid)[:, 1]

        pr_auc = average_precision_score(y_valid, probabilities)
        roc_auc = roc_auc_score(y_valid, probabilities)

        print(f"{name} PR-AUC: {pr_auc:.4f}")
        print(f"{name} ROC-AUC: {roc_auc:.4f}")

        report_lines.append(f"\n{name}")
        report_lines.append(f"PR-AUC: {pr_auc:.4f}")
        report_lines.append(f"ROC-AUC: {roc_auc:.4f}")

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_model = model
            best_probabilities = probabilities

    threshold, cost = choose_best_threshold(y_valid, best_probabilities)

    final_predictions = (best_probabilities >= threshold).astype(int)

    final_report = classification_report(y_valid, final_predictions)
    final_cm = confusion_matrix(y_valid, final_predictions)

    print("\nBest Model:", best_model_name)
    print("Best PR-AUC:", round(best_pr_auc, 4))
    print("Best Threshold:", round(threshold, 3))
    print("Estimated Cost:", int(cost))

    print("\nClassification Report:")
    print(final_report)

    print("\nConfusion Matrix:")
    print(final_cm)

    joblib.dump(
        {
            "model": best_model,
            "threshold": float(threshold),
            "model_name": best_model_name
        },
        "models/fraud_model.joblib"
    )

    with open("outputs/model_report.txt", "w", encoding="utf-8") as f:
        f.write("Credit Card Fraud Detection Model Report\n")
        f.write("=" * 50)
        f.write("\n\n")
        f.write("\n".join(report_lines))
        f.write("\n\nBest Model: " + best_model_name)
        f.write("\nBest PR-AUC: " + str(round(best_pr_auc, 4)))
        f.write("\nBest Threshold: " + str(round(threshold, 3)))
        f.write("\nEstimated Cost: " + str(int(cost)))
        f.write("\n\nClassification Report:\n")
        f.write(final_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(final_cm))

    print("\nModel saved at models/fraud_model.joblib")
    print("Report saved at outputs/model_report.txt")


if __name__ == "__main__":
    train()