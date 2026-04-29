from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "merchant_cat",
    "city",
    "country",
    "device_type",
    "channel",
    "is_international",
    "is_night",
    "is_weekend",
    "merchant_cat_rare"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ],
    remainder="drop"
)