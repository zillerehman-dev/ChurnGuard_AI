from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco_customer_churn.xlsx"
SHEET = "Telco_Churn"
TARGET = "Churn Value"

NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]
FEATURES = NUMERIC + CATEGORICAL

# Never model inputs: identifiers, target-derived fields, constant and geographic columns
EXCLUDED = [
    "CustomerID", "Churn Label", "Churn Value", "Churn Score", "Churn Reason",
    "CLTV", "Count", "Country", "State", "City", "Zip Code", "Lat Long",
    "Latitude", "Longitude",
]


def load_raw(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place Telco_customer_churn.xlsx in the data/ folder."
        )
    return pd.read_excel(path, sheet_name=SHEET)


def clean_features(df):
    """Coerce numeric columns; blank Total Charges (new customers) become NaN and are imputed in the pipeline."""
    out = df.copy()
    for col in NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CATEGORICAL:
        out[col] = out[col].astype("string").str.strip().astype(object)
    return out


def load_xy(path=DATA_PATH):
    raw = load_raw(path)
    X = clean_features(raw[FEATURES])
    return X, raw[TARGET].astype(int), raw


def validate_input(df):
    """Return a cleaned frame with only model features, or raise ValueError listing missing columns."""
    if df is None or len(df) == 0:
        raise ValueError("The uploaded file contains no rows.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    X = clean_features(df[FEATURES])
    if X[FEATURES].isna().all(axis=1).any():
        raise ValueError("Some rows have no usable feature values.")
    return X
