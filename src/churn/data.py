"""Loading and cleaning the IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"

# Raw input columns a customer record must have (everything except ID and target).
NUMERIC_INPUTS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_INPUTS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
INPUT_COLUMNS = NUMERIC_INPUTS + CATEGORICAL_INPUTS


def load_raw(path: str | Path, url: str = "") -> pd.DataFrame:
    """Read the dataset from disk, downloading and caching it on first use."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    if not url:
        raise FileNotFoundError(f"{path} not found and no download URL configured")
    df = pd.read_csv(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types and missing values.

    TotalCharges is blank for 11 customers, all with tenure == 0: they signed up
    but haven't been billed yet. Their true total is $0, so we fill it instead of
    dropping the rows.
    """
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    new_customers = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customers, "TotalCharges"] = 0.0
    df = df.dropna(subset=["TotalCharges"])

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    return df.reset_index(drop=True)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[INPUT_COLUMNS].copy()
    y = df[TARGET].astype(int)
    return X, y


def load_dataset(path: str | Path, url: str = "") -> tuple[pd.DataFrame, pd.Series]:
    return split_xy(clean(load_raw(path, url)))
