"""Shared fixtures. Tests use a synthetic dataset so they run offline and fast."""

from __future__ import annotations

from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import pytest

from churn.config import Economics
from churn.data import clean, split_xy
from churn.models import make_pipeline

CHOICES = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def make_raw(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Raw rows shaped like the Telco CSV, with churn driven by contract and tenure."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in CHOICES.items()})
    df.insert(0, "customerID", [f"{i:04d}-TEST" for i in range(n)])
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    total = (df["MonthlyCharges"] * df["tenure"]).round(2).astype(str)
    df["TotalCharges"] = total.where(df["tenure"] > 0, " ")  # blank like the real data
    logit = -1 + 1.5 * (df["Contract"] == "Month-to-month") - 0.04 * df["tenure"]
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def xy(raw_df):
    return split_xy(clean(raw_df))


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, xy):
    X, y = xy
    pipe = make_pipeline("logistic_regression").fit(X, y)
    bundle = {
        "pipeline": pipe,
        "model_name": "logistic_regression",
        "params": {},
        "policy": "expected_value",
        "threshold": 0.3,
        "economics": asdict(Economics()),
        "background": X.head(50),
        "metrics": {"test_at_chosen_threshold": {"roc_auc": 0.8}},
        "version": "test",
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(bundle, path)
    return path
