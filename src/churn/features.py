"""Feature engineering and preprocessing.

Everything lives inside the sklearn Pipeline, so the exact same transforms run
during cross-validation, on the test set, and on live API requests. Nothing is
fit on data the model is evaluated on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import CATEGORICAL_INPUTS, NUMERIC_INPUTS

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ENGINEERED_NUMERIC = ["num_services", "avg_monthly_spend", "price_increase", "auto_pay"]
ENGINEERED_CATEGORICAL = ["tenure_group"]

NUMERIC_FEATURES = NUMERIC_INPUTS + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = CATEGORICAL_INPUTS + ENGINEERED_CATEGORICAL

TENURE_BINS = [-1, 6, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    # How many products the customer has. More products usually means more lock-in.
    X["num_services"] = (X[SERVICE_COLS] == "Yes").sum(axis=1)
    # Lifetime average bill. New customers fall back to their current bill.
    tenure = X["tenure"].replace(0, np.nan)
    X["avg_monthly_spend"] = (X["TotalCharges"] / tenure).fillna(X["MonthlyCharges"])
    # Positive means they now pay more than their historical average (a price hike).
    X["price_increase"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
    X["auto_pay"] = X["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    X["tenure_group"] = pd.cut(X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer wrapping `add_features` for use in a Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_features(pd.DataFrame(X))


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def output_to_source_feature(preprocessor: ColumnTransformer) -> list[str]:
    """For each column the fitted preprocessor outputs, name the input feature it came from.

    Used to roll one-hot SHAP values back up to the original feature
    (e.g. Contract_Month-to-month and Contract_Two year both map to Contract).
    """
    sources: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            sources.extend(cols)
        elif name == "cat":
            for col, cats in zip(cols, transformer.categories_):
                sources.extend([col] * len(cats))
    return sources
