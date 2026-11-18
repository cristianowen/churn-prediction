import numpy as np
import pandas as pd

from churn.data import INPUT_COLUMNS, clean, split_xy
from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features, build_preprocessor


def test_clean_fills_blank_total_charges_for_new_customers(raw_df):
    df = clean(raw_df)
    new = df[df["tenure"] == 0]
    assert len(new) > 0
    assert (new["TotalCharges"] == 0).all()
    assert df["TotalCharges"].notna().all()
    assert len(df) == len(raw_df)  # nothing dropped


def test_clean_maps_target_to_binary(raw_df):
    df = clean(raw_df)
    assert set(df["Churn"].unique()) <= {0, 1}
    assert df["Churn"].sum() == (raw_df["Churn"] == "Yes").sum()


def test_split_xy_excludes_id_and_target(xy):
    X, y = xy
    assert list(X.columns) == INPUT_COLUMNS
    assert "customerID" not in X and "Churn" not in X
    assert len(X) == len(y)


def test_add_features_values():
    row = {c: "No" for c in INPUT_COLUMNS}
    row.update(
        PhoneService="Yes", StreamingTV="Yes", OnlineSecurity="Yes",
        PaymentMethod="Credit card (automatic)", Contract="One year",
        SeniorCitizen=0, tenure=10, MonthlyCharges=80.0, TotalCharges=700.0,
    )
    new = dict(row, tenure=0, TotalCharges=0.0, PaymentMethod="Electronic check")
    out = add_features(pd.DataFrame([row, new]))
    assert out.loc[0, "num_services"] == 3
    assert out.loc[0, "avg_monthly_spend"] == 70.0
    assert out.loc[0, "price_increase"] == 10.0
    assert out.loc[0, "auto_pay"] == 1 and out.loc[1, "auto_pay"] == 0
    assert out.loc[0, "tenure_group"] == "7-12m"
    # tenure 0 must not divide by zero
    assert out.loc[1, "avg_monthly_spend"] == 80.0
    assert out.loc[1, "tenure_group"] == "0-6m"


def test_preprocessor_handles_unseen_categories(xy):
    X, _ = xy
    feats = add_features(X)
    prep = build_preprocessor().fit(feats)
    odd = feats.head(2).copy()
    odd["PaymentMethod"] = "Crypto"
    out = prep.transform(odd)
    assert np.isfinite(out).all()
    assert out.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
