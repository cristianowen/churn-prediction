import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn.evaluate import classification_metrics, gains_table
from churn.explain import ChurnExplainer
from churn.models import CANDIDATES, make_pipeline
from churn.predict import ChurnPredictor


@pytest.mark.parametrize("name", CANDIDATES)
def test_every_candidate_fits_and_predicts(xy, name):
    X, y = xy
    kw = {"n_estimators": 30} if name == "random_forest" else {}
    p = make_pipeline(name, **kw).fit(X, y).predict_proba(X)[:, 1]
    assert p.shape == (len(X),)
    assert ((p >= 0) & (p <= 1)).all()


def test_model_learns_signal(xy):
    X, y = xy
    p = make_pipeline("logistic_regression").fit(X, y).predict_proba(X)[:, 1]
    assert roc_auc_score(y, p) > 0.7


@pytest.mark.parametrize("name", ["logistic_regression", "gradient_boosting"])
def test_grouped_shap_adds_up_to_model_output(xy, name):
    X, y = xy
    pipe = make_pipeline(name).fit(X, y)
    exp = ChurnExplainer(pipe, X.head(50))
    grouped = exp.grouped_shap(X.head(5))
    # one column per original feature, and the grouped values still sum to the full SHAP total
    assert "Contract" in grouped.columns
    assert np.allclose(grouped.sum(axis=1), exp.shap_values(X.head(5)).sum(axis=1))
    reasons = exp.top_reasons(X.head(2), k=3)
    assert len(reasons) == 2 and len(reasons[0]) == 3


def test_metrics_and_gains(xy):
    X, y = xy
    p = make_pipeline("logistic_regression").fit(X, y).predict_proba(X)[:, 1]
    m = classification_metrics(y, p, 0.4)
    assert sum(m["confusion"].values()) == len(y)
    g = gains_table(y, p)
    assert len(g) == 10
    assert g["cum_pct_churners"].iloc[-1] == pytest.approx(1.0)
    assert g["lift"].iloc[0] > 1  # top decile beats average


def test_predictor_output(model_path, xy):
    X, _ = xy
    pred = ChurnPredictor.load(model_path)
    out = pred.predict(X.head(4), explain=True)
    assert list(out.columns) == ["churn_probability", "risk_tier", "send_offer", "expected_offer_value", "top_reasons"]
    assert out["risk_tier"].isin(["low", "medium", "high"]).all()
    # send_offer follows the expected-value policy
    assert (out["send_offer"] == (out["expected_offer_value"] > 0)).all()
