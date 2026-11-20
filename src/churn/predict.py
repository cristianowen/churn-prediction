"""Load a trained model bundle and score customers."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.business import expected_offer_value, target_mask
from churn.config import Economics
from churn.data import INPUT_COLUMNS
from churn.explain import ChurnExplainer


def risk_tier(p: float) -> str:
    if p >= 0.6:
        return "high"
    if p >= 0.3:
        return "medium"
    return "low"


class ChurnPredictor:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.economics = Economics(**bundle["economics"])
        self.policy = bundle["policy"]
        self.threshold = bundle["threshold"]
        self._explainer: ChurnExplainer | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ChurnPredictor":
        return cls(joblib.load(path))

    @property
    def explainer(self) -> ChurnExplainer:
        if self._explainer is None:
            self._explainer = ChurnExplainer(self.pipeline, self.bundle["background"])
        return self._explainer

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[INPUT_COLUMNS])[:, 1]

    def predict(self, X: pd.DataFrame, explain: bool = False, top_k: int = 5) -> pd.DataFrame:
        X = X[INPUT_COLUMNS].reset_index(drop=True)
        p = self.predict_proba(X)
        ev = expected_offer_value(p, X["MonthlyCharges"], self.economics)
        offer = target_mask(self.policy, p, X["MonthlyCharges"], self.economics, self.threshold)
        out = pd.DataFrame(
            {
                "churn_probability": p.round(4),
                "risk_tier": [risk_tier(v) for v in p],
                "send_offer": offer,
                "expected_offer_value": ev.round(2),
            }
        )
        if explain:
            out["top_reasons"] = self.explainer.top_reasons(X, k=top_k)
        return out
