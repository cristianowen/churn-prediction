"""SHAP explanations, rolled up from one-hot columns to the original features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.features import output_to_source_feature


class ChurnExplainer:
    def __init__(self, pipeline: Pipeline, background: pd.DataFrame):
        self.engineer = pipeline.named_steps["features"]
        self.transform = pipeline[:-1]
        self.model = pipeline.named_steps["model"]
        self.sources = output_to_source_feature(pipeline.named_steps["prep"])
        self.source_names = list(dict.fromkeys(self.sources))

        bg = self.transform.transform(background)
        if isinstance(self.model, LogisticRegression):
            masker = shap.maskers.Independent(bg, max_samples=len(bg))
            self._explainer = shap.LinearExplainer(self.model, masker)
        elif isinstance(self.model, (HistGradientBoostingClassifier, RandomForestClassifier)):
            # Tree-path-dependent mode is exact for sklearn's histogram GBM; the
            # interventional (background-data) mode fails SHAP's additivity check on it.
            self._explainer = shap.TreeExplainer(self.model)
        else:
            self._explainer = shap.Explainer(lambda a: self.model.predict_proba(a)[:, 1], bg)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """SHAP values for the churn class, one column per preprocessed feature."""
        values = self._explainer(self.transform.transform(X)).values
        if values.ndim == 3:  # (n, features, classes)
            values = values[:, :, 1]
        return values

    def grouped_shap(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP values summed per original feature, so each row adds to the model's output."""
        values = pd.DataFrame(self.shap_values(X), columns=self.sources)
        return values.T.groupby(level=0, sort=False).sum().T[self.source_names]

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        return self.grouped_shap(X).abs().mean().sort_values(ascending=False)

    def top_reasons(self, X: pd.DataFrame, k: int = 5) -> list[list[dict]]:
        """For each customer, the k features pushing their churn risk hardest (either way)."""
        grouped = self.grouped_shap(X)
        feats = self.engineer.transform(X).reset_index(drop=True)
        out = []
        for i in range(len(grouped)):
            row = grouped.iloc[i]
            top = row.reindex(row.abs().sort_values(ascending=False).index[:k])
            out.append(
                [
                    {
                        "feature": name,
                        "value": _jsonable(feats.at[i, name]),
                        "impact": round(float(impact), 4),
                        "direction": "increases risk" if impact > 0 else "decreases risk",
                    }
                    for name, impact in top.items()
                ]
            )
        return out


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 2)
    return v
