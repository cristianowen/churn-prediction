"""Candidate models and their Optuna search spaces.

No class reweighting: we keep predicted probabilities calibrated and handle
class imbalance at decision time by choosing a profit-optimal threshold
(see business.py). Reweighting would inflate probabilities and break the
expected-value math.
"""

from __future__ import annotations

from typing import Any, Callable

import optuna
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.features import FeatureEngineer, build_preprocessor


def make_estimator(name: str, seed: int = 42, **params: Any):
    if name == "baseline":
        return DummyClassifier(strategy="prior")
    if name == "logistic_regression":
        return LogisticRegression(max_iter=5000, **params)
    if name == "random_forest":
        defaults = dict(n_estimators=400, min_samples_leaf=5, n_jobs=-1)
        return RandomForestClassifier(random_state=seed, **{**defaults, **params})
    if name == "gradient_boosting":
        defaults = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0)
        return HistGradientBoostingClassifier(random_state=seed, **{**defaults, **params})
    raise ValueError(f"unknown model: {name}")


def make_pipeline(name: str, seed: int = 42, **params: Any) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("prep", build_preprocessor()),
            ("model", make_estimator(name, seed, **params)),
        ]
    )


CANDIDATES = ["baseline", "logistic_regression", "random_forest", "gradient_boosting"]


def _lr_space(trial: optuna.Trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}


def _rf_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
    }


def _gb_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 48),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 120),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "logistic_regression": _lr_space,
    "random_forest": _rf_space,
    "gradient_boosting": _gb_space,
}
