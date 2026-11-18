"""Turning churn probabilities into retention decisions worth money.

Profit model for one customer we send a retention offer to:
  - would have churned:     success_rate * customer_value - offer_cost
  - would have stayed:      -offer_cost   (they take the offer anyway)
Customers we don't target cost and earn nothing (relative to doing nothing).

customer_value = MonthlyCharges * horizon_months, the revenue we keep if they stay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import Economics


def customer_value(monthly_charges, econ: Economics) -> np.ndarray:
    return np.asarray(monthly_charges, dtype=float) * econ.horizon_months


def expected_offer_value(proba, monthly_charges, econ: Economics) -> np.ndarray:
    """Expected $ gain from sending this customer an offer, given churn probability."""
    proba = np.asarray(proba, dtype=float)
    return proba * econ.success_rate * customer_value(monthly_charges, econ) - econ.offer_cost


def realized_profit(y_true, targeted, monthly_charges, econ: Economics) -> float:
    """Actual campaign profit (vs. no campaign) for a given set of targeted customers."""
    y_true = np.asarray(y_true).astype(bool)
    targeted = np.asarray(targeted).astype(bool)
    value = customer_value(monthly_charges, econ)
    gain = np.where(y_true, econ.success_rate * value, 0.0) - econ.offer_cost
    return float(gain[targeted].sum())


def profit_curve(y_true, proba, monthly_charges, econ: Economics, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    proba = np.asarray(proba)
    rows = []
    for t in thresholds:
        targeted = proba >= t
        rows.append(
            {
                "threshold": t,
                "profit": realized_profit(y_true, targeted, monthly_charges, econ),
                "pct_targeted": targeted.mean(),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(y_true, proba, monthly_charges, econ: Economics) -> float:
    curve = profit_curve(y_true, proba, monthly_charges, econ)
    return float(curve.loc[curve["profit"].idxmax(), "threshold"])


def target_mask(policy: str, proba, monthly_charges, econ: Economics, threshold: float = 0.5):
    """Which customers get an offer under a targeting policy."""
    proba = np.asarray(proba)
    if policy == "expected_value":
        return expected_offer_value(proba, monthly_charges, econ) > 0
    if policy == "threshold":
        return proba >= threshold
    if policy == "everyone":
        return np.ones_like(proba, dtype=bool)
    if policy == "nobody":
        return np.zeros_like(proba, dtype=bool)
    raise ValueError(f"unknown policy: {policy}")
