import numpy as np
import pytest

from churn.business import (
    best_threshold,
    expected_offer_value,
    profit_curve,
    realized_profit,
    target_mask,
)
from churn.config import Economics

ECON = Economics(offer_cost=50, success_rate=0.5, horizon_months=10)


def test_realized_profit_by_hand():
    # value = 100 * 10 = 1000; churner saved: 0.5 * 1000 - 50 = 450; non-churner: -50
    y = [1, 0, 1, 0]
    targeted = [True, True, False, False]
    assert realized_profit(y, targeted, [100] * 4, ECON) == pytest.approx(400)


def test_expected_offer_value_break_even():
    # break-even probability = cost / (success * value) = 50 / 500 = 0.1
    ev = expected_offer_value([0.05, 0.1, 0.5], [100] * 3, ECON)
    assert ev[0] < 0
    assert ev[1] == pytest.approx(0)
    assert ev[2] == pytest.approx(200)


def test_best_threshold_separates_perfect_scores():
    y = np.array([1] * 10 + [0] * 90)
    proba = np.r_[np.full(10, 0.9), np.full(90, 0.1)]
    t = best_threshold(y, proba, np.full(100, 100.0), ECON)
    assert 0.1 < t <= 0.9


def test_profit_curve_shape():
    curve = profit_curve([1, 0], [0.8, 0.2], [100, 100], ECON)
    assert {"threshold", "profit", "pct_targeted"} <= set(curve.columns)
    assert curve["pct_targeted"].is_monotonic_decreasing


@pytest.mark.parametrize("policy,expected", [("everyone", 3), ("nobody", 0), ("threshold", 2), ("expected_value", 2)])
def test_target_mask(policy, expected):
    mask = target_mask(policy, [0.05, 0.6, 0.9], [100] * 3, ECON, threshold=0.5)
    assert mask.sum() == expected
