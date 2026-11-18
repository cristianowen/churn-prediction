"""Metrics, lift/gains tables and report figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 3),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def gains_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Rank customers by risk, cut into deciles, and show how many churners each catches."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    g = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"), avg_score=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["customers"]
    g["cum_pct_customers"] = g["customers"].cumsum() / g["customers"].sum()
    g["cum_pct_churners"] = g["churners"].cumsum() / g["churners"].sum()
    g["lift"] = g["churn_rate"] / df["y"].mean()
    return g.reset_index()


# ---------------------------------------------------------------- figures

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_roc_pr(curves: dict[str, tuple], path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax1.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        ax2.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    base = np.mean(next(iter(curves.values()))[0])
    ax2.axhline(base, color="k", ls="--", lw=0.8, label=f"base rate {base:.2f}")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    _save(fig, path)


def plot_calibration(y, proba, path: Path):
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set(xlabel="Predicted churn probability", ylabel="Observed churn rate", title="Calibration (test set)")
    ax.legend()
    _save(fig, path)


def plot_gains(gains: pd.DataFrame, path: Path):
    x = np.r_[0, gains["cum_pct_customers"]]
    y = np.r_[0, gains["cum_pct_churners"]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(x * 100, y * 100, "o-", label="model")
    ax.plot([0, 100], [0, 100], "k--", lw=0.8, label="random targeting")
    ax.set(xlabel="% of customers contacted (highest risk first)", ylabel="% of churners caught",
           title="Cumulative gains (test set)")
    ax.legend()
    _save(fig, path)


def plot_profit_curve(curve: pd.DataFrame, chosen: float, path: Path, per: int):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(curve["threshold"], curve["profit"], lw=2)
    ax.axvline(chosen, color="tab:red", ls="--", label=f"chosen threshold {chosen:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="default 0.50")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="Churn probability threshold for sending an offer",
           ylabel=f"Campaign profit ($) on {per:,} test customers", title="Profit vs. threshold (test set)")
    ax.legend()
    _save(fig, path)


def plot_importance(importance: pd.Series, path: Path, top: int = 15):
    imp = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(imp.index, imp.values)
    ax.set(xlabel="Mean |SHAP value|", title="What drives churn predictions")
    _save(fig, path)
