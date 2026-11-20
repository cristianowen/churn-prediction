"""End-to-end training: compare models, tune, pick a targeting policy, evaluate, save.

    python -m churn.train            # full run
    python -m churn.train --fast     # quick smoke run (few trials, 3 folds)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from churn import __version__
from churn import evaluate as ev
from churn.business import best_threshold, profit_curve, realized_profit, target_mask
from churn.config import DEFAULT_CONFIG, Config, load_config
from churn.data import load_dataset
from churn.explain import ChurnExplainer
from churn.models import CANDIDATES, SEARCH_SPACES, make_pipeline

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "accuracy": "accuracy", "brier": "neg_brier_score"}


def compare_models(X, y, cv, seed) -> pd.DataFrame:
    rows = []
    for name in CANDIDATES:
        res = cross_validate(make_pipeline(name, seed), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name, "tuned": False}
        for k in SCORING:
            vals = res[f"test_{k}"] * (-1 if k == "brier" else 1)
            row[k] = vals.mean()
            row[f"{k}_std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows)


def tune(name, X, y, cv, n_trials, seed) -> tuple[dict, float]:
    def objective(trial):
        pipe = make_pipeline(name, seed, **SEARCH_SPACES[name](trial))
        res = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return res["test_score"].mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def run(cfg: Config, n_trials: int | None = None, folds: int | None = None) -> dict:
    t0 = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = cfg.n_trials if n_trials is None else n_trials
    folds = folds or cfg.cv_folds
    econ = cfg.economics
    figs = cfg.reports_dir / "figures"

    X, y = load_dataset(cfg.data_path, cfg.data_url)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    cv = StratifiedKFold(folds, shuffle=True, random_state=cfg.seed)
    print(f"Data: {len(X):,} customers, churn rate {y.mean():.1%}. Train {len(X_tr):,} / test {len(X_te):,}")

    # 1. Compare model families with default settings
    print(f"\n[1/5] Comparing models with {folds}-fold CV...")
    comparison = compare_models(X_tr, y_tr, cv, cfg.seed)

    # 2. Tune each model family
    tuned = {}
    for name in SEARCH_SPACES:
        trials = n_trials if name != "random_forest" else max(1, n_trials // 3)
        print(f"[2/5] Tuning {name} ({trials} Optuna trials)...")
        params, score = tune(name, X_tr, y_tr, cv, trials, cfg.seed)
        tuned[name] = (params, score)
        res = cross_validate(make_pipeline(name, cfg.seed, **params), X_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name, "tuned": True}
        for k in SCORING:
            vals = res[f"test_{k}"] * (-1 if k == "brier" else 1)
            row[k], row[f"{k}_std"] = vals.mean(), vals.std()
        comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True)

    best_name = max(tuned, key=lambda n: tuned[n][1])
    best_params = tuned[best_name][0]
    print(f"      Best: {best_name} (CV ROC-AUC {tuned[best_name][1]:.4f})")

    # 3. Pick a targeting policy using out-of-fold predictions on the training set only
    print("[3/5] Choosing a targeting policy on out-of-fold predictions...")
    final = make_pipeline(best_name, cfg.seed, **best_params)
    oof = cross_val_predict(final, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    mc_tr = X_tr["MonthlyCharges"].values
    threshold = best_threshold(y_tr, oof, mc_tr, econ)
    oof_profit = {
        "threshold": realized_profit(y_tr, target_mask("threshold", oof, mc_tr, econ, threshold), mc_tr, econ),
        "expected_value": realized_profit(y_tr, target_mask("expected_value", oof, mc_tr, econ), mc_tr, econ),
    }
    policy = max(oof_profit, key=oof_profit.get)
    print(f"      Profit-optimal threshold {threshold:.2f}; chosen policy: {policy}")

    # 4. Fit on the full training set and evaluate once on the held-out test set
    print("[4/5] Final fit and test-set evaluation...")
    final.fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    mc_te = X_te["MonthlyCharges"].values
    per = 1000 / len(X_te)

    policies = {
        "No campaign": ("nobody", 0.5),
        "Offer everyone": ("everyone", 0.5),
        "Model, default 0.5 threshold": ("threshold", 0.5),
        f"Model, profit-tuned threshold ({threshold:.2f})": ("threshold", threshold),
        "Model, per-customer expected value": ("expected_value", 0.5),
    }
    policy_rows = []
    for label, (pol, t) in policies.items():
        mask = target_mask(pol, p_te, mc_te, econ, t)
        profit = realized_profit(y_te, mask, mc_te, econ)
        policy_rows.append({
            "policy": label,
            "pct_targeted": mask.mean(),
            "churners_reached": int((mask & (y_te.values == 1)).sum()),
            "profit_test": profit,
            "profit_per_1000_customers": profit * per,
        })
    policy_df = pd.DataFrame(policy_rows)

    baseline_p = make_pipeline("baseline").fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    lr_v1 = make_pipeline("logistic_regression").fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    gains = ev.gains_table(y_te, p_te)
    deployed_mask = target_mask(policy, p_te, mc_te, econ, threshold)

    metrics = {
        "model": best_name,
        "params": best_params,
        "policy": policy,
        "threshold": threshold,
        "economics": asdict(econ),
        "test_at_default_threshold": ev.classification_metrics(y_te, p_te, 0.5),
        "test_at_chosen_threshold": ev.classification_metrics(y_te, p_te, threshold),
        "test_majority_class_accuracy": float(1 - y_te.mean()),
        "test_logreg_untuned_roc_auc": float(ev.roc_auc_score(y_te, lr_v1)),
        "top_20pct_churners_caught": float(np.interp(0.2, gains["cum_pct_customers"], gains["cum_pct_churners"])),
        "top_decile_lift": float(gains.loc[0, "lift"]),
        "deployed_policy_profit_per_1000": float(realized_profit(y_te, deployed_mask, mc_te, econ) * per),
        "default_threshold_profit_per_1000": float(policy_df.iloc[2]["profit_per_1000_customers"]),
    }

    # 5. Explanations, figures, artifacts
    print("[5/5] SHAP explanations, figures and artifacts...")
    background = X_tr.sample(200, random_state=cfg.seed)
    explainer = ChurnExplainer(final, background)
    importance = explainer.global_importance(X_te)
    metrics["top_drivers"] = importance.head(8).round(4).to_dict()

    ev.plot_roc_pr({best_name: (y_te, p_te), "logistic_regression (untuned)": (y_te, lr_v1),
                    "baseline (always 'no churn')": (y_te, baseline_p)}, figs / "roc_pr.png")
    ev.plot_calibration(y_te, p_te, figs / "calibration.png")
    ev.plot_gains(gains, figs / "gains.png")
    ev.plot_profit_curve(profit_curve(y_te, p_te, mc_te, econ), threshold, figs / "profit_curve.png", len(X_te))
    ev.plot_importance(importance, figs / "shap_importance.png")

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(cfg.reports_dir / "model_comparison.csv", index=False)
    policy_df.round(3).to_csv(cfg.reports_dir / "policy_comparison.csv", index=False)
    gains.round(4).to_csv(cfg.reports_dir / "gains.csv", index=False)
    with open(cfg.reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    bundle = {
        "pipeline": final,
        "model_name": best_name,
        "params": best_params,
        "policy": policy,
        "threshold": threshold,
        "economics": asdict(econ),
        "background": background,
        "metrics": metrics,
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cfg.model_path)

    _print_summary(comparison, policy_df, metrics, time.time() - t0)
    return metrics


def _print_summary(comparison, policy_df, m, secs):
    pd.set_option("display.width", 140)
    cols = ["model", "tuned", "roc_auc", "pr_auc", "accuracy", "brier"]
    print("\nModel comparison (cross-validated on training set):")
    print(comparison[cols].round(4).to_string(index=False))
    print("\nTargeting policies (held-out test set):")
    print(policy_df.round(3).to_string(index=False))
    t = m["test_at_chosen_threshold"]
    print(
        f"\nTest ROC-AUC {t['roc_auc']:.3f} | PR-AUC {t['pr_auc']:.3f} | "
        f"majority-class accuracy {m['test_majority_class_accuracy']:.1%}\n"
        f"Top 20% riskiest customers contain {m['top_20pct_churners_caught']:.0%} of churners "
        f"(top-decile lift {m['top_decile_lift']:.1f}x)\n"
        f"Deployed policy ({m['policy']}): ${m['deployed_policy_profit_per_1000']:,.0f} per 1,000 customers "
        f"vs ${m['default_threshold_profit_per_1000']:,.0f} at the default 0.5 threshold\n"
        f"Done in {secs:.0f}s."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--trials", type=int, default=None, help="Optuna trials per model")
    ap.add_argument("--fast", action="store_true", help="quick run: 5 trials, 3 folds")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.fast:
        run(cfg, n_trials=args.trials or 5, folds=3)
    else:
        run(cfg, n_trials=args.trials)


if __name__ == "__main__":
    main()
