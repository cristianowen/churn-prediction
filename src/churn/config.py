"""Project configuration loaded from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Economics:
    """Assumptions for a retention campaign."""

    offer_cost: float = 60.0
    success_rate: float = 0.35
    horizon_months: int = 12


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_trials: int = 40
    data_path: Path = PROJECT_ROOT / "data" / "telco_churn.csv"
    data_url: str = ""
    model_path: Path = PROJECT_ROOT / "models" / "churn_model.joblib"
    reports_dir: Path = PROJECT_ROOT / "reports"
    economics: Economics = field(default_factory=Economics)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        seed=raw.get("seed", 42),
        test_size=raw.get("test_size", 0.2),
        cv_folds=raw.get("cv_folds", 5),
        n_trials=raw.get("n_trials", 40),
        data_path=_resolve(raw["data"]["path"]),
        data_url=raw["data"].get("url", ""),
        model_path=_resolve(raw["paths"]["model"]),
        reports_dir=_resolve(raw["paths"]["reports"]),
        economics=Economics(**raw.get("economics", {})),
    )
