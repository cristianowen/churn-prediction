"""REST API for churn scoring.

    uvicorn api.main:app --reload
    # docs at http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn.config import PROJECT_ROOT
from churn.predict import ChurnPredictor

YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 171.0,
            }
        }
    }


class Reason(BaseModel):
    feature: str
    value: str | int | float
    impact: float
    direction: str


class Prediction(BaseModel):
    churn_probability: float
    risk_tier: Literal["low", "medium", "high"]
    send_offer: bool
    expected_offer_value: float = Field(description="Expected $ gain from sending a retention offer")
    top_reasons: list[Reason] | None = None


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("CHURN_MODEL_PATH", PROJECT_ROOT / "models" / "churn_model.joblib")
    state["predictor"] = ChurnPredictor.load(path)
    yield
    state.clear()


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


def _predictor() -> ChurnPredictor:
    if "predictor" not in state:
        raise HTTPException(503, "model not loaded")
    return state["predictor"]


def _score(customers: list[Customer], explain: bool) -> list[Prediction]:
    df = pd.DataFrame([c.model_dump() for c in customers])
    out = _predictor().predict(df, explain=explain)
    return [Prediction(**row) for row in out.to_dict(orient="records")]


@app.get("/health")
def health():
    p = _predictor()
    return {
        "status": "ok",
        "model": p.bundle["model_name"],
        "policy": p.policy,
        "trained_at": p.bundle["trained_at"],
        "test_roc_auc": round(p.bundle["metrics"]["test_at_chosen_threshold"]["roc_auc"], 4),
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, explain: bool = True):
    return _score([customer], explain)[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer], explain: bool = False):
    if len(customers) > 10_000:
        raise HTTPException(413, "max 10,000 customers per request")
    return _score(customers, explain)
