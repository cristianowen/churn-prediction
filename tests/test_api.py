import pytest
from fastapi.testclient import TestClient

from api.main import Customer, app

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def client(model_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(model_path))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_single_with_reasons(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert len(body["top_reasons"]) == 5


def test_predict_batch(client):
    r = client.post("/predict/batch", json=[EXAMPLE, dict(EXAMPLE, Contract="Two year", tenure=60)])
    assert r.status_code == 200
    a, b = r.json()
    assert a["churn_probability"] > b["churn_probability"]
    assert a["top_reasons"] is None


def test_rejects_invalid_input(client):
    r = client.post("/predict", json=dict(EXAMPLE, Contract="Forever"))
    assert r.status_code == 422
