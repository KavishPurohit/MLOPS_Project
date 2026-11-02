import pytest

prometheus_client = pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_reports_missing_real_model_as_unavailable(monkeypatch):
    import api.main as service
    monkeypatch.setattr(service, "_bundle", None)
    monkeypatch.setenv("MODEL_PATH", "missing-model-artifact.joblib")
    response = client.get("/health")
    assert response.status_code == 503
    assert "Train with" in response.json()["detail"]

def test_prediction_rejects_incomplete_sensor_schema():
    response = client.post("/predict", files={"file": ("history.csv", b"cycle,sensor_1\n1,0.4\n", "text/csv")})
    assert response.status_code == 400
    assert "must include" in response.json()["detail"]

def test_metrics_endpoint_is_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rul_prediction_requests_total" in response.text
