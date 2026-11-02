import pytest
from io import BytesIO

prometheus_client = pytest.importorskip("prometheus_client")
from fastapi import HTTPException
from fastapi import UploadFile
from api.main import app, health, predict_csv

def test_health_reports_missing_real_model_as_unavailable(monkeypatch):
    import api.main as service
    monkeypatch.setattr(service, "_bundle", None)
    monkeypatch.setenv("MODEL_PATH", "missing-model-artifact.joblib")
    with pytest.raises(HTTPException) as error:
        health()
    assert error.value.status_code == 503
    assert "Train with" in error.value.detail

def test_prediction_rejects_incomplete_sensor_schema():
    file = UploadFile(filename="history.csv", file=BytesIO(b"cycle,sensor_1\n1,0.4\n"))
    with pytest.raises(HTTPException) as error:
        predict_csv(file)
    assert error.value.status_code == 400
    assert "must include" in error.value.detail

def test_metrics_endpoint_is_exposed():
    assert any(getattr(route, "path", None) == "/metrics" for route in app.routes)
