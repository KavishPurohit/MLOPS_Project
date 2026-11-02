from io import BytesIO
from pathlib import Path
import joblib
import numpy as np
import pytest
from fastapi import UploadFile
from src.config import FEATURES, MODEL_DIR, RAW_DIR
from src.data import load_fd001

READY = (MODEL_DIR / "best_model.joblib").exists() and all(
    (RAW_DIR / name).exists()
    for name in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
)
pytestmark = pytest.mark.skipif(not READY, reason="CI prepares NASA FD001 and trains the model before pytest")

def test_selected_model_predicts_on_real_nasa_test_history():
    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    _, test, _ = load_fd001(RAW_DIR)
    history = test[test.engine_id == 1]
    features = bundle["preprocessor"].transform(history).iloc[[-1]]
    prediction = float(bundle["model"].predict(features)[0])
    assert np.isfinite(prediction)
    assert prediction >= 0

def test_fastapi_prediction_handler_uses_real_model_and_history(monkeypatch):
    import api.main as service
    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    _, test, _ = load_fd001(RAW_DIR)
    history = test[test.engine_id == 1][["engine_id", "cycle", *FEATURES]]
    monkeypatch.setattr(service, "_bundle", bundle)
    upload = UploadFile(filename="fd001-engine-1.csv",
                        file=BytesIO(history.to_csv(index=False).encode("utf-8")))
    response = service.predict_csv(upload)
    assert response["model"] == "random_forest"
    assert response["engine_id"] == 1
    assert np.isfinite(response["estimated_rul_cycles"])
