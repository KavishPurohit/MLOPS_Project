import io
import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from src.config import MODEL_DIR, FEATURES
from src.features import SequentialFeatureEngineer

app = FastAPI(title="C-MAPSS RUL Prediction API", version="0.1.0")
REQUESTS = Counter("rul_prediction_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram("rul_prediction_latency_seconds", "Prediction request latency")
DRIFT = Gauge("rul_input_mean_abs_zscore", "Mean absolute per-feature shift from training feature distribution")
app.mount("/metrics", make_asgi_app())
_bundle = None

def get_bundle():
    global _bundle
    path = os.getenv("MODEL_PATH", str(MODEL_DIR / "best_model.joblib"))
    if _bundle is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact missing: {path}. Train with `python -m src.train` first.")
        _bundle = joblib.load(path)
        if "preprocessor" not in _bundle:
            _bundle["preprocessor"] = SequentialFeatureEngineer()
    return _bundle

@app.get("/health")
def health():
    try:
        b = get_bundle(); return {"status": "ok", "model": b.get("model_name", "unknown")}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/model-info")
def model_info():
    try:
        b = get_bundle(); return {"model": b.get("model_name"), "metrics": b.get("metrics", {}), "features": b.get("feature_names", [])}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.post("/predict")
def predict_csv(file: UploadFile = File(...)):
    start = time.perf_counter()
    try:
        raw = pd.read_csv(io.BytesIO(file.file.read()))
        if not {"cycle", *FEATURES}.issubset(raw.columns):
            raise ValueError("CSV must include cycle, 3 operating settings and 21 sensors.")
        if "engine_id" not in raw: raw["engine_id"] = 1
        if raw.empty: raise ValueError("CSV must contain at least one sensor-history row.")
        if raw.engine_id.nunique() != 1: raise ValueError("Upload one engine history per request.")
        raw[FEATURES + ["cycle"]] = raw[FEATURES + ["cycle"]].apply(pd.to_numeric, errors="coerce")
        values = raw[FEATURES + ["cycle"]].to_numpy(dtype=float)
        if not np.isfinite(values).all(): raise ValueError("CSV has missing or non-numeric sensor values.")
        if (raw["cycle"] < 1).any(): raise ValueError("Cycle values must be positive.")
        b = get_bundle(); x = b["preprocessor"].transform(raw).iloc[[-1]]
        if list(x.columns) != b["feature_names"]: raise ValueError("Uploaded history has an incompatible feature schema.")
        pred = max(0.0, float(b["model"].predict(x)[0]))
        if b.get("reference_mean") and b.get("reference_std"):
            ref_mean = pd.Series(b["reference_mean"])
            ref_std = pd.Series(b["reference_std"]).replace(0, 1)
            aligned = x.iloc[0].reindex(ref_mean.index)
            DRIFT.set(float(((aligned - ref_mean).abs() / ref_std).mean()))
        REQUESTS.labels(status="success").inc()
        return {"estimated_rul_cycles": round(pred, 2), "model": b.get("model_name"), "engine_id": int(raw.engine_id.iloc[-1])}
    except FileNotFoundError as exc:
        REQUESTS.labels(status="error").inc(); raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        REQUESTS.labels(status="error").inc(); raise HTTPException(status_code=400, detail=str(exc))
    finally:
        LATENCY.observe(time.perf_counter() - start)
