"""Leakage-safe per-engine rolling and lifetime summary features."""
import numpy as np
import pandas as pd
from .config import FEATURES

WINDOWS = (5, 10, 20)

def engineer_features(frame: pd.DataFrame, sensors: list[str] | None = None) -> pd.DataFrame:
    sensors = sensors or FEATURES
    required = {"engine_id", "cycle", *sensors}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    f = frame.sort_values(["engine_id", "cycle"]).copy()
    feature_columns = {"cycle": f["cycle"].astype(float)}
    # Do not normalize by engine lifetime: the failure cycle is unknown at inference.
    feature_columns["log_cycle"] = np.log1p(f["cycle"].astype(float))
    for col in sensors:
        s = f[col].astype(float)
        g = f.assign(_v=s).groupby("engine_id", sort=False)["_v"]
        feature_columns[f"{col}_current"] = s
        feature_columns[f"{col}_mean_life"] = g.expanding().mean().reset_index(level=0, drop=True).reindex(f.index)
        feature_columns[f"{col}_std_life"] = g.expanding().std().reset_index(level=0, drop=True).reindex(f.index).fillna(0)
        for w in WINDOWS:
            feature_columns[f"{col}_mean_{w}"] = g.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).reindex(f.index)
            feature_columns[f"{col}_std_{w}"] = g.rolling(w, min_periods=2).std().reset_index(level=0, drop=True).reindex(f.index).fillna(0)
        feature_columns[f"{col}_trend_10"] = g.diff(10).fillna(0) / 10.0
    result = pd.DataFrame(feature_columns, index=f.index)
    return result.replace([np.inf, -np.inf], 0).fillna(0).sort_index()

def latest_engine_features(history: pd.DataFrame, sensors: list[str] | None = None) -> pd.DataFrame:
    if history.empty:
        raise ValueError("Sensor history must contain at least one row.")
    features = engineer_features(history, sensors)
    return features.iloc[[-1]]

class SequentialFeatureEngineer:
    """Stateless, serializable preprocessing step stored with the model bundle."""
    def __init__(self, sensors=None):
        self.sensors = sensors or FEATURES

    def fit(self, frame, y=None):
        return self

    def transform(self, frame):
        return engineer_features(frame, self.sensors)
