"""C-MAPSS ingestion and label construction (NASA text files are whitespace delimited)."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import FEATURES, ID_COLUMNS, RAW_DIR

COLS = ID_COLUMNS + [f"sensor_{i}" for i in range(1, 22)]

def load_partition(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}. Download and extract CMAPSSData first.")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == 27:
        df = df.iloc[:, :26]
    if df.shape[1] != len(COLS):
        raise ValueError(f"Expected 26 columns in {path.name}, found {df.shape[1]}.")
    df.columns = COLS
    if df.isna().any().any() or (df["cycle"] < 1).any():
        raise ValueError(f"Invalid or missing values found in {path.name}.")
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite sensor values found in {path.name}.")
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

def load_fd001(raw_dir: str | Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    raw_dir = Path(raw_dir)
    train = load_partition(raw_dir / "train_FD001.txt")
    test = load_partition(raw_dir / "test_FD001.txt")
    rul_path = raw_dir / "RUL_FD001.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing {rul_path.name}; NASA test labels are required for evaluation.")
    test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None).iloc[:, 0].to_numpy(dtype=float)
    if len(test_rul) != test.engine_id.nunique():
        raise ValueError("RUL_FD001.txt must contain one label per test engine, in engine-id order.")
    return train, test, test_rul

def add_training_rul(train: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    out = train.copy()
    last_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = last_cycle - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out

def attach_test_rul(test: pd.DataFrame, labels: np.ndarray, cap: int | None = None) -> pd.DataFrame:
    out = test.copy()
    maxima = out.groupby("engine_id")["cycle"].max()
    ordered_ids = maxima.index.to_numpy()
    if len(ordered_ids) != len(labels):
        raise ValueError("Test labels and test engines do not align.")
    remaining = dict(zip(ordered_ids, labels))
    out["rul"] = out["engine_id"].map(maxima + pd.Series(remaining, index=ordered_ids)) - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out
