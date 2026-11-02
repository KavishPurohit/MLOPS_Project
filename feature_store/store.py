from pathlib import Path
import pandas as pd
from src.config import ROOT, RAW_DIR
from src.data import load_fd001, add_training_rul
from src.features import engineer_features

def build_snapshot(raw_dir: str | Path = RAW_DIR, output: str | Path = ROOT / "data/processed/fd001_features.csv") -> Path:
    train, _, _ = load_fd001(raw_dir)
    labeled = add_training_rul(train)
    features = engineer_features(labeled).drop(columns=["cycle"])
    snapshot = pd.concat([labeled[["engine_id", "cycle", "rul"]], features], axis=1)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(output, index=False)
    return output

if __name__ == "__main__":
    print(build_snapshot())
