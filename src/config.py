from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
FEATURES = ["op_setting_1", "op_setting_2", "op_setting_3", *[f"sensor_{i}" for i in range(1, 22)]]
ID_COLUMNS = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
RUL_CAP = 125
