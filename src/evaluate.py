"""Final evaluation against the separate NASA test labels."""
from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from .config import RAW_DIR, RUL_CAP, MODEL_DIR
from .data import attach_test_rul, load_fd001
from .features import SequentialFeatureEngineer
from .metrics import nasa_score, regression_metrics

def evaluate_test(model, preprocessor=None, raw_dir=RAW_DIR):
    _, test, labels = load_fd001(raw_dir)
    preprocessor = preprocessor or SequentialFeatureEngineer()
    features = preprocessor.transform(test)
    last_features = features.groupby(test.engine_id, sort=True).tail(1)
    truth_frame = attach_test_rul(test, labels, RUL_CAP).groupby("engine_id", sort=True).tail(1)
    truth = truth_frame["rul"].to_numpy()
    prediction = model.predict(last_features)
    metrics = regression_metrics(truth, prediction)
    metrics["nasa_score"] = nasa_score(truth, prediction)
    predictions = pd.DataFrame({"engine_id": truth_frame.engine_id.to_numpy(),
                                "actual_rul": truth,
                                "predicted_rul": prediction,
                                "error": prediction - truth})
    return metrics, predictions

def write_test_evaluation(model, model_dir, preprocessor=None, raw_dir=RAW_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics, predictions = evaluate_test(model, preprocessor, raw_dir)
    predictions.to_csv(model_dir / "test_predictions.csv", index=False)
    return metrics, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    bundle = joblib.load(model_dir / "best_model.joblib")
    metrics, predictions = write_test_evaluation(bundle["model"], model_dir,
                                                  bundle.get("preprocessor"), args.raw_dir)
    report_path = model_dir / "evaluation.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["test"] = metrics
    report["test_engines"] = int(len(predictions))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
