"""Reproducible engine-wise validation and MLflow model comparison."""
import argparse
import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import MODEL_DIR, RUL_CAP, ROOT
from .data import add_training_rul, load_fd001
from .evaluate import write_test_evaluation
from .features import engineer_features, SequentialFeatureEngineer
from .metrics import regression_metrics, nasa_score

MODELS = {
    "linear_regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "random_forest": lambda: RandomForestRegressor(n_estimators=250, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=42),
    "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=160, learning_rate=0.04, max_depth=2, loss="huber", random_state=42),
}

def train(raw_dir="data/raw", model_dir=MODEL_DIR):
    train_df, test_df, test_labels = load_fd001(raw_dir)
    labeled = add_training_rul(train_df, RUL_CAP)
    X = engineer_features(labeled)
    y = labeled["rul"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, va = next(splitter.split(X, y, groups=labeled.engine_id))
    # Keep the classroom tracking store portable and project-local even when a
    # machine has a global MLFLOW_TRACKING_URI configured.
    mlflow.set_tracking_uri("sqlite:///./mlflow.db")
    experiment = mlflow.get_experiment_by_name("cmapss-fd001-rul")
    if experiment is None:
        mlflow.create_experiment("cmapss-fd001-rul", artifact_location=(ROOT / "mlruns").resolve().as_uri())
    mlflow.set_experiment("cmapss-fd001-rul")
    results = {}
    for name, factory in MODELS.items():
        with mlflow.start_run(run_name=name) as run:
            model = factory()
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
            metrics = regression_metrics(y[va], pred)
            metrics["nasa_score"] = nasa_score(y[va], pred)
            mlflow.log_params({"model": name, "rul_cap": RUL_CAP, "split": "engine-wise GroupShuffleSplit", "random_state": 42})
            mlflow.log_metrics(metrics)
            mlflow.log_param("validation_engines", int(labeled.iloc[va].engine_id.nunique()))
            # These models are produced locally by this run; use cloudpickle to
            # keep estimator artifacts reproducible across the supported sklearn
            # regressors without silently trusting arbitrary tree payloads.
            mlflow.sklearn.log_model(model, "model", serialization_format="cloudpickle")
            results[name] = {**metrics, "run_id": run.info.run_id}
    best_name = min(results, key=lambda n: results[n]["rmse"])
    # Refit selected estimator on all labeled training engines.
    best_model = MODELS[best_name]().fit(X, y)
    model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    test_metrics, test_predictions = write_test_evaluation(best_model, model_dir,
                                                            SequentialFeatureEngineer(), raw_dir)
    reference_mean = X.mean()
    reference_std = X.std().replace(0, 1).fillna(1)
    joblib.dump({"model": best_model, "preprocessor": SequentialFeatureEngineer(),
                 "feature_names": list(X.columns), "model_name": best_name,
                 "metrics": test_metrics, "reference_mean": reference_mean.to_dict(),
                 "reference_std": reference_std.to_dict()}, model_dir / "best_model.joblib")
    report = {"validation": results, "selected_model": best_name, "test": test_metrics, "test_engines": int(len(test_predictions))}
    (model_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args(); print(json.dumps(train(args.raw_dir), indent=2))
