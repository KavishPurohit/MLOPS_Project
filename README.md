# AeroPulse: NASA C-MAPSS RUL MLOps project

An integrated classroom project estimating aircraft-engine Remaining Useful Life from sequential sensor measurements. FD001 is the starter subset; the code is structured for follow-on subset support.

## Scope and data status

The seven source experiment documents mentioned in the request were not present anywhere under the active Documents tree. This implementation follows the pasted specification, and does not alter any original documents. The NASA archive is downloaded locally but ignored by Git; trained model/evaluation artifacts are likewise generated locally and ignored. No predictions, scores, or deployments are fabricated.

Download NASA C-MAPSS from the [NASA dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) or its [official ZIP resource](https://data.nasa.gov/docs/legacy/CMAPSSData.zip), follow the terms shown there (the NASA catalog currently lists the license as unspecified), extract `CMAPSSData.zip`, and copy `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into `data/raw/`.

NASA rows are whitespace-delimited: engine ID, cycle, three operating settings, and 21 sensors. Training RUL is the engine's last observed cycle minus the current cycle (training target capped at 125). NASA's test RUL file contains one label per test engine: cycles remaining after its final observed row. The loader adds each row's cycles remaining until that final row. The dataset's engine ordering pairs test histories with the labels.

## Architecture and layout

Raw data → validation and RUL labels → shared sequential feature transforms and local feature snapshot → engine-wise validation of three sklearn models tracked with MLflow → selected joblib model → FastAPI plus Prometheus → single Streamlit dashboard and Grafana. GitHub Actions runs tests. Docker Compose, Kubernetes, and Terraform are alternative deployment paths.

`src/` holds data, features, metrics and training; `feature_store/` produces a reusable CSV snapshot; `api/` serves predictions; `streamlit_app/` is the nine-section dashboard; `monitoring/` configures Prometheus/Grafana; `kubernetes/` and `terraform/` deploy the API; `docs/` contains Scrum and practical evidence guidance.

## Windows setup

In PowerShell from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put the official files in `data/raw/`, then run:

```powershell
python -m pytest -q
python -m feature_store.store
python -m src.train --raw-dir data/raw
```

The training script compares scaled Linear Regression, Random Forest, and Gradient Boosting. `GroupShuffleSplit` keeps engines separate across train/validation; features use current and past readings only. Lowest validation RMSE is refit on all training engines, then evaluated once against official test labels. MAE, RMSE, R² and the asymmetric NASA PHM08 score go to MLflow and `models/evaluation.json`.

Latest actual FD001 run (seed 42, RUL cap 125): held-out engine-wise validation RMSE was 14.58 for Linear Regression, 12.47 for Random Forest, and 12.60 for Gradient Boosting. Random Forest was selected. On the separate 100-engine NASA test set, the measured MAE was 8.88 cycles, RMSE 13.01 cycles, R² 0.895, and NASA score 306.00. These outputs are local run artifacts, not committed results; rerunning training replaces them.

## Run locally

Backend (requires trained model): `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000`. API docs: `http://127.0.0.1:8000/docs`; Prometheus metrics: `http://127.0.0.1:8000/metrics/`.

Dashboard: `streamlit run streamlit_app/app.py` (default `http://localhost:8501`). MLflow UI: `mlflow ui --backend-store-uri sqlite:///./mlflow.db --host 127.0.0.1 --port 5000`.

Dashboard pages: Project Overview, Dataset Explorer, EDA, ETL & Feature Store, Model Experiments, Model Evaluation, RUL Prediction from a selected test engine or CSV upload, Monitoring, and MLOps Pipeline. Data/model/monitoring sections show setup states until real data and artifacts exist.

## Container and infrastructure commands

With Docker Desktop running and a trained model: `docker compose up --build`. Services: Streamlit 8501, API 8000, Prometheus 9090, Grafana 3000. Stop with `docker compose down`. Local Grafana default is admin/admin; change credentials for shared use.

For kind/Minikube, first build and load image, then apply and scale:

```powershell
docker build -t cmapss-rul:latest .
kind load docker-image cmapss-rul:latest
kubectl apply -f kubernetes/api.yaml
kubectl get deploy,svc,pods
kubectl scale deployment/rul-api --replicas=5
kubectl get deployment rul-api
kubectl port-forward service/rul-api 8000:8000
kubectl delete -f kubernetes/api.yaml
```

The manifest starts at 3 replicas. Supply the model at `/mnt/cmapss-model` or adapt the volume for your cluster. Verify ready replicas before claiming scaling success.

Terraform manages a separately named Docker container; don't run it with Compose on the same port. From `terraform/`: `terraform init`, `terraform validate`, `terraform plan -out tfplan`, `terraform apply tfplan`, `terraform output`, `terraform destroy`.

## Git and CI

Initialize this folder as its own repository, not the parent Documents workspace:

```powershell
git init -b main
git add .
git commit -m "Build C-MAPSS RUL MLOps project"
git switch -c feature/etl-model
git switch main
git merge --no-ff feature/etl-model
```

Create an empty GitHub repository named `cmapss-rul-mlops`, then set its remote and push:

```powershell
git remote add origin https://github.com/<your-username>/cmapss-rul-mlops.git
git push -u origin main
```

`.github/workflows/ci.yml` downloads the official FD001 files, trains and logs real experiments, and runs data, model-prediction, and API tests on each push and pull request. Check the repository's Actions tab for actual hosted run status. The practical checklist explains how to record a successful run and a deliberately failing test on a temporary branch, then restore green.

## Monitoring and limitations

Prometheus tracks request counts/status, latency, and per-request mean absolute feature z-score against the training feature distribution; Grafana provisions panels. Those are operational/input-shift diagnostics, not model quality. Quality needs labeled actual RUL and is measured offline. Treat the one-row z-score as a screening signal, not a calibrated alert, until a reference and thresholds are validated for the operating population.

FD002–FD004 are not yet validated. The feature store and MLflow store are local, not remote/online. The simulated dataset may not transfer to real fleets without domain validation. See `docs/scrum.md` and `docs/practical-checklist.md` for Agile artifacts and evidence capture.
