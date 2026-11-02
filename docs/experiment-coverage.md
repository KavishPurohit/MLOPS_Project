# Experiment Coverage for the C-MAPSS Project

This note maps the supplied college experiment material to the integrated C-MAPSS Remaining Useful Life project. The source files remain unchanged in `Downloads/MLOPS`. Generic Iris, marks-classification, and NGINX examples in the source material are illustrative lab examples; this project uses the NASA C-MAPSS aircraft-engine task and its trained regression model.

| Experiment topic | Project implementation | Verified status and remaining evidence |
|---|---|---|
| Git and GitHub | Local Git repository, branch and merge history, project commits, and Git instructions in `README.md` | Local Git history is present. No GitHub remote is configured and no push or hosted run is claimed. |
| Agile and Scrum | `docs/scrum.md` defines the C-MAPSS product goal, roles, backlogs, sprint goals, ceremonies, and completion criteria | Project planning is documented. Capture the practical ceremony evidence listed in `docs/practical-checklist.md`; do not present planning notes as proof that meetings occurred. |
| GitHub Actions CI | `.github/workflows/ci.yml` installs dependencies, obtains FD001 data, and runs the project tests on pushes and pull requests | The local test suite passed 13 tests. Hosted success and deliberate-failure runs require publishing the repository and are pending. |
| ETL, feature store, and MLflow | `src/data.py`, `src/features.py`, `feature_store/store.py`, and `src/train.py` implement C-MAPSS ingestion, engine-local sequential features, reusable feature snapshots, model comparisons, and MLflow logging | FD001 training and separate NASA test-label evaluation ran locally. Training/inference share the feature transformer; engine-wise validation avoids engine leakage. FD002–FD004 remain unvalidated. |
| Docker and Kubernetes | `Dockerfile`, `docker-compose.yml`, `Dockerfile.k8s`, and `kubernetes/api.yaml` package and deploy the actual FastAPI model service | Compose services ran locally. Docker Desktop Kubernetes started at 3 replicas, passed the API health check, and scaled to 5/5 ready replicas. |
| Terraform | `terraform/main.tf` provisions the actual API container with the Docker provider, separately from Compose and Kubernetes | Init, validation, plan, apply, output, and destroy completed locally. Apply used the prebuilt `cmapss-rul:latest` image; build that image with `Dockerfile.k8s` before running Terraform. |
| Prometheus and Grafana | `monitoring/prometheus.yml` scrapes the API; provisioned Grafana panels cover request rate/status, errors, p95 latency, and input-shift z-score | The target reported `up`, API metrics incremented after prediction, and Grafana health returned successfully. The z-score is an input-shift diagnostic; it is not model-accuracy monitoring. Capture dashboard-panel screenshots for the practical submission. |

## Supplied source-file notes

The supplied folder contains seven distinct experiment topics. It also contains two byte-identical copies of the Terraform experiment; both originals are preserved. A separate Git/GitHub PDF is present alongside the Agile `.doc` file and the five topic-specific `.docx` materials. None of the source files were modified.

The project follows the overall request for one connected, real-data C-MAPSS system. Example code in the teaching documents is not treated as project data or as verified execution evidence. Local verification and cloud-hosted GitHub results are reported separately.
