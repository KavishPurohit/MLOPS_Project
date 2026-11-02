# Practical demonstration and evidence checklist

Capture screenshots after completing each real step; include timestamps or run IDs where useful.

1. Git: repository tree, `git log --oneline --graph --all`, feature branch, merged pull request/merge commit.
2. Agile: Product Backlog and Sprint Backlog tables, Sprint Planning notes, Daily Scrum record, Review demo, Retrospective action.
3. CI: GitHub Actions successful run and a deliberately failing test run; keep the failed-run evidence and then restore the test.
4. Data/MLflow: extracted FD001 folder, validation output, EDA charts, MLflow experiment comparison, artifact page and `models/evaluation.json`.
5. Dashboard/API: Streamlit overview, dataset chart, evaluation, successful upload prediction, FastAPI `/docs`, `/health`, `/metrics`.
6. Containers: `docker compose ps`, API response, Prometheus targets, Grafana dashboard panels.
7. Kubernetes: `kubectl get deploy,svc,pods`, scale command and observed 5/5 ready replicas; only claim success when shown by cluster output.
8. Terraform: terminal outputs for `init`, `validate`, `plan`, `apply`, `output`, and `destroy`.

Avoid screenshots that expose credentials, personal paths, or private repository tokens.
