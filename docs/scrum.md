# Agile and Scrum plan

## Product goal

Deliver a reproducible aircraft-engine RUL workflow for FD001, with an API, operator dashboard, observable service, and repeatable deployment artifacts. Expand to FD002–FD004 after validating multi-condition feature behavior.

## Roles

- Product Owner: course/project owner; orders backlog by learning value and acceptance criteria.
- Scrum Master: student team member facilitating ceremonies, clearing blockers, and tracking sprint goal.
- Developers: data, modeling, API, UI, CI, infrastructure, and documentation contributors. Small teams can combine roles transparently.

## Product backlog

| Priority | Story | Acceptance evidence |
|---|---|---|
| P0 | Ingest and validate FD001 | Shape/schema checks and actionable missing-data errors |
| P0 | Construct engine-local RUL labels and features | RUL unit tests; training/inference schema agrees |
| P0 | Compare regressors with engine-wise validation | MLflow runs include MAE, RMSE, R² and NASA score |
| P0 | Serve engine history estimates | FastAPI schema validation, health and metrics endpoints |
| P1 | Explore data and predictions | Streamlit sections work with clear empty-data states |
| P1 | Automate tests | Actions runs on push and pull request |
| P1 | Package and observe service | Compose, Kubernetes, Terraform, Prometheus/Grafana configs |
| P2 | Extend FD002–FD004 | subset selection and operating-condition analysis |

## Sprint plan (one-week teaching sprint)

### Sprint 1 goal

Turn FD001 from raw files into a tested baseline artifact and local prediction API.

Sprint backlog: data validation and labels (2 points), feature transformations (3), model comparison and MLflow (3), API (2), automated tests and review docs (2). Definition of Done: code reviewed, tests pass, no fabricated results, instructions updated, and a demoable workflow with actual data.

### Sprint 2 goal

Deliver a usable dashboard and operational deployment/monitoring path.

Sprint backlog: dashboard sections (3), CI (2), Compose (2), Kubernetes and Terraform (3), Prometheus/Grafana (2), practical capture guide (1). Definition of Done: configuration validates; runtime claims are recorded only after actual execution.

## Ceremonies

- Sprint Planning: review backlog order, capacity, risks (dataset download, Windows container runtime), and define one sprint goal with testable acceptance evidence.
- Daily Scrum: 15 minutes: what moved the sprint goal, next work, blockers. Update board after meeting.
- Sprint Review: demonstrate a real raw-to-prediction run and dashboard/API; inspect actual CI/deployment evidence and collect stakeholder feedback.
- Retrospective: discuss one practice to keep, one friction to fix, and one action with an owner and due date. Do not label planned configuration as a successful deployment.
