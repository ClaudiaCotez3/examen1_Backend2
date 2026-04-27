# AI Service — Supervisor Insights

FastAPI sidecar that runs unsupervised models (scikit-learn) over the
Workflow Engine Mongo collections to surface bottlenecks, classify
operators by efficiency tier, and flag time-anomaly trámites.

## Running locally

```bash
cd ai-service
python -m venv .venv
.venv\Scripts\activate          # PowerShell
# (or: source .venv/bin/activate on Linux / macOS)
pip install -r requirements.txt
copy .env.example .env          # then edit if needed
uvicorn main:app --reload --port 8001
```

The service expects:

- Mongo running on `MONGODB_URI` (defaults to localhost).
- `JWT_SECRET` matching the Spring Boot value (defaults to the dev secret
  in `application.properties`).

## Endpoints

All `/insights/*` routes require `Authorization: Bearer <jwt>` with
`role` ∈ {SUPERVISOR, ADMIN}.

| Route                       | Returns |
| --------------------------- | ------- |
| `GET /healthz`              | Liveness probe (no auth). |
| `GET /insights/bottlenecks` | Activities ranked by combined wait+service z-score. |
| `GET /insights/operators`   | Operators clustered into eficiente / promedio / lento (KMeans). |
| `GET /insights/anomalies`   | Per-instance lead-time outliers (IsolationForest). |
| `GET /insights/summary`     | Three-line natural-language summary across the above. |

## Algorithms

* **Bottleneck detection** — z-score on `(avgWait + avgService)` per
  activity. Threshold +1.5σ is `CRITICAL`, +0.5σ is `WARNING`.
* **Operator clustering** — `KMeans(n_clusters=3)` over scaled
  `(avgService, completedCount)`. Centroids are sorted by efficiency
  to assign the human-readable label.
* **Anomaly detection** — `IsolationForest(contamination=0.1)` on the
  per-instance lead time of every finalised activity. Outliers are
  reported with their `caseId` and time taken.
