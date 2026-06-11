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
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

> **`--host 0.0.0.0` es obligatorio para la app móvil en teléfono físico.**
> Sin él, uvicorn enlaza solo a `127.0.0.1` y el servicio queda accesible
> únicamente desde la propia PC (la web en `localhost` funciona, pero el
> teléfono recibe «No se pudo conectar con el servidor» al clasificar un
> trámite). El backend Spring (`:8080`) ya enlaza a todas las interfaces,
> por eso login/consultas sí funcionan desde el teléfono.
>
> En Windows, además, permite el puerto 8001 entrante en el Firewall:
> ```powershell
> New-NetFirewallRule -DisplayName "ai-service 8001" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8001
> ```

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

## Motor inteligente de enrutamiento y riesgos (TensorFlow)

Módulo del Parcial 2. Tres modelos Keras entrenados sobre el historial de
`instancias_actividad` predicen, por trámite/actividad: **demora** (ETA),
**riesgo** de atraso, **prioridad** (derivada) y **anomalías**; y
recomiendan la **mejor asignación** de operador.

### Entrenar (offline)

```bash
# Con datos reales de Mongo:
python train_models.py
# Demo / poco historial — genera 800 muestras sintéticas en memoria:
python train_models.py --synthetic 800 --epochs 40
```

Esto guarda `eta.keras`, `risk.keras`, `anomaly.keras` y `encoders.json`
en `MODELS_DIR` (por defecto `./models`). El servicio solo los **carga e
infiere**; nunca reentrena en caliente. Si aún no hay modelos, los
endpoints responden con una heurística transparente (`model: "heuristic"`).

### Endpoints

| Route | Rol | Returns |
| ----- | --- | ------- |
| `GET /engine/status` | SUP/ADMIN | Si los modelos TF están entrenados/cargados. |
| `GET /engine/predict/{tramiteId}` | OP/SUP/ADMIN | ETA, riesgo y prioridad de un trámite en curso. |
| `GET /engine/priorities?limit=20` | SUP/ADMIN | Cola de activos ordenada por riesgo × demora. |
| `GET /engine/anomalies` | SUP/ADMIN | Anomalías de duración vía autoencoder TF. |
| `POST /engine/recommend-assignment` | SUP/ADMIN | Mejor operador para una actividad (`{activityId, candidateOperatorIds}`). |

### Modelos

* **ETA / demora** — red densa con *embeddings* de actividad, operador y
  calle + features numéricos (hora/día cíclicos, backlog). Objetivo
  `log1p(minutos)`, pérdida MSE.
* **Riesgo** — mismo tronco, cabeza sigmoide. Etiqueta = el lead superó
  `mediana_actividad × 1.5`.
* **Anomalías** — autoencoder sobre `(lead, wait, service)`; umbral =
  `media + 2σ` del error de reconstrucción. Reemplaza/complementa al
  IsolationForest.
* **Mejor asignación** — corre el modelo de ETA para cada operador
  candidato y recomienda el de menor demora esperada. No decide la rama
  sí/no de un nodo DECISION (eso lo define la regla de negocio); optimiza
  **a quién** se enruta la tarea.
