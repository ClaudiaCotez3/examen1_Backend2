"""
Bottleneck + operator-performance analytics for the supervisor.

Two unsupervised models from scikit-learn drive the insights:

  * KMeans on (avgService, completedCount) clusters operators into
    three productivity tiers — eficiente / promedio / lento — so the
    UI can colour them at a glance.
  * IsolationForest on the per-instance lead time flags trámites that
    are sitting unusually long in the same area; those become the
    "anomalías" panel.

Both run on data we read straight from Mongo. Deterministic enough to
be reproducible across runs (we seed the random_state) and fast enough
to evaluate on every request without caching.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from bson import ObjectId
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from db import get_db


# ── Data loading ──────────────────────────────────────────────────────────


def _load_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[ObjectId, dict[str, Any]],
                            dict[ObjectId, dict[str, Any]], dict[ObjectId, dict[str, Any]],
                            dict[ObjectId, dict[str, Any]]]:
    """Pulls every collection we care about into pandas. Returns dataframes
    plus lookup dicts keyed by Mongo `_id`.
    """
    db = get_db()
    instances = list(db.instancias_actividad.find())
    activities = list(db.actividades.find())
    procedures = list(db.tramites.find())
    lanes = list(db.calles.find())
    policies = list(db.politicas_negocio.find())
    users = list(db.usuarios.find())

    inst_df = pd.DataFrame(instances)
    proc_df = pd.DataFrame(procedures)
    activity_by_id = {a["_id"]: a for a in activities}
    lane_by_id = {l["_id"]: l for l in lanes}
    policy_by_id = {p["_id"]: p for p in policies}
    user_by_id = {u["_id"]: u for u in users}
    return inst_df, proc_df, activity_by_id, lane_by_id, policy_by_id, user_by_id


def _minutes(start, end) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    delta = (end - start).total_seconds() / 60.0
    return delta if delta >= 0 else None


# ── Bottleneck insights ──────────────────────────────────────────────────


@dataclass
class BottleneckInsight:
    activityId: str
    activityName: str
    laneName: str | None
    policyName: str | None
    avgServiceMinutes: float
    avgWaitMinutes: float
    backlog: int
    severity: str  # CRITICAL | WARNING | OK
    score: float   # z-score against the population mean
    explanation: str


def compute_bottlenecks() -> dict[str, Any]:
    inst_df, _proc_df, activity_by_id, lane_by_id, policy_by_id, _users = _load_frames()
    if inst_df.empty:
        return {"items": [], "summary": "No hay datos suficientes para analizar."}

    # Restrict to TASK activities (DECISION/START/END have no service time).
    inst_df = inst_df[inst_df["actividad_id"].isin([
        a_id for a_id, a in activity_by_id.items() if a.get("tipo") == "TASK"
    ])]
    if inst_df.empty:
        return {"items": [], "summary": "No hay tareas con datos suficientes."}

    inst_df["service"] = inst_df.apply(
        lambda r: _minutes(r.get("fecha_inicio"), r.get("fecha_fin")), axis=1
    )
    inst_df["wait"] = inst_df.apply(
        lambda r: _minutes(r.get("fecha_creacion"), r.get("fecha_inicio")), axis=1
    )

    rows: list[BottleneckInsight] = []
    for activity_id, group in inst_df.groupby("actividad_id"):
        activity = activity_by_id.get(activity_id, {})
        lane = lane_by_id.get(activity.get("calle_id"), {})
        policy = policy_by_id.get(activity.get("politica_id"), {})
        avg_service = float(group["service"].dropna().mean() or 0.0)
        avg_wait = float(group["wait"].dropna().mean() or 0.0)
        backlog = int(group["estado"].isin(["en_espera", "bloqueada"]).sum())
        rows.append(BottleneckInsight(
            activityId=str(activity_id),
            activityName=activity.get("nombre", "—"),
            laneName=lane.get("nombre"),
            policyName=policy.get("nombre"),
            avgServiceMinutes=round(avg_service, 2),
            avgWaitMinutes=round(avg_wait, 2),
            backlog=backlog,
            severity="OK",
            score=0.0,
            explanation="",
        ))

    if not rows:
        return {"items": [], "summary": "Sin datos suficientes."}

    # Compute z-score on combined wait+service. Anything > +1σ is a flag.
    feature = np.array([[r.avgWaitMinutes + r.avgServiceMinutes] for r in rows])
    if len(feature) >= 2:
        scaler = StandardScaler()
        zs = scaler.fit_transform(feature).flatten()
    else:
        zs = np.zeros(len(rows))
    for r, z in zip(rows, zs):
        r.score = round(float(z), 2)
        if z >= 1.5:
            r.severity = "CRITICAL"
            r.explanation = (
                f"Toma en promedio {r.avgServiceMinutes:.0f} min (espera {r.avgWaitMinutes:.0f} min) y "
                f"acumula {r.backlog} pendientes. Está {z:.1f} desviaciones por encima de la media — "
                f"considera reasignar carga o sumar un operador."
            )
        elif z >= 0.5:
            r.severity = "WARNING"
            r.explanation = (
                f"Tiempo medio elevado ({r.avgServiceMinutes:.0f} min) y {r.backlog} pendientes; "
                f"vigílala antes de que acumule más backlog."
            )
        else:
            r.explanation = (
                f"Funciona dentro del rango esperado ({r.avgServiceMinutes:.0f} min, "
                f"{r.backlog} pendientes)."
            )

    rows.sort(key=lambda r: r.score, reverse=True)
    top = [r for r in rows if r.severity != "OK"][:3]
    summary = _bottleneck_summary(top, len(rows))
    return {
        "items": [asdict(r) for r in rows],
        "summary": summary,
        "topAlerts": [asdict(r) for r in top],
    }


def _bottleneck_summary(top: list[BottleneckInsight], total: int) -> str:
    if not top:
        return f"Las {total} actividades operan dentro del rango esperado. No se detectan cuellos de botella."
    head = top[0]
    if len(top) == 1:
        return (
            f"Se identificó 1 cuello de botella sobre {total} actividades. La principal es "
            f"«{head.activityName}» — {head.explanation}"
        )
    names = ", ".join(f"«{r.activityName}»" for r in top)
    return (
        f"Se identificaron {len(top)} actividades problemáticas sobre {total}. "
        f"Las más críticas son {names}. Foco inmediato: «{head.activityName}» "
        f"(score {head.score:+.2f})."
    )


# ── Operator clustering ──────────────────────────────────────────────────


@dataclass
class OperatorCluster:
    userId: str
    fullName: str | None
    email: str | None
    completedCount: int
    avgServiceMinutes: float
    cluster: str  # EFICIENTE | PROMEDIO | LENTO
    explanation: str


def cluster_operators() -> dict[str, Any]:
    inst_df, _proc_df, _act, _lane, _pol, user_by_id = _load_frames()
    if inst_df.empty or "asignado_a" not in inst_df.columns:
        return {"items": [], "summary": "No hay datos de operadores."}

    inst_df = inst_df.dropna(subset=["asignado_a"])
    inst_df["service"] = inst_df.apply(
        lambda r: _minutes(r.get("fecha_inicio"), r.get("fecha_fin")), axis=1
    )

    rows: list[OperatorCluster] = []
    features: list[list[float]] = []
    for op_id, group in inst_df.groupby("asignado_a"):
        completed = int((group["estado"] == "finalizado").sum())
        avg_service = float(group["service"].dropna().mean() or 0.0)
        if completed == 0:
            continue  # not enough data to cluster on
        user = user_by_id.get(op_id, {})
        rows.append(OperatorCluster(
            userId=str(op_id),
            fullName=user.get("nombre"),
            email=user.get("email"),
            completedCount=completed,
            avgServiceMinutes=round(avg_service, 2),
            cluster="PROMEDIO",
            explanation="",
        ))
        features.append([avg_service, float(completed)])

    if not rows:
        return {"items": [], "summary": "Aún no hay tareas finalizadas para evaluar."}

    if len(rows) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(np.array(features))
        # Three clusters → eficiente / promedio / lento. We assign labels
        # by sorting cluster centroids on the (lower service, higher
        # completedCount) axis: lowest service-time + highest throughput
        # is "EFICIENTE".
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        centroids = scaler.inverse_transform(km.cluster_centers_)
        # Score each centroid: lower service + higher throughput = better.
        centroid_scores = [(-c[0], c[1]) for c in centroids]
        ordered = sorted(range(3), key=lambda i: centroid_scores[i], reverse=True)
        cluster_labels = {ordered[0]: "EFICIENTE", ordered[1]: "PROMEDIO", ordered[2]: "LENTO"}
        for r, lbl in zip(rows, labels):
            r.cluster = cluster_labels[int(lbl)]
    else:
        # Fewer than 3 operators → label by service-time relative to median.
        services = sorted(r.avgServiceMinutes for r in rows)
        med = services[len(services) // 2]
        for r in rows:
            if r.avgServiceMinutes <= med * 0.85:
                r.cluster = "EFICIENTE"
            elif r.avgServiceMinutes >= med * 1.25:
                r.cluster = "LENTO"
            else:
                r.cluster = "PROMEDIO"

    median_completed = float(np.median([r.completedCount for r in rows]))
    median_service = float(np.median([r.avgServiceMinutes for r in rows]))
    for r in rows:
        if r.cluster == "EFICIENTE":
            r.explanation = (
                f"{r.completedCount} tareas finalizadas con un tiempo medio de "
                f"{r.avgServiceMinutes:.0f} min, por debajo de la mediana ({median_service:.0f} min)."
            )
        elif r.cluster == "LENTO":
            r.explanation = (
                f"Tiempo medio {r.avgServiceMinutes:.0f} min vs mediana {median_service:.0f}; "
                f"completó {r.completedCount} tareas. Considera capacitación o redistribuir carga."
            )
        else:
            r.explanation = (
                f"Rendimiento dentro del promedio del equipo ({median_service:.0f} min, "
                f"~{median_completed:.0f} tareas)."
            )

    rows.sort(key=lambda r: (r.cluster != "LENTO", r.avgServiceMinutes), reverse=True)
    summary = _operator_summary(rows)
    return {"items": [asdict(r) for r in rows], "summary": summary}


def _operator_summary(rows: list[OperatorCluster]) -> str:
    by_cluster: dict[str, list[OperatorCluster]] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster, []).append(r)
    parts = []
    for label in ("EFICIENTE", "PROMEDIO", "LENTO"):
        if by_cluster.get(label):
            names = ", ".join(r.fullName or r.email or r.userId for r in by_cluster[label])
            parts.append(f"{label.capitalize()}: {names}")
    if not parts:
        return "Sin operadores activos para clasificar."
    return " · ".join(parts)


# ── Anomalies (per-instance) ─────────────────────────────────────────────


@dataclass
class AnomalyCase:
    caseId: str
    code: str | None
    activityName: str | None
    laneName: str | None
    leadMinutes: float
    explanation: str


def detect_anomalies() -> dict[str, Any]:
    inst_df, proc_df, activity_by_id, lane_by_id, _pol, _users = _load_frames()
    if inst_df.empty:
        return {"items": [], "summary": "Sin datos."}

    inst_df["lead"] = inst_df.apply(
        lambda r: _minutes(r.get("fecha_creacion") or r.get("fecha_inicio"), r.get("fecha_fin")),
        axis=1,
    )
    finalized = inst_df[inst_df["estado"] == "finalizado"].dropna(subset=["lead"])
    if len(finalized) < 5:
        return {"items": [], "summary": "Datos insuficientes para detectar anomalías (mínimo 5 instancias finalizadas)."}

    iso = IsolationForest(random_state=42, contamination=0.1)
    preds = iso.fit_predict(finalized[["lead"]].to_numpy())
    finalized = finalized.assign(_anomaly=preds)
    anomalies = finalized[finalized["_anomaly"] == -1]

    if anomalies.empty:
        return {
            "items": [],
            "summary": "Ninguna instancia se desvía significativamente de la duración esperada.",
        }

    proc_by_id = {p["_id"]: p for p in proc_df.to_dict(orient="records")}
    rows: list[AnomalyCase] = []
    for _, r in anomalies.iterrows():
        proc = proc_by_id.get(r.get("tramite_id"), {})
        activity = activity_by_id.get(r.get("actividad_id"), {})
        lane = lane_by_id.get(activity.get("calle_id"), {})
        rows.append(AnomalyCase(
            caseId=str(r.get("tramite_id")),
            code=proc.get("codigo"),
            activityName=activity.get("nombre"),
            laneName=lane.get("nombre"),
            leadMinutes=round(float(r.get("lead", 0.0)), 2),
            explanation=(
                f"Tomó {float(r.get('lead', 0.0)):.0f} min, fuera del rango habitual de la actividad."
            ),
        ))

    return {
        "items": [asdict(r) for r in rows],
        "summary": (
            f"Se detectaron {len(rows)} instancias atípicas mediante Isolation Forest. "
            f"Revísalas para entender por qué tardaron más de lo esperado."
        ),
    }


# ── Top-of-page summary ──────────────────────────────────────────────────


def render_summary() -> dict[str, Any]:
    bottlenecks = compute_bottlenecks()
    operators = cluster_operators()
    anomalies = detect_anomalies()
    return {
        "bottlenecks": bottlenecks.get("summary"),
        "operators": operators.get("summary"),
        "anomalies": anomalies.get("summary"),
    }
