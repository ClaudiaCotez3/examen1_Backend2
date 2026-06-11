"""
Motor inteligente de enrutamiento y riesgos — TensorFlow (Parcial 2).

Predice, por trámite/actividad, las cuatro salidas que pidió el negocio:

  * DEMORA   — cuántos minutos tardará una actividad (regresión Keras).
  * RIESGO   — probabilidad de que se atrase respecto a su umbral
               histórico (clasificación binaria Keras).
  * ANOMALÍA — instancias cuya duración no encaja con el patrón normal
               (autoencoder Keras; sustituye/complementa al
               IsolationForest de `insights.py`).
  * PRIORIDAD — métrica derivada (riesgo × demora) para ordenar la cola.

Y la pieza de "enrutamiento" propiamente dicha:

  * MEJOR ASIGNACIÓN — dado un conjunto de operadores candidatos para una
    actividad, recomienda a quién asignarla para minimizar la demora
    esperada. (No "elige" la rama sí/no de un nodo DECISION —eso lo
    define la regla de negocio—; optimiza a QUIÉN se enruta la tarea.)

Diseño:
  - `train_models.py` entrena OFFLINE y guarda los `.keras` + `encoders.json`
    en MODELS_DIR. Este módulo solo CARGA e INFIERE (rápido, sin reentrenar
    en cada request).
  - Si todavía no hay modelos entrenados, cada función degrada a una
    heurística transparente (campo `model: "heuristic"`), para que los
    endpoints respondan algo útil desde el día cero.
  - TensorFlow se importa de forma perezosa: el resto del ai-service
    arranca aunque TF no esté instalado.

La fuente de verdad es Mongo (solo lectura), igual que `insights.py`.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np

from db import get_db

# ── Constantes de dominio ─────────────────────────────────────────────

ESTADO_FINALIZADO = "finalizado"
ESTADO_ACTIVOS = ("en_espera", "en_proceso", "bloqueada")
TIPO_TASK = "TASK"
TIPO_DECISION = "DECISION"

UNKNOWN = "UNKNOWN"

# Campos categóricos que alimentan capas Embedding y campos numéricos
# (en este orden fijo) que alimentan la rama densa. Train e inferencia
# DEBEN compartir este contrato.
CATEGORICAL_FIELDS = ("activity", "operator", "lane")
NUMERIC_FIELDS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "backlog")
ANOMALY_FIELDS = ("lead", "wait", "service")

# ── Rutas de artefactos ───────────────────────────────────────────────


def _models_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, os.getenv("MODELS_DIR", "models"))


def _path(name: str) -> str:
    return os.path.join(_models_dir(), name)


ETA_MODEL_PATH = lambda: _path("eta.keras")          # noqa: E731
RISK_MODEL_PATH = lambda: _path("risk.keras")        # noqa: E731
ANOMALY_MODEL_PATH = lambda: _path("anomaly.keras")  # noqa: E731
ENCODERS_PATH = lambda: _path("encoders.json")       # noqa: E731


# ── Encoders (vocabularios + escaladores), persistidos en JSON ─────────


@dataclass
class Encoders:
    """Mapea categorías → índices enteros (idx 0 = desconocido) y guarda
    la media/desviación para estandarizar los features numéricos, los
    umbrales de riesgo por actividad y el umbral del autoencoder."""

    vocabs: dict[str, dict[str, int]] = field(default_factory=dict)
    numeric_mean: list[float] = field(default_factory=list)
    numeric_std: list[float] = field(default_factory=list)
    risk_threshold: dict[str, float] = field(default_factory=dict)
    anomaly_mean: list[float] = field(default_factory=list)
    anomaly_std: list[float] = field(default_factory=list)
    anomaly_threshold: float = 0.0

    # -- categóricos --
    def encode(self, field_name: str, value: str | None) -> int:
        vocab = self.vocabs.get(field_name, {})
        return vocab.get(str(value), 0) if value is not None else 0

    def vocab_size(self, field_name: str) -> int:
        # +1 por el índice 0 reservado a desconocido/relleno.
        return len(self.vocabs.get(field_name, {})) + 1

    # -- numéricos --
    def scale_numeric(self, vec: list[float]) -> list[float]:
        if not self.numeric_mean:
            return vec
        out = []
        for v, m, s in zip(vec, self.numeric_mean, self.numeric_std):
            out.append((v - m) / (s if s else 1.0))
        return out

    def scale_anomaly(self, vec: list[float]) -> list[float]:
        if not self.anomaly_mean:
            return vec
        return [
            (v - m) / (s if s else 1.0)
            for v, m, s in zip(vec, self.anomaly_mean, self.anomaly_std)
        ]

    # -- (de)serialización --
    def to_json(self) -> dict[str, Any]:
        return {
            "vocabs": self.vocabs,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "risk_threshold": self.risk_threshold,
            "anomaly_mean": self.anomaly_mean,
            "anomaly_std": self.anomaly_std,
            "anomaly_threshold": self.anomaly_threshold,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Encoders":
        return cls(
            vocabs=data.get("vocabs", {}),
            numeric_mean=data.get("numeric_mean", []),
            numeric_std=data.get("numeric_std", []),
            risk_threshold=data.get("risk_threshold", {}),
            anomaly_mean=data.get("anomaly_mean", []),
            anomaly_std=data.get("anomaly_std", []),
            anomaly_threshold=data.get("anomaly_threshold", 0.0),
        )

    def save(self) -> None:
        os.makedirs(_models_dir(), exist_ok=True)
        with open(ENCODERS_PATH(), "w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, ensure_ascii=False, indent=2)


@lru_cache(maxsize=1)
def load_encoders() -> Encoders | None:
    try:
        with open(ENCODERS_PATH(), encoding="utf-8") as fh:
            return Encoders.from_json(json.load(fh))
    except FileNotFoundError:
        return None


# ── Helpers de tiempo ─────────────────────────────────────────────────


def _minutes(start, end) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    delta = (end - start).total_seconds() / 60.0
    return delta if delta >= 0 else None


def _time_features(when: datetime | None) -> tuple[float, float, float, float]:
    """Codifica hora-del-día y día-de-la-semana de forma cíclica (sin/cos),
    para que '23h' y '0h' queden cerca."""
    if not isinstance(when, datetime):
        return 0.0, 1.0, 0.0, 1.0
    hour = when.hour + when.minute / 60.0
    dow = when.weekday()
    return (
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * dow / 7.0),
        math.cos(2 * math.pi * dow / 7.0),
    )


# ── Carga de datos desde Mongo ────────────────────────────────────────


def _load_raw() -> dict[str, Any]:
    db = get_db()
    instances = list(db.instancias_actividad.find())
    activities = {a["_id"]: a for a in db.actividades.find()}
    lanes = {l["_id"]: l for l in db.calles.find()}
    users = {u["_id"]: u for u in db.usuarios.find()}
    procedures = {p["_id"]: p for p in db.tramites.find()}
    return {
        "instances": instances,
        "activities": activities,
        "lanes": lanes,
        "users": users,
        "procedures": procedures,
    }


def _backlog_at(instances: list[dict], activity_id, at: datetime, exclude_id=None) -> float:
    """Cuántas instancias de la MISMA actividad estaban abiertas (creadas
    antes y aún sin terminar) en el instante `at`. Aproxima la carga de la
    cola que veía esa tarea al momento de entrar."""
    if not isinstance(at, datetime):
        return 0.0
    count = 0
    for inst in instances:
        if inst.get("actividad_id") != activity_id:
            continue
        if exclude_id is not None and inst.get("_id") == exclude_id:
            continue
        creado = inst.get("fecha_creacion")
        fin = inst.get("fecha_fin")
        if not isinstance(creado, datetime) or creado > at:
            continue
        if isinstance(fin, datetime) and fin <= at:
            continue
        count += 1
    return float(count)


# ── Construcción del dataset de entrenamiento ─────────────────────────


def build_training_frame() -> list[dict[str, Any]]:
    """Una fila por instancia de actividad TASK finalizada, con sus
    features y etiquetas (lead/wait/service en minutos). Es lo que
    consume `train_models.py`."""
    raw = _load_raw()
    instances = raw["instances"]
    activities = raw["activities"]

    rows: list[dict[str, Any]] = []
    for inst in instances:
        activity = activities.get(inst.get("actividad_id"))
        if not activity or activity.get("tipo") != TIPO_TASK:
            continue
        if inst.get("estado") != ESTADO_FINALIZADO:
            continue

        creado = inst.get("fecha_creacion") or inst.get("fecha_inicio")
        lead = _minutes(creado, inst.get("fecha_fin"))
        if lead is None:
            continue
        wait = _minutes(inst.get("fecha_creacion"), inst.get("fecha_inicio")) or 0.0
        service = _minutes(inst.get("fecha_inicio"), inst.get("fecha_fin")) or 0.0

        rows.append({
            "instance_id": str(inst.get("_id")),
            "activity_id": str(inst.get("actividad_id")),
            "operator_id": str(inst.get("asignado_a")) if inst.get("asignado_a") else UNKNOWN,
            "lane_id": str(activity.get("calle_id")) if activity.get("calle_id") else UNKNOWN,
            "created": creado,
            "backlog": _backlog_at(instances, inst.get("actividad_id"), creado, inst.get("_id")),
            "lead": float(lead),
            "wait": float(wait),
            "service": float(service),
        })
    return rows


# ── Featurización (compartida train ↔ inferencia) ─────────────────────


def featurize(rows: list[dict[str, Any]], encoders: Encoders) -> dict[str, np.ndarray]:
    """Convierte filas (dicts con activity_id/operator_id/lane_id/created/
    backlog) en los tensores de entrada que esperan los modelos Keras."""
    activity = np.array([encoders.encode("activity", r["activity_id"]) for r in rows], dtype=np.int32)
    operator = np.array([encoders.encode("operator", r["operator_id"]) for r in rows], dtype=np.int32)
    lane = np.array([encoders.encode("lane", r["lane_id"]) for r in rows], dtype=np.int32)

    numeric = []
    for r in rows:
        hs, hc, ds, dc = _time_features(r.get("created"))
        vec = [hs, hc, ds, dc, float(r.get("backlog", 0.0))]
        numeric.append(encoders.scale_numeric(vec))
    numeric_arr = np.array(numeric, dtype=np.float32)

    return {
        "activity": activity,
        "operator": operator,
        "lane": lane,
        "numeric": numeric_arr,
    }


# ── Carga perezosa de modelos Keras ───────────────────────────────────


@lru_cache(maxsize=1)
def _load_keras_models() -> dict[str, Any] | None:
    """Carga los tres modelos. Devuelve None si faltan o si TF no está
    instalado — los llamadores caen a la heurística."""
    try:
        from tensorflow import keras  # import perezoso
    except Exception:  # pragma: no cover - TF ausente
        return None
    try:
        return {
            "eta": keras.models.load_model(ETA_MODEL_PATH()),
            "risk": keras.models.load_model(RISK_MODEL_PATH()),
            "anomaly": keras.models.load_model(ANOMALY_MODEL_PATH()),
        }
    except (OSError, ValueError):
        return None


def models_ready() -> bool:
    return load_encoders() is not None and _load_keras_models() is not None


def engine_status() -> dict[str, Any]:
    enc = load_encoders()
    return {
        "modelsTrained": models_ready(),
        "encodersLoaded": enc is not None,
        "tensorflowAvailable": _tf_available(),
        "activitiesKnown": (len(enc.vocabs.get("activity", {})) if enc else 0),
        "operatorsKnown": (len(enc.vocabs.get("operator", {})) if enc else 0),
        "modelsDir": _models_dir(),
    }


def _tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


# ── Inferencia de bajo nivel ──────────────────────────────────────────


def _predict_eta_risk(rows: list[dict[str, Any]]) -> list[tuple[float, float, str]]:
    """Devuelve (etaMinutos, riesgo 0..1, fuente) por fila."""
    enc = load_encoders()
    models = _load_keras_models()
    if enc is None or models is None:
        return [(_heuristic_eta(r), _heuristic_risk(r), "heuristic") for r in rows]

    inputs = featurize(rows, enc)
    # El modelo de ETA predice log1p(minutos): revertimos con expm1.
    eta_log = models["eta"].predict(inputs, verbose=0).reshape(-1)
    eta = np.expm1(np.clip(eta_log, 0, None))
    risk = models["risk"].predict(inputs, verbose=0).reshape(-1)
    return [(float(max(0.0, e)), float(np.clip(rk, 0.0, 1.0)), "tensorflow")
            for e, rk in zip(eta, risk)]


def _heuristic_eta(row: dict[str, Any]) -> float:
    """Sin modelo: usa el promedio histórico de la actividad (si existe)
    inflado por el backlog actual."""
    avg = _activity_avg_lead().get(row["activity_id"], 0.0)
    backlog = float(row.get("backlog", 0.0))
    return round(avg * (1.0 + 0.1 * backlog), 1)


def _heuristic_risk(row: dict[str, Any]) -> float:
    backlog = float(row.get("backlog", 0.0))
    return float(min(1.0, backlog / 10.0))


@lru_cache(maxsize=1)
def _activity_avg_lead() -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for r in build_training_frame():
        sums.setdefault(r["activity_id"], []).append(r["lead"])
    return {k: (sum(v) / len(v)) for k, v in sums.items() if v}


def _priority_bucket(score: float) -> str:
    if score >= 0.66:
        return "ALTA"
    if score >= 0.33:
        return "MEDIA"
    return "BAJA"


# ── API pública: por trámite ──────────────────────────────────────────


def predict_case(tramite_id: str) -> dict[str, Any]:
    """Predice demora, riesgo y prioridad de un trámite EN CURSO, mirando
    sus actividades activas."""
    from bson import ObjectId

    raw = _load_raw()
    try:
        oid = ObjectId(tramite_id)
    except Exception:
        return {"tramiteId": tramite_id, "error": "id inválido"}

    active = [
        inst for inst in raw["instances"]
        if inst.get("tramite_id") == oid and inst.get("estado") in ESTADO_ACTIVOS
    ]
    if not active:
        return {
            "tramiteId": tramite_id,
            "etaMinutos": 0.0,
            "riesgo": 0.0,
            "prioridad": "BAJA",
            "actividades": [],
            "model": "n/a",
            "summary": "El trámite no tiene actividades activas.",
        }

    rows = []
    for inst in active:
        activity = raw["activities"].get(inst.get("actividad_id"), {})
        rows.append({
            "activity_id": str(inst.get("actividad_id")),
            "activity_name": activity.get("nombre") or "—",
            "operator_id": str(inst.get("asignado_a")) if inst.get("asignado_a") else UNKNOWN,
            "lane_id": str(activity.get("calle_id")) if activity.get("calle_id") else UNKNOWN,
            "created": inst.get("fecha_creacion"),
            "backlog": _backlog_at(raw["instances"], inst.get("actividad_id"),
                                   inst.get("fecha_creacion") or datetime.now()),
        })

    preds = _predict_eta_risk(rows)
    actividades = []
    for r, (eta, risk, src) in zip(rows, preds):
        actividades.append({
            "actividad": r["activity_name"],
            "etaMinutos": round(eta, 1),
            "riesgo": round(risk, 3),
        })

    eta_total = round(sum(p[0] for p in preds), 1)
    riesgo_max = round(max(p[1] for p in preds), 3)
    # Prioridad = combinación de riesgo y demora relativa. Normalizamos la
    # demora contra una jornada (480 min) para acotarla a 0..1.
    eta_norm = min(1.0, eta_total / 480.0)
    score = round(0.6 * riesgo_max + 0.4 * eta_norm, 3)

    return {
        "tramiteId": tramite_id,
        "etaMinutos": eta_total,
        "riesgo": riesgo_max,
        "prioridad": _priority_bucket(score),
        "prioridadScore": score,
        "actividades": actividades,
        "model": preds[0][2],
        "summary": _case_summary(eta_total, riesgo_max, _priority_bucket(score)),
    }


def _case_summary(eta: float, risk: float, prio: str) -> str:
    horas = eta / 60.0
    dur = f"{horas:.1f} h" if eta >= 120 else f"{eta:.0f} min"
    nivel = {"ALTA": "alto", "MEDIA": "medio", "BAJA": "bajo"}[prio]
    return (f"Demora estimada ~{dur}, riesgo de atraso {risk*100:.0f}% "
            f"(prioridad {prio.lower()}, nivel {nivel}).")


# ── API pública: mejor asignación (enrutamiento) ──────────────────────


def recommend_assignment(activity_id: str, candidate_operator_ids: list[str]) -> dict[str, Any]:
    """Dado una actividad y operadores candidatos, recomienda a quién
    asignarla para minimizar la demora esperada. ESTE es el 'enrutamiento'
    real: no decide la rama sí/no del flujo, decide a quién va la tarea."""
    from bson import ObjectId

    raw = _load_raw()
    try:
        act_oid = ObjectId(activity_id)
    except Exception:
        return {"activityId": activity_id, "error": "id inválido"}
    activity = raw["activities"].get(act_oid, {})
    lane_id = str(activity.get("calle_id")) if activity.get("calle_id") else UNKNOWN
    now = datetime.now()
    backlog = _backlog_at(raw["instances"], act_oid, now)

    if not candidate_operator_ids:
        return {"activityId": activity_id, "candidates": [],
                "summary": "No se pasaron operadores candidatos."}

    rows = [{
        "activity_id": activity_id,
        "operator_id": str(op),
        "lane_id": lane_id,
        "created": now,
        "backlog": backlog,
    } for op in candidate_operator_ids]

    preds = _predict_eta_risk(rows)
    users = raw["users"]
    candidates = []
    for op, (eta, risk, src) in zip(candidate_operator_ids, preds):
        from bson import ObjectId as _OID
        try:
            user = users.get(_OID(op), {})
        except Exception:
            user = {}
        candidates.append({
            "operatorId": op,
            "operator": user.get("nombre") or user.get("email") or op,
            "etaMinutos": round(eta, 1),
            "riesgo": round(risk, 3),
        })
    candidates.sort(key=lambda c: (c["etaMinutos"], c["riesgo"]))

    best = candidates[0]
    return {
        "activityId": activity_id,
        "activityName": activity.get("nombre"),
        "recommended": best,
        "candidates": candidates,
        "model": preds[0][2],
        "summary": (f"Asignar a {best['operator']}: menor demora esperada "
                    f"(~{best['etaMinutos']:.0f} min, riesgo {best['riesgo']*100:.0f}%)."),
    }


# ── API pública: anomalías (autoencoder TF) ───────────────────────────


def detect_anomalies_tf() -> dict[str, Any]:
    """Versión TensorFlow del detector de anomalías: el autoencoder
    aprende la 'forma normal' (lead, wait, service) y marca lo que
    reconstruye mal. Mismo shape de salida que insights.detect_anomalies
    para que el frontend lo reuse."""
    enc = load_encoders()
    models = _load_keras_models()
    raw = _load_raw()
    activities = raw["activities"]
    lanes = raw["lanes"]
    procedures = raw["procedures"]

    finalized = [
        inst for inst in raw["instances"]
        if inst.get("estado") == ESTADO_FINALIZADO
    ]
    samples = []
    for inst in finalized:
        lead = _minutes(inst.get("fecha_creacion") or inst.get("fecha_inicio"), inst.get("fecha_fin"))
        if lead is None:
            continue
        wait = _minutes(inst.get("fecha_creacion"), inst.get("fecha_inicio")) or 0.0
        service = _minutes(inst.get("fecha_inicio"), inst.get("fecha_fin")) or 0.0
        samples.append((inst, [float(lead), float(wait), float(service)]))

    if len(samples) < 5:
        return {"items": [], "summary": "Datos insuficientes para detectar anomalías (mínimo 5).",
                "model": "n/a"}

    if enc is None or models is None or not enc.anomaly_mean:
        return {"items": [], "summary": "Modelo de anomalías no entrenado todavía.",
                "model": "heuristic"}

    X = np.array([enc.scale_anomaly(vec) for _, vec in samples], dtype=np.float32)
    recon = models["anomaly"].predict(X, verbose=0)
    errors = np.mean(np.square(X - recon), axis=1)

    rows = []
    for (inst, vec), err in zip(samples, errors):
        if err <= enc.anomaly_threshold:
            continue
        activity = activities.get(inst.get("actividad_id"), {})
        lane = lanes.get(activity.get("calle_id"), {})
        proc = procedures.get(inst.get("tramite_id"), {})
        rows.append({
            "caseId": str(inst.get("tramite_id")),
            "code": proc.get("codigo"),
            "activityName": activity.get("nombre"),
            "laneName": lane.get("nombre"),
            "leadMinutes": round(vec[0], 2),
            "reconError": round(float(err), 4),
            "explanation": (f"Tomó {vec[0]:.0f} min; el autoencoder no logra "
                            f"reconstruir su patrón (error {err:.3f})."),
        })
    rows.sort(key=lambda r: r["reconError"], reverse=True)
    if not rows:
        return {"items": [], "model": "tensorflow",
                "summary": "Ninguna instancia se desvía del patrón normal aprendido."}
    return {
        "items": rows,
        "model": "tensorflow",
        "summary": (f"Se detectaron {len(rows)} instancias atípicas mediante un "
                    f"autoencoder. Revísalas para entender la desviación."),
    }


# ── API pública: cola priorizada ──────────────────────────────────────


def priorities(limit: int = 20) -> dict[str, Any]:
    """Ordena los trámites en curso por prioridad (riesgo × demora) para
    que el supervisor sepa qué atender primero."""
    raw = _load_raw()
    active_case_ids = {
        inst.get("tramite_id")
        for inst in raw["instances"]
        if inst.get("estado") in ESTADO_ACTIVOS
    }
    items = []
    for case_oid in active_case_ids:
        if case_oid is None:
            continue
        pred = predict_case(str(case_oid))
        proc = raw["procedures"].get(case_oid, {})
        pred["code"] = proc.get("codigo")
        items.append(pred)
    items.sort(key=lambda p: p.get("prioridadScore", 0.0), reverse=True)
    items = items[:limit]
    model = items[0]["model"] if items else "n/a"
    return {
        "items": items,
        "model": model,
        "summary": (f"{len(items)} trámites activos priorizados por riesgo y demora."
                    if items else "No hay trámites activos."),
    }
