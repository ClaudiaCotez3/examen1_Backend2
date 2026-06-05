"""
Módulo 4 — Reportes Inteligentes (supervisores).

El supervisor escribe una pregunta en lenguaje natural ("¿qué área tuvo
más demoras este mes?", "ranking de operadores por tareas completadas")
y el servicio devuelve un reporte estructurado: título, resumen con
cifras, tabla, sugerencia de gráfico e insights accionables.

Diseño (escalable y seguro):
  1. Python construye un DATASET agregado y determinista leyendo Mongo
     (solo lectura) — trámites, áreas, actividades, operadores, clientes
     y series mensuales. La IA NUNCA consulta la base directamente ni
     inventa cifras: solo selecciona, combina y narra números que ya
     vienen calculados.
  2. Claude recibe el dataset + la pregunta y responde mediante una
     herramienta forzada (`emit_report`) con salida 100% estructurada,
     lista para que Angular renderice tabla + gráfico sin parsing.

Stateless a propósito: cada pregunta es independiente (un reporte no
necesita memoria conversacional, y así el endpoint escala horizontal).
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from ai_chat import _anthropic_client, _model_name
from db import get_db


# ── Dataset determinista (agregado en Python, no por la IA) ────────────


def _minutes(start: datetime | None, end: datetime | None) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    delta = (end - start).total_seconds() / 60.0
    return round(delta, 1) if delta >= 0 else None


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def build_dataset() -> dict[str, Any]:
    """Aggregated, JSON-safe snapshot of the workflow's operational data."""
    db = get_db()
    procedures = list(db.tramites.find())
    instances = list(db.instancias_actividad.find())
    activities = {a["_id"]: a for a in db.actividades.find()}
    lanes = {l["_id"]: l for l in db.calles.find()}
    policies = {p["_id"]: p for p in db.politicas_negocio.find()}
    versions = {v["_id"]: v for v in db.versiones_politica.find()}
    users = {u["_id"]: u for u in db.usuarios.find()}
    customers = {c["_id"]: c for c in db.clientes.find()}
    documents = list(db.documentos.find())

    # ── Trámites ──────────────────────────────────────────────────────
    def policy_name_of(proc: dict) -> str:
        version = versions.get(proc.get("version_politica_id"))
        policy = policies.get(version.get("politica_id")) if version else None
        return (policy or {}).get("nombre") or "(política desconocida)"

    cases_by_policy: dict[str, dict[str, Any]] = {}
    cases_by_month: dict[str, dict[str, int]] = {}
    case_durations: list[float] = []
    active_cases = 0
    finished_cases = 0

    for proc in procedures:
        estado = proc.get("estado")
        if estado == "finalizado":
            finished_cases += 1
        else:
            active_cases += 1

        pname = policy_name_of(proc)
        bucket = cases_by_policy.setdefault(
            pname, {"policy": pname, "total": 0, "active": 0,
                    "finished": 0, "durations": []})
        bucket["total"] += 1
        bucket["finished" if estado == "finalizado" else "active"] += 1

        dur = _minutes(proc.get("fecha_inicio"), proc.get("fecha_fin"))
        if dur is not None:
            case_durations.append(dur)
            bucket["durations"].append(dur)

        started = proc.get("fecha_inicio")
        if isinstance(started, datetime):
            month = started.strftime("%Y-%m")
            row = cases_by_month.setdefault(month, {"started": 0, "finished": 0})
            row["started"] += 1
        ended = proc.get("fecha_fin")
        if isinstance(ended, datetime):
            month = ended.strftime("%Y-%m")
            row = cases_by_month.setdefault(month, {"started": 0, "finished": 0})
            row["finished"] += 1

    by_policy = []
    for bucket in cases_by_policy.values():
        durations = bucket.pop("durations")
        bucket["avgCaseDurationMinutes"] = _avg(durations)
        by_policy.append(bucket)
    by_policy.sort(key=lambda b: b["total"], reverse=True)

    by_month = [
        {"month": m, **counts}
        for m, counts in sorted(cases_by_month.items())
    ][-12:]

    # ── Actividades y áreas (tiempos) ─────────────────────────────────
    act_stats: dict[Any, dict[str, Any]] = {}
    lane_stats: dict[Any, dict[str, Any]] = {}
    op_stats: dict[Any, dict[str, Any]] = {}

    for inst in instances:
        activity = activities.get(inst.get("actividad_id"))
        if not activity or activity.get("tipo") != "TASK":
            continue
        lane = lanes.get(activity.get("calle_id"))
        estado = inst.get("estado")
        wait = _minutes(inst.get("fecha_creacion"), inst.get("fecha_inicio"))
        service = _minutes(inst.get("fecha_inicio"), inst.get("fecha_fin"))

        a = act_stats.setdefault(activity["_id"], {
            "activity": activity.get("nombre") or "(sin nombre)",
            "area": (lane or {}).get("nombre"),
            "policy": (policies.get(activity.get("politica_id")) or {}).get("nombre"),
            "total": 0, "completed": 0, "backlog": 0,
            "waits": [], "services": [],
        })
        a["total"] += 1
        if estado == "finalizado":
            a["completed"] += 1
        elif estado in ("en_espera", "en_proceso", "bloqueada"):
            a["backlog"] += 1
        if wait is not None:
            a["waits"].append(wait)
        if service is not None:
            a["services"].append(service)

        if lane is not None:
            l = lane_stats.setdefault(lane["_id"], {
                "area": lane.get("nombre") or "(sin nombre)",
                "total": 0, "completed": 0, "backlog": 0,
                "waits": [], "services": [],
            })
            l["total"] += 1
            if estado == "finalizado":
                l["completed"] += 1
            elif estado in ("en_espera", "en_proceso", "bloqueada"):
                l["backlog"] += 1
            if wait is not None:
                l["waits"].append(wait)
            if service is not None:
                l["services"].append(service)

        operator_id = inst.get("asignado_a")
        if operator_id is not None:
            user = users.get(operator_id)
            o = op_stats.setdefault(operator_id, {
                "operator": (user or {}).get("nombre")
                            or (user or {}).get("email") or "(desconocido)",
                "completed": 0, "inProgress": 0, "services": [],
            })
            if estado == "finalizado":
                o["completed"] += 1
            elif estado == "en_proceso":
                o["inProgress"] += 1
            if service is not None:
                o["services"].append(service)

    def finalize(rows: dict[Any, dict[str, Any]], top: int, sort_key: str) -> list[dict]:
        out = []
        for row in rows.values():
            row["avgWaitMinutes"] = _avg(row.pop("waits", []))
            row["avgServiceMinutes"] = _avg(row.pop("services", []))
            out.append(row)
        out.sort(key=lambda r: r.get(sort_key, 0), reverse=True)
        return out[:top]

    areas = finalize(lane_stats, top=30, sort_key="total")
    activity_rows = finalize(act_stats, top=40, sort_key="total")

    operator_rows = []
    for row in op_stats.values():
        row["avgServiceMinutes"] = _avg(row.pop("services", []))
        operator_rows.append(row)
    operator_rows.sort(key=lambda r: r["completed"], reverse=True)

    # ── Clientes (Opción B) y documentos ──────────────────────────────
    cases_per_client: dict[Any, int] = {}
    for proc in procedures:
        cid = proc.get("cliente_id")
        if cid is not None:
            cases_per_client[cid] = cases_per_client.get(cid, 0) + 1
    top_clients = sorted(
        (
            {"client": (customers.get(cid) or {}).get("nombre")
                       or (customers.get(cid) or {}).get("email") or "(sin nombre)",
             "cases": count}
            for cid, count in cases_per_client.items()
        ),
        key=lambda r: r["cases"],
        reverse=True,
    )[:15]

    return {
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "today": date.today().isoformat(),
        "cases": {
            "total": len(procedures),
            "active": active_cases,
            "finished": finished_cases,
            "avgCaseDurationMinutes": _avg(case_durations),
            "byPolicy": by_policy,
            "byMonth": by_month,
        },
        "areas": areas,
        "activities": activity_rows,
        "operators": operator_rows,
        "customers": {
            "total": len(customers),
            "topByCases": top_clients,
        },
        "documents": {"total": len(documents)},
    }


# ── Salida estructurada (herramienta forzada) ──────────────────────────


_TOOL_NAME = "emit_report"

_TOOL_SCHEMA: dict[str, Any] = {
    "name": _TOOL_NAME,
    "description": (
        "Emit the structured report that answers the supervisor's question. "
        "Always call this tool — never reply in plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Título corto del reporte en español (máx. 10 palabras).",
            },
            "summary": {
                "type": "string",
                "description": (
                    "Resumen ejecutivo en español (2-4 oraciones) con las cifras "
                    "CONCRETAS del dataset que responden la pregunta. Si el "
                    "dataset no alcanza para responder, dilo aquí y sugiere qué "
                    "sí se puede consultar."
                ),
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Encabezados de la tabla, en español.",
            },
            "rows": {
                "type": "array",
                "items": {"type": "array", "items": {"type": ["string", "number"]}},
                "description": (
                    "Filas de la tabla (mismo orden que columns). Usa SOLO datos "
                    "del dataset; números redondeados a 1 decimal."
                ),
            },
            "chart": {
                "type": "object",
                "description": (
                    "Gráfico sugerido cuando la respuesta compara categorías o "
                    "muestra evolución temporal. Omite el campo si una tabla basta."
                ),
                "properties": {
                    "type": {"enum": ["bar", "line", "pie"]},
                    "label": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "values": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["type", "label", "labels", "values"],
            },
            "insights": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "1-3 recomendaciones accionables en español, basadas en los "
                    "números (p. ej. reasignar operadores, revisar un área)."
                ),
            },
        },
        "required": ["title", "summary", "columns", "rows", "insights"],
    },
}


def _build_system_prompt(dataset: dict[str, Any]) -> str:
    dataset_json = json.dumps(dataset, ensure_ascii=False, default=str)
    return f"""Eres el "Analista de reportes" de un sistema de gestión de
trámites (workflow BPMN). Un SUPERVISOR te hace preguntas en lenguaje
natural sobre trámites, áreas (departamentos), actividades, tiempos,
rendimiento de operadores, clientes y documentos. Respondes SIEMPRE en
español llamando a la herramienta `emit_report`.

REGLAS DURAS
1. Usa EXCLUSIVAMENTE los números del DATASET. Nunca inventes cifras,
   nombres ni porcentajes que no puedas derivar aritméticamente de él.
2. Los tiempos del dataset están en MINUTOS. Si un valor supera 120,
   conviértelo a horas (1 decimal) y dilo explícitamente ("4.5 horas").
3. Elige las columnas y filas MÁS relevantes para la pregunta (máx. 15
   filas). Ordena la tabla por la métrica que responde la pregunta.
4. Incluye `chart` cuando compares categorías (bar/pie) o muestres
   evolución temporal (line, usando cases.byMonth). Las labels/values
   del chart deben ser un subconjunto coherente de la tabla.
5. Si la pregunta NO se puede responder con el dataset (p. ej. pide
   datos que no existen), no inventes: explica en `summary` qué falta y
   qué reportes sí están disponibles; deja `columns`/`rows` con lo más
   cercano que sí puedas mostrar.
6. `insights` siempre con recomendaciones concretas atadas a los
   números (qué área reforzar, qué operador reconocer, qué proceso
   revisar). Nada genérico tipo "seguir monitoreando".
7. Vocabulario del dominio: "trámite" = caso/expediente; "área" =
   departamento/calle/lane; "actividad" = tarea del proceso;
   "backlog" = tareas en espera o en proceso.

DATASET (agregado al {dataset.get("today")})
{dataset_json}
"""


def generate_report(question: str) -> dict[str, Any]:
    """Answers one natural-language question with a structured report."""
    text = (question or "").strip()
    if not text:
        return {
            "title": "Pregunta vacía",
            "summary": "Escribe una pregunta, por ejemplo: «¿Qué área acumula más demoras?»",
            "columns": [],
            "rows": [],
            "chart": None,
            "insights": [],
        }

    dataset = build_dataset()
    client = _anthropic_client()
    response = client.messages.create(
        model=_model_name(),
        max_tokens=2048,
        temperature=0.1,
        system=_build_system_prompt(dataset),
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[{"role": "user", "content": text}],
    )

    payload: dict[str, Any] | None = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _TOOL_NAME:
            payload = dict(getattr(block, "input", {}) or {})
            break

    if payload is None:
        return {
            "title": "Sin respuesta",
            "summary": "No pude generar el reporte. Intenta reformular la pregunta.",
            "columns": [],
            "rows": [],
            "chart": None,
            "insights": [],
        }

    # Defensive shaping so the Angular renderer never breaks.
    chart = payload.get("chart")
    if not isinstance(chart, dict) or not chart.get("labels") or not chart.get("values"):
        chart = None
    return {
        "title": str(payload.get("title") or "Reporte"),
        "summary": str(payload.get("summary") or ""),
        "columns": _as_str_list(payload.get("columns")),
        "rows": [list(r) for r in (payload.get("rows") or []) if isinstance(r, list)],
        "chart": chart,
        "insights": _as_str_list(payload.get("insights")),
    }


def _as_str_list(value: Any) -> list[str]:
    """Coerces a tool output that SHOULD be a list of strings.

    The model occasionally emits a single string instead of a list;
    iterating a string would explode it into one-character "items"
    (a real bug observed: 596 single-char insights), so strings are
    wrapped, lists are stringified per-item, anything else is dropped.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []
