"""
Módulo 3 — Clasificación Inteligente de Trámites (app móvil del cliente).

El cliente describe su necesidad en lenguaje natural ("se me dañó el
medidor de luz y necesito que lo cambien") y la IA identifica la política
de negocio adecuada del catálogo para que la app cargue su formulario
inicial.

Diseño:
  - El catálogo de políticas (nombre + descripción + áreas/actividades)
    se lee de Mongo y se inyecta como contexto — la IA SOLO puede elegir
    una política existente (tool forzada con los ids reales), nunca
    inventar una.
  - El endpoint es consumido por clientes móviles SIN cuenta de usuario,
    así que la puerta es el mismo par correo + CI del portal móvil,
    verificado contra la colección `clientes` (Opción B).
  - Stateless: cada descripción se clasifica de forma independiente.
"""
from __future__ import annotations

from typing import Any

from ai_chat import _anthropic_client, _model_name
from db import get_db


# ── Verificación del cliente (correo + CI, colección `clientes`) ───────


def verify_customer(email: str, ci: str) -> bool:
    clean_email = (email or "").strip().lower()
    clean_ci = (ci or "").strip().lower()
    if not clean_email or not clean_ci:
        return False
    db = get_db()
    for candidate in db.clientes.find({"email": {"$regex": f"^{_escape(clean_email)}$", "$options": "i"}}):
        stored_ci = str(candidate.get("ci") or "").strip().lower()
        if stored_ci and stored_ci == clean_ci:
            return True
    return False


def _escape(value: str) -> str:
    import re

    return re.escape(value)


# ── Catálogo de políticas (contexto para la IA) ────────────────────────


def _load_catalog() -> list[dict[str, Any]]:
    """Active policies with enough context for semantic matching."""
    db = get_db()
    lanes_by_policy: dict[Any, list[str]] = {}
    for lane in db.calles.find():
        lanes_by_policy.setdefault(lane.get("politica_id"), []).append(
            lane.get("nombre") or "")
    activities_by_policy: dict[Any, list[str]] = {}
    for act in db.actividades.find():
        if act.get("tipo") == "TASK":
            activities_by_policy.setdefault(act.get("politica_id"), []).append(
                act.get("nombre") or "")

    catalog = []
    for policy in db.politicas_negocio.find():
        if policy.get("estado") == "ARCHIVED":
            continue
        pid = policy["_id"]
        catalog.append({
            "policyId": str(pid),
            "name": policy.get("nombre") or "(sin nombre)",
            "description": policy.get("descripcion") or "",
            "areas": [a for a in lanes_by_policy.get(pid, []) if a],
            "activities": [a for a in activities_by_policy.get(pid, []) if a][:10],
        })
    return catalog


# ── Salida estructurada ────────────────────────────────────────────────


_TOOL_NAME = "emit_classification"


def _tool_schema(valid_ids: list[str]) -> dict[str, Any]:
    return {
        "name": _TOOL_NAME,
        "description": (
            "Emit the classification of the customer's need. Always call "
            "this tool — never reply in plain text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "policyId": {
                    # La IA solo puede elegir un id REAL del catálogo (o
                    # cadena vacía cuando nada aplica) — imposible inventar.
                    "enum": valid_ids + [""],
                    "description": (
                        "Id de la política que mejor atiende la necesidad. "
                        "Cadena vacía si NINGUNA aplica razonablemente."
                    ),
                },
                "confidence": {
                    "enum": ["ALTA", "MEDIA", "BAJA"],
                    "description": "Qué tan segura es la clasificación.",
                },
                "reply": {
                    "type": "string",
                    "description": (
                        "Mensaje breve y cálido en español para el CLIENTE "
                        "(2-3 oraciones): confirma qué entendiste y qué "
                        "trámite le recomiendas, o pide más detalle si nada "
                        "aplica. Sin jerga técnica ni ids."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "1-2 oraciones (para auditoría interna) de por qué "
                        "esa política es la adecuada."
                    ),
                },
                "alternativeIds": {
                    "type": "array",
                    "items": {"enum": valid_ids},
                    "description": (
                        "Hasta 2 políticas alternativas si la necesidad es "
                        "ambigua. Vacío si la elección es clara."
                    ),
                },
            },
            "required": ["policyId", "confidence", "reply", "reasoning"],
        },
    }


def _build_system_prompt(catalog: list[dict[str, Any]]) -> str:
    lines = []
    for entry in catalog:
        lines.append(
            f"- policyId: {entry['policyId']}\n"
            f"  nombre: {entry['name']}\n"
            f"  descripción: {entry['description'] or '(sin descripción)'}\n"
            f"  áreas: {', '.join(entry['areas']) or '—'}\n"
            f"  actividades: {', '.join(entry['activities']) or '—'}"
        )
    catalog_block = "\n".join(lines) or "(catálogo vacío)"

    return f"""Eres el "Asistente de trámites" de la app móvil de una empresa
de servicios. Un CLIENTE (no un empleado) describe en sus propias
palabras qué necesita, y tu trabajo es identificar cuál de los TRÁMITES
del catálogo atiende esa necesidad. Respondes SIEMPRE llamando a la
herramienta `emit_classification`.

REGLAS DURAS
1. Elige la política comparando la NECESIDAD del cliente con el nombre,
   la descripción, las áreas y las actividades de cada entrada del
   catálogo. El significado importa más que las palabras exactas
   ("se cortó la luz" puede mapear a "Reclamo técnico").
2. Si ninguna política aplica razonablemente, devuelve policyId: "" y
   explica en `reply` (amable, en español) que no encontraste un trámite
   para eso y qué tipos de trámite sí están disponibles.
3. Si dos políticas compiten, elige la mejor, marca confidence MEDIA o
   BAJA y lista las otras en alternativeIds.
4. `reply` habla DIRECTO al cliente: cálido, claro, sin tecnicismos, sin
   mencionar ids ni la palabra "política" (usa "trámite").
5. Nunca inventes trámites que no estén en el catálogo.

CATÁLOGO DE TRÁMITES DISPONIBLES
{catalog_block}
"""


def classify(description: str) -> dict[str, Any]:
    """Classifies a customer need against the policy catalog."""
    text = (description or "").strip()
    if not text:
        return {
            "policyId": None,
            "policyName": None,
            "confidence": "BAJA",
            "reply": "Cuéntame qué necesitas y te indico el trámite adecuado.",
            "reasoning": "",
            "alternatives": [],
        }

    catalog = _load_catalog()
    if not catalog:
        return {
            "policyId": None,
            "policyName": None,
            "confidence": "BAJA",
            "reply": "Por ahora no hay trámites disponibles para iniciar desde la app.",
            "reasoning": "Catálogo vacío.",
            "alternatives": [],
        }
    names_by_id = {entry["policyId"]: entry["name"] for entry in catalog}

    client = _anthropic_client()
    response = client.messages.create(
        model=_model_name(),
        max_tokens=1024,
        temperature=0.1,
        system=_build_system_prompt(catalog),
        tools=[_tool_schema(list(names_by_id.keys()))],
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
            "policyId": None,
            "policyName": None,
            "confidence": "BAJA",
            "reply": "No pude interpretar tu solicitud. ¿Puedes describirla de otra forma?",
            "reasoning": "",
            "alternatives": [],
        }

    policy_id = str(payload.get("policyId") or "").strip() or None
    alternatives = [
        {"policyId": alt, "policyName": names_by_id.get(alt)}
        for alt in (payload.get("alternativeIds") or [])
        if isinstance(alt, str) and alt in names_by_id and alt != policy_id
    ][:2]

    return {
        "policyId": policy_id if policy_id in names_by_id else None,
        "policyName": names_by_id.get(policy_id or ""),
        "confidence": str(payload.get("confidence") or "MEDIA"),
        "reply": str(payload.get("reply") or ""),
        "reasoning": str(payload.get("reasoning") or ""),
        "alternatives": alternatives,
    }
