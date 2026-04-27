"""
Operator-side assistant: given a dynamic form schema and the user's
voice/text instructions, returns the values the operator wants to
write into each field.

Uses the same Anthropic Claude client as `ai_chat`, but with a
different forced tool (`emit_form_values`) whose JSON schema is
deliberately permissive: a free-form `values` dictionary that maps
field names to whatever value the model thinks fits. The frontend
validates names against the form schema before applying anything.

Conversation memory is intentionally NOT shared with the designer's
session — operators and admins are different roles, and mixing
contexts would leak diagram talk into form filling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ai_chat import _anthropic_client, _model_name


# ── In-memory session store (separate from the designer's) ────────────


@dataclass
class _FillTurn:
    role: str
    content: Any
    ts: float = field(default_factory=time.time)


@dataclass
class _FillSession:
    turns: list[_FillTurn] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)


_SESSION_TTL_SECONDS = 60 * 60 * 2
_MAX_HISTORY_TURNS = 12

_sessions: dict[str, _FillSession] = {}


def _get_session(session_id: str) -> _FillSession:
    now = time.time()
    stale = [k for k, s in _sessions.items() if now - s.last_used > _SESSION_TTL_SECONDS]
    for k in stale:
        _sessions.pop(k, None)
    session = _sessions.get(session_id)
    if session is None:
        session = _FillSession()
        _sessions[session_id] = session
    session.last_used = now
    return session


# ── Tool schema ───────────────────────────────────────────────────────


_TOOL_NAME = "emit_form_values"

_TOOL_SCHEMA: dict[str, Any] = {
    "name": _TOOL_NAME,
    "description": (
        "Emit a friendly Spanish reply for the operator and the dictionary "
        "of values that should be written into the form. Always call this "
        "tool — never reply in plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": (
                    "Mensaje breve y conversacional en español para el operador "
                    "(1-2 oraciones). Confirma qué campos llenaste o pide la "
                    "información que falte. Sin jerga técnica."
                ),
            },
            "values": {
                "type": "object",
                "description": (
                    "Diccionario { nombreDelCampo: valor } SOLO con los campos "
                    "que vas a actualizar este turno. Omite los campos que ya "
                    "tienen el valor correcto. Para tipos: text/textarea/date/"
                    "datetime/select/radio usa string; number usa number; "
                    "checkbox usa boolean; tags usa array de strings. Nunca "
                    "inventes nombres de campo: usa exactamente los que están "
                    "en CAMPOS DEL FORMULARIO."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["reply", "values"],
    },
}


def _build_system_prompt(
    fields: list[dict[str, Any]],
    current_values: dict[str, Any],
) -> str:
    def fmt_field(f: dict[str, Any]) -> str:
        line = f"- {f.get('name', '?')} (etiqueta: \"{f.get('label', '')}\", tipo: {f.get('type', 'text')})"
        opts = f.get("options")
        if opts:
            line += f" — opciones: {', '.join(str(o) for o in opts)}"
        return line

    fields_block = "\n".join(fmt_field(f) for f in fields) or "  (sin campos)"
    current_block = (
        "\n".join(f"- {k}: {v!r}" for k, v in (current_values or {}).items())
        or "  (todos los campos están vacíos)"
    )

    return f"""Eres "Asistente de formularios" de un sistema de tareas. Hablas
SIEMPRE en español, en tono natural y breve, como un colega ayudando.
No uses jerga técnica ("campo", "control", "form"); usa los nombres
visibles para el usuario (las "etiquetas" de cada campo).

OBJETIVO
Cada turno el operador te dirá (por voz o texto) qué información quiere
escribir en el formulario. Tú debes devolver el diccionario `values` con
las parejas {{ nombre: valor }} que el frontend escribirá en los campos.

REGLAS DURAS
1. Usa SIEMPRE los nombres EXACTOS de la sección "CAMPOS DEL FORMULARIO".
   No inventes nombres ni los traduzcas. La etiqueta se muestra al
   usuario; el `name` es la clave técnica.
2. Solo incluye en `values` los campos que vas a cambiar este turno.
   Si un campo ya tiene el valor correcto, no lo repitas.
3. Respeta el TIPO declarado del campo:
   - text / textarea / date / datetime → string
   - number → número (no string)
   - checkbox → true / false
   - select / radio → uno de los valores listados en `opciones`
   - tags → array de strings
4. Si el operador pide algo ambiguo o falta información para un campo
   obligatorio, devuelve `values: {{}}` y pídele la información en `reply`.
5. Si el operador adjunta una imagen (foto de un documento, recibo,
   formulario en papel), extrae los datos visibles y úsalos para llenar
   los campos cuando coincidan con el formulario.
6. El `reply` confirma en lenguaje natural qué llenaste. Ejemplo bueno:
   "Listo, anoté Pedro García como nombre y 1234 como número de medidor."
   Ejemplo malo: "He establecido el campo nombre con valor Pedro García."

CAMPOS DEL FORMULARIO
{fields_block}

VALORES ACTUALES (lo que el operador ya escribió)
{current_block}
"""


# ── Public API ────────────────────────────────────────────────────────


def fill(
    session_id: str,
    user_message: str,
    fields: list[dict[str, Any]],
    current_values: dict[str, Any],
    image: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One turn of the form-fill assistant. Returns `{reply, values}`."""
    text_part = (user_message or "").strip()
    if not text_part and not image:
        return {"reply": "Cuéntame qué quieres llenar.", "values": {}}

    session = _get_session(session_id)
    client = _anthropic_client()

    messages: list[dict[str, Any]] = [
        {"role": t.role, "content": t.content}
        for t in session.turns[-_MAX_HISTORY_TURNS * 2 :]
    ]

    if image:
        fallback_text = (
            text_part or "Lee este documento y llena los campos del formulario."
        )
        user_content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.get("mime_type", "image/png"),
                    "data": image["data"],
                },
            },
            {"type": "text", "text": fallback_text},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": text_part})

    response = client.messages.create(
        model=_model_name(),
        max_tokens=1024,
        temperature=0.2,
        system=_build_system_prompt(fields, current_values),
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=messages,
    )

    payload: dict[str, Any] | None = None
    tool_use_id: str | None = None
    assistant_blocks: list[dict[str, Any]] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "tool_use" and getattr(block, "name", None) == _TOOL_NAME:
            payload = dict(getattr(block, "input", {}) or {})
            tool_use_id = getattr(block, "id", None)
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_use_id or "",
                    "name": _TOOL_NAME,
                    "input": payload or {},
                }
            )
        elif btype == "text":
            assistant_blocks.append(
                {"type": "text", "text": getattr(block, "text", "")}
            )

    if payload is None:
        return {
            "reply": "No pude interpretar tu pedido. ¿Puedes repetirlo?",
            "values": {},
        }

    # Persist a compact transcript so follow-up turns stay coherent.
    history_text = text_part or (
        "(usuario adjuntó una imagen)" if image else "(sin texto)"
    )
    session.turns.append(_FillTurn(role="user", content=history_text))
    session.turns.append(_FillTurn(role="assistant", content=assistant_blocks))
    if tool_use_id:
        session.turns.append(
            _FillTurn(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": "ok",
                    }
                ],
            )
        )

    raw_values = payload.get("values") or {}
    # Defensive filter — only let through field names that the schema
    # actually declares. Keeps stray hallucinations out of the form.
    valid_names = {f.get("name") for f in fields if f.get("name")}
    cleaned = {k: v for k, v in raw_values.items() if k in valid_names}

    return {
        "reply": payload.get("reply", ""),
        "values": cleaned,
    }


def reset(session_id: str) -> None:
    _sessions.pop(session_id, None)
