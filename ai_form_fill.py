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

import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
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
    today = date.today()
    today_block = today.strftime("%Y-%m-%d")

    return f"""Eres "Asistente de formularios" de un sistema de tareas. Hablas
SIEMPRE en español, en tono natural y breve, como un colega ayudando.
No uses jerga técnica ("campo", "control", "form"); usa los nombres
visibles para el usuario (las "etiquetas" de cada campo).

FECHA DE HOY: {today_block}

OBJETIVO
Cada turno el operador te dirá (por voz o texto) qué información quiere
escribir en el formulario. Suele ser un DICTADO NARRADO completo, por
ejemplo: "Hola, soy Juan Pérez. Realicé la inspección el día 20 de junio.
Visité la vivienda en la avenida Busch número 125. La instalación fue
correcta. No se requiere reparación." Tu trabajo es repartir TODA esa
información entre los campos correctos en un solo turno. Devuelve el
diccionario `values` con las parejas {{ nombre: valor }}.

REGLAS DURAS
1. Usa SIEMPRE los nombres EXACTOS de la sección "CAMPOS DEL FORMULARIO".
   No inventes nombres ni los traduzcas. La etiqueta se muestra al
   usuario; el `name` es la clave técnica.
2. Solo incluye en `values` los campos que vas a cambiar este turno.
   Si un campo ya tiene el valor correcto, no lo repitas.
3. Respeta el TIPO declarado del campo:
   - text / textarea → string
   - date → string en formato ISO "YYYY-MM-DD" SIEMPRE. Si el operador
     dice "el 20 de junio" sin año, usa el año de la FECHA DE HOY.
     "Ayer" / "hoy" / "mañana" se resuelven contra la FECHA DE HOY.
   - datetime → string ISO "YYYY-MM-DDTHH:mm" (misma regla para el año).
   - number → número (no string). Convierte números en palabras a
     dígitos ("ciento veinticinco" → 125).
   - checkbox → true / false. ATENCIÓN a las negaciones del español:
     "NO se requiere reparación", "sin inconvenientes", "no aplica"
     → false. Afirmaciones ("sí se necesita", "requiere revisión")
     → true. Una negación explícita SÍ es información: llena el campo
     con false, no lo dejes vacío.
   - select / radio → EXACTAMENTE uno de los valores listados en
     `opciones`. Si el operador parafrasea ("la instalación fue
     realizada correctamente"), elige la opción MÁS CERCANA en
     significado (p. ej. "Instalado correctamente"). NUNCA inventes una
     opción nueva ni copies la frase literal del operador.
   - tags → array de strings.
4. Frases de identidad como "soy Juan Pérez" o "mi nombre es..." van al
   campo de nombre del técnico/operador/responsable si el formulario
   tiene uno. Direcciones dictadas ("avenida Busch número 125") se
   escriben en forma compacta habitual ("Av. Busch #125").
5. Si una parte del dictado es ambigua, llena los campos claros y pide
   en `reply` SOLO lo que falte. Devuelve `values: {{}}` únicamente
   cuando no haya nada interpretable.
6. Si el operador adjunta una imagen (foto de un documento, recibo,
   formulario en papel), extrae los datos visibles y úsalos para llenar
   los campos cuando coincidan con el formulario.
7. El `reply` confirma en lenguaje natural qué llenaste. Ejemplo bueno:
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
    fields_by_name = {f.get("name"): f for f in fields if f.get("name")}
    cleaned = {
        name: _coerce_value(fields_by_name[name], value)
        for name, value in raw_values.items()
        if name in fields_by_name
    }

    return {
        "reply": payload.get("reply", ""),
        "values": cleaned,
    }


# ── Defensive type coercion ───────────────────────────────────────────
#
# Second safety net behind the prompt rules: even at temperature 0.2 the
# model occasionally emits "true" as a string, a dd/mm/yyyy date, or a
# select value with different casing. Coercing here keeps the Angular
# reactive form (and the Spring validators downstream) happy without
# bouncing the request back to the operator.

_TRUTHY = {"true", "sí", "si", "yes", "1", "verdadero"}
_FALSY = {"false", "no", "0", "falso"}

_DATE_PATTERNS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y")


def _coerce_value(field_def: dict[str, Any], value: Any) -> Any:
    ftype = str(field_def.get("type") or "text").lower()

    if ftype == "checkbox":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _TRUTHY:
                return True
            if lowered in _FALSY:
                return False
        return bool(value)

    if ftype == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            try:
                num = float(value.replace(",", "."))
                return int(num) if num.is_integer() else num
            except ValueError:
                return value
        return value

    if ftype == "date":
        return _coerce_date(value)

    if ftype in ("select", "radio"):
        return _coerce_option(field_def.get("options") or [], value)

    if ftype == "tags" and isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]

    return value


def _coerce_date(value: Any) -> Any:
    """Normalizes common date shapes to ISO YYYY-MM-DD."""
    if not isinstance(value, str):
        return value
    raw = value.strip()
    # Already ISO (possibly with a time suffix) → keep the date part.
    iso_match = re.match(r"^(\d{4}-\d{2}-\d{2})", raw)
    if iso_match:
        return iso_match.group(1)
    for pattern in _DATE_PATTERNS:
        try:
            return datetime.strptime(raw, pattern).date().isoformat()
        except ValueError:
            continue
    return value


def _coerce_option(options: list[Any], value: Any) -> Any:
    """Snaps a select/radio value onto the declared options list.

    Exact match wins; then case/accent-insensitive; then substring
    containment in either direction (handles paraphrases the prompt
    rules didn't fully normalize). Unmatchable values pass through —
    the Angular form simply won't select anything, which is visible
    (and fixable) by the operator.
    """
    if not options or not isinstance(value, str):
        return value
    if value in options:
        return value

    def norm(s: Any) -> str:
        text = str(s).strip().lower()
        replacements = str.maketrans("áéíóúüñ", "aeiouun")
        return text.translate(replacements)

    target = norm(value)
    for opt in options:
        if norm(opt) == target:
            return opt
    for opt in options:
        n = norm(opt)
        if n and (n in target or target in n):
            return opt
    return value


def reset(session_id: str) -> None:
    _sessions.pop(session_id, None)
