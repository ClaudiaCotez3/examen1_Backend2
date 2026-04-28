"""
Conversational policy-designer assistant powered by Anthropic Claude.

Round-trip protocol with the Angular client:

  request  → { sessionId, message, diagram: { lanes, nodes, edges } }
  response ← { reply: str, operations: [DiagramOp, ...] }

The model is forced to call a single tool (`emit_designer_response`)
whose JSON-Schema describes both the natural-language reply and the
list of structured BPMN operations the frontend applies. Forcing the
tool use is Anthropic's official way of getting deterministic,
schema-validated JSON out of the model — no string parsing, no
"forgot the closing brace" surprises.

Conversation memory lives in-process (`_sessions`) keyed by
`sessionId`. We don't persist it: the assistant is a designer aide,
not a transcript repository.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic


# ── Models ────────────────────────────────────────────────────────────


@dataclass
class _Turn:
    """One conversation turn as we hand it back to Anthropic.

    We store the full content list (Anthropic supports multimodal /
    tool_use blocks) so resuming the conversation reproduces the model's
    exact prior reply, including the tool call.
    """
    role: str           # "user" | "assistant"
    content: Any        # str OR list[dict] (Anthropic content blocks)
    ts: float = field(default_factory=time.time)


@dataclass
class _Session:
    turns: list[_Turn] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)


# Drop transcripts that haven't been touched for this many seconds so
# memory doesn't grow unbounded.
_SESSION_TTL_SECONDS = 60 * 60 * 4

# Cap conversation history to keep token use predictable. The model
# always sees the system prompt + last N user/assistant pairs.
_MAX_HISTORY_TURNS = 16


_sessions: dict[str, _Session] = {}


def _get_session(session_id: str) -> _Session:
    _evict_stale()
    session = _sessions.get(session_id)
    if session is None:
        session = _Session()
        _sessions[session_id] = session
    session.last_used = time.time()
    return session


def _evict_stale() -> None:
    now = time.time()
    stale = [k for k, s in _sessions.items() if now - s.last_used > _SESSION_TTL_SECONDS]
    for k in stale:
        _sessions.pop(k, None)


# ── Anthropic plumbing ───────────────────────────────────────────────


def _anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-ant-api03-replace"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Set it in ai-service/.env "
            "before using the AI assistant."
        )
    return anthropic.Anthropic(api_key=api_key)


def _model_name() -> str:
    # Opus 4.7 is the strongest reasoner in the family — it produces the
    # complete multi-pool flows (assigning tasks to every department, not
    # just one) that smaller models occasionally collapse. Override via
    # ANTHROPIC_MODEL if you want to trade quality for cost.
    return os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")


# Anthropic input schema for the single tool we force the model to call.
# Notes on schema design:
#   * `additionalProperties: false` is intentionally OMITTED — Anthropic's
#     tool schema is more permissive than OpenAI's "strict" mode and
#     emits warnings if you include it on every level.
#   * Each operation only declares the fields it actually uses; the
#     frontend reads them defensively (anything missing is undefined).
_TOOL_NAME = "emit_designer_response"

_OPERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": [
                "addLane",
                "addNode",
                "renameNode",
                "removeNode",
                "connect",
                "disconnect",
                "setBranchLabel",
                "assignUsers",
            ],
            "description": "Type of edit to apply on the BPMN canvas.",
        },
        "name": {
            "type": "string",
            "description": (
                "Name of the affected element. For renameNode/removeNode/"
                "addNode/addLane it identifies (or labels) the element."
            ),
        },
        "newName": {
            "type": "string",
            "description": "New name when renaming a node.",
        },
        "laneName": {
            "type": "string",
            "description": "Lane (departamento) the new node belongs to.",
        },
        "afterNode": {
            "type": "string",
            "description": (
                "Optional: place the new node right after this existing one. "
                "The frontend uses appendShape so the connection is drawn "
                "automatically."
            ),
        },
        "nodeType": {
            "type": "string",
            "enum": ["TASK", "START", "END", "DECISION"],
            "description": "Required for `addNode`. Anything else ignores it.",
        },
        "fromNode": {
            "type": "string",
            "description": "Source node for connect/disconnect/setBranchLabel.",
        },
        "toNode": {
            "type": "string",
            "description": "Target node for connect/disconnect/setBranchLabel.",
        },
        "branchLabel": {
            "type": "string",
            "enum": ["APROBADO", "RECHAZADO"],
            "description": (
                "Label written on a flow leaving a DECISION node. The engine "
                "uses this at runtime to keep the chosen branch and discard "
                "the rest."
            ),
        },
        "userNames": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Solo para `assignUsers`. Nombres EXACTOS de operadores tomados "
                "del bloque OPERADORES DISPONIBLES del prompt. NO inventes "
                "nombres: si el usuario menciona alguien que no aparece en la "
                "lista, omite la operación y avísalo en `reply`."
            ),
        },
    },
    "required": ["op"],
}

_TOOL_SCHEMA: dict[str, Any] = {
    "name": _TOOL_NAME,
    "description": (
        "Emit the assistant's reply to the user together with the ordered list "
        "of BPMN edit operations the frontend should apply to the canvas. "
        "Always call this tool — never reply in plain text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": (
                    "Mensaje breve y claro para el usuario en español. Resume "
                    "qué hiciste o qué hace falta. NO incluyas el JSON de "
                    "operaciones aquí; el campo `operations` ya las lleva."
                ),
            },
            "operations": {
                "type": "array",
                "description": (
                    "Lista ORDENADA de operaciones a aplicar. Vacío si la "
                    "pregunta del usuario no requiere editar el diagrama."
                ),
                "items": _OPERATION_SCHEMA,
            },
        },
        "required": ["reply", "operations"],
    },
}


def _build_system_prompt(diagram: dict[str, Any]) -> str:
    """Builds the deterministic context the model needs every turn:
    available operations, their semantics, and a snapshot of the
    current diagram so it can reference existing nodes by name.
    """
    lanes = diagram.get("lanes", [])
    nodes = diagram.get("nodes", [])
    edges = diagram.get("edges", [])
    operators = diagram.get("availableOperators", []) or []

    def fmt_lane(lane: dict) -> str:
        return f"- {lane.get('name', '?')}"

    def fmt_node(node: dict) -> str:
        lane = node.get("laneName") or "—"
        return f"- [{node.get('type', '?')}] {node.get('name', '?')} (lane: {lane})"

    def fmt_edge(edge: dict) -> str:
        label = f" [{edge.get('branchLabel')}]" if edge.get("branchLabel") else ""
        return f"- {edge.get('sourceName', '?')} → {edge.get('targetName', '?')}{label}"

    def fmt_operator(op: dict) -> str:
        return f"- {op.get('name', '?')} <{op.get('email', '?')}>"

    diagram_snapshot = (
        "Lanes:\n"
        + ("\n".join(fmt_lane(l) for l in lanes) or "  (sin lanes)")
        + "\n\nNodos:\n"
        + ("\n".join(fmt_node(n) for n in nodes) or "  (sin nodos)")
        + "\n\nFlechas:\n"
        + ("\n".join(fmt_edge(e) for e in edges) or "  (sin flechas)")
    )

    operators_block = (
        "\n".join(fmt_operator(o) for o in operators)
        if operators
        else "  (no hay operadores cargados; no uses assignUsers)"
    )

    return f"""Eres "Asistente de diseño BPMN" para una plataforma de gestión de procesos.

OBJETIVO
Ayudas al administrador a construir y editar un diagrama BPMN. Hablas
en español, eres breve y SIEMPRE devuelves tu respuesta llamando a la
herramienta `{_TOOL_NAME}` (nunca respondas en texto plano).

ACTITUD DE TRABAJO — MUY IMPORTANTE
- Eres un colaborador proactivo, no un mero ejecutor. Si el usuario te
  pide "un proceso de ventas completo" o "un proceso típico de RR.HH.",
  asume el rol de un analista experto y DISEÑA el flujo completo: tú
  decides cuántas áreas, qué tareas tiene cada una, en qué orden van y
  cómo se conectan. No le devuelvas el problema al usuario pidiendo
  más detalles, salvo que de plano sea imposible decidir.
- Si el usuario menciona N áreas (ej. "ventas, finanzas, almacén,
  logística"), TIENES que dejar nodos en CADA UNA. Un proceso donde un
  área queda vacía es un proceso inválido. Ante la duda, agrega al
  menos una tarea representativa al área "huérfana" para que aporte
  valor al flujo (verificar, registrar, notificar, archivar, etc.).
- Cuando inventes un proceso completo, reparte las responsabilidades
  como lo haría una empresa real: Ventas recibe y registra, Finanzas
  cobra y factura, Almacén verifica/prepara stock, Logística envía,
  Servicio al cliente da seguimiento. Adapta los nombres a la jerga
  del dominio que pidió el usuario.
- Las decisiones (rombos) son lo que hace interesante un proceso: si
  hay aprobación, validación o verificación de stock, mete un DECISION
  con dos ramas (APROBADO / RECHAZADO) o (DISPONIBLE / SIN STOCK). No
  hagas flujos lineales aburridos cuando un rombo es lo natural.

CADA RESPUESTA TIENE DOS PARTES
1. `reply`      — Mensaje conversacional en español sencillo, como si
                  fueras un asistente humano hablando con un usuario que
                  NO es programador. Reglas estrictas:
                    * NO uses jerga: nada de "nodo", "flow", "lane", "BPMN",
                      "evento START/END", "sequenceFlow", "pool",
                      "operations", "ops".
                    * Tradúcelo a lenguaje cotidiano: "el inicio del
                      proceso", "el fin", "la tarea Realizar Venta",
                      "el área de Almacén", "conecté A con B".
                    * No describas la lista de operaciones ("apliqué N
                      cambios"); el usuario ya las ve por separado.
                    * Sé breve (1-3 oraciones). Confirma lo que hiciste y,
                      si tomaste alguna decisión libre (por ejemplo,
                      interpretar un croquis), explícala en una frase.
2. `operations` — una lista ORDENADA de cambios a aplicar. Si el
                  usuario solo pregunta o conversa, devuelve `operations: []`.

OPERACIONES DISPONIBLES
- `addLane`           : crea un ÁREA / departamento. En este sistema cada
                        área es un POOL independiente (un `bpmn:Participant`
                        con su propio Process). DOS áreas = DOS pools, no
                        un pool dividido en swim-lanes. El frontend ya se
                        encarga de apilar los pools verticalmente.
                        Campos: `name`.
- `addNode`           : crea un nodo dentro de una lane existente.
                        Campos: `name`, `nodeType` (TASK | START | END | DECISION),
                        `laneName`. Opcional: `afterNode` para ubicarlo
                        después de un nodo existente (cualquier nombre del
                        diagrama actual).
- `renameNode`        : cambia el nombre de un nodo.
                        Campos: `name` (nombre actual), `newName`.
- `removeNode`        : elimina un nodo y sus flechas.
                        Campos: `name`.
- `connect`           : conecta dos nodos con una flecha (sequence flow).
                        Campos: `fromNode`, `toNode`. Opcional:
                        `branchLabel` (APROBADO | RECHAZADO) si la flecha
                        sale de un DECISION.
- `disconnect`        : elimina la flecha entre dos nodos.
                        Campos: `fromNode`, `toNode`.
- `setBranchLabel`    : pone APROBADO/RECHAZADO sobre una flecha existente
                        que sale de un DECISION.
                        Campos: `fromNode`, `toNode`, `branchLabel`.
- `assignUsers`       : asigna uno o más operadores responsables a una
                        tarea (`name`). Campos: `name` (nombre de la tarea),
                        `userNames` (lista de nombres EXACTOS tomados de
                        OPERADORES DISPONIBLES). NO inventes nombres.

REGLAS DURAS — debes cumplirlas SIEMPRE
1. Cada diagrama representa un FLUJO COMPLETO. Por lo tanto:
   - Debe existir UN evento START y UN evento END.
   - La PRIMERA tarea (en orden lógico) debe estar conectada al START
     (`connect: START → primera tarea`).
   - La ÚLTIMA tarea debe estar conectada al END
     (`connect: última tarea → END`).
   - TODAS las tareas deben quedar enlazadas en una cadena continua. Aunque
     el usuario no pida explícitamente las conexiones intermedias, tú las
     deduces y las creas: nunca dejes tareas huérfanas.
   - Si las tareas están en lanes distintas, conecta la última tarea de la
     lane N con la primera tarea de la lane N+1 siguiendo el orden en que
     el usuario las nombró.
   - Si creaste N áreas, CADA UNA debe tener al menos un `addNode`. Si te
     queda alguna vacía, vuelve a leer el flujo y reasigna o agrega
     tareas hasta que todas participen. Antes de devolver tu respuesta,
     verifica mentalmente: "¿hay nodos en TODAS las áreas que creé?".
2. Para `addLane`: el lienzo arranca vacío (sin pools). Cada `addLane`
   crea un POOL (Participant) NUEVO; nunca reutilices uno existente
   "dividiéndolo". Si el usuario pide N áreas, emite exactamente N
   `addLane` en orden. El frontend los apila uno debajo del otro
   automáticamente.
3. `addNode` requiere `laneName` cuando hay lanes. START y END también
   son nodos: ubícalos en una lane razonable (la primera lane suele ser
   la indicada para START; END puede ir en la última).
4. Para crear conexiones usa SIEMPRE los nombres exactos que ves en el
   diagrama o que vas a crear. El frontend resuelve nombres a IDs.
5. Para DECISION genera dos `connect` con `branchLabel: APROBADO` y
   `branchLabel: RECHAZADO`. Si las tareas destino aún no existen,
   créalas con `addNode` ANTES de los `connect`.
6. Si el usuario menciona asignar a alguien (ej: "asigna a Claudia"),
   busca el nombre en OPERADORES DISPONIBLES (case-insensitive) y emite
   `assignUsers`. Si el nombre no aparece, NO uses assignUsers y
   menciónalo en `reply`.
7. No inventes lanes ni nodos que no existen; créalos con `addLane` /
   `addNode` antes de referenciarlos.
8. Solo incluye los campos relevantes en cada operación.

REGLAS DE LAYOUT (para que el diagrama se vea limpio)
- Crea las tareas dentro de cada área en el ORDEN en que el usuario las
  mencionó. El frontend las coloca en una fila horizontal centrada
  verticalmente, así que el orden de tus `addNode` define la secuencia
  visual de izquierda a derecha. No las desordenes.
- Cuando conectes tareas dentro del mismo área, hazlo siempre en el
  sentido de la fila (la anterior con la siguiente). Eso evita que las
  flechas crucen el área de un lado a otro.
- Para conectar entre áreas distintas, conecta SOLO la última tarea del
  área de origen con la primera del área de destino. Una sola flecha
  vertical entre pools — no varias.
- NUNCA emitas el mismo `connect` dos veces. Cada par (fromNode, toNode)
  debe aparecer COMO MUCHO una sola vez en `operations`. Esto incluye:
    * No repetir A→B en dos ops distintas.
    * No emitir A→B y B→A (eso es un ciclo, no un flujo).
    * No duplicar conexiones que ya existían en el diagrama actual.
  Antes de agregar un `connect`, recórreLo mentalmente: si ya está en
  el snapshot del diagrama o ya lo emitiste en este turno, omítelo.
- START siempre es el primer `addNode` y va en la primera área. END
  siempre es el último `addNode` y va en el último área. Esto deja la
  línea principal del proceso lo más recta posible.

REGLAS PARA DECISIONES (rombo / ExclusiveGateway)
- Los rombos llevan un NOMBRE PLANO, sin signos de interrogación ni de
  exclamación. NO uses "¿Stock disponible?", "¿Aprobado?", "¿Es válido?".
  Usa formas verbales o sustantivas neutras: "Validar Stock",
  "Decidir Aprobación", "Verificar Datos", "Evaluar Pago". Los signos
  `¿` y `?` confunden a bpmn-js cuando intenta dibujar la etiqueta y a
  veces corrompen las flechas que salen del rombo.
- Cuando crees una decisión, las tareas que salen de ella se abren
  visualmente: la PRIMERA rama queda ARRIBA del rombo y la SEGUNDA queda
  ABAJO. El frontend las coloca automáticamente en esas posiciones según
  el ORDEN en que emitas los `addNode`. Por eso debes emitir primero la
  rama que quieras visualmente arriba, luego la que va abajo.
- Para CADA tarea que sale del rombo usa `afterNode: <nombre del rombo>`
  en su `addNode`. Eso le dice al frontend "este nodo nace del rombo".
- Después emite los `connect` con `branchLabel: APROBADO` (rama de
  arriba) y `branchLabel: RECHAZADO` (rama de abajo). El branchLabel SOLO
  va en el `connect`, NUNCA en el `addNode`.
- Cuando ambas ramas deben converger al END, NO uses `afterNode` en el
  `addNode` del END — déjalo sin ese campo. El frontend lo ancla en el
  extremo derecho del pool al nivel del centro, y los dos `connect`
  (rama1 → END, rama2 → END) llegan a un único END por arriba y por
  abajo. Si el usuario pide explícitamente otra ubicación para el fin
  (por ejemplo "a la izquierda"), describe eso en `reply` para que el
  usuario lo ajuste a mano; tú igual emite el END sin `afterNode`.

EJEMPLO de decisión con dos ramas:
  addNode  rombo "¿Aprobado?"  nodeType=DECISION   afterNode=tareaAnterior
  addNode  "Instalar Medidor"  nodeType=TASK       afterNode=¿Aprobado?
  addNode  "Observaciones"     nodeType=TASK       afterNode=¿Aprobado?
  addNode  END                  nodeType=END       (sin afterNode)
  connect  ¿Aprobado? → Instalar Medidor   branchLabel=APROBADO
  connect  ¿Aprobado? → Observaciones      branchLabel=RECHAZADO
  connect  Instalar Medidor → END
  connect  Observaciones → END

CUANDO EL USUARIO ADJUNTA UNA IMAGEN
- Probablemente es un boceto/croquis del proceso. Analízalo: identifica
  áreas (rectángulos grandes etiquetados), tareas (cajas internas),
  decisiones (rombos), eventos (círculos) y flechas.
- Tradúcelo a operaciones siguiendo las MISMAS REGLAS DURAS de arriba
  (un solo START y END, todas las tareas conectadas, áreas como pools).
- Si el croquis es ambiguo, escoge una interpretación razonable y
  explícala en `reply`. No le pidas al usuario que vuelva a dibujar.

ORDEN RECOMENDADO de operaciones para "crear un proceso nuevo":
  1) `addLane` por cada área (en el orden que el usuario las nombró).
  2) `addNode` START en la primera lane.
  3) `addNode` por cada tarea en su lane correspondiente, en orden.
  4) `addNode` END en la última lane.
  5) `connect` START → tarea1.
  6) `connect` tarea_i → tarea_{{i+1}} para todas las tareas (incluso entre
     lanes distintas).
  7) `connect` última_tarea → END.
  8) `assignUsers` por cada tarea con responsables.

CHECKLIST mental que DEBES hacer antes de devolver `operations`:
  □ ¿Hay exactamente 1 START y 1 END?
  □ ¿Cada `addLane` que creé tiene al menos una tarea adentro?
  □ ¿La cadena de `connect` va desde START hasta END sin huecos?
  □ ¿Las decisiones tienen dos ramas con APROBADO y RECHAZADO?
  □ ¿Estoy referenciando nombres EXACTOS (no inventé un nombre que no
    existe)?

EJEMPLO CANÓNICO de proceso completo de ventas con 4 áreas
(úsalo como plantilla mental cuando el usuario pida algo similar):

  addLane Ventas
  addLane Almacén
  addLane Finanzas
  addLane Logística
  addNode "Inicio"                nodeType=START   laneName=Ventas
  addNode "Recibir Pedido"        nodeType=TASK    laneName=Ventas
  addNode "Registrar Pedido"      nodeType=TASK    laneName=Ventas
  addNode "Verificar Stock"       nodeType=TASK    laneName=Almacén
  addNode "Validar Stock"         nodeType=DECISION laneName=Almacén
  addNode "Reservar Stock"        nodeType=TASK    laneName=Almacén afterNode="Validar Stock"
  addNode "Notificar Falta Stock" nodeType=TASK    laneName=Almacén afterNode="Validar Stock"
  addNode "Cobrar Pago"           nodeType=TASK    laneName=Finanzas
  addNode "Emitir Factura"        nodeType=TASK    laneName=Finanzas
  addNode "Coordinar Envío"       nodeType=TASK    laneName=Logística
  addNode "Entregar al Cliente"   nodeType=TASK    laneName=Logística
  addNode "Fin"                   nodeType=END     laneName=Logística
  connect "Inicio" → "Recibir Pedido"
  connect "Recibir Pedido" → "Registrar Pedido"
  connect "Registrar Pedido" → "Verificar Stock"
  connect "Verificar Stock" → "Validar Stock"
  connect "Validar Stock" → "Reservar Stock"        branchLabel=APROBADO
  connect "Validar Stock" → "Notificar Falta Stock" branchLabel=RECHAZADO
  connect "Reservar Stock" → "Cobrar Pago"
  connect "Cobrar Pago" → "Emitir Factura"
  connect "Emitir Factura" → "Coordinar Envío"
  connect "Coordinar Envío" → "Entregar al Cliente"
  connect "Entregar al Cliente" → "Fin"
  connect "Notificar Falta Stock" → "Fin"

Observa cómo CADA una de las 4 áreas tiene tareas, hay un único START
en Ventas y un único END en Logística, y todas las flechas forman una
sola cadena (con dos ramas convergiendo al fin). Ese es el estándar.

DIAGRAMA ACTUAL
{diagram_snapshot}

OPERADORES DISPONIBLES (usa estos nombres EXACTOS en `assignUsers`)
{operators_block}
"""


def _coerce_history(turns: list[_Turn]) -> list[dict[str, Any]]:
    """Anthropic expects {role, content}. We trim to the last N pairs
    to bound token use. `content` can be a string or a list of blocks
    (we stored whichever form Anthropic gave us)."""
    trimmed = turns[-_MAX_HISTORY_TURNS * 2:]
    return [{"role": t.role, "content": t.content} for t in trimmed]


# ── Public API ───────────────────────────────────────────────────────


def chat(
    session_id: str,
    user_message: str,
    diagram: dict[str, Any],
    image: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Runs one turn of the assistant. Returns `{reply, operations}`.

    `image`, when provided, is `{"data": <base64>, "mime_type": "image/png"}`
    (no data-URL prefix). It's sent to Claude as an image content block so
    the model can analyze sketches / croquis the admin attaches.
    """
    text_part = (user_message or "").strip()
    if not text_part and not image:
        return {"reply": "¿En qué te ayudo?", "operations": []}

    session = _get_session(session_id)
    client = _anthropic_client()

    # Compose the messages: history + the new user turn. Anthropic takes
    # the system prompt as a top-level argument, NOT as a role. When an
    # image is attached the user content becomes a list of blocks
    # (image first so the model "sees" before reading the question).
    messages: list[dict[str, Any]] = list(_coerce_history(session.turns))

    if image:
        fallback_text = (
            text_part
            or "Analiza este croquis y construye el diagrama BPMN siguiendo las reglas."
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
        max_tokens=2048,
        temperature=0.2,
        system=_build_system_prompt(diagram or {}),
        tools=[_TOOL_SCHEMA],
        # Force the model to call our tool — guarantees structured output.
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=messages,
    )

    tool_payload: dict[str, Any] | None = None
    for block in response.content:
        # Anthropic SDK returns a typed object; both attribute and dict
        # access work, but attribute access is the official API.
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == _TOOL_NAME:
            tool_payload = dict(block.input or {})
            break

    if tool_payload is None:
        # Fallback: model ignored the forced tool_choice (extremely rare
        # but theoretically possible). Return whatever text it gave so the
        # user at least sees something instead of an empty reply.
        text_blocks = [
            getattr(b, "text", "") for b in response.content if getattr(b, "type", None) == "text"
        ]
        return {
            "reply": "\n".join(t for t in text_blocks if t) or "No pude generar una respuesta.",
            "operations": [],
        }

    # Persist the conversation. We keep the full assistant content
    # (including the tool_use block) so subsequent turns see Claude's
    # actual previous output, not a regenerated approximation. For
    # image-bearing turns we store a text-only summary in history (the
    # raw image bytes would balloon every subsequent request) — Claude
    # already extracted the relevant info into its reply, so the
    # transcript stays useful for follow-up turns.
    history_text = text_part if text_part else "(imagen analizada)"
    if image and not text_part:
        history_text = "(usuario adjuntó una imagen / croquis)"
    elif image and text_part:
        history_text = f"{text_part}  [+ imagen adjunta]"
    session.turns.append(_Turn(role="user", content=history_text))
    session.turns.append(
        _Turn(
            role="assistant",
            content=[
                # Re-serialise content blocks back to plain dicts that
                # the next API call will accept.
                _block_to_dict(b) for b in response.content
            ],
        )
    )

    # Anthropic must follow each `tool_use` with a `tool_result`, so the
    # next user turn includes a synthetic ack pointing at the tool_use_id
    # of the assistant block we just stored. Without this, Anthropic
    # rejects the next request with a 400.
    tool_use_id = next(
        (
            getattr(b, "id", None)
            for b in response.content
            if getattr(b, "type", None) == "tool_use"
        ),
        None,
    )
    if tool_use_id:
        session.turns.append(
            _Turn(
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

    # Sanitise operations — the model occasionally emits keys with empty
    # strings; the frontend is happier when those simply aren't there.
    operations = []
    for op in tool_payload.get("operations") or []:
        cleaned = {k: v for k, v in op.items() if v not in (None, "")}
        if cleaned.get("op"):
            operations.append(cleaned)

    return {
        "reply": tool_payload.get("reply", ""),
        "operations": operations,
    }


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Turn an Anthropic content block back into a JSON-serialisable dict
    so we can store it in our in-memory session and replay it on the
    next turn."""
    btype = getattr(block, "type", None)
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", ""),
            "name": getattr(block, "name", ""),
            "input": getattr(block, "input", {}) or {},
        }
    # Fallback: best-effort dict() — shouldn't happen with our schema.
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": btype or "unknown"}


def reset(session_id: str) -> None:
    _sessions.pop(session_id, None)
