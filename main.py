"""
FastAPI sidecar for AI-driven supervisor insights.

Listens on :8001 by default and exposes four read-only endpoints under
/insights/*. Calls Mongo directly (read-only) and runs scikit-learn
locally — no external API key needed. Authenticates with the same
HS256 JWT Spring Boot signs.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8001
"""
from __future__ import annotations

import logging
import traceback

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env before anything else reads env vars.
load_dotenv()

import ai_chat
import ai_form_fill
from insights import compute_bottlenecks, cluster_operators, detect_anomalies, render_summary
from pydantic import BaseModel, Field
from security import require_role

app = FastAPI(
    title="Workflow Engine — Insights",
    description="Supervisor analytics powered by scikit-learn (KMeans + IsolationForest).",
    version="1.0.0",
)

# CORS: explicit origins because the Angular dev server lives on a
# different port. Browsers reject `allow_origins=["*"]` together with
# credentials, and they also drop the `Authorization` header when the
# response wildcard mismatches — so we list the dev origin explicitly
# and skip credentials (we authenticate via header, not cookies).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)


_ALLOWED_ORIGINS = {"http://localhost:4200", "http://127.0.0.1:4200"}
_log = logging.getLogger("ai-service")


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Make sure 500s carry CORS headers so the browser surfaces the real
    error instead of swallowing it as an opaque CORS failure."""
    _log.error("Unhandled error on %s %s\n%s", request.method, request.url.path,
               "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    origin = request.headers.get("origin", "")
    headers: dict[str, str] = {}
    if origin in _ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
        headers=headers,
    )


supervisor_only = require_role(["SUPERVISOR", "ADMIN"])
admin_only = require_role(["ADMIN"])
operator_or_admin = require_role(["OPERATOR", "ADMIN"])


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Plain liveness probe — no auth so Docker/Kubernetes can hit it."""
    return {"status": "ok"}


@app.get("/insights/bottlenecks", dependencies=[Depends(supervisor_only)])
async def bottlenecks() -> dict:
    return compute_bottlenecks()


@app.get("/insights/operators", dependencies=[Depends(supervisor_only)])
async def operators() -> dict:
    return cluster_operators()


@app.get("/insights/anomalies", dependencies=[Depends(supervisor_only)])
async def anomalies() -> dict:
    return detect_anomalies()


@app.get("/insights/summary", dependencies=[Depends(supervisor_only)])
async def summary() -> dict:
    return render_summary()


# ── Policy-designer chat assistant ───────────────────────────────────


class DiagramLane(BaseModel):
    id: str | None = None
    name: str | None = None


class DiagramNode(BaseModel):
    id: str | None = None
    name: str | None = None
    type: str | None = None
    laneId: str | None = None
    laneName: str | None = None


class DiagramEdge(BaseModel):
    id: str | None = None
    source: str | None = None
    target: str | None = None
    sourceName: str | None = None
    targetName: str | None = None
    branchLabel: str | None = None


class DiagramOperator(BaseModel):
    name: str
    email: str


class DiagramSnapshot(BaseModel):
    lanes: list[DiagramLane] = Field(default_factory=list)
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    availableOperators: list[DiagramOperator] = Field(default_factory=list)


class ChatRequest(BaseModel):
    sessionId: str
    message: str
    diagram: DiagramSnapshot | None = None
    # Optional image attachment — base64-encoded payload (without the
    # `data:...;base64,` prefix) plus its MIME type. When present, the
    # sidecar forwards it to Claude as an image content block so the
    # model can analyze sketches / hand-drawn diagrams alongside the
    # user's text. Anthropic supports png, jpeg, gif and webp.
    imageData: str | None = None
    imageMimeType: str | None = None


class ChatResponse(BaseModel):
    reply: str
    operations: list[dict]


@app.post("/ai/chat", dependencies=[Depends(admin_only)], response_model=ChatResponse)
async def ai_chat_endpoint(req: ChatRequest) -> ChatResponse:
    diagram = req.diagram.model_dump() if req.diagram else {}
    image = None
    if req.imageData and req.imageMimeType:
        image = {"data": req.imageData, "mime_type": req.imageMimeType}
    from fastapi import HTTPException
    try:
        result = ai_chat.chat(req.sessionId, req.message, diagram, image=image)
    except RuntimeError as exc:
        # Configuration problems (missing API key, etc.).
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        # Anthropic SDK errors (auth, rate limit, bad model name, …).
        # Surface them through HTTPException so CORS headers stick.
        _log.exception("Anthropic call failed")
        raise HTTPException(
            status_code=502,
            detail=f"AI backend error ({type(exc).__name__}): {exc}",
        )
    return ChatResponse(**result)


class ChatResetRequest(BaseModel):
    sessionId: str


@app.post("/ai/chat/reset", dependencies=[Depends(admin_only)])
async def ai_chat_reset(req: ChatResetRequest) -> dict[str, bool]:
    ai_chat.reset(req.sessionId)
    return {"ok": True}


# ── Operator-side: assistant that fills dynamic forms ────────────────


class FormFieldSchema(BaseModel):
    name: str
    label: str
    type: str
    options: list[str] | None = None


class FormFillRequest(BaseModel):
    sessionId: str
    message: str
    fields: list[FormFieldSchema] = Field(default_factory=list)
    currentValues: dict[str, object] = Field(default_factory=dict)
    imageData: str | None = None
    imageMimeType: str | None = None


class FormFillResponse(BaseModel):
    reply: str
    values: dict[str, object]


@app.post(
    "/ai/form-fill",
    dependencies=[Depends(operator_or_admin)],
    response_model=FormFillResponse,
)
async def ai_form_fill_endpoint(req: FormFillRequest) -> FormFillResponse:
    image = None
    if req.imageData and req.imageMimeType:
        image = {"data": req.imageData, "mime_type": req.imageMimeType}
    from fastapi import HTTPException
    try:
        result = ai_form_fill.fill(
            session_id=req.sessionId,
            user_message=req.message,
            fields=[f.model_dump() for f in req.fields],
            current_values=req.currentValues,
            image=image,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        _log.exception("Anthropic form-fill call failed")
        raise HTTPException(
            status_code=502,
            detail=f"AI backend error ({type(exc).__name__}): {exc}",
        )
    return FormFillResponse(**result)
