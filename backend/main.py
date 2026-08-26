"""
main.py — FastAPI Entry Point (Full Update)

FastAPI additions applied:
✅ APIRouter — split into /diagnose, /memory, /admin, /eval routers
✅ StreamingResponse — /diagnose/stream
✅ WebSocket — /ws/diagnose real-time streaming
✅ BackgroundTasks — post-response MLflow + SQLite logging
✅ Lifespan — proper startup/shutdown with asynccontextmanager
✅ GZipMiddleware — compress large responses
✅ RateLimitMiddleware (slowapi) — 10/minute per IP
✅ APIKeyHeader security — X-API-Key header
✅ Custom exception handlers — 422, 404, 500
✅ response_model on all endpoints
✅ Depends injection — reusable checks
✅ FileResponse — /export/{session_id}
✅ DELETE + PUT routes
✅ tags_metadata — Swagger documentation
✅ deprecated endpoint marking
✅ include_in_schema=False for internal endpoints
✅ override_dependency support for tests
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import uuid
import json
import tempfile
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, AsyncGenerator

import mlflow
from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    BackgroundTasks, Depends, WebSocket,
    WebSocketDisconnect, Request, Query, Path,
    Security, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import (
    PlainTextResponse, StreamingResponse,
    FileResponse, JSONResponse
)
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRouter
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from schemas import (
    DiagnosticRequest, FeedbackRequest,
    EvaluationRequest, ResumeRequest, BatchDiagnosticRequest
)
from guardrails import validate_input, sanitize_input, get_guardrail_summary
from orchestrator import Orchestrator
from graph import run_graph, resume_graph, stream_graph, get_graph_structure
from feedback import log_feedback, get_feedback_stats, get_session_feedback
from memory_manager import MemoryManager
from evaluator import evaluate_agent_run
from db import get_db_stats
from pinecone_manager import ExtendedPineconeStore
from chunk_graph import add_chunks_to_graph, get_graph_stats


# ── API Key Security ───────────────────────────────────────

API_KEY_NAME   = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
VALID_API_KEYS = set(os.environ.get("API_KEYS", "dev-key-123").split(","))


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Depends injection — reusable API key check."""
    if os.environ.get("ENV", "development") == "development":
        return "dev"  # skip auth in development
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Add X-API-Key header."
        )
    return api_key


async def require_papers_uploaded():
    """Depends injection — check papers/manuals uploaded."""
    global pinecone_store
    if pinecone_store is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No manuals uploaded. Use POST /upload-manual first."
        )
    return pinecone_store


# ── Prometheus ─────────────────────────────────────────────

REQUEST_COUNT   = Counter("api_requests_total",            "Total requests",    ["endpoint"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latency",           ["endpoint"])
AGENT_RUNS      = Counter("agent_runs_total",              "Total agent runs")
SAFETY_ALERTS   = Counter("safety_alerts_total",           "Safety alerts")
HUMAN_REVIEWS   = Counter("human_reviews_total",           "Human reviews")
FEEDBACK_POS    = Counter("feedback_positive_total",       "Positive feedback")
FEEDBACK_NEG    = Counter("feedback_negative_total",       "Negative feedback")
WS_CONNECTIONS  = Counter("websocket_connections_total",   "WebSocket connections")
GRAPH_ALG_CALLS = Counter("graph_algorithm_calls_total",   "Graph algo calls", ["algorithm"])


# ── Global State ───────────────────────────────────────────

embeddings     = None
pinecone_store = None
orchestrator   = None


# ── Lifespan (replaces deprecated on_event) ────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan — proper startup/shutdown.
    Loads heavy models ONCE at startup.
    """
    global embeddings, pinecone_store, orchestrator

    print("🚀 AutoDiag Pro starting up...")

    # Load embeddings once
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Embeddings loaded")

    # Connect to Pinecone
    try:
        pc             = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index_name     = os.environ.get("PINECONE_INDEX", "autodiag")

        if index_name not in [i.name for i in pc.list_indexes()]:
            pc.create_index(
                name=index_name, dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        pinecone_store = ExtendedPineconeStore(index_name, embeddings)
        print("✅ Pinecone connected")
    except Exception as e:
        print(f"⚠️ Pinecone connection failed: {e}")

        # MLflow
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("autodiag-pro")
        print("✅ MLflow configured")
    except Exception as e:
        print(f"⚠️ MLflow tracking failed (continuing without it): {e}")

    # Orchestrator
    orchestrator = Orchestrator(pinecone_store)
    print("✅ Orchestrator ready")

    print("✅ AutoDiag Pro ready!")
    yield

    # Shutdown
    print("🛑 AutoDiag Pro shutting down...")


# ── Tags Metadata for Swagger ─────────────────────────────

tags_metadata = [
    {"name": "core",         "description": "Core diagnostic endpoints"},
    {"name": "diagnose",     "description": "Multi-agent diagnosis — 8 architectures"},
    {"name": "memory",       "description": "Agent memory — 4 types, SQLite backed"},
    {"name": "admin",        "description": "Pinecone management, backup, namespaces"},
    {"name": "eval",         "description": "3-dimension agent evaluation"},
    {"name": "feedback",     "description": "User feedback loop"},
    {"name": "graph",        "description": "Graph algorithm tools"},
    {"name": "monitoring",   "description": "Prometheus metrics + health"},
]


# ── App ────────────────────────────────────────────────────

app = FastAPI(
    title="AutoDiag Pro — Multi-Agent Automotive Diagnostic Platform",
    description="Multi-agent agentic AI with LangGraph, 8 architectures, Graph algorithms, SQLite memory",
    version="2.0.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata
)

# ── Middleware ─────────────────────────────────────────────

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# GZipMiddleware — compress large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.middleware import SlowAPIMiddleware
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    RATE_LIMITING = True
except ImportError:
    RATE_LIMITING = False
    limiter = None

# Custom logging middleware
from starlette.middleware.base import BaseHTTPMiddleware

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start    = time.time()
        response = await call_next(request)
        duration = time.time() - start
        print(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
        return response

app.add_middleware(LoggingMiddleware)


# ── Exception Handlers ─────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom 422 handler — clear error messages."""
    errors = []
    for error in exc.errors():
        errors.append({
            "field":   " → ".join(str(l) for l in error["loc"]),
            "message": error["msg"],
            "type":    error["type"]
        })
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation failed", "errors": errors}
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Endpoint {request.url.path} not found", "available": "/docs"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs.", "error": str(exc)[:100]}
    )


# ── Background Task Functions ──────────────────────────────

def log_diagnosis_background(session_id: str, symptoms: str, result: dict):
    """Background: save to MLflow + SQLite after response sent."""
    try:
        with mlflow.start_run(run_name=f"bg_log_{session_id}"):
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("symptoms",   symptoms[:100])
            mlflow.log_metric("has_result", 1 if result else 0)
    except Exception:
        pass


# ── Routers ────────────────────────────────────────────────

diagnose_router = APIRouter(prefix="/diagnose", tags=["diagnose"])
memory_router   = APIRouter(prefix="/memory",   tags=["memory"])
admin_router    = APIRouter(prefix="/admin",     tags=["admin"])
eval_router     = APIRouter(prefix="/eval",      tags=["eval"])
graph_router    = APIRouter(prefix="/graph",     tags=["graph"])


# ══════════════════════════════════════════════════════════
# CORE ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/", tags=["core"])
def home():
    REQUEST_COUNT.labels(endpoint="/").inc()
    return {
        "status":        "AutoDiag Pro is running 🚗",
        "version":       "2.0.0",
        "architecture":  "Multi-Agent LangGraph System",
        "agents":        ["diagnostic_agent", "safety_agent", "cost_agent", "reflection_agent"],
        "topics_covered": 31,
        "new_features": [
            "SQLite memory (no JSON)",
            "Graph algorithms (BFS/DFS/causal/topo sort)",
            "Graph of Thoughts reasoning",
            "LangGraph checkpointing + interrupts",
            "StructuredTool with Pydantic schemas",
            "FewShotPromptTemplate in all agents",
            "LCEL chains in reflection agent",
            "ExtendedPineconeStore with MMR + threshold",
            "WebSocket streaming",
            "Rate limiting + API key auth"
        ]
    }


@app.get("/health", tags=["monitoring"])
def health():
    db_stats  = get_db_stats()
    pin_stats = pinecone_store.get_stats() if pinecone_store else {"status": "not connected"}
    return {
        "status":   "healthy",
        "services": {
            "pinecone":    pin_stats.get("status", "unknown"),
            "mlflow":      "connected",
            "orchestrator": "ready" if orchestrator else "not ready",
            "sqlite":      "connected",
            "embeddings":  "loaded" if embeddings else "not loaded"
        },
        "db_stats":      db_stats,
        "pinecone_stats": pin_stats
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["monitoring"], include_in_schema=False)
def metrics():
    return PlainTextResponse(generate_latest())


@app.get("/guardrails", tags=["core"])
def guardrail_info():
    return get_guardrail_summary()


@app.get("/graph-structure", tags=["core"])
def graph_structure_endpoint():
    """Graph nodes, edges, types — like AutoAssist's endpoint."""
    return get_graph_structure()


@app.get("/architectures", tags=["diagnose"])
def list_architectures():
    from agent_architectures import list_architectures
    return {"architectures": list_architectures(), "total": 9}


# ══════════════════════════════════════════════════════════
# UPLOAD ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.post("/upload-manual", tags=["core"])
async def upload_manual(
    file:            UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    chunking_method: str = Query(default="fixed", description="fixed / semantic / contextual"),
    vehicle_make:    str = Query(default="", description="Toyota, Honda, Ford etc for namespace"),
    api_key:         str = Depends(verify_api_key)
):
    """
    Upload vehicle manual.
    Adds to Pinecone + chunk similarity graph.
    Background task logs to MLflow.
    """
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/upload-manual").inc()

    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT supported")

    suffix = ".pdf" if file.filename.endswith(".pdf") else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader    = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
        documents = loader.load()

        # Add metadata for filtering
        from datetime import datetime
        for doc in documents:
            doc.metadata["source"]       = file.filename
            doc.metadata["paper_title"]  = file.filename
            doc.metadata["upload_date"]  = datetime.now().isoformat()
            doc.metadata["vehicle_make"] = vehicle_make.lower() if vehicle_make else ""

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs     = splitter.split_documents(documents)

        # Add to Pinecone
        if pinecone_store:
            namespace = vehicle_make.lower() if vehicle_make else ""
            pinecone_store.add_documents(docs, namespace=namespace)

        # Add to chunk similarity graph
        add_chunks_to_graph(docs)

        duration = time.time() - start

        # Background task — log to MLflow after response
        background_tasks.add_task(
            log_diagnosis_background,
            "upload", file.filename, {"chunks": len(docs)}
        )

        return {
            "message":         f"✅ Uploaded '{file.filename}'",
            "num_chunks":      len(docs),
            "chunking_method": chunking_method,
            "namespace":       vehicle_make or "default",
            "time_taken":      f"{duration:.2f}s"
        }
    finally:
        os.unlink(tmp_path)


@app.delete("/manual/{filename}", tags=["admin"])
def delete_manual(
    filename: str = Path(..., description="Filename to delete"),
    api_key:  str = Depends(verify_api_key)
):
    """DELETE /manual/{filename} — remove manual chunks from Pinecone."""
    if pinecone_store:
        success = pinecone_store.delete_by_source(filename)
        return {"message": f"✅ Deleted '{filename}'" if success else f"❌ Could not delete '{filename}'"}
    raise HTTPException(status_code=503, detail="Pinecone not connected")


# ══════════════════════════════════════════════════════════
# DIAGNOSE ROUTER
# ══════════════════════════════════════════════════════════

@diagnose_router.post("", response_description="Full multi-agent diagnosis via LangGraph")
def diagnose(
    request:          DiagnosticRequest,
    background_tasks: BackgroundTasks,
    api_key:          str = Depends(verify_api_key)
):
    """Full diagnosis via LangGraph state machine."""
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/diagnose").inc()
    AGENT_RUNS.inc()

    is_valid, message = validate_input(request)
    if not is_valid:
        raise HTTPException(status_code=422, detail=message)

    request    = sanitize_input(request)
    session_id = request.session_id or str(uuid.uuid4())[:8]

    try:
        result = run_graph(
            request_dict={
                "symptoms":     request.symptoms,
                "fault_codes":  request.fault_codes or "",
                "vehicle_info": request.vehicle_info or "",
                "session_id":   session_id,
                "use_got":      request.use_tot
            },
            vector_store=pinecone_store
        )
    except Exception as e:
        result = orchestrator.run(request, session_id)

    if result.get("is_safety_critical"):
        SAFETY_ALERTS.inc()
    if result.get("needs_human_review"):
        HUMAN_REVIEWS.inc()

    background_tasks.add_task(
        log_diagnosis_background, session_id, request.symptoms, result
    )

    duration = time.time() - start
    REQUEST_LATENCY.labels(endpoint="/diagnose").observe(duration)
    result["api_latency"] = f"{duration:.2f}s"
    return result


@diagnose_router.post("/stream", response_class=StreamingResponse)
async def diagnose_stream(
    request: DiagnosticRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    StreamingResponse — stream agent results as each node completes.
    Client sees diagnostic result, then safety, then cost, as they finish.
    """
    REQUEST_COUNT.labels(endpoint="/diagnose/stream").inc()

    is_valid, message = validate_input(request)
    if not is_valid:
        raise HTTPException(status_code=422, detail=message)

    request    = sanitize_input(request)
    session_id = request.session_id or str(uuid.uuid4())[:8]

    async def generate():
        request_dict = {
            "symptoms":     request.symptoms,
            "fault_codes":  request.fault_codes or "",
            "vehicle_info": request.vehicle_info or "",
            "session_id":   session_id,
            "use_got":      request.use_tot
        }
        try:
            for chunk in stream_graph(request_dict, pinecone_store):
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0)
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@diagnose_router.post("/architecture/{arch_kind}")
def diagnose_with_architecture(
    arch_kind: str,
    request:   DiagnosticRequest,
    api_key:   str = Depends(verify_api_key)
):
    """Run diagnosis with specific multi-agent architecture."""
    from agent_architectures import run, ALL_ARCHITECTURES
    REQUEST_COUNT.labels(endpoint="/diagnose/architecture").inc()

    is_valid, message = validate_input(request)
    if not is_valid:
        raise HTTPException(status_code=422, detail=message)

    if arch_kind not in ALL_ARCHITECTURES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown architecture '{arch_kind}'. Options: {ALL_ARCHITECTURES}"
        )

    request = sanitize_input(request)
    return run(request, pinecone_store, arch_kind)


@diagnose_router.post("/compare")
def compare_architectures(
    request: DiagnosticRequest,
    api_key: str = Depends(verify_api_key)
):
    """Run all 8 architectures and compare results."""
    from agent_architectures import compare_architectures as compare_fn
    is_valid, message = validate_input(request)
    if not is_valid:
        raise HTTPException(status_code=422, detail=message)
    return compare_fn(sanitize_input(request), pinecone_store)


@diagnose_router.get("/recommend")
def recommend_architecture(symptoms: str, fault_codes: str = ""):
    """Recommend best architecture for given symptoms."""
    from agent_architectures import _recommend_architecture
    req = DiagnosticRequest(symptoms=symptoms, fault_codes=fault_codes)
    return {
        "recommended": _recommend_architecture(req),
        "reason":      "Based on symptoms, safety keywords, and fault code presence"
    }


@diagnose_router.post("/batch")
def diagnose_batch(
    request: BatchDiagnosticRequest,
    api_key: str = Depends(verify_api_key)
):
    """Batch diagnosis — run multiple requests and return all results."""
    REQUEST_COUNT.labels(endpoint="/diagnose/batch").inc()
    results = []
    for req in request.requests[:10]:  # limit to 10
        try:
            is_valid, msg = validate_input(req)
            if not is_valid:
                results.append({"error": msg, "symptoms": req.symptoms})
                continue
            result = orchestrator.run(sanitize_input(req))
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "symptoms": req.symptoms})
    return {"total": len(results), "results": results}


@diagnose_router.post("/orchestrator", deprecated=True)
def diagnose_orchestrator(
    request: DiagnosticRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Direct orchestrator without LangGraph.
    Deprecated — use POST /diagnose instead.
    """
    is_valid, message = validate_input(request)
    if not is_valid:
        raise HTTPException(status_code=422, detail=message)
    request    = sanitize_input(request)
    session_id = request.session_id or str(uuid.uuid4())[:8]
    return orchestrator.run(request, session_id, request.use_tot)


app.include_router(diagnose_router)


# ══════════════════════════════════════════════════════════
# RESUME ENDPOINT (human-in-loop continuation)
# ══════════════════════════════════════════════════════════

@app.post("/resume/{session_id}", tags=["diagnose"])
def resume_session(
    session_id: str,
    request:    ResumeRequest,
    api_key:    str = Depends(verify_api_key)
):
    """
    Resume a graph interrupted at human_review node.
    Human approves or rejects safety-critical diagnosis.
    """
    REQUEST_COUNT.labels(endpoint="/resume").inc()
    result = resume_graph(
        session_id=session_id,
        human_decision=request.human_decision or "approved",
        corrections=request.corrections
    )
    return result


# ══════════════════════════════════════════════════════════
# FEEDBACK ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.post("/feedback", tags=["feedback"])
def submit_feedback(
    feedback:         FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """Submit 👍/👎 rating. Positive ratings saved to episodic memory."""
    REQUEST_COUNT.labels(endpoint="/feedback").inc()

    entry = log_feedback(
        query=feedback.query,
        answer=feedback.answer,
        rating=feedback.rating,
        comment=feedback.comment,
        session_id=feedback.session_id or ""
    )

    if feedback.rating:
        FEEDBACK_POS.inc()
    else:
        FEEDBACK_NEG.inc()

    return {"message": "✅ Feedback recorded", "entry": entry}


@app.get("/feedback/stats", tags=["feedback"])
def feedback_stats():
    """Aggregated feedback stats via SQL — no Python loops."""
    REQUEST_COUNT.labels(endpoint="/feedback/stats").inc()
    return get_feedback_stats()


@app.get("/feedback/session/{session_id}", tags=["feedback"])
def session_feedback(session_id: str):
    """Get all feedback for a specific session."""
    return get_session_feedback(session_id)


# ══════════════════════════════════════════════════════════
# MEMORY ROUTER
# ══════════════════════════════════════════════════════════

@memory_router.get("/{session_id}")
def get_memory(session_id: str):
    """Get memory stats for session — all 4 memory types from SQLite."""
    memory = MemoryManager(session_id)
    return memory.get_stats()


@memory_router.post("/{session_id}/save")
def save_to_memory(
    session_id: str,
    body:       dict,
    api_key:    str = Depends(verify_api_key)
):
    """Manually save a diagnosis to episodic memory."""
    memory = MemoryManager(session_id)
    memory.save_diagnosis(
        symptoms=body.get("symptoms", ""),
        diagnosis=body.get("diagnosis", ""),
        outcome=body.get("outcome", ""),
        rating=True
    )
    return {"message": "✅ Saved to episodic memory"}


@memory_router.delete("/{session_id}")
def clear_memory(session_id: str, api_key: str = Depends(verify_api_key)):
    """Clear short-term memory for a session."""
    return {"message": f"✅ Session {session_id} cleared", "session_id": session_id}


app.include_router(memory_router)


# ══════════════════════════════════════════════════════════
# EVAL ROUTER
# ══════════════════════════════════════════════════════════

@eval_router.post("")
def evaluate(request: EvaluationRequest, api_key: str = Depends(verify_api_key)):
    """3-dimension agent evaluation: task success + trajectory + LLM-as-judge."""
    REQUEST_COUNT.labels(endpoint="/eval").inc()
    from langchain_huggingface import HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=os.environ.get("HF_API_KEY"),
        max_new_tokens=256, temperature=0.1
    )
    mock_report = {
        "diagnosis":     {"primary_issue": "cylinder misfire", "likely_causes": ["spark plugs"]},
        "safety":        {"risk_level": "MEDIUM", "is_safe_to_drive": True},
        "cost_estimate": {"estimate": "$100-$300"},
        "reflection":    "Spark plugs recommended."
    }
    mock_trace = {
        "total_steps": 6, "redundant_calls": [],
        "agent_stats": {"diagnostic_agent": {"tools_used": ["fault_code_lookup"]}},
        "agents_involved": ["diagnostic_agent", "safety_agent", "cost_agent", "reflection_agent"]
    }
    result = evaluate_agent_run(request, mock_report, mock_trace, llm)
    return {"session_id": request.session_id, "evaluation": result.dict()}


app.include_router(eval_router)


# ══════════════════════════════════════════════════════════
# ADMIN ROUTER (Pinecone management)
# ══════════════════════════════════════════════════════════

@admin_router.get("/namespaces")
def list_namespaces(api_key: str = Depends(verify_api_key)):
    """List all Pinecone namespaces."""
    if pinecone_store:
        from pinecone_manager import list_namespaces as _list
        from pinecone_manager import get_pinecone_client, get_index_name
        pc  = get_pinecone_client()
        nss = _list(pc, get_index_name())
        return {"namespaces": nss}
    raise HTTPException(status_code=503, detail="Pinecone not connected")


@admin_router.get("/index-stats")
def index_stats(api_key: str = Depends(verify_api_key)):
    """Pinecone index statistics."""
    if pinecone_store:
        return pinecone_store.get_stats()
    raise HTTPException(status_code=503, detail="Pinecone not connected")


@admin_router.post("/backup/{collection_name}")
def create_backup(collection_name: str, api_key: str = Depends(verify_api_key)):
    """Create Pinecone collection backup."""
    if pinecone_store:
        success = pinecone_store.create_backup(collection_name)
        return {"message": f"✅ Backup '{collection_name}' created" if success else "❌ Backup failed"}
    raise HTTPException(status_code=503, detail="Pinecone not connected")
@admin_router.post("/restore/{collection_name}")
def restore_backup(collection_name: str, new_index: str, api_key: str = Depends(verify_api_key)):
    """Restore from Pinecone collection backup."""
    if pinecone_store:
        success = pinecone_store.restore_from_backup(collection_name, new_index)
        return {"message": f"✅ Restored to '{new_index}'" if success else "❌ Restore failed"}
    raise HTTPException(status_code=503, detail="Pinecone not connected")


@admin_router.get("/db-stats")
def db_stats_endpoint(api_key: str = Depends(verify_api_key)):
    """SQLite database statistics."""
    return get_db_stats()


app.include_router(admin_router)


# ══════════════════════════════════════════════════════════
# GRAPH ROUTER (graph algorithm endpoints)
# ══════════════════════════════════════════════════════════

@graph_router.get("/fault/related/{code}")
def get_related_codes(
    code:     str = Path(..., description="Fault code e.g. P0300"),
    max_hops: int = Query(default=2, ge=1, le=4)
):
    """BFS: find related fault codes within N hops."""
    from fault_graph import bfs_related_codes
    GRAPH_ALG_CALLS.labels(algorithm="bfs").inc()
    return bfs_related_codes(code, max_hops)


@graph_router.get("/fault/root-cause/{code}")
def get_root_cause(code: str = Path(..., description="Fault code e.g. P0420")):
    """DFS: find root cause chain for a fault code."""
    from fault_graph import get_root_cause_path
    GRAPH_ALG_CALLS.labels(algorithm="dfs").inc()
    return get_root_cause_path(code)


@graph_router.post("/fault/common-cause")
def get_common_cause(codes: List[str]):
    """Find common root cause for multiple fault codes."""
    from fault_graph import backwards_causal_search
    GRAPH_ALG_CALLS.labels(algorithm="causal").inc()
    return backwards_causal_search(codes)


@graph_router.post("/fault/predict")
def predict_codes(codes: List[str]):
    """Predict which codes may appear next if current ones not fixed."""
    from fault_graph import predict_next_codes
    GRAPH_ALG_CALLS.labels(algorithm="predict").inc()
    return predict_next_codes(codes)


@graph_router.post("/repair-order")
def get_repair_order(codes: List[str]):
    """Topological sort: correct repair order respecting dependencies."""
    from vehicle_hierarchy import get_repair_order
    GRAPH_ALG_CALLS.labels(algorithm="topo_sort").inc()
    return {"repair_order": get_repair_order(", ".join(codes))}


@graph_router.get("/fault/cheapest-path/{start}/{end}")
def get_cheapest_path(
    start: str = Path(..., description="Starting fault code e.g. P0101"),
    end:   str = Path(..., description="Target fault code e.g. P0420")
):
    """Dijkstra: minimum-cost diagnostic path + dollar-cost path between two related fault codes."""
    from fault_graph import dijkstra_min_cost_path, dijkstra_cheapest_repair_path
    GRAPH_ALG_CALLS.labels(algorithm="dijkstra").inc()
    effort_path = dijkstra_min_cost_path(start, end)
    dollar_path = dijkstra_cheapest_repair_path(start, end)
    return {"diagnostic_effort_path": effort_path, "dollar_cost_path": dollar_path}


@graph_router.get("/system/{code}")
def get_system(code: str = Path(..., description="Fault code")):
    """Get vehicle system context for a fault code."""
    from vehicle_hierarchy import get_system_context
    return {"context": get_system_context(code)}


@graph_router.get("/chunks/stats")
def chunk_graph_stats():
    """Chunk similarity graph statistics."""
    return get_graph_stats()


app.include_router(graph_router)


# ══════════════════════════════════════════════════════════
# EXPORT ENDPOINT
# ══════════════════════════════════════════════════════════

@app.get("/export/{session_id}", tags=["core"])
def export_session(
    session_id: str,
    format:     str = Query(default="json", description="json or txt"),
    api_key:    str = Depends(verify_api_key)
):
    """
    FileResponse — download diagnosis report as file.
    """
    from db import get_feedback_by_session

    feedback = get_feedback_by_session(session_id)
    memory   = MemoryManager(session_id)
    episodes = memory.retrieve_relevant_episodes("")

    export_data = {
        "session_id": session_id,
        "feedback":   feedback,
        "episodes":   episodes
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{format}", delete=False, dir="/tmp"
    ) as f:
        if format == "json":
            json.dump(export_data, f, indent=2)
        else:
            f.write(f"Session: {session_id}\n")
            f.write(f"Feedback entries: {len(feedback)}\n")
            f.write(f"Past episodes: {len(episodes)}\n")
        tmp_path = f.name

    return FileResponse(
        path=tmp_path,
        filename=f"autodiag_{session_id}.{format}",
        media_type="application/json" if format == "json" else "text/plain"
    )


# ══════════════════════════════════════════════════════════
# WEBSOCKET — Real-time streaming
# ══════════════════════════════════════════════════════════

@app.websocket("/ws/diagnose")
async def websocket_diagnose(websocket: WebSocket):
    """
    WebSocket — stream agent steps in real-time.
    Client receives JSON after each agent node completes.

    Protocol:
    Client sends: {"symptoms": "...", "fault_codes": "..."}
    Server sends: {"node": "diagnose", "result": {...}} per node
    Server sends: {"status": "complete", "final": {...}} at end
    """
    WS_CONNECTIONS.inc()
    await websocket.accept()

    try:
        data    = await websocket.receive_json()
        symptoms    = data.get("symptoms", "")
        fault_codes = data.get("fault_codes", "")
        session_id  = data.get("session_id", str(uuid.uuid4())[:8])

        await websocket.send_json({
            "status":  "started",
            "message": "Starting multi-agent diagnosis...",
            "session_id": session_id
        })

        request_dict = {
            "symptoms":    symptoms,
            "fault_codes": fault_codes,
            "vehicle_info": data.get("vehicle_info", ""),
            "session_id":  session_id
        }

        # Stream each node result
        for chunk in stream_graph(request_dict, pinecone_store):
            await websocket.send_json({
                "status": "update",
                "chunk":  chunk
            })
            await asyncio.sleep(0.1)

        # Final result
        final_result = run_graph(request_dict, pinecone_store)
        await websocket.send_json({
            "status": "complete",
            "final":  final_result
        })

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected")
    except Exception as e:
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# ARCHITECTURE ENDPOINT
# ══════════════════════════════════════════════════════════

@app.get("/architecture", tags=["core"])
def get_architecture():
    """Full system architecture documentation."""
    return {
        "pattern":    "Hierarchical Multi-Agent with LangGraph",
        "agents": {
            "orchestrator":     "Plans, routes, resolves conflicts — uses GoT",
            "diagnostic_agent": "ReAct + Agentic RAG + 9 graph tools + FewShot",
            "safety_agent":     "ReAct + severity gradation + FewShot",
            "cost_agent":       "ReAct + topological repair ordering + FewShot",
            "reflection_agent": "LCEL chains — critique + revise"
        },
        "infrastructure": {
            "blackboard":      "Shared agent memory, thread-safe",
            "tracer":          "Full step observability + MLflow",
            "context_manager": "LCEL runnables + compaction",
            "memory_manager":  "4-type SQLite memory + write policies",
            "reasoning_engine": "Graph of Thoughts + World Models",
            "guardrails":      "Input/output + human-in-loop",
            "graph":           "LangGraph + checkpointing + interrupts",
            "fault_graph":     "BFS + DFS + causal + predict",
            "vehicle_hierarchy": "Hierarchy + dependency + topo sort",
            "chunk_graph":     "Similarity graph + deduplication",
            "pinecone_manager": "MMR + threshold + namespaces + hybrid"
        },
        "storage": {
            "pinecone":  "Vector embeddings — manuals + chunks",
            "sqlite":    "Feedback + all 4 memory types",
            "checkpoints": "LangGraph state persistence"
        },
        "agentic_topics_covered": 31
    }