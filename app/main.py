"""
main.py — FastAPI entrypoint.

Exposes:
  POST /policy   — ingest / update a policy document
  POST /ask      — ask a question against ingested policies
  POST /audit    — upload a bill (PDF or image), extract + parse it, and
                   check its per-category totals against ingested policy
  GET  /health   — liveness + corpus stats

  /v2 — probabilistic RAPTOR + multi-hop pipeline (app/advanced/probabilistic/),
        using a local LLM through Ollama (qwen3:4b by default):
  POST /v2/policy — ingest a policy and build its tree
  POST /v2/policy/file — same, from a PDF / image; text via Docling (docling-serve),
                    falling back to Tesseract
  POST /v2/ask    — cached, multi-hop, LLM-judged answer with a confidence
  POST /v2/audit  — scan a bill and check each category against the policy
                    using /v2 judgments
  GET  /v2/health — LLM / cache / document status
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.advanced.probabilistic.audit import run_probabilistic_audit
from app.advanced.probabilistic.pipeline import ProbabilisticRagPipeline
from app.core.simple_rag import SimpleRagPipeline
from app.core.audit_pipeline import run_audit
from app.core.docling_client import DoclingClient, DoclingUnavailable
from app.core.policy_extraction import EXTRACTORS, extract_policy_text
from app.models.schemas import (
    PolicyUpdateRequest,
    PolicyUpdateResponse,
    AskRequest,
    AskResponse,
    AuditResponse,
    V2AskRequest,
    V2PolicyRequest,
)

app = FastAPI(
    title="Hodos RAG Microservice",
    description=(
        "An enterprise RAG service featuring overlap chunking, RAPTOR tree "
        "indexing, multi-hop retrieval, caching, and automated document audit capabilities."
    ),
    version="1.0.0",
)

pipeline = SimpleRagPipeline()
v2_pipeline = ProbabilisticRagPipeline()

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
UI_PATH = Path(__file__).parent / "static" / "index.html"


@app.get("/ui", include_in_schema=False)
def ui():
    """Browser test console for the v1 and /v2 endpoints."""
    return FileResponse(UI_PATH)


@app.post("/policy", response_model=PolicyUpdateResponse)
def update_policy(req: PolicyUpdateRequest):
    """Ingest or re-ingest a policy document (chunk + embed + store)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Policy text cannot be empty.")
    result = pipeline.update_policy(req.doc_id, req.text)
    return result


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question against currently ingested policies."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = pipeline.ask(req.question, top_k=req.top_k or 4)
    return result


@app.post("/audit", response_model=AuditResponse)
async def audit(file: UploadFile = File(...)):
    """
    Upload a bill (PDF or image), scan it, and check it against ingested
    policy. See app/core/audit_pipeline.py for exactly what's real
    (extraction, parsing) versus heuristic (limit extraction from a policy
    chunk) in this flow.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = run_audit(tmp_path, pipeline.store)
    finally:
        tmp_path.unlink(missing_ok=True)

    return result.to_dict()


@app.get("/health")
def health():
    """Basic liveness check + corpus stats, not part of the core endpoint contract."""
    return {"status": "ok", "store_stats": pipeline.store.stats()}


# ---------- /v2: probabilistic RAPTOR + multi-hop ----------

@app.post("/v2/policy")
def v2_policy(req: V2PolicyRequest):
    """Ingest a policy: chunk, embed, build the probabilistic RAPTOR tree."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Policy text cannot be empty.")
    return v2_pipeline.ingest(req.doc_id, req.text)


@app.post("/v2/policy/file")
async def v2_policy_file(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    extractor: str = Form("auto"),
):
    """
    Ingest a policy PDF / image. extractor: auto (Docling, Tesseract if Docling
    is down) | docling | tesseract. The response carries the extraction method,
    any fallback warning, and the structured table rules read from the text.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    if extractor not in EXTRACTORS:
        raise HTTPException(status_code=400, detail=f"extractor must be one of {list(EXTRACTORS)}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        policy = extract_policy_text(tmp_path, extractor)
    except DoclingUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    if not policy.text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file.")
    result = v2_pipeline.ingest(doc_id, policy.text)
    return {**result, "extraction": {"method": policy.method, "chars": len(policy.text),
                                     "warnings": policy.warnings, **policy.details}}


@app.post("/v2/ask")
def v2_ask(req: V2AskRequest):
    """Answer with a structured rule, citations and a probability-based confidence."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return v2_pipeline.ask(
            req.question, req.doc_id, category=req.category, grade=req.grade,
            location=req.location, version=req.version, use_cache=req.use_cache,
            use_structured=req.use_structured,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0]))


@app.post("/v2/audit")
async def v2_audit(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    grade: str = Form("any"),
    location: str = Form("any"),
    days: int = Form(1),
    version: str = Form("v1"),
):
    """Upload a bill (PDF or image) and audit it against a /v2-ingested policy."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    if doc_id not in v2_pipeline.trees:
        raise HTTPException(status_code=404, detail=f"No policy ingested with doc_id='{doc_id}'.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return run_probabilistic_audit(
            tmp_path, v2_pipeline, doc_id, grade=grade, location=location, days=days, version=version
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/v2/health")
def v2_health():
    from app.core.document_extraction import _get_router, _get_vlm

    router, vlm, docling = _get_router(), _get_vlm(), DoclingClient()
    return {
        "status": "ok",
        **v2_pipeline.stats(),
        "bill_router": {"cnn": "Qdrant/resnet50-onnx", "trained": router.available()},
        "vlm": {"model": vlm.model, "available": vlm.available()},
        "docling": {"url": docling.url, "available": docling.available()},
    }
