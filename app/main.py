"""
main.py — FastAPI entrypoint.

Exposes:
  POST /policy   — ingest / update a policy document
  POST /ask      — ask a question against ingested policies
  POST /audit    — upload a bill (PDF or image), extract + parse it, and
                   check its per-category totals against ingested policy
  GET  /health   — liveness + corpus stats
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File

from app.core.simple_rag import SimpleRagPipeline
from app.core.audit_pipeline import run_audit
from app.models.schemas import (
    PolicyUpdateRequest,
    PolicyUpdateResponse,
    AskRequest,
    AskResponse,
    AuditResponse,
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

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


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
