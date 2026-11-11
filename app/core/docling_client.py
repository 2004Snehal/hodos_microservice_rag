"""
docling_client.py — Policy PDF -> markdown through a docling-serve sidecar.

WHY DOCLING FOR POLICIES
On the image-only COMPANY XX policy (eval/run_table_eval.py), Tesseract
plain text gave 0 of 4 table rules: it flattens tables and drops light grey
header rows. Docling (layout + TableFormer, default OCR) returned both rule
tables cell for cell, and app.core.table_rules read all 4 rules and all 11
metro cities from its markdown. Bills keep the existing router
(app/core/document_extraction.py); Docling hasn't been evaluated on bills.

WHY A SIDECAR
Docling needs torch and numpy 2; this service pins numpy 1.26 /
scikit-learn 1.5. docling-serve (Docling's own HTTP API) runs separately:
  docker compose: the `docling` service (quay.io/docling-project/docling-serve)
  local:          .venv-docling/bin/docling-serve run --port 5001

Conversions use the async endpoint and poll, since OCR of a scanned policy
can outlast the sync endpoint's wait (7 scanned pages took ~94s on CPU).
If docling-serve is unreachable the caller falls back to Tesseract.
"""

import html
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_DOCLING_URL = os.getenv("HODOS_DOCLING_URL", "http://localhost:5001")
DEFAULT_TIMEOUT = float(os.getenv("HODOS_DOCLING_TIMEOUT", "900"))


class DoclingUnavailable(RuntimeError):
    pass


@dataclass
class DoclingResult:
    markdown: str
    status: str
    processing_time: Optional[float]
    errors: List[Any] = field(default_factory=list)


class DoclingClient:
    def __init__(self, url: str = DEFAULT_DOCLING_URL, timeout: float = DEFAULT_TIMEOUT, poll_seconds: float = 2.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.poll_seconds = poll_seconds

    def available(self) -> bool:
        try:
            return requests.get(f"{self.url}/health", timeout=3).ok
        except requests.RequestException:
            return False

    def convert(self, path: Path) -> DoclingResult:
        """PDF/image -> markdown, with OCR and accurate table structure (Docling's defaults)."""
        form = {"to_formats": "md", "do_ocr": "true", "table_mode": "accurate",
                "image_export_mode": "placeholder"}
        try:
            with open(path, "rb") as f:
                response = requests.post(f"{self.url}/v1/convert/file/async", data=form,
                                         files={"files": (path.name, f)}, timeout=60)
            response.raise_for_status()
            task_id = response.json()["task_id"]

            deadline = time.time() + self.timeout
            while True:
                status = requests.get(f"{self.url}/v1/status/poll/{task_id}", timeout=30).json()["task_status"]
                if status in ("success", "partial_success"):
                    break
                if status in ("failure", "skipped", "revoked"):
                    raise DoclingUnavailable(f"docling-serve task {task_id} ended with status {status!r}")
                if time.time() > deadline:
                    raise DoclingUnavailable(f"docling-serve task {task_id} still {status!r} after {self.timeout:.0f}s")
                time.sleep(self.poll_seconds)

            result = requests.get(f"{self.url}/v1/result/{task_id}", timeout=60)
            result.raise_for_status()
            body: Dict[str, Any] = result.json()
        except requests.RequestException as exc:
            raise DoclingUnavailable(f"docling-serve call failed: {exc}") from exc

        markdown = (body.get("document") or {}).get("md_content") or ""
        if not markdown.strip():
            raise DoclingUnavailable(f"docling-serve returned no markdown (status {body.get('status')!r})")
        return DoclingResult(clean_markdown(markdown), body.get("status", ""),
                             body.get("processing_time"), body.get("errors") or [])


def clean_markdown(markdown: str) -> str:
    """
    Undo Docling's HTML escaping (&amp; -> &) except an escaped pipe (&#124;),
    which stays escaped so a "|" inside a table cell can't split the cell.
    Image placeholders are dropped.
    """
    markdown = markdown.replace("&#124;", "\x00")
    markdown = html.unescape(markdown).replace("\x00", "&#124;")
    return re.sub(r"^\s*<!-- image -->\s*$", "", markdown, flags=re.MULTILINE)
