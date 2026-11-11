"""
policy_extraction.py — Policy file (PDF / image) -> text for /v2/policy/file.

  extractor="auto"       docling-serve if reachable, else Tesseract (with a warning)
  extractor="docling"    docling-serve only; fails if it's down
  extractor="tesseract"  the existing per-page router (direct text layer /
                         Tesseract / VLM) from app.core.document_extraction

Docling's markdown keeps tables as pipe tables and headings as "#" lines,
which is what app.core.table_rules reads structured rules from.

NUMBER CROSS-CHECK
Docling text is checked against the Tesseract / text-layer read of the same
file before it's used: every number in it must also appear there.
  - Since Docling 2.118.0 a line-final hyphen is deleted when a digit follows
    ("7-" + "10 days" -> "710 days"; docling-project/docling#4151, fixes in
    open PRs #4211 / #4387). A number missing from the reference whose
    digits are a hyphenated pair there ("7-10") is restored to that form.
  - Any other number missing from the reference is left as is and listed in
    `unverified_numbers`, with a warning: one of the two reads got it wrong.
Drop the repair once a Docling release with the #4151 fix is in use; keep the
check, since it also catches OCR digit errors.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.docling_client import DoclingClient, DoclingUnavailable
from app.core.document_extraction import extract_document

EXTRACTORS = ("auto", "docling", "tesseract")

_NUMBER = re.compile(r"(?<![\w.#])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
_URL = re.compile(r"https?://[^\s)\]]+")
_HYPHEN_PAIR = re.compile(r"(?<![\w.])(\d+)\s*[-–]\s*(\d+)(?![\w])")


@dataclass
class PolicyText:
    text: str
    method: str  # "docling" | "tesseract"
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _normalize_number(token: str) -> str:
    return f"{float(token.replace(',', '')):g}"


def reconcile_numbers(text: str, reference: str) -> Tuple[str, List[Dict[str, str]], List[str]]:
    """
    Check every number in `text` against `reference` (see module docstring).
    Returns (text with repairs, repairs made, numbers still unverified).
    """
    known = {_normalize_number(m.group(0)) for m in _NUMBER.finditer(reference)}
    joined = {a + b: f"{a}-{b}" for a, b in _HYPHEN_PAIR.findall(reference)}

    repairs: List[Dict[str, str]] = []
    unverified: List[str] = []

    def check(match: re.Match) -> str:
        token = match.group(0)
        if _normalize_number(token) in known:
            return token
        if token in joined:
            repairs.append({"from": token, "to": joined[token],
                            "reason": "Docling dropped a line-break hyphen (docling#4151)"})
            return joined[token]
        if token not in unverified:
            unverified.append(token)
        return token

    # Numbers inside URLs (link targets Docling keeps, e.g. SEC filing ids) aren't checked.
    parts, last = [], 0
    for url in _URL.finditer(text):
        parts.append(_NUMBER.sub(check, text[last:url.start()]))
        parts.append(url.group(0))
        last = url.end()
    parts.append(_NUMBER.sub(check, text[last:]))
    return "".join(parts), repairs, unverified


def extract_policy_text(path: Path, extractor: str = "auto", client: Optional[DoclingClient] = None) -> PolicyText:
    if extractor not in EXTRACTORS:
        raise ValueError(f"extractor must be one of {EXTRACTORS}")
    warnings: List[str] = []
    docling_result = None
    if extractor in ("auto", "docling"):
        client = client or DoclingClient()
        try:
            docling_result = client.convert(path)
        except DoclingUnavailable as exc:
            if extractor == "docling":
                raise
            warnings.append(f"Docling unavailable ({exc}); used Tesseract, which flattens tables, "
                            "so table rules may be missing.")

    extraction = extract_document(path)
    reference = {"methods_used": extraction.methods_used, "page_routes": extraction.page_routes}
    if docling_result is None:
        return PolicyText(extraction.full_text, "tesseract", reference, warnings)

    text, repairs, unverified = reconcile_numbers(docling_result.markdown, extraction.full_text)
    if repairs:
        warnings.append(f"Restored {len(repairs)} number(s) Docling merged across a line-break hyphen: "
                        + ", ".join(f"{r['from']} -> {r['to']}" for r in repairs))
    if unverified:
        warnings.append(f"{len(unverified)} number(s) in the Docling text are not in the Tesseract / text-layer "
                        f"read and could be misread: {', '.join(unverified[:20])}")
    return PolicyText(text, "docling",
                      {"status": docling_result.status, "processing_time": docling_result.processing_time,
                       "errors": docling_result.errors, "url": client.url,
                       "number_check": {"repairs": repairs, "unverified_numbers": unverified,
                                        "reference": reference}},
                      warnings)
