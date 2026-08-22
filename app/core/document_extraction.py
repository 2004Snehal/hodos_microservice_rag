"""
document_extraction.py — ACTIVE, real document ingestion (PDF or image -> text).

WHY THIS EXISTS
The original design (see docs/ARCHITECTURE.md) called for a handwriting
classifier routing between OCR and a vision-language model. Neither of those
can actually run in a locked-down/offline environment: a handwriting
classifier and a VLLM both need model weights pulled from somewhere like
Hugging Face, which most sandboxed or air-gapped deployments won't have
network access to (this one doesn't, for example — see the note in
docs/ARCHITECTURE.md). Rather than ship something that only works if you
happen to have that access, this module uses a genuinely offline-capable
heuristic instead of an ML classifier:

    Does this PDF page have an extractable text layer?
      YES -> it's a born-digital PDF (or a digitally-produced bill/invoice) —
             pull the text directly. Fast, perfect accuracy, no OCR needed.
      NO  -> it's scanned/image-only (or a plain image file) — rasterize the
             page and run Tesseract OCR on the pixels.

This is not a compromise unique to this project — checking for a text layer
before deciding whether to OCR at all is a completely standard first step in
real document-processing pipelines, independent of whether a fancier
handwriting-vs-printed classifier sits downstream of it. It doesn't
distinguish handwritten from printed text the way the original design's
classifier would — it distinguishes "this page already has machine-readable
text" from "this page is just an image," which is the more fundamental
question and the one that actually decides whether OCR is needed at all.

WHAT THIS DOES NOT DO
It does not tell you whether an image-only page is handwritten vs. printed —
Tesseract runs on it either way. Tesseract's accuracy on genuinely messy
handwriting is poor (it's an OCR engine designed for printed/typed text); if
you need real handwriting recognition, that's exactly the kind of task a
VLLM was originally meant to handle, and doing that for real requires either
network access to a hosted VLLM API or a locally-downloaded open model —
neither of which this module assumes you have. This is the honest
limitation being traded for "runs fully offline, no model download."
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # older PyMuPDF versions expose the module as `fitz`


# Below this many extractable characters on a page, treat it as having no
# usable text layer (a page with a stray text box but otherwise scanned
# content shouldn't be treated as "digital").
MIN_CHARS_FOR_DIGITAL_PAGE = 20

# DPI used when rasterizing a page for OCR — higher improves OCR accuracy at
# the cost of speed; 300 is a common default for OCR pipelines.
OCR_RASTER_DPI = 300


@dataclass
class PageExtraction:
    page_number: int
    method: str  # "direct_text" | "ocr"
    text: str


@dataclass
class DocumentExtraction:
    source_path: str
    pages: List[PageExtraction]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def methods_used(self) -> List[str]:
        return sorted({p.method for p in self.pages})


def extract_pdf(path: Path) -> DocumentExtraction:
    """
    Open a PDF and extract text page by page, routing each page to direct
    text extraction or OCR based on whether it has a usable text layer.
    """
    doc = fitz.open(str(path))
    pages: List[PageExtraction] = []

    for i, page in enumerate(doc):
        direct_text = page.get_text().strip()

        if len(direct_text) >= MIN_CHARS_FOR_DIGITAL_PAGE:
            pages.append(PageExtraction(page_number=i, method="direct_text", text=direct_text))
            continue

        # No usable text layer -> rasterize and OCR.
        zoom = OCR_RASTER_DPI / 72  # PDF points are 72 per inch
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        ocr_text = pytesseract.image_to_string(img).strip()
        pages.append(PageExtraction(page_number=i, method="ocr", text=ocr_text))

    doc.close()
    return DocumentExtraction(source_path=str(path), pages=pages)


def extract_image(path: Path) -> DocumentExtraction:
    """
    A plain image file (jpg/png/etc.) has no concept of a text layer — it's
    always routed straight to OCR.
    """
    img = Image.open(path)
    text = pytesseract.image_to_string(img).strip()
    return DocumentExtraction(
        source_path=str(path),
        pages=[PageExtraction(page_number=0, method="ocr", text=text)],
    )


def extract_document(path: Path) -> DocumentExtraction:
    """Entry point: dispatches by file extension to the right extractor."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return extract_image(path)
    raise ValueError(f"Unsupported file type: {suffix}")
