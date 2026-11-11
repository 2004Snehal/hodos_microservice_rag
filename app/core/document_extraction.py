"""
document_extraction.py — Document ingestion (PDF or image -> text), with
handwritten/printed routing.

ROUTING, per page
  1. PDF page with a usable text layer -> "direct_text". Born-digital bills
     need no model at all; this check always runs first.
  2. Otherwise the page is an image (photo, scan, image-only PDF). The CNN
     router (app/core/bill_router/) estimates P(handwritten):
       handwritten -> VLM (app/llm/vlm.py), if one is installed;
                      else Tesseract, with a note that it's unreliable here
       printed     -> Tesseract; escalated to the VLM if Tesseract's mean
                      word confidence is below OCR_MIN_CONFIDENCE
       uncertain   -> same as printed
     If the router's weights are missing, every image page is treated as
     "uncertain".

Each page records the route, P(handwritten), the OCR confidence and any
notes, so callers (the /audit endpoints) can show why a page was read the
way it was.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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

# Mean Tesseract word confidence (0-100) below which a printed/uncertain
# page is re-read by the VLM, when one is available.
OCR_MIN_CONFIDENCE = 60

_router = None
_vlm = None


def _get_router():
    global _router
    if _router is None:
        from app.core.bill_router.router import HandwritingRouter
        _router = HandwritingRouter()
    return _router


def _get_vlm():
    global _vlm
    if _vlm is None:
        from app.llm.vlm import OllamaVLM
        _vlm = OllamaVLM()
    return _vlm


@dataclass
class PageExtraction:
    page_number: int
    method: str  # "direct_text" | "ocr" | "vlm"
    text: str
    route: str = "digital"  # "digital" | "printed" | "handwritten" | "uncertain"
    p_handwritten: Optional[float] = None
    ocr_confidence: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "page": self.page_number, "route": self.route, "method": self.method,
            "p_handwritten": self.p_handwritten, "ocr_confidence": self.ocr_confidence,
            "notes": self.notes,
        }


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

    @property
    def page_routes(self) -> List[dict]:
        return [p.summary() for p in self.pages]


def _ocr(img: Image.Image):
    """Tesseract text plus mean word confidence (0-100)."""
    text = pytesseract.image_to_string(img).strip()
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, w in zip(data["conf"], data["text"]) if float(c) >= 0 and w.strip()]
    return text, (round(sum(confs) / len(confs), 1) if confs else 0.0)


def extract_raster(img: Image.Image, page_number: int = 0) -> PageExtraction:
    """Route an image page to Tesseract or the VLM."""
    img = img.convert("RGB")
    router = _get_router()
    decision = router.route(img) if router.available() else None
    route = decision.label if decision else "uncertain"
    p_hw = decision.p_handwritten if decision else None
    notes = [] if decision else ["Router weights missing — run scripts/train_bill_router.py."]

    vlm = _get_vlm()
    if route == "handwritten":
        if vlm.available():
            return _vlm_page(vlm, img, page_number, route, p_hw, None, notes)
        notes.append(f"Handwriting detected but no VLM installed (ollama pull {vlm.model}); "
                     "Tesseract is unreliable on handwriting.")

    text, confidence = _ocr(img)
    if route != "handwritten" and confidence < OCR_MIN_CONFIDENCE and vlm.available():
        notes.append(f"OCR confidence {confidence} < {OCR_MIN_CONFIDENCE}; re-read with VLM.")
        return _vlm_page(vlm, img, page_number, route, p_hw, confidence, notes)
    return PageExtraction(page_number, "ocr", text, route, p_hw, confidence, notes)


def _vlm_page(vlm, img, page_number, route, p_hw, ocr_confidence, notes) -> PageExtraction:
    try:
        result = vlm.transcribe(img)
        return PageExtraction(page_number, "vlm", result.text, route, p_hw, ocr_confidence, notes)
    except Exception as exc:  # VLM down / bad output: fall back to OCR rather than fail the audit
        text, confidence = _ocr(img)
        notes.append(f"VLM failed ({type(exc).__name__}: {exc}); used Tesseract.")
        return PageExtraction(page_number, "ocr", text, route, p_hw, confidence, notes)


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

        # No usable text layer -> rasterize and route (OCR or VLM).
        zoom = OCR_RASTER_DPI / 72  # PDF points are 72 per inch
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(extract_raster(img, page_number=i))

    doc.close()
    return DocumentExtraction(source_path=str(path), pages=pages)


def extract_image(path: Path) -> DocumentExtraction:
    """A plain image file has no text layer — always routed (OCR or VLM)."""
    return DocumentExtraction(source_path=str(path), pages=[extract_raster(Image.open(path))])


def extract_document(path: Path) -> DocumentExtraction:
    """Entry point: dispatches by file extension to the right extractor."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return extract_image(path)
    raise ValueError(f"Unsupported file type: {suffix}")
