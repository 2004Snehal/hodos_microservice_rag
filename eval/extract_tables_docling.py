"""
extract_tables_docling.py — Convert a policy PDF with Docling and cache the result.

Runs in its own venv, because Docling pulls torch and numpy 2, which the
service's pinned numpy 1.26 / scikit-learn stack doesn't take:

  python3.12 -m venv .venv-docling && .venv-docling/bin/pip install docling
  .venv-docling/bin/python eval/extract_tables_docling.py eval/policies/company_xx_travel_india.pdf
  .venv-docling/bin/python eval/extract_tables_docling.py <pdf> docling_tesseract   # one variant

Writes, per variant, eval/extractions/<pdf stem>/<variant>/:
  document.md   — Docling's markdown export (tables as pipe tables)
  pages.json    — the same, one markdown string per page (page 1 first)
  tables.json   — every detected table as a cell grid, with its page
  meta.json     — Docling version, options, seconds

eval/run_table_eval.py (main venv) scores these against the gold tables.
"""

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

ROOT = Path(__file__).resolve().parents[1]


def variants():
    # Default OCR engine (Docling picks one) vs Tesseract — the same OCR
    # engine as the current system, so the comparison isolates layout/table
    # structure from OCR quality.
    default = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
    default.table_structure_options.mode = TableFormerMode.ACCURATE

    tesseract = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
    tesseract.table_structure_options.mode = TableFormerMode.ACCURATE
    tesseract.ocr_options = TesseractCliOcrOptions(lang=["eng"], force_full_page_ocr=True)
    return {"docling_default_ocr": default, "docling_tesseract": tesseract}


def main(pdf: Path, only=None) -> None:
    for name, options in variants().items():
        if only and name not in only:
            continue
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        started = time.time()
        document = converter.convert(pdf).document
        seconds = time.time() - started

        tables = []
        for i, table in enumerate(document.tables):
            grid = [[cell.text for cell in row] for row in table.data.grid]
            page = table.prov[0].page_no if table.prov else None
            tables.append({"index": i, "page": page, "rows": table.data.num_rows,
                           "cols": table.data.num_cols, "grid": grid})

        out = ROOT / "eval" / "extractions" / pdf.stem / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "document.md").write_text(document.export_to_markdown())
        pages = [document.export_to_markdown(page_no=n) for n in range(1, document.num_pages() + 1)]
        (out / "pages.json").write_text(json.dumps(pages, ensure_ascii=False))
        (out / "tables.json").write_text(json.dumps(tables, indent=1, ensure_ascii=False))
        (out / "meta.json").write_text(json.dumps({
            "system": name, "docling_version": version("docling"), "seconds": round(seconds, 2),
            "ocr_engine": type(options.ocr_options).__name__,
            "table_mode": str(options.table_structure_options.mode),
            "pdf": str(pdf.relative_to(ROOT)) if pdf.is_relative_to(ROOT) else str(pdf),
        }, indent=1))
        print(f"{name}: {len(tables)} tables in {seconds:.1f}s -> {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), set(sys.argv[2:]))
