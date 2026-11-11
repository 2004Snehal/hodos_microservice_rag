"""
build_financebench_subset.py — A slice of FinanceBench (open subset) in the eval format.

FinanceBench (Islam et al. 2023; PatronusAI/financebench, CC BY-NC 4.0):
questions about public company filings (10-K, 10-Q, 8-K, earnings), each with
a gold answer and the evidence text + page it came from.

  .venv/bin/python -m eval.build_financebench_subset

SLICE
The 8 filings with the most questions in the 150-question open subset:
40 questions. PDFs come from github.com/patronus-ai/financebench.

WRITES
  eval/policies/financebench/<doc>.pdf          — source PDF (gitignored; re-downloaded)
  eval/policies/financebench/<doc>.txt          — text from app.core.document_extraction,
                                                  one block per page
  eval/policies/financebench/<doc>.pages.json   — the same text as a list of pages,
                                                  for page-level retrieval scoring
  eval/datasets/financebench_subset.json        — one item per question:
      evidence_pages   0-based pages of the gold evidence, found by text overlap
                       with evidence_text_full_page (FinanceBench's own page
                       numbers aren't assumed to be 0- or 1-based)
      expected         answer_numbers for numeric answers (metrics-generated, or
                       short answers with numbers); other questions are
                       retrieval-only
"""

import json
import re
from pathlib import Path

import requests

from app.core.document_extraction import extract_document

ROOT = Path(__file__).resolve().parents[1]
ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=PatronusAI/financebench&config=default&split=train"
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{}.pdf"

DOCS = [
    "AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K", "PEPSICO_2022_10K",
    "AMCOR_2023_10K", "ULTABEAUTY_2023Q4_EARNINGS", "3M_2022_10K", "3M_2023Q2_10Q",
]


def _rows():
    rows = []
    for offset in (0, 100):
        page = requests.get(f"{ROWS_URL}&offset={offset}&length=100", timeout=60).json()
        rows += [r["row"] for r in page["rows"]]
    return rows


def _words(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()))


def _evidence_page(full_page: str, pages) -> int:
    """0-based page whose words overlap the gold evidence page most (Jaccard)."""
    gold = _words(full_page)
    scores = [len(gold & _words(p)) / (len(gold | _words(p)) or 1) for p in pages]
    return max(range(len(pages)), key=scores.__getitem__)


def _numbers(text: str):
    """Amounts in an answer; years and digits glued to letters ("FY2022", "Q4") are not amounts."""
    found = []
    for m in re.finditer(r"(?<![A-Za-z\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        value = float(m.group(0).replace(",", ""))
        if re.fullmatch(r"(19|20)\d\d", m.group(0)):
            continue
        found.append(value)
    return found


def main() -> None:
    rows = [r for r in _rows() if r["doc_name"] in DOCS]
    out_dir = ROOT / "eval" / "policies" / "financebench"
    out_dir.mkdir(parents=True, exist_ok=True)

    policies, pages_files, page_texts = {}, {}, {}
    for doc in DOCS:
        pdf = out_dir / f"{doc}.pdf"
        if not pdf.exists():
            pdf.write_bytes(requests.get(PDF_URL.format(doc), timeout=300).content)
        extraction = extract_document(pdf)
        pages = [p.text for p in extraction.pages]
        # Blank line between pages so the chunker treats each page boundary as a paragraph break.
        (out_dir / f"{doc}.txt").write_text("\n\n".join(pages) + "\n")
        (out_dir / f"{doc}.pages.json").write_text(json.dumps(pages, ensure_ascii=False))
        policies[doc] = str((out_dir / f"{doc}.txt").relative_to(ROOT))
        pages_files[doc] = str((out_dir / f"{doc}.pages.json").relative_to(ROOT))
        page_texts[doc] = pages
        print(f"{doc}: {len(pages)} pages, methods {extraction.methods_used}", flush=True)

    items, offsets = [], []
    for r in rows:
        evidence_pages = sorted({_evidence_page(e["evidence_text_full_page"], page_texts[r["doc_name"]])
                                 for e in r["evidence"]})
        offsets += [int(e["evidence_page_num"]) - p for e, p in zip(r["evidence"], evidence_pages)
                    if str(e.get("evidence_page_num", "")).isdigit()]
        item = {
            "id": r["financebench_id"],
            "doc_id": r["doc_name"],
            "type": r["question_type"],
            "reasoning": r["question_reasoning"],
            "question": r["question"],
            "category": "any",
            "evidence_pages": evidence_pages,
            "gold_answer": r["answer"],
            "justification": r["justification"],
        }
        numbers = _numbers(r["answer"])
        # Pure numeric answers, plus short answers whose numbers are the answer
        # ("The consumer segment shrunk by 0.9% organically.").
        if numbers and (r["question_type"] == "metrics-generated" or len(r["answer"]) <= 80):
            item["expected"] = {"answer_numbers": numbers}
        items.append(item)

    dataset = {
        "name": "financebench_subset",
        "policies": policies,
        "pages": pages_files,
        "source": "PatronusAI/financebench open subset (CC BY-NC 4.0); PDFs from github.com/patronus-ai/financebench",
        "gold_from": "FinanceBench annotations",
        "notes": [
            "Financial filings, not expense policies: tests long-PDF retrieval and numeric answers.",
            "Retrieval is scored by page: a retrieved leaf hits when it lies on a gold evidence page.",
            "Judgment is scored only for numeric answers: every gold number must appear in the "
            "pipeline's answer or limit_amount (1% tolerance). The pipeline's judge is built for expense "
            "rules, so this measures how it handles out-of-domain numeric questions.",
            f"FinanceBench evidence_page_num minus resolved 0-based page: {sorted(set(offsets))}",
        ],
        "items": items,
    }
    out = ROOT / "eval" / "datasets" / "financebench_subset.json"
    out.write_text(json.dumps(dataset, indent=1, ensure_ascii=False))
    judged = sum("expected" in i for i in items)
    print(f"{len(items)} questions ({judged} numerically judged) over {len(policies)} filings -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
