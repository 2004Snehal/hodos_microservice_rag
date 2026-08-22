"""
audit_pipeline.py — ACTIVE, wires together extraction + parsing + retrieval
into a verdict. This is what POST /audit calls.

PIPELINE
  1. document_extraction.extract_document()  — PDF/image -> raw text
  2. bill_parser.parse_line_items()          — raw text -> categorized line items
  3. for each category found on the bill:
       retrieve the most relevant policy chunk for that category
       (reuses the SAME SimpleVectorStore the simple RAG pipeline already
       built for /ask — no separate index)
       resolve the applicable rule via app.core.rule_extraction — a numeric
       cap, an explicit waiver, or a pre-approval requirement, not just a
       number search
  4. compare the bill's per-category total against the resolved rule
  5. return a verdict per category, with citations back to the exact policy
     chunk used

WHAT'S REAL VS. HEURISTIC HERE
Steps 1-2 are real, working code (Tesseract OCR / direct PDF text extraction,
regex-based line-item parsing) — see each module's own docstring for their
specific limitations. Step 3's rule resolution (app.core.rule_extraction)
is real, tested pattern-matching code — not an LLM call, but genuinely more
than "grab the first number": it distinguishes a numeric cap from an
explicit waiver from a pre-approval requirement, checked in the text window
immediately following the category's own keyword. It is still NOT the
multi-hop/RAPTOR reasoning described in docs/ARCHITECTURE.md — it cannot
resolve a grade override two sentences away, or follow a cross-reference to
a different chunk. Swap point: replace `resolve_rule` with a real LLM call
that reads the retrieved chunk and returns the same
`{rule_type, limit_amount}` structure — the rest of this pipeline doesn't
need to change, since it already treats rule resolution as a black box with
that output contract.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core.document_extraction import extract_document
from app.core.bill_parser import parse_line_items, CATEGORY_KEYWORDS
from app.core.rule_extraction import resolve_rule
from app.core.vectorstore import SimpleVectorStore


@dataclass
class CategoryVerdict:
    category: str
    claimed_total: float
    rule_type: str  # "daily_cap" | "waived" | "requires_preapproval" | "unresolved"
    retrieved_limit: Optional[float]
    verdict: str  # "pass" | "fail" | "needs_review" | "no_policy_found"
    citation_chunk_id: Optional[str]
    citation_text: Optional[str]
    citation_score: Optional[float]


@dataclass
class AuditResult:
    source_file: str
    extraction_methods_used: List[str]
    vendor_guess: Optional[str]
    date_guess: Optional[str]
    raw_extracted_text: str
    category_verdicts: List[CategoryVerdict]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "extraction_methods_used": self.extraction_methods_used,
            "vendor_guess": self.vendor_guess,
            "date_guess": self.date_guess,
            "raw_extracted_text": self.raw_extracted_text,
            "category_verdicts": [asdict(v) for v in self.category_verdicts],
        }


def _verdict_for_rule(rule_type: str, claimed_total: float, limit_amount: Optional[float]) -> str:
    """
    Real rule-type-aware verdict logic — this is the behavioral improvement
    over the earlier version, which only ever compared an amount to a
    number. A waived category should never fail on amount; a
    pre-approval-required category can't be resolved as pass/fail from text
    alone (approval status isn't in the bill), so it correctly routes to
    human review instead of guessing.
    """
    if rule_type == "waived":
        return "pass"
    if rule_type == "requires_preapproval":
        return "needs_review"
    if rule_type == "daily_cap" and limit_amount is not None:
        return "fail" if claimed_total > limit_amount else "pass"
    return "no_policy_found"


def run_audit(file_path: Path, store: SimpleVectorStore) -> AuditResult:
    """
    Args:
        file_path: path to the uploaded bill (PDF or image)
        store: the SAME SimpleVectorStore used by /policy and /ask — the
               audit pipeline retrieves against whatever policies have
               already been ingested, it doesn't maintain a separate index.
    """
    extraction = extract_document(file_path)
    parsed = parse_line_items(extraction.full_text)

    verdicts: List[CategoryVerdict] = []
    for category, claimed_total in parsed.total_by_category.items():
        results = store.search(f"reimbursement limit for {category} expenses", top_k=1)

        if not results:
            verdicts.append(
                CategoryVerdict(
                    category=category,
                    claimed_total=claimed_total,
                    rule_type="unresolved",
                    retrieved_limit=None,
                    verdict="no_policy_found",
                    citation_chunk_id=None,
                    citation_text=None,
                    citation_score=None,
                )
            )
            continue

        top_chunk = results[0]
        keywords = CATEGORY_KEYWORDS.get(category, [category])
        resolved = resolve_rule(top_chunk["text"], keywords)

        verdict_str = _verdict_for_rule(resolved.rule_type, claimed_total, resolved.limit_amount)

        verdicts.append(
            CategoryVerdict(
                category=category,
                claimed_total=claimed_total,
                rule_type=resolved.rule_type,
                retrieved_limit=resolved.limit_amount,
                verdict=verdict_str,
                citation_chunk_id=top_chunk["chunk_id"],
                citation_text=top_chunk["text"],
                citation_score=round(top_chunk["score"], 4),
            )
        )

    return AuditResult(
        source_file=str(file_path),
        extraction_methods_used=extraction.methods_used,
        vendor_guess=parsed.vendor_guess,
        date_guess=parsed.date_guess,
        raw_extracted_text=extraction.full_text,
        category_verdicts=verdicts,
    )
