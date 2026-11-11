"""
audit.py — Bill audit on top of the probabilistic pipeline (POST /v2/audit).

  1. document_extraction.extract_document()  — PDF text layer / Tesseract OCR
  2. bill_parser.parse_line_items()          — categorized line items
  3. per category on the bill: ProbabilisticRagPipeline.ask() with a question
     built from the category, grade and location — so the cap comes from
     multi-hop retrieval + LLM rule extraction (exceptions, "1500 rupees"
     wording) with a confidence, instead of the regex engine.
  4. compare the claim against the rule, respecting the rule's period:
       per_trip            -> each line item checked against the cap
       per_day / per_night -> category total vs cap × days
       anything else       -> category total vs cap
  5. a judgment the pipeline marks needs_review (low confidence, number not
     in the evidence) never produces a hard pass/fail — the verdict becomes
     needs_review, with the computed outcome kept as `provisional_verdict`.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.advanced.probabilistic.pipeline import ProbabilisticRagPipeline
from app.core.bill_parser import parse_line_items
from app.core.document_extraction import extract_document


def run_probabilistic_audit(
    file_path: Path,
    pipeline: ProbabilisticRagPipeline,
    doc_id: str,
    grade: str = "any",
    location: str = "any",
    days: int = 1,
    version: str = "v1",
) -> Dict[str, Any]:
    extraction = extract_document(file_path)
    parsed = parse_line_items(extraction.full_text)

    items_by_category: Dict[str, List[Any]] = {}
    for item in parsed.line_items:
        items_by_category.setdefault(item.category, []).append(item)

    verdicts = []
    for category, items in items_by_category.items():
        question = _question(category, grade, location)
        judgment = pipeline.ask(question, doc_id, category=category, grade=grade,
                                location=location, version=version)
        verdicts.append(_verdict(category, items, judgment, days))

    return {
        "doc_id": doc_id,
        "grade": grade,
        "location": location,
        "days": days,
        "extraction_methods_used": extraction.methods_used,
        "page_routes": extraction.page_routes,
        "vendor_guess": parsed.vendor_guess,
        "date_guess": parsed.date_guess,
        "raw_extracted_text": extraction.full_text,
        "line_items": [
            {"description": i.description, "category": i.category, "amount": i.amount}
            for i in parsed.line_items
        ],
        "category_verdicts": verdicts,
        "overall": _overall([v["verdict"] for v in verdicts]),
    }


def _question(category: str, grade: str, location: str) -> str:
    who = f" for a grade {grade} employee" if grade and grade != "any" else ""
    where = f" in {location}" if location and location != "any" else ""
    return f"What is the {category} expense limit{who}{where}?"


def _verdict(category: str, items: List[Any], judgment: Dict[str, Any], days: int) -> Dict[str, Any]:
    claimed_total = round(sum(i.amount for i in items), 2)
    rule = judgment.get("rule_type")
    limit = judgment.get("limit_amount")
    period = judgment.get("period")
    over_limit_items: List[Dict[str, Any]] = []
    allowed: Optional[float] = None

    if rule == "waived":
        computed = "pass"
    elif rule == "requires_preapproval":
        computed = "needs_review"
    elif rule == "cap" and limit is not None:
        limit = float(limit)
        if period == "per_trip":
            allowed = limit
            over_limit_items = [
                {"description": i.description, "amount": i.amount} for i in items if i.amount > limit
            ]
            computed = "fail" if over_limit_items else "pass"
        else:
            allowed = limit * max(days, 1) if period in ("per_day", "per_night") else limit
            computed = "fail" if claimed_total > allowed else "pass"
    else:
        computed = "no_policy_found"

    needs_review = judgment.get("status") != "ok" and computed in ("pass", "fail")
    return {
        "category": category,
        "claimed_total": claimed_total,
        "line_items": len(items),
        "verdict": "needs_review" if needs_review else computed,
        "provisional_verdict": computed if needs_review else None,
        "rule_type": rule,
        "limit_amount": limit,
        "period": period,
        "allowed_amount": allowed,
        "over_limit_items": over_limit_items,
        "confidence": judgment.get("confidence"),
        "answer": judgment.get("answer"),
        "conditions": judgment.get("conditions", []),
        "citations": judgment.get("citations", []),
        "warnings": judgment.get("warnings", []),
        "judgment_source": judgment.get("source"),
    }


def _overall(verdicts: List[str]) -> str:
    if "fail" in verdicts:
        return "fail"
    if "needs_review" in verdicts or "no_policy_found" in verdicts:
        return "needs_review"
    return "pass" if verdicts else "no_items_found"
