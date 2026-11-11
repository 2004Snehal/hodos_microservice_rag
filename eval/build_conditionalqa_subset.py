"""
build_conditionalqa_subset.py — A policy-like slice of ConditionalQA dev, in the eval format.

ConditionalQA (Sun et al., ACL 2022; github.com/haitian-sun/ConditionalQA,
BSD-2-Clause, research use) asks questions about UK gov.uk guidance pages, each
with the asker's scenario; answers hold only under conditions, and some
questions have no answer in the page.

  .venv/bin/python -m eval.build_conditionalqa_subset

SELECTION
The 15 dev documents below were picked by money amounts and allowance /
claim / eligibility / limit wording per word, then adjusted so the slice
keeps conditional answers, unanswerable questions and tables: 68 questions,
27 with conditional answers, 4 unanswerable, 9 documents with tables.

WRITES
  eval/policies/conditionalqa/<slug>.txt        — page as plain text: headings,
                                                  paragraphs, "- " list items,
                                                  "| a | b |" table rows
  eval/datasets/conditionalqa_dev_subset.json   — one item per question:
      question          scenario + question, as the pipeline would be asked
      evidence_phrases  the gold evidence sentences (retrieval scoring)
      expected          no_rule_found for unanswerable questions; answerable
                        ones are retrieval-only until the judge has a general
                        answer mode
      cqa_answers       the original [answer, [conditions]] pairs, kept for that
"""

import hashlib
import html
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/haitian-sun/ConditionalQA/master/v1_0/"

URLS = [
    "https://www.gov.uk/residential-support-scheme",
    "https://www.gov.uk/challenge-solicitors-bill",
    "https://www.gov.uk/funeral-payments",
    "https://www.gov.uk/bereavement-support-payment",
    "https://www.gov.uk/compensation-road-property-value",
    "https://www.gov.uk/claim-gift-aid",
    "https://www.gov.uk/directors-loans",
    "https://www.gov.uk/tax-on-shopping",
    "https://www.gov.uk/support-for-foster-parents",
    "https://www.gov.uk/maternity-pay-leave",
    "https://www.gov.uk/running-payroll",
    "https://www.gov.uk/state-pension-if-you-retire-abroad",
    "https://www.gov.uk/financial-help-disabled",
    "https://www.gov.uk/vaccine-damage-payment",
    "https://www.gov.uk/vat-registration",
]


def _plain(fragment: str) -> str:
    """One HTML fragment from ConditionalQA -> one line of plain text."""
    tag = re.match(r"<(\w+)", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()
    kind = tag.group(1) if tag else ""
    if kind == "tr":
        return "| " + " | ".join(c.strip() for c in text.split("|")) + " |"
    if kind == "li":
        return "- " + text
    if kind in ("h1", "h2", "h3", "h4"):
        return "#" * int(kind[1]) + " " + text
    return text


def _document_text(contents) -> str:
    """Blocks separated by blank lines; consecutive list items / table rows stay together."""
    lines, previous = [], None
    for fragment in contents:
        line = _plain(fragment)
        kind = line[:2] if line[:2] in ("- ", "| ") else "p"
        if lines and not (kind == previous and kind != "p"):
            lines.append("")
        lines.append(line)
        previous = kind
    return "\n".join(lines) + "\n"


def _slug(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def main() -> None:
    raw = {name: requests.get(BASE + name, timeout=60).content for name in ("documents.json", "dev.json")}
    documents = {d["url"]: d for d in json.loads(raw["documents.json"])}
    questions = json.loads(raw["dev.json"])

    policy_dir = ROOT / "eval" / "policies" / "conditionalqa"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policies = {}
    for url in URLS:
        path = policy_dir / f"{_slug(url)}.txt"
        path.write_text(_document_text(documents[url]["contents"]))
        policies[_slug(url)] = str(path.relative_to(ROOT))

    items = []
    for q in questions:
        if q["url"] not in URLS:
            continue
        conditional = any(conditions for _, conditions in q["answers"])
        item = {
            "id": q["id"],
            "doc_id": _slug(q["url"]),
            "type": "unanswerable" if q["not_answerable"] else ("conditional" if conditional else "unconditional"),
            "question": f"{q['scenario']} {q['question']}",
            "category": "any",
            "evidence_phrases": [_plain(e).lstrip("-# ").strip("| ").strip() for e in q["evidences"]],
            "cqa_answers": [[a, [_plain(c) for c in conds]] for a, conds in q["answers"]],
        }
        if q["not_answerable"]:
            item["expected"] = {"rule_type": "no_rule_found", "limit_amount": None}
        items.append(item)

    dataset = {
        "name": "conditionalqa_dev_subset",
        "policies": policies,
        "source": "ConditionalQA v1_0 dev (github.com/haitian-sun/ConditionalQA), BSD-2-Clause, research use",
        "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        "gold_from": "ConditionalQA annotations (not verified by legal professionals, per the dataset README)",
        "notes": [
            "Questions are scenario + question, as ConditionalQA intends.",
            "Answerable questions are scored on retrieval only: the pipeline's judge returns an expense rule_type, not a span/yes-no with conditions.",
            "Amounts are in £; app.core.table_rules only reads ₹/Rs/INR/$/USD as currency.",
        ],
        "items": items,
    }
    out = ROOT / "eval" / "datasets" / "conditionalqa_dev_subset.json"
    out.write_text(json.dumps(dataset, indent=1, ensure_ascii=False))
    types = {t: sum(i["type"] == t for i in items) for t in ("unconditional", "conditional", "unanswerable")}
    print(f"{len(policies)} policies, {len(items)} questions {types} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
