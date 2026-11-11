"""
run_table_eval.py — Score table extraction systems against gold tables.

  .venv/bin/python -m eval.run_table_eval                     # every gold file in eval/datasets/tables/

SYSTEMS
  current_tesseract_text — today's path: Tesseract plain text
                           (app/core/document_extraction.py) -> table_rules._find_tables
  docling_*              — eval/extractions/<pdf stem>/<variant>/ from
                           eval/extract_tables_docling.py (markdown + cell grids)

LEVELS, per system
  1. tables   — detection recall / precision against the gold tables; a
                predicted table matches a gold one when ≥50% of the gold
                cells appear in it. Cell accuracy = gold cells found at the
                same column of the best-aligned predicted row.
  2. rules    — app.core.table_rules.extract_table_rules over the system's
                text: exact (category, city tier, rule type, cap, unit) and
                lenient (category, rule type, cap) recall, precision, skips.
  3. tiers    — gold metro cities mapped to the "metro" tier.
  4. lookups  — TableRuleStore.lookup() for the gold requests: hit with the
                right cap, or not.

OUTPUT
  metrics/eval_runs/<UTC time>_table_extraction.json  — everything, per table / rule / lookup
  metrics/eval_history.jsonl                          — one line per system ("eval": "table_extraction")
"""

import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.rule_store import TableRuleStore
from app.core.table_rules import ALL, _find_tables, extract_table_rules

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "eval" / "datasets" / "tables"
EXTRACTIONS = ROOT / "eval" / "extractions"
RUNS_DIR = ROOT / "metrics" / "eval_runs"
HISTORY = ROOT / "metrics" / "eval_history.jsonl"


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower())


def _systems(gold: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    text = (ROOT / gold["current_system_text"]).read_text()
    systems = {"current_tesseract_text": {
        "text": text,
        "grids": [[cells for _, cells in rows] for _, rows in _find_tables(text.splitlines())],
        "meta": {"system": "current_tesseract_text", "ocr_engine": "tesseract image_to_string"},
    }}
    for variant in sorted((EXTRACTIONS / Path(gold["source_pdf"]).stem).glob("*/")):
        systems[variant.name] = {
            "text": (variant / "document.md").read_text(),
            "grids": [t["grid"] for t in json.loads((variant / "tables.json").read_text())],
            "meta": json.loads((variant / "meta.json").read_text()),
        }
    return systems


# ---------------------------------------------------------------- 1. tables

def _score_tables(gold_tables: List[Dict[str, Any]], grids: List[List[List[str]]]) -> Dict[str, Any]:
    used, per_table = set(), []
    for gt in gold_tables:
        gold_rows = [gt["header"]] + gt["rows"]
        gold_cells = [_norm(c) for row in gold_rows for c in row]
        best, best_overlap = None, 0.0
        for i, grid in enumerate(grids):
            if i in used:
                continue
            predicted = {_norm(c) for row in grid for c in row}
            overlap = sum(c in predicted for c in gold_cells) / len(gold_cells)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        detected = best is not None and best_overlap >= 0.5
        record = {"id": gt["id"], "rule_bearing": gt["rule_bearing"], "detected": detected,
                  "cell_overlap": round(best_overlap, 4), "predicted_index": best if detected else None}
        if detected:
            used.add(best)
            grid = [[_norm(c) for c in row] for row in grids[best]]
            matched, aligned = 0, []
            for row in gold_rows:
                g = [_norm(c) for c in row]
                scores = [sum(j < len(p) and p[j] == c for j, c in enumerate(g)) for p in grid]
                k = max(range(len(grid)), key=scores.__getitem__)
                matched += scores[k]
                aligned.append(k)
            record.update({"cell_accuracy": round(matched / len(gold_cells), 4),
                           "gold_row_to_predicted_row": aligned,
                           "predicted_shape": [len(grids[best]), max(len(r) for r in grids[best])],
                           "predicted_grid": grids[best]})
        else:
            record["cell_accuracy"] = 0.0
        per_table.append(record)
    found = [t for t in per_table if t["detected"]]
    return {
        "predicted_tables": len(grids),
        "recall": round(len(found) / len(gold_tables), 4),
        "precision": round(len(found) / len(grids), 4) if grids else None,
        "rule_table_recall": round(sum(t["detected"] for t in per_table if t["rule_bearing"])
                                   / sum(t["rule_bearing"] for t in per_table), 4),
        "mean_cell_accuracy": round(sum(t["cell_accuracy"] for t in per_table) / len(per_table), 4),
        "per_table": per_table,
    }


# ---------------------------------------------------------------- 2-4. rules, tiers, lookups

def _rule_matches(gold: Dict[str, Any], rule, exact: bool) -> bool:
    if (rule.category, rule.rule_type) != (gold["category"], gold["rule_type"]):
        return False
    if (rule.cap or None) != (float(gold["cap"]) if gold["cap"] is not None else None):
        return False
    if not exact:
        return True
    tier_ok = rule.city_tier == gold["city_tier"] or (gold["city_tier"] == ALL and rule.city_tier == ALL)
    return tier_ok and rule.unit == gold["unit"] and rule.grade_ranges is None


def _score_rules(gold: Dict[str, Any], text: str) -> Dict[str, Any]:
    extraction = extract_table_rules(text)
    rules = extraction.rules
    exact = [g["id"] for g in gold["rules"] if any(_rule_matches(g, r, True) for r in rules)]
    lenient = [g["id"] for g in gold["rules"] if any(_rule_matches(g, r, False) for r in rules)]
    correct_rows = [r for r in rules if any(_rule_matches(g, r, False) for g in gold["rules"])]

    metros = gold["city_tiers"]["metro"]
    tiers = extraction.city_tiers
    tier_hits = [c for c in metros if tiers.get(c) == "metro"]

    with tempfile.TemporaryDirectory() as tmp:
        store = TableRuleStore(Path(tmp) / "rules.db")
        store.replace("doc", "eval", extraction)
        lookups = []
        for q in gold["lookups"]:
            match = store.lookup("doc", q["category"], q["grade"], q["location"])
            cap = match.rules[0].cap if match.hit else None
            lookups.append({"id": q["id"], "status": match.status, "reason": match.reason, "cap": cap,
                            "correct": match.hit and cap == q["expect"]["cap"]})

    return {
        "rules": {
            "extracted": len(rules),
            "exact_recall": round(len(exact) / len(gold["rules"]), 4),
            "lenient_recall": round(len(lenient) / len(gold["rules"]), 4),
            "precision": round(len(correct_rows) / len(rules), 4) if rules else None,
            "exact_ids": exact, "lenient_ids": lenient,
            "rows": [r.public() for r in rules],
            "skipped": extraction.skipped,
            "tables_seen_by_rule_parser": extraction.tables,
        },
        "city_tiers": {"metro_recall": round(len(tier_hits) / len(metros), 4),
                       "rest_mapped": tiers.get("*"), "extracted": tiers},
        "lookups": {"accuracy": round(sum(l["correct"] for l in lookups) / len(lookups), 4), "per_lookup": lookups},
    }


def main() -> None:
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + "_table_extraction"
    run = {"run_id": run_id, "started_at": started_at.isoformat(), "results": []}

    for gold_path in sorted(GOLD_DIR.glob("*.json")):
        gold = json.loads(gold_path.read_text())
        for name, system in _systems(gold).items():
            result = {"dataset": gold["name"], "system": name, "meta": system["meta"],
                      "tables": _score_tables(gold["tables"], system["grids"]),
                      **_score_rules(gold, system["text"])}
            run["results"].append(result)
            print(f"{gold['name']} / {name}: tables recall={result['tables']['recall']} "
                  f"cells={result['tables']['mean_cell_accuracy']} rules exact={result['rules']['exact_recall']} "
                  f"lenient={result['rules']['lenient_recall']} tiers={result['city_tiers']['metro_recall']} "
                  f"lookups={result['lookups']['accuracy']}", flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.json"
    run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False, default=str))
    with open(HISTORY, "a") as f:
        for r in run["results"]:
            t = r["tables"]
            f.write(json.dumps({
                "run_id": run_id, "started_at": run["started_at"], "eval": "table_extraction",
                "dataset": r["dataset"], "system": r["system"], "seconds": r["meta"].get("seconds"),
                "table_recall": t["recall"], "table_precision": t["precision"],
                "rule_table_recall": t["rule_table_recall"], "mean_cell_accuracy": t["mean_cell_accuracy"],
                "rules_extracted": r["rules"]["extracted"], "rule_exact_recall": r["rules"]["exact_recall"],
                "rule_lenient_recall": r["rules"]["lenient_recall"], "rule_precision": r["rules"]["precision"],
                "metro_city_recall": r["city_tiers"]["metro_recall"], "lookup_accuracy": r["lookups"]["accuracy"],
                "details": str(run_path.relative_to(ROOT)),
            }) + "\n")
    print(f"\nsaved {run_path.relative_to(ROOT)} and appended {HISTORY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
