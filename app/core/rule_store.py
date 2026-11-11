"""
rule_store.py — SQLite store for rules parsed from policy tables.

Kept apart from the vector index and RAPTOR trees: rows here are looked up
by exact key, never by similarity. See app/core/table_rules.py for how rows
are produced.

LOOKUP
A request (category, grade, location) is a hit only when the answer can't
depend on anything the request left out:
  - category must be given;
  - a grade / location the request leaves as "any" only matches if no row
    for that category varies by it (otherwise the answer depends on who's
    asking, so it's a miss);
  - every row that matches must agree on (rule_type, cap, unit) — two rows
    with different caps (say an "All grades" row and an "L3–L4" row) are a
    conflict, and the caller falls back to retrieval.
"""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.table_rules import (
    ALL, TableExtraction, TableRule, location_tiers, normalize_category, parse_request_grade,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    doc_id TEXT NOT NULL,
    doc_hash TEXT,
    ref TEXT NOT NULL,
    category TEXT NOT NULL,
    grade TEXT NOT NULL,
    grade_ranges TEXT,
    city_tier TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    cap REAL,
    unit TEXT,
    currency TEXT,
    source TEXT
);
CREATE INDEX IF NOT EXISTS rules_doc_category ON rules (doc_id, category);
CREATE TABLE IF NOT EXISTS city_tiers (
    doc_id TEXT NOT NULL,
    city TEXT NOT NULL,
    tier TEXT NOT NULL
);
"""


@dataclass
class TableMatch:
    status: str                      # hit | miss | conflict
    reason: str = ""
    rules: List[TableRule] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        return self.status == "hit"

    def summary(self) -> Dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "rows": [r.public() for r in self.rules]}


class TableRuleStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self._connect()) as conn, conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    # ---------------------------------------------------------------- write

    def replace(self, doc_id: str, doc_hash: str, extraction: TableExtraction) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM rules WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM city_tiers WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                "INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(doc_id, doc_hash, r.ref, r.category, r.grade,
                  None if r.grade_ranges is None else json.dumps(r.grade_ranges),
                  r.city_tier, r.rule_type, r.cap, r.unit, r.currency, r.source)
                 for r in extraction.rules],
            )
            conn.executemany(
                "INSERT INTO city_tiers VALUES (?, ?, ?)",
                [(doc_id, city, tier) for city, tier in extraction.city_tiers.items()],
            )

    # ---------------------------------------------------------------- read

    def rules(self, doc_id: str, category: Optional[str] = None) -> List[TableRule]:
        sql = ("SELECT ref, category, grade, grade_ranges, city_tier, rule_type, cap, unit, currency, source "
               "FROM rules WHERE doc_id = ?")
        args: List[Any] = [doc_id]
        if category:
            sql += " AND category = ?"
            args.append(category)
        with closing(self._connect()) as conn:
            rows = conn.execute(sql + " ORDER BY rowid", args).fetchall()
        return [
            TableRule(category=c, grade=g, grade_ranges=None if gr is None else [tuple(x) for x in json.loads(gr)],
                      city_tier=t, rule_type=rt, cap=cap, unit=u, currency=cur, ref=ref, source=src)
            for ref, c, g, gr, t, rt, cap, u, cur, src in rows
        ]

    def city_tiers(self, doc_id: str) -> Dict[str, str]:
        with closing(self._connect()) as conn:
            return dict(conn.execute("SELECT city, tier FROM city_tiers WHERE doc_id = ?", (doc_id,)).fetchall())

    def lookup(self, doc_id: str, category: str, grade: str = "any", location: str = "any") -> TableMatch:
        cat = normalize_category(category)
        if not cat:
            return TableMatch("miss", "category not specified")
        candidates = self.rules(doc_id, cat)
        if not candidates:
            return TableMatch("miss", f"no table row for category '{cat}'")

        req_grade = parse_request_grade(grade)
        req_tiers = location_tiers(location, self.city_tiers(doc_id))
        if req_grade is None and any(r.grade_ranges for r in candidates):
            return TableMatch("miss", f"table varies '{cat}' by grade, but no single grade was given")
        if req_tiers is None and any(r.city_tier != ALL for r in candidates):
            return TableMatch("miss", f"table varies '{cat}' by city tier, but no location was given")

        matched = [r for r in candidates if _grade_ok(r, req_grade) and _tier_ok(r, req_tiers)]
        if not matched:
            return TableMatch("miss", f"no '{cat}' row matches grade={grade!r}, location={location!r}")
        if len({(r.rule_type, r.cap, r.unit) for r in matched}) > 1:
            return TableMatch("conflict", "matching rows disagree", matched)
        # Most specific row first (named grade and tier), for the citation.
        matched.sort(key=lambda r: (r.grade_ranges is None, r.city_tier == ALL))
        return TableMatch("hit", "", matched)

    def stats(self) -> Dict[str, int]:
        with closing(self._connect()) as conn:
            return dict(conn.execute("SELECT doc_id, COUNT(*) FROM rules GROUP BY doc_id").fetchall())


def _grade_ok(rule: TableRule, req) -> bool:
    if rule.grade_ranges is None:
        return True
    if req is None:
        return False
    prefix, n = req
    return any(p == prefix and lo <= n <= hi for p, lo, hi in rule.grade_ranges)


def _tier_ok(rule: TableRule, req_tiers) -> bool:
    if rule.city_tier == ALL:
        return True
    return req_tiers is not None and rule.city_tier in req_tiers
