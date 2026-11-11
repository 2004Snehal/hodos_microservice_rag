"""
table_rules.py — deterministic rules read from tables in policy text.

At ingest, finds tables (markdown pipes, tabs, or columns separated by two
or more spaces) and turns each value cell into a structured row:

  {category: meal, grade: L3–L4, city_tier: metro, cap: 1800, unit: per_day}

Two table shapes are read:
  long — one rule per line:           Category | Grade | City tier | Cap | Unit
  wide — value columns named by tier
         or category:                  Grade | Metro | Non-metro
A dimension the table doesn't have comes from the caption line just above it
("Meal allowance (₹ per day)"), and a missing category from the nearest
markdown heading above it; a dimension nothing names applies to everyone.

A cell becomes a row only when it can't be misread: a bare amount with an
optional currency and unit, "at actuals" / "no limit" (waived), or
"pre-approval". Anything else ("1800, more with approval") is skipped and left
to the LLM path. A cap with no unit anywhere is skipped too: a guessed
per_day vs per_trip would turn into a wrong audit verdict.

Policy lines like "Metro cities: Mumbai, Delhi, Chennai" are read into a
city -> tier map so a request for location=Mumbai can match a "Metro" row.
A table row label that lists its cities ("Metros + Top cities: (Mumbai, Pune)",
"Rest of the cities") is read the same way.

The header is the row with no value cells and some role word (limit,
allowance, grade, city ...), not necessarily the first: some PDFs print the
header under the data row, and layout tools (Docling) keep that order.

One inference is made, and reported in `inferred`: a lodging cap with no unit
anywhere, in a table about hotels / rooms / accommodation, is per_night —
hotel tariffs are nightly. Every other unit-less cap is still skipped.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.core.bill_parser import CATEGORY_KEYWORDS

ALL = "all"
OPEN_HIGH = 999.0  # upper bound for "L5 & above"

# Used only when the policy doesn't list its own cities. These are the
# Indian "X" (metro) cities from the 7th CPC HRA classification.
DEFAULT_METROS = {
    "mumbai", "bombay", "delhi", "new delhi", "kolkata", "calcutta", "chennai", "madras",
    "bengaluru", "bangalore", "hyderabad", "pune", "ahmedabad",
}

GradeRange = Tuple[str, float, float]  # (prefix, lo, hi), inclusive


@dataclass
class TableRule:
    category: str
    grade: str                 # as written in the table, or "all"
    grade_ranges: Optional[List[GradeRange]]  # None = every grade
    city_tier: str             # tier label, "city:<name>", or "all"
    rule_type: str             # cap | waived | requires_preapproval
    cap: Optional[float]
    unit: Optional[str]        # per_day | per_night | per_trip | per_month | per_year | one_time
    currency: Optional[str]
    ref: str                   # "table1:row2:col3"
    source: str                # the table line the rule came from

    def public(self) -> Dict:
        return {"category": self.category, "grade": self.grade, "city_tier": self.city_tier,
                "rule_type": self.rule_type, "cap": self.cap, "unit": self.unit,
                "currency": self.currency, "ref": self.ref}


@dataclass
class TableExtraction:
    tables: int = 0
    rules: List[TableRule] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    city_tiers: Dict[str, str] = field(default_factory=dict)  # city -> tier, "*" = every other city
    inferred: List[Dict[str, str]] = field(default_factory=list)  # values not stated in the table


# ---------------------------------------------------------------- normalizers

_UNIT_WORDS = {
    "day": "per_day", "daily": "per_day", "night": "per_night", "nightly": "per_night",
    "trip": "per_trip", "month": "per_month", "monthly": "per_month",
    "year": "per_year", "annum": "per_year", "yearly": "per_year", "annually": "per_year",
    "diem": "per_day",
}
_UNIT_PHRASE = (r"(?:(?:\b(?:per|a|an|each)\s+|/\s*)(?:day|night|trip|month|year|annum|diem)\b"
                r"|\b(?:daily|nightly|monthly|yearly|annually)\b|\bone[\s-]?time\b)")
_UNIT_RE = re.compile(_UNIT_PHRASE, re.IGNORECASE)
_CURRENCY_RE = re.compile(r"₹|\brs\b\.?|\binr\b|\$|\busd\b", re.IGNORECASE)

_CAP_CELL = re.compile(
    rf"^\s*(?P<cur1>₹|rs\.?|inr|\$|usd)?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?:/-)?\s*"
    rf"(?P<cur2>inr|usd|rupees)?\s*(?P<unit>{_UNIT_PHRASE})?\s*$",
    re.IGNORECASE,
)
_WAIVED_CELLS = {"actuals", "actual", "at actuals", "at actual", "as per actuals", "no limit",
                 "no cap", "unlimited", "waived"}
_PREAPPROVAL_CELL = re.compile(
    r"^(?:requires?\s+|needs\s+)?(?:pre\s*approval|prior approval|approval required|with approval|on approval)$"
)
_EMPTY_CELLS = {"", "-", "–", "—", "n/a", "na", "nil"}


def parse_unit(text: str) -> Optional[str]:
    m = _UNIT_RE.search(text or "")
    if not m:
        return None
    phrase = m.group(0).lower()
    if "one" in phrase:
        return "one_time"
    return _UNIT_WORDS[re.findall(r"[a-z]+", phrase)[-1]]


def parse_currency(text: str) -> Optional[str]:
    m = _CURRENCY_RE.search(text or "")
    if not m:
        return None
    return "USD" if m.group(0).lower() in ("$", "usd") else "INR"


def category_of(text: str) -> Optional[str]:
    """The one known category named in text, or None (none, or more than one)."""
    lowered = (text or "").lower()
    found = {cat for cat, words in CATEGORY_KEYWORDS.items()
             if any(re.search(rf"\b{re.escape(w)}", lowered) for w in words)}
    return found.pop() if len(found) == 1 else None


def normalize_category(text: str) -> Optional[str]:
    """Known category, else a slug of the text ("Internet" -> "internet")."""
    if not text or text.strip().lower() == "any":
        return None
    return category_of(text) or _slug(text)


def _core(text: str) -> str:
    """Header/caption text without parentheticals, units and currency marks."""
    text = re.sub(r"\(.*?\)", " ", text or "")
    text = _UNIT_RE.sub(" ", text)
    text = _CURRENCY_RE.sub(" ", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def normalize_tier(text: str) -> Optional[str]:
    t = _core(text)
    suffix = r"(?: cities| city| locations| areas)?"
    if t in ("all", "any", "all cities", "all locations", "everywhere"):
        return ALL
    if re.fullmatch(rf"(?:non|other than) ?metro(?:politan)?{suffix}", t):
        return "non_metro"
    if re.fullmatch(rf"metro(?:politan)?s?{suffix}", t):
        return "metro"
    m = re.fullmatch(rf"tier ?(1|2|3|i|ii|iii){suffix}", t)
    if m:
        return "tier_" + {"i": "1", "ii": "2", "iii": "3"}.get(m.group(1), m.group(1))
    if re.fullmatch(rf"(?:other|others|rest){suffix}", t):
        return "other"
    return None


_ALL_GRADES = {"all", "any", "all grades", "all levels", "all bands", "everyone", "all employees"}
_GRADE_TOKEN = re.compile(r"\b([A-Za-z]{1,3})\s*-?\s*(\d{1,2})\b")
_GRADE_FILLER = re.compile(
    r"\b(?:grades?|bands?|levels?|and|or|to|up|upto|till|above|below|onwards)\b|[&+,/\-–—\s]"
)


def parse_grade(text: str) -> Optional[List[GradeRange]]:
    """
    "L3–L4" -> [("L", 3, 4)], "L1, L2" -> two points, "L5 & above" -> [("L", 5, 999)].
    Returns [] for "all grades" and None when the text can't be read exactly
    (bare "above L5" is skipped: whether it includes L5 is ambiguous).
    """
    raw = (text or "").strip()
    if raw.lower() in _ALL_GRADES:
        return []
    tokens = list(_GRADE_TOKEN.finditer(raw))
    if not tokens or _GRADE_FILLER.sub("", _GRADE_TOKEN.sub("", raw)):
        return None
    lowered = raw.lower()
    open_high = bool(re.search(r"(?:and|&|or)\s*above|\+|onwards", lowered))
    open_low = bool(re.search(r"(?:and|&|or)\s*below|\bup\s*to\b|\bupto\b|\btill\b", lowered))
    if re.search(r"\b(?:above|below)\b", lowered) and not (open_high or open_low):
        return None

    points = [(t.group(1).upper(), float(t.group(2))) for t in tokens]
    if len(points) == 1:
        prefix, n = points[0]
        return [(prefix, 0.0 if open_low else n, OPEN_HIGH if open_high else n)]
    if open_high or open_low:
        return None
    between = [raw[a.end():b.start()].strip().lower() for a, b in zip(tokens, tokens[1:])]
    if len(points) == 2 and between[0] in ("-", "–", "—", "to") and points[0][0] == points[1][0]:
        return [(points[0][0], min(points[0][1], points[1][1]), max(points[0][1], points[1][1]))]
    if all(s in (",", "/", "and", "&", "or") for s in between):
        return [(p, n, n) for p, n in points]
    return None


def parse_request_grade(grade: str) -> Optional[Tuple[str, float]]:
    ranges = parse_grade(grade or "")
    if not ranges or len(ranges) != 1 or ranges[0][1] != ranges[0][2]:
        return None
    return ranges[0][0], ranges[0][1]


def location_tiers(location: str, city_tiers: Dict[str, str]) -> Optional[Set[str]]:
    """Tier labels a requested location belongs to; None when it isn't specified."""
    if not location or location.strip().lower() == "any":
        return None
    tier = normalize_tier(location)
    if tier == ALL:
        return None
    if tier:
        return {tier}
    city = _city(location)
    labels = {f"city:{city}"}
    if city_tiers:
        mapped = city_tiers.get(city) or city_tiers.get("*")
        if mapped:
            labels.add(mapped)
    elif city in DEFAULT_METROS:
        labels |= {"metro", "tier_1"}
    return labels


def _city(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())).strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


# ---------------------------------------------------------------- table detection

_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> Optional[Tuple[str, List[str]]]:
    if "|" in line:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        return ("pipe", cells) if len(cells) >= 2 else None
    if "\t" in line:
        cells = [c.strip() for c in re.split(r"\t+", line.strip())]
        return ("tab", cells) if len(cells) >= 2 else None
    cells = [c.strip() for c in re.split(r"\s{2,}", line.strip())]
    return ("space", cells) if len(cells) >= 2 else None


def _find_tables(lines: List[str]) -> List[Tuple[int, List[Tuple[int, List[str]]]]]:
    """[(header line index, [(line index, cells), ...])] — header row first."""
    tables, i = [], 0
    while i < len(lines):
        first = _split_row(lines[i]) if lines[i].strip() else None
        if not first:
            i += 1
            continue
        kind, width = first[0], len(first[1])
        rows, j = [(i, first[1])], i + 1
        while j < len(lines):
            if kind == "pipe" and _SEPARATOR_ROW.match(lines[j]):
                j += 1
                continue
            nxt = _split_row(lines[j]) if lines[j].strip() else None
            # Space-aligned columns are only trusted when every line has the same width.
            if not nxt or nxt[0] != kind or (kind == "space" and len(nxt[1]) != width):
                break
            rows.append((j, nxt[1]))
            j += 1
        if len(rows) >= 2:
            tables.append((i, rows))
        i = j
    return tables


# ---------------------------------------------------------------- header roles

# Checked in order. city_tier comes before category so "Type of City" is a
# city column, not an expense-type column.
_ROLE_WORDS = [
    ("cap", ["cap", "limit", "amount", "max", "maximum", "entitlement", "allowance", "ceiling", "rate", "eligibility"]),
    ("grade", ["grade", "band", "level", "designation"]),
    ("city_tier", ["city", "cities", "tier", "location", "class"]),
    ("category", ["category", "expense", "head", "item", "type"]),
    ("unit", ["unit", "period", "frequency", "basis"]),
]


@dataclass
class _Column:
    role: str                         # "value", a dimension name, or "unknown"
    category: Optional[str] = None    # fixed by the header, for value columns
    city_tier: Optional[str] = None
    unit: Optional[str] = None
    currency: Optional[str] = None


def _header_role(cell: str) -> Optional[str]:
    words = set(_core(cell).split())
    for role, keys in _ROLE_WORDS:
        if words & set(keys):
            return role
    return None


def _header_category(cell: str) -> Optional[str]:
    """
    The category a header names. A header naming two known categories ("Own
    Accommodation Allowance (Inclusive of Meal Allowance)") is its own
    category, by its name without parentheticals ("own_accommodation_allowance"),
    so a lodging or meal lookup never picks it up by accident.
    """
    lowered = (cell or "").lower()
    found = {cat for cat, words in CATEGORY_KEYWORDS.items()
             if any(re.search(rf"\b{re.escape(w)}", lowered) for w in words)}
    if len(found) == 1:
        return found.pop()
    return _slug(_core(cell)) if len(found) > 1 else None


def _classify(cell: str) -> _Column:
    role, tier, cat = _header_role(cell), normalize_tier(cell), _header_category(cell)
    fixed = dict(category=cat, unit=parse_unit(cell), currency=parse_currency(cell))
    if role == "cap":
        return _Column("value", city_tier=tier, **fixed)
    if tier and tier != ALL:
        return _Column("value", city_tier=tier, **fixed)
    if role:
        return _Column(role)
    if cat:
        return _Column("value", city_tier=None, **fixed)
    return _Column("unknown")


def _infer_first_column(values: List[str]) -> str:
    """A label column with no header ("" | Metro | Non-metro) holds grades or categories."""
    values = [v for v in values if v]
    if values and all(parse_grade(v) is not None for v in values):
        return "grade"
    if values and all(category_of(v) for v in values):
        return "category"
    return "unknown"


def _parse_value(cell: str) -> Optional[Tuple[str, Optional[float], Optional[str], Optional[str]]]:
    """(rule_type, cap, unit, currency) for an unambiguous cell, else None."""
    lowered = re.sub(r"\s+", " ", cell.strip().lower())
    if lowered in _WAIVED_CELLS:
        return "waived", None, None, None
    if _PREAPPROVAL_CELL.match(re.sub(r"[-.]", " ", lowered).strip()):
        return "requires_preapproval", None, None, None
    m = _CAP_CELL.match(cell)
    if not m:
        return None
    cur = m.group("cur1") or m.group("cur2")
    return "cap", float(m.group("num").replace(",", "")), parse_unit(m.group("unit") or ""), parse_currency(cur or "")


def _row_tier(label: str) -> Tuple[Optional[str], List[str]]:
    """
    A city-tier row label -> (tier, cities it lists).
    "Metros +Top cities + Holiday destinations: (Mumbai, Pune)" -> ("metro", ["mumbai", "pune"])
    "Rest of the cities" -> ("other", ["*"])
    """
    head = re.split(r"[:(]", label, maxsplit=1)[0]
    tier = normalize_tier(head)
    if not tier:
        core = _core(head)
        if re.search(r"\bnon ?metro", core):
            tier = "non_metro"
        elif re.search(r"\bmetro", core):
            tier = "metro"
        elif re.match(r"(?:rest|other|others|remaining|all other)\b", core):
            tier = "other"
    if not tier or tier == ALL:
        return None, []
    if tier == "other" and not re.search(r"[(:]", label):
        return tier, ["*"]
    listed = re.search(r"\((.*?)\)", label) or re.search(r":(.*)$", label)
    cities = [_city(c) for c in re.split(r",|;|/|\band\b", listed.group(1))] if listed else []
    return tier, [c for c in cities if c]


def _header_row(rows: List[Tuple[int, List[str]]]) -> int:
    """
    Index of the header row: the first row with no value cell and some role
    word. Falls back to the first row, as before.
    """
    for i, (_, cells) in enumerate(rows):
        has_value = any(_parse_value(c) for c in cells if c.strip().lower() not in _EMPTY_CELLS)
        has_role = any(_header_role(c) or normalize_tier(c) for c in cells)
        if not has_value and has_role:
            return i
    return 0


_HOTEL_CONTEXT = re.compile(r"\b(?:hotel|room|accommodation|lodging|stay|tariff)", re.IGNORECASE)


# ---------------------------------------------------------------- extraction

def extract_city_tiers(text: str) -> Dict[str, str]:
    """Lines like "Metro cities: Mumbai, Delhi" or "Non-metro: all other cities"."""
    mapping: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        tier = normalize_tier(re.sub(r"^\s*[-*•#]+\s*", "", left))
        if not tier or tier == ALL:
            continue
        if re.search(r"\b(?:all\s+)?(?:other|remaining|rest)\b", right.lower()) and len(right.split()) <= 5:
            mapping["*"] = tier
            continue
        for name in re.split(r",|;|/|\band\b", right):
            city = _city(re.sub(r"\betc\b", "", name, flags=re.IGNORECASE))
            if city:
                mapping[city] = tier
    return mapping


def extract_table_rules(text: str) -> TableExtraction:
    lines = text.splitlines()
    out = TableExtraction(city_tiers=extract_city_tiers(text))

    for header_idx, rows in _find_tables(lines):
        h = _header_row(rows)
        header, body = rows[h][1], rows[:h] + rows[h + 1:]
        columns = [_classify(c) for c in header]
        if columns[0].role == "unknown":
            columns[0] = _Column(_infer_first_column([r[1][0] if r[1] else "" for r in body]))
        if not any(c.role == "value" for c in columns):
            continue

        caption = next((lines[k] for k in range(header_idx - 1, max(header_idx - 3, -1), -1) if lines[k].strip()), "")
        cap_category, cap_unit, cap_currency = category_of(caption), parse_unit(caption), parse_currency(caption)
        # Nearest markdown heading above the table: layout tools can place a
        # table several lines away from its caption, but under its section.
        section = next((lines[k] for k in range(header_idx - 1, -1, -1) if lines[k].lstrip().startswith("#")), "")
        section_category = category_of(section)
        out.tables += 1
        table_no = out.tables

        for row_no, (line_idx, cells) in enumerate(body, start=1):
            cells = (cells + [""] * len(columns))[: len(columns)]
            dims = {c.role: cells[i] for i, c in enumerate(columns) if c.role in ("category", "grade", "city_tier", "unit")}
            source = lines[line_idx].strip()

            # A row label listing cities maps them to its tier even when the
            # row's amount can't be read: otherwise those cities would fall
            # through to the "rest of the cities" row and get its cap.
            tier_text = dims.get("city_tier", "").strip()
            row_tier = normalize_tier(tier_text) if tier_text else None
            if tier_text and not row_tier:
                row_tier, cities = _row_tier(tier_text)
                for city in cities:
                    out.city_tiers.setdefault(city, row_tier)
                row_tier = row_tier or f"city:{_city(tier_text)}"

            for col_no, (col, cell) in enumerate(zip(columns, cells), start=1):
                if col.role != "value" or cell.strip().lower() in _EMPTY_CELLS:
                    continue
                ref = f"table{table_no}:row{row_no}:col{col_no}"

                def skip(reason: str) -> None:
                    out.skipped.append({"ref": ref, "cell": cell, "line": source, "reason": reason})

                value = _parse_value(cell)
                if value is None:
                    skip("cell is not a bare amount, 'at actuals' or 'pre-approval'")
                    continue
                rule_type, cap, cell_unit, cell_currency = value

                category = normalize_category(dims.get("category", "")) or col.category or cap_category or section_category
                if not category:
                    skip("no category in the row, column header, caption or section heading")
                    continue

                grade = dims.get("grade", "").strip() or ALL
                ranges = parse_grade(grade)
                if ranges is None:
                    skip(f"grade '{grade}' can't be read exactly")
                    continue

                city_tier = row_tier or col.city_tier or ALL

                unit = cell_unit or parse_unit(dims.get("unit", "")) or _unit_word(dims.get("unit", "")) or col.unit or cap_unit
                if rule_type == "cap" and not unit and category == "lodging" \
                        and _HOTEL_CONTEXT.search(" ".join([caption, section, " ".join(header), source])):
                    unit = "per_night"
                    out.inferred.append({"ref": ref, "field": "unit", "value": unit,
                                         "reason": "lodging cap with no stated unit in a hotel/room table"})
                if rule_type == "cap" and not unit:
                    skip("no unit (per day / per night / per trip ...) in the cell, row, header or caption")
                    continue

                out.rules.append(TableRule(
                    category=category, grade=grade, grade_ranges=ranges or None, city_tier=city_tier,
                    rule_type=rule_type, cap=cap, unit=unit,
                    currency=cell_currency or col.currency or cap_currency,
                    ref=ref, source=source,
                ))
    return out


def _unit_word(text: str) -> Optional[str]:
    """A unit column holding a bare word: "day", "night", "per_trip"."""
    word = (text or "").strip().lower().replace("per_", "").replace("per ", "")
    if word in ("one_time", "one time", "one-time"):
        return "one_time"
    return _UNIT_WORDS.get(word)
