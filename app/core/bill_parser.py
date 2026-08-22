"""
bill_parser.py — Parser: raw extracted text -> structured line items.

Parses bill lines using regex and category keyword matching.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# Keyword -> category. First match wins per line. Extend this list for your
# own bill formats; this is intentionally small and readable rather than
# exhaustive.
CATEGORY_KEYWORDS = {
    "meal": ["meal", "restaurant", "lunch", "dinner", "breakfast", "food", "cafe", "dining"],
    "lodging": ["hotel", "lodging", "room", "stay", "accommodation", "inn"],
    "transport": ["taxi", "cab", "uber", "ola", "transport", "fare", "ride"],
}

# Matches amounts like "₹1,600", "Rs. 650", "INR 450", "650.00", "$120", "1500"
AMOUNT_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr|\$)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)


@dataclass
class LineItem:
    description: str
    category: Optional[str]
    amount: Optional[float]
    raw_line: str


@dataclass
class ParsedBill:
    vendor_guess: Optional[str]
    date_guess: Optional[str]
    line_items: List[LineItem]

    @property
    def total_by_category(self) -> dict:
        totals: dict = {}
        for item in self.line_items:
            if item.category and item.amount is not None:
                totals[item.category] = totals.get(item.category, 0.0) + item.amount
        return totals


def _guess_category(line: str) -> Optional[str]:
    lower = line.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return category
    return None


def _extract_amount(line: str) -> Optional[float]:
    """
    Take the LAST amount-shaped number on a line — bills commonly show a unit
    price and a line total on the same line ("2 x Naan  100  200"), and the
    final number is usually the line total. This is a real simplification:
    it can misfire on bills laid out differently (see module docstring).
    """
    matches = AMOUNT_PATTERN.findall(line)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def parse_line_items(raw_text: str) -> ParsedBill:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    vendor_guess = lines[0] if lines else None
    date_guess = None
    for line in lines[:5]:  # dates usually appear near the top of a bill
        match = DATE_PATTERN.search(line)
        if match:
            date_guess = match.group(0)
            break

    items: List[LineItem] = []
    for line in lines:
        category = _guess_category(line)
        amount = _extract_amount(line)
        if category is not None and amount is not None:
            items.append(
                LineItem(description=line, category=category, amount=amount, raw_line=line)
            )

    return ParsedBill(vendor_guess=vendor_guess, date_guess=date_guess, line_items=items)
