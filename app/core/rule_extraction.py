"""
rule_extraction.py — Shared limit/rule extraction logic.

WHAT THIS DOES
Given a chunk of policy text and a category's keyword list, finds:
  - `rule_type`: whether the applicable rule is a numeric cap
    ("daily_cap"), an explicit waiver ("waived"), or a pre-approval
    requirement ("requires_preapproval").
  - `limit_amount`: the currency-prefixed number in that same window, if
    `rule_type` is "daily_cap".
"""

import re
from dataclasses import dataclass
from typing import List, Optional


LIMIT_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)

# Keyword sets checked within the same text window as the category keyword.
# Order matters: waiver is checked before pre-approval, before falling back
# to a numeric cap — an explicit waiver should win even if a stray number
# also appears nearby (e.g. "capped at Rs. 1500, waived for international
# travel exceeding five days").
WAIVER_KEYWORDS = ["waived", "waive", "no limit", "not applicable", "exempt"]
PREAPPROVAL_KEYWORDS = [
    "pre-approval", "preapproval", "prior approval", "requires approval",
    "manager approval", "pre-approved",
]

DEFAULT_WINDOW = 80  # characters of context searched after a keyword match


@dataclass
class ResolvedRule:
    rule_type: str  # "daily_cap" | "waived" | "requires_preapproval" | "unresolved"
    limit_amount: Optional[float]
    matched_window: Optional[str]  # the text window the decision was based on, for citation/debugging


def _keyword_windows(text: str, keywords: List[str], window: int = DEFAULT_WINDOW) -> List[str]:
    """All text windows following each occurrence of any keyword in `keywords`."""
    lower_text = text.lower()
    windows = []
    for keyword in keywords:
        start = 0
        while True:
            idx = lower_text.find(keyword, start)
            if idx == -1:
                break
            windows.append(text[idx: idx + len(keyword) + window])
            start = idx + len(keyword)
    return windows


def resolve_rule(text: str, category_keywords: List[str], window: int = DEFAULT_WINDOW) -> ResolvedRule:
    """
    Find the applicable rule for a category within `text`, checked in the
    windows immediately following the category's own keyword occurrences —
    not the whole chunk indiscriminately (see module docstring on why that
    distinction matters).

    Windows are resolved IN DOCUMENT ORDER, and WITHIN each window, whichever
    signal (waiver keyword, pre-approval keyword, or numeric cap) occurs
    EARLIEST in that window's text wins — not a fixed category priority
    applied regardless of position. This went through two real bugs before
    landing here, both surfaced by testing against the served /audit
    endpoint, not found by inspection:

      Bug 1 (all-windows-first-pass): checking every window for a waiver
      BEFORE checking any window for a cap let an unrelated, later clause
      ("international travel meal caps are waived") override an earlier,
      directly-relevant one ("meal expenses are limited to Rs. 1500") that
      happened to share the same chunk. Fixed by resolving one window at a
      time, in order, stopping at the first window that resolves anything.

      Bug 2 (fixed-priority-within-a-window): with windows fixed to resolve
      in order, a SINGLE window could still be wide enough to span both a
      relevant number ("Rs. 1000") and an unrelated later waiver
      ("waived") — and checking waiver-before-cap unconditionally inside
      that one window still let the later, irrelevant "waived" win even
      though "1000" appeared earlier in the same window. Fixed by finding
      the actual character position of each candidate match within the
      window and taking whichever occurs earliest, rather than a fixed
      category order.

    Both bugs are the same underlying lesson twice: resolving by CATEGORY
    PRIORITY instead of by ACTUAL TEXTUAL POSITION silently favors whichever
    condition your code happened to check first, not whichever condition
    the text actually states first. A real LLM reasoning step wouldn't be
    vulnerable to this at all, since it isn't scanning for keyword
    positions in the first place — this remains a documented ceiling of the
    pattern-matching approach, not a claim that this module now reasons
    about text.
    """
    windows = _keyword_windows(text, category_keywords, window)

    if not windows:
        # Category keyword not found at all in this chunk — nothing to resolve.
        return ResolvedRule(rule_type="unresolved", limit_amount=None, matched_window=None)

    for w in windows:
        lower_w = w.lower()

        candidates = []  # (position, rule_type, limit_amount)

        for kw in WAIVER_KEYWORDS:
            pos = lower_w.find(kw)
            if pos != -1:
                candidates.append((pos, "waived", None))

        for kw in PREAPPROVAL_KEYWORDS:
            pos = lower_w.find(kw)
            if pos != -1:
                candidates.append((pos, "requires_preapproval", None))

        match = LIMIT_PATTERN.search(w)
        if match:
            try:
                amount = float(match.group(1).replace(",", ""))
                candidates.append((match.start(), "daily_cap", amount))
            except ValueError:
                pass

        if candidates:
            candidates.sort(key=lambda c: c[0])  # earliest position wins
            _, rule_type, limit_amount = candidates[0]
            return ResolvedRule(rule_type=rule_type, limit_amount=limit_amount, matched_window=w)

    # No window resolved to anything — fall back to a whole-chunk number
    # search as a last resort (can still misfire on a multi-category
    # chunk; documented as a known limitation, same as before this module
    # existed).
    match = LIMIT_PATTERN.search(text)
    if match:
        try:
            amount = float(match.group(1).replace(",", ""))
            return ResolvedRule(rule_type="daily_cap", limit_amount=amount, matched_window=text[:200])
        except ValueError:
            pass

    return ResolvedRule(rule_type="unresolved", limit_amount=None, matched_window=windows[0] if windows else None)
