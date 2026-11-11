"""
chunking.py — structure-aware recursive chunking.

Strategy (in order of preference for where a chunk may end):
  1. Paragraph boundaries (blank lines)
  2. Sentence boundaries (pysbd, plus a domain-abbreviation merge — see below)
  3. Clause boundaries (; : ,) — only for a single sentence longer than chunk_size
  4. Word boundaries — last resort, still never splits inside a word or number

Sentences are packed greedily into chunks of up to `chunk_size` units.
Overlap is made of WHOLE trailing sentences from the previous chunk (up to
`overlap` units), so a clause like "capped at Rs. 6000 per night" is
never cut in half — the old fixed-character-window chunker split it into
"capped at Rs. 6" / "000 per night", and rule extraction read the cap as 6.

LENGTH UNITS
`length` decides what a unit is: characters by default (len), or embedding
model tokens when the caller passes a token counter (the /v2 pipeline passes
dense_embeddings.count_tokens, so chunks are sized for bge-small's 512-token
window). A final chunk shorter than `min_chunk` is merged into the previous
one when the result stays within `max_chunk`.

TABLES (chunk_document)
chunk_text treats a table as one long "sentence" and cuts it at commas and
spaces, so a header lands in one leaf, its rows in another, and a markdown
separator row ("|-----|") becomes a leaf of dashes. chunk_document lifts
pipe / tab tables out of the text first and gives each one its own leaves:
  - each row is written out with its column names ("Class: B; Amount: 2500."),
    so a leaf of rows still says what its numbers are; separator rows are dropped
  - every leaf of a table starts with where it sits: its "Table N" id and
    caption, the markdown heading path above it, and up to `description_budget`
    units of sentences elsewhere in the document that mention "Table N"
    (a table described on page 5 but printed on page 9 still embeds with
    that description)
  - rows are grouped up to chunk_size per leaf, the context repeated in each
The prose around the tables is chunked by chunk_text as before. Leaves record
the table ids they are or mention, so the caller can link a sentence citing
"Table 14.3" to the table's leaves (app/advanced/probabilistic/pipeline.py).
Columns separated only by spaces are left as prose: that pattern misfires on
OCR text.

Hard line wraps inside a paragraph (common in PDF / plain-text policies) are
joined back into a single line before sentence splitting.

SENTENCE SPLITTING
pysbd (rule-based, the Golden Rules test set) does the base split. It gets
numbered lists ("1. Meals ... 2. Lodging"), "No." as a one-word answer, and
sentences starting in lowercase right. It doesn't know policy/currency
abbreviations and splits "capped at INR. 1000 per day" into "...INR." /
"1000 per day." — cutting a currency from its amount, the exact failure this
module exists to prevent. So a split is undone when the previous sentence
ends in one of _ABBREVIATIONS and the next one starts with a digit or a
lowercase letter ("INR. 1000", "Approx. 5", "max. per"). "No. The cap ..."
stays split: the next sentence starts with an uppercase word.

This is the ACTIVE chunking strategy used by the simple RAG pipeline
(app/core/simple_rag.py), the RAPTOR leaf layer
(app/advanced/integrated/pipeline.py) and the /v2 probabilistic pipeline
(app/advanced/probabilistic/pipeline.py).
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pysbd

from app.core.table_rules import _SEPARATOR_ROW, _find_tables, _header_row

# Abbreviations whose trailing "." must not end a sentence when the next
# sentence starts with a digit or lowercase letter. Matched
# case-insensitively against the word immediately before the period.
_ABBREVIATIONS = {
    "rs", "inr", "no", "nos", "sec", "secs", "e.g", "i.e", "etc", "vs", "approx",
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "dept", "max", "min", "incl", "excl",
    "fig", "ref", "art", "cl", "para",
}

_CLAUSE_END = re.compile(r"(?<=[;:,])\s+")


def _normalize_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [re.sub(r"\s+", " ", p).strip() for p in paragraphs if p.strip()]


_segmenter = None


def _get_segmenter() -> "pysbd.Segmenter":
    global _segmenter
    if _segmenter is None:
        # clean=False: keep the text exactly as given (no whitespace/newline rewriting).
        _segmenter = pysbd.Segmenter(language="en", clean=False)
    return _segmenter


def split_sentences(paragraph: str) -> List[str]:
    sentences: List[str] = []
    for piece in _get_segmenter().segment(paragraph):
        piece = piece.strip()
        if not piece:
            continue
        if sentences and _ends_in_abbreviation(sentences[-1]) and re.match(r"[a-z0-9]", piece):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _ends_in_abbreviation(sentence: str) -> bool:
    preceding_word = re.search(r"([\w.]+)\.$", sentence)
    return bool(preceding_word) and preceding_word.group(1).lower() in _ABBREVIATIONS


def _split_oversized(sentence: str, chunk_size: int, length: Callable[[str], int] = len) -> List[str]:
    """Break a single sentence longer than chunk_size at clause, then word, boundaries."""
    pieces: List[str] = []
    for clause in _CLAUSE_END.split(sentence):
        if length(clause) <= chunk_size:
            pieces.append(clause)
            continue
        current = ""
        for word in clause.split(" "):
            candidate = f"{current} {word}".strip()
            if length(candidate) > chunk_size and current:
                pieces.append(current)
                current = word
            else:
                current = candidate
        if current:
            pieces.append(current)

    # Re-pack small clauses back together so we don't emit tiny fragments.
    packed: List[str] = []
    for piece in pieces:
        if packed and length(f"{packed[-1]} {piece}") <= chunk_size:
            packed[-1] = f"{packed[-1]} {piece}"
        else:
            packed.append(piece)
    return packed


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    min_chunk: int = 0,
    max_chunk: Optional[int] = None,
    length: Callable[[str], int] = len,
) -> List[str]:
    """
    Split `text` into chunks of whole sentences.

    Args:
        text: the raw document text to split.
        chunk_size: soft maximum units per chunk. A chunk only exceeds it
                    when a single word is longer than chunk_size, or when a
                    short final chunk is merged into it (see min_chunk).
        overlap: maximum units of trailing whole sentences repeated at the
                 start of the next chunk. 0 disables overlap.
        min_chunk: a final chunk with fewer units of new (non-overlap)
                   text than this is merged into the previous chunk.
        max_chunk: hard ceiling for that merge; defaults to chunk_size.
        length: measures a string in units — len for characters, or a
                token counter. Must be additive over space-joined text
                (true for characters and for BERT WordPiece tokens).

    Returns:
        List of chunk strings, in document order.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    max_chunk = chunk_size if max_chunk is None else max_chunk
    if max_chunk < chunk_size:
        raise ValueError("max_chunk must be at least chunk_size")
    sep = length(" ")  # cost of the joining space: 1 character, 0 tokens

    units: List[tuple] = []  # (sentence, units, starts_new_paragraph)
    for paragraph in _normalize_paragraphs(text):
        for i, sentence in enumerate(split_sentences(paragraph)):
            size = length(sentence)
            parts = [sentence] if size <= chunk_size else _split_oversized(sentence, chunk_size, length)
            for j, part in enumerate(parts):
                units.append((part, size if len(parts) == 1 else length(part), i == 0 and j == 0))

    chunks: List[tuple] = []  # (sentences, sizes, carried)
    current: List[str] = []
    sizes: List[int] = []
    current_len = 0
    carried = 0  # how many sentences at the head of `current` are overlap

    def flush():
        nonlocal current, sizes, current_len, carried
        if len(current) > carried:
            chunks.append((current, sizes, carried))
        # Carry whole trailing sentences forward as overlap.
        tail_len = 0
        keep = 0
        for size in reversed(sizes):
            if tail_len + size + sep > overlap:
                break
            tail_len += size + sep
            keep += 1
        current, sizes = current[len(current) - keep:], sizes[len(sizes) - keep:]
        current_len, carried = tail_len, keep

    for sentence, size, new_paragraph in units:
        needed = size + (sep if current else 0)
        # Prefer to end a chunk at a paragraph boundary once it's half full.
        at_paragraph_break = new_paragraph and current_len >= chunk_size // 2 and len(current) > carried
        if current and (current_len + needed > chunk_size or at_paragraph_break):
            flush()
            # Drop overlap if it would push this sentence over the limit.
            while current and current_len + size + sep > chunk_size:
                current.pop(0)
                current_len -= sizes.pop(0) + sep
                carried -= 1
            needed = size + (sep if current else 0)
        current.append(sentence)
        sizes.append(size)
        current_len += needed

    if len(current) > carried:
        chunks.append((current, sizes, carried))

    # Merge a short final chunk into the previous one (dropping its overlap,
    # which the previous chunk already holds) if that stays under max_chunk.
    if len(chunks) >= 2 and min_chunk:
        last, last_sizes, last_carried = chunks[-1]
        new_sentences, new_sizes = last[last_carried:], last_sizes[last_carried:]
        new_len = sum(new_sizes) + sep * (len(new_sizes) - 1)
        prev, prev_sizes, prev_carried = chunks[-2]
        merged_len = sum(prev_sizes) + sum(new_sizes) + sep * (len(prev_sizes) + len(new_sizes) - 1)
        if new_len < min_chunk and merged_len <= max_chunk:
            chunks[-2:] = [(prev + new_sentences, prev_sizes + new_sizes, prev_carried)]

    return [" ".join(sentences) for sentences, _, _ in chunks]


# ---------------------------------------------------------------- tables

_TABLE_ID = re.compile(r"\btable\s+(\d+(?:[.\-]\d+)*[a-z]?|[a-z](?:[.\-]?\d+)*)\b", re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")
_MAX_CAPTION_CHARS = 200


@dataclass
class Leaf:
    text: str
    kind: str = "text"                                 # "text" | "table"
    table_ids: List[str] = field(default_factory=list)  # the table a "table" leaf belongs to
    mentions: List[str] = field(default_factory=list)   # tables a "text" leaf mentions
    table_key: Optional[int] = None                     # which table (all leaves of one table share it)
    table_info: dict = field(default_factory=dict)      # caption, section, header, row labels: for summaries


def table_ids_in(text: str) -> List[str]:
    """Normalized "Table N" ids mentioned in text: "Table 14.3" -> "14.3"."""
    return sorted({m.group(1).lower() for m in _TABLE_ID.finditer(text)})


def _is_table_line(line: str) -> bool:
    return "|" in line or "\t" in line


def _heading_path(lines: List[str], end: int) -> str:
    """
    Heading breadcrumb above line `end`. Docling writes every heading at the
    same level ("##"), which would leave only the last one, so when the path
    has a single heading the one before it is kept too ("C.1 Hotel
    Accommodation > 1. Guest Houses ...").
    """
    path: List[tuple] = []
    previous = None
    for line in lines[:end]:
        m = _HEADING.match(line)
        if m:
            if path:
                previous = path[-1]
            level = len(m.group(1))
            path = [(lvl, title) for lvl, title in path if lvl < level] + [(level, m.group(2))]
    if len(path) == 1 and previous is not None:
        path = [previous] + path
    return " > ".join(title for _, title in path)


def _caption(lines: List[str], start: int) -> str:
    """
    Nearest non-empty line above the table, accepted only if it reads like a
    caption: it names a "Table N" or ends with ":" ("... are as under:").
    Layout tools can put an unrelated paragraph right above a table, and a
    wrong caption is worse than none.
    """
    for k in range(start - 1, max(start - 4, -1), -1):
        line = lines[k].strip()
        if not line:
            continue
        if _HEADING.match(line) or _is_table_line(line):
            return ""
        if table_ids_in(line) or line.endswith(":"):
            return line[:_MAX_CAPTION_CHARS]
        return ""
    return ""


def _linearize(header: List[str], cells: List[str]) -> str:
    parts = []
    for name, cell in zip(header + [""] * len(cells), cells):
        cell = re.sub(r"\s+", " ", cell).strip()
        if not cell:
            continue
        name = re.sub(r"\s+", " ", name).strip()
        parts.append(f"{name}: {cell}" if name else cell)
    return "; ".join(parts) + "." if parts else ""


def chunk_document(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    min_chunk: int = 0,
    max_chunk: Optional[int] = None,
    length: Callable[[str], int] = len,
    description_budget: int = 60,
) -> List[Leaf]:
    """
    Table-aware chunking: prose through chunk_text, each pipe / tab table
    as its own self-describing leaves. See the module docstring.
    """
    max_chunk = chunk_size if max_chunk is None else max_chunk
    lines = text.splitlines()
    tables = [(start, rows) for start, rows in _find_tables(lines) if _is_table_line(lines[start])]

    prose_lines = list(lines)
    for start, rows in tables:
        for k in range(start, rows[-1][0] + 1):
            prose_lines[k] = ""
    prose = "\n".join(prose_lines)

    leaves = [Leaf(t, "text", mentions=table_ids_in(t))
              for t in chunk_text(prose, chunk_size=chunk_size, overlap=overlap,
                                  min_chunk=min_chunk, max_chunk=max_chunk, length=length)]

    prose_sentences = [s for paragraph in _normalize_paragraphs(prose) for s in split_sentences(paragraph)]

    for table_key, (start, rows) in enumerate(tables):
        rows = [(i, cells) for i, cells in rows if not _SEPARATOR_ROW.match(lines[i])]
        if not rows:
            continue
        caption = _caption(lines, start)
        path = _heading_path(lines, start)
        ids = table_ids_in(caption) or table_ids_in(path)

        context = []
        if caption:
            context.append(caption if table_ids_in(caption) or not ids else f"Table {ids[0]}: {caption}")
        if path:
            context.append(f"Section: {path}")
        if ids:
            described, used = [], 0
            for sentence in prose_sentences:
                if sentence == caption or not set(table_ids_in(sentence)) & set(ids):
                    continue
                size = length(sentence)
                if used + size > description_budget:
                    break
                described.append(sentence)
                used += size
            if described:
                context.append("About this table: " + " ".join(described))
        prefix = "\n".join(context)

        h = _header_row(rows)
        header = rows[h][1]
        body = [_linearize(header, cells) for _, cells in rows[:h] + rows[h + 1:]]
        body = [b for b in body if b] or [_linearize([], header)]
        info = {"caption": caption, "section": path, "header": [c.strip() for c in header],
                "row_labels": [cells[0].strip() for _, cells in rows[:h] + rows[h + 1:] if cells and cells[0].strip()]}

        group: List[str] = []
        for row in body:
            candidate = "\n".join(filter(None, [prefix, *group, row]))
            if group and length(candidate) > chunk_size:
                leaves.append(Leaf("\n".join(filter(None, [prefix, *group])), "table", ids,
                                   table_key=table_key, table_info=info))
                group = []
            group.append(row)
        leaves.append(Leaf("\n".join(filter(None, [prefix, *group])), "table", ids,
                           table_key=table_key, table_info=info))

    return leaves

