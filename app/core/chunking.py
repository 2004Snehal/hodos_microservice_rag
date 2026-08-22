"""
chunking.py — basic fixed-size overlap chunking.

This is deliberately the simplest possible chunking strategy: split text into
fixed-size character windows with a fixed overlap between consecutive windows.
No sentence boundaries, no semantic awareness, no clause-level parsing.

This is the ACTIVE chunking strategy used by the simple RAG pipeline
(app/core/simple_rag.py). RAPTOR (app/advanced/raptor/) uses this same
function to produce its leaf layer, then clusters and summarizes on top of it
— chunking strategy is a separate concern from tree-building.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[str]:
    """
    Split `text` into overlapping fixed-size chunks.

    Args:
        text: the raw document text to split.
        chunk_size: number of characters per chunk.
        overlap: number of characters shared between consecutive chunks.
                 Must be smaller than chunk_size or you'll loop forever.

    Returns:
        List of chunk strings, in order, each <= chunk_size characters
        (the last chunk may be shorter).

    Example:
        >>> chunk_text("A" * 1200, chunk_size=500, overlap=100)
        # produces chunks covering [0:500], [400:900], [800:1200]
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step

    return chunks
