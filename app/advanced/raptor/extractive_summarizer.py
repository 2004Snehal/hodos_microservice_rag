"""
extractive_summarizer.py — Centroid-based extractive summarization.

WHAT THIS DOES
Classic centroid-based sentence extraction:
  1. Split the cluster's combined text into sentences.
  2. Compute a local TF-IDF vector per sentence.
  3. Compute the centroid (mean vector) of all sentences in the cluster.
  4. Score each sentence by cosine similarity to that centroid.
  5. Select the top-scoring sentences, re-ordered back into their ORIGINAL sequence.
"""

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Requires a period/!/? followed by whitespace AND an uppercase letter (or
# end of string) before treating it as a sentence boundary — this avoids
# splitting on abbreviations like "Rs." where the next token ("1500") isn't
# capitalized, unlike the naive ". " split that caused a real bug elsewhere
# in this project (see app/core/audit_pipeline.py's docstring history).
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z]|$)")

DEFAULT_COMPRESSION_RATIO = 0.4  # keep ~40% of sentences, min 1
MIN_SENTENCES_FOR_SCORING = 3  # below this, just return the text as-is


def _split_sentences(text: str) -> List[str]:
    sentences = _SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def summarize_extractive(
    member_texts: List[str],
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
) -> str:
    """
    Args:
        member_texts: the text of every leaf/summary node in this cluster.
        compression_ratio: fraction of sentences to keep, rounded up,
                            minimum 1. The paper's own measured compression
                            ratio is ~0.28 by TOKEN count for abstractive
                            summaries (tech doc §2.4.3) — this is a
                            different unit (sentence count, extractive), so
                            the two numbers aren't directly comparable; this
                            default is a reasonable starting point, not a
                            reproduction of the paper's figure.

    Returns:
        A summary built from the highest-centroid-similarity sentences,
        restored to their original order.
    """
    combined_text = " ".join(t.strip() for t in member_texts if t.strip())
    sentences = _split_sentences(combined_text)

    if len(sentences) < MIN_SENTENCES_FOR_SCORING:
        # Too few sentences for centroid scoring to be meaningful — just
        # return the combined text directly rather than force a
        # near-total "summary" that keeps almost everything anyway.
        return combined_text

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        sentence_vectors = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        # e.g. every sentence is pure stopwords/empty after vectorization —
        # fall back to the raw combined text rather than crash.
        return combined_text

    centroid = sentence_vectors.mean(axis=0, keepdims=True)
    scores = cosine_similarity(centroid, sentence_vectors)[0]

    n_keep = max(1, int(np.ceil(len(sentences) * compression_ratio)))
    top_indices = sorted(np.argsort(scores)[::-1][:n_keep])  # sorted -> restores original order

    return " ".join(sentences[i] for i in top_indices)
