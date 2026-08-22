"""
embeddings.py — TF-IDF based "embedding" model.

Why TF-IDF instead of a neural embedding model (e.g. sentence-transformers)?
This microservice is meant to run standalone, offline, with no dependency on
downloading model weights at runtime. TF-IDF is deterministic, fast, has zero
external dependencies beyond scikit-learn, and is good enough to demonstrate
the RAG *architecture* (chunking -> embed -> store -> retrieve -> generate).

Swap point: replace `TfidfEmbedder` with a real embedding model (OpenAI,
Anthropic, sentence-transformers, etc.) by implementing the same two methods
(`fit_transform` / `transform`) — nothing else in the pipeline needs to change,
since `vectorstore.py` only depends on this interface, not on TF-IDF specifics.
"""

import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbedder:
    """
    Wraps a scikit-learn TfidfVectorizer with fit/transform/save/load,
    exposing the minimal interface the rest of the pipeline depends on.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
        )
        self._fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit the vectorizer on `texts` and return their embeddings."""
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        """Embed `texts` using the already-fitted vectorizer."""
        if not self._fitted:
            raise RuntimeError("Embedder has not been fit yet — call fit_transform first.")
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "fitted": self._fitted}, f)

    @classmethod
    def load(cls, path: Path) -> "TfidfEmbedder":
        instance = cls()
        if path.exists():
            with open(path, "rb") as f:
                state = pickle.load(f)
            instance.vectorizer = state["vectorizer"]
            instance._fitted = state["fitted"]
        return instance
