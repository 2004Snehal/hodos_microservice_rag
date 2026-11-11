"""
dense_embeddings.py — Neural sentence embeddings (BAAI/bge-small-en-v1.5 via
fastembed: local ONNX, 384-dim, no API calls).

Unlike app/core/embeddings.py's TF-IDF, this embedding space is fixed — it
doesn't need fitting per corpus — so vectors from different documents, the
semantic cache and the RAPTOR tree are all directly comparable, and
synonyms ("food" / "meal") land close together.

Vectors are L2-normalized, so a dot product is cosine similarity.
"""

import os
import threading
from typing import List

import numpy as np

EMBEDDING_MODEL = os.getenv("HODOS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384


class Embedder:
    _shared_models = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def _model(self):
        # One loaded model per process, shared by every Embedder instance.
        with self._lock:
            if self.model_name not in self._shared_models:
                from fastembed import TextEmbedding
                self._shared_models[self.model_name] = TextEmbedding(self.model_name)
            return self._shared_models[self.model_name]

    def embed_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        vectors = np.asarray(list(self._model().embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a search query against passages. bge models expect queries to
        carry a retrieval instruction prefix (fastembed's query_embed adds
        it); passages and question-vs-question comparisons (the semantic
        cache) use plain `embed`.
        """
        vector = np.asarray(next(iter(self._model().query_embed([text]))), dtype=np.float32)
        return vector / (np.linalg.norm(vector) or 1.0)



_counting_tokenizer = None


def count_tokens(text: str) -> int:
    """
    bge-small WordPiece tokens in `text`, without [CLS]/[SEP]. The model
    truncates at 512 tokens, so chunks are sized in these units. Uses a copy
    of the model's tokenizer with truncation off — the model's own one stops
    counting at 512.
    """
    global _counting_tokenizer
    if _counting_tokenizer is None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_str(Embedder()._model().model.tokenizer.to_str())
        tokenizer.no_truncation()
        tokenizer.no_padding()
        _counting_tokenizer = tokenizer
    return len(_counting_tokenizer.encode(text, add_special_tokens=False).ids)
