"""
vectorstore.py — a minimal, file-persisted vector store.

No external vector DB (Chroma / FAISS / pgvector / Qdrant) is used here on
purpose — this keeps the microservice runnable with zero infra dependencies
and makes the retrieval mechanics fully transparent for learning purposes.

Storage:
  - chunk metadata (id, doc_id, text, position) -> JSON file
  - embedding matrix                             -> numpy .npy file
  - fitted embedder                               -> pickle (via embeddings.py)

Swap point: replace this class with a real vector DB client (Chroma, Qdrant,
pgvector...) behind the same `add_chunks` / `search` interface — nothing in
simple_rag.py or main.py needs to know the difference. This is exactly the
seam described in the accompanying ARCHITECTURE.md as where RAPTOR's tree
storage would plug in for the advanced/raptor module.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.embeddings import TfidfEmbedder


class SimpleVectorStore:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.storage_dir / "chunks.json"
        self.matrix_path = self.storage_dir / "embeddings.npy"
        self.embedder_path = self.storage_dir / "embedder.pkl"

        self.chunks: List[Dict[str, Any]] = []
        self.matrix: np.ndarray = np.zeros((0, 0))
        self.embedder = TfidfEmbedder()

        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if self.meta_path.exists():
            with open(self.meta_path, "r") as f:
                self.chunks = json.load(f)
        if self.matrix_path.exists():
            self.matrix = np.load(self.matrix_path)
        if self.embedder_path.exists():
            self.embedder = TfidfEmbedder.load(self.embedder_path)

    def _persist(self) -> None:
        with open(self.meta_path, "w") as f:
            json.dump(self.chunks, f, indent=2)
        np.save(self.matrix_path, self.matrix)
        self.embedder.save(self.embedder_path)

    # ---------- writes ----------

    def add_chunks(self, doc_id: str, texts: List[str]) -> int:
        """
        Add chunks for a document. Because TF-IDF's vocabulary depends on the
        full corpus, adding new text means re-fitting the vectorizer and
        re-embedding *all* chunks — not just the new ones. This is a known
        cost of the TF-IDF choice; see docs/ARCHITECTURE.md for the tradeoff
        and what a production embedding model would change here.
        """
        new_entries = [
            {"chunk_id": f"{doc_id}::{i}", "doc_id": doc_id, "text": t}
            for i, t in enumerate(texts)
        ]
        self.chunks.extend(new_entries)

        all_texts = [c["text"] for c in self.chunks]
        self.matrix = self.embedder.fit_transform(all_texts)

        self._persist()
        return len(new_entries)

    def replace_document(self, doc_id: str, texts: List[str]) -> int:
        """Remove any existing chunks for doc_id, then add the new ones (an update)."""
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        return self.add_chunks(doc_id, texts)

    # ---------- reads ----------

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if not self.chunks or self.matrix.shape[0] == 0:
            return []

        query_vec = self.embedder.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if sims[idx] <= 0:
                continue
            entry = dict(self.chunks[idx])
            entry["score"] = float(sims[idx])
            results.append(entry)
        return results

    def stats(self) -> Dict[str, Any]:
        doc_ids = sorted({c["doc_id"] for c in self.chunks})
        return {
            "total_chunks": len(self.chunks),
            "total_documents": len(doc_ids),
            "document_ids": doc_ids,
        }
