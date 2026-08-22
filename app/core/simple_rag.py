"""
simple_rag.py — RAG pipeline.

Provides:
  ingest:  text -> overlap chunks -> embed -> store
  ask:     query -> embed -> top-k cosine similarity -> extractive answer
"""

from pathlib import Path
from typing import List, Dict, Any

from app.core.chunking import chunk_text
from app.core.vectorstore import SimpleVectorStore

VECTOR_STORE_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_store"
POLICY_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4


class SimpleRagPipeline:
    def __init__(self):
        self.store = SimpleVectorStore(VECTOR_STORE_DIR)
        POLICY_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- update policy ----------

    def update_policy(self, doc_id: str, text: str) -> Dict[str, Any]:
        """
        Ingest (or re-ingest) a policy document:
          1. persist the raw text
          2. chunk it with fixed-size overlap chunking
          3. store embeddings across vectorstore
        """
        raw_path = POLICY_RAW_DIR / f"{doc_id}.txt"
        raw_path.write_text(text)

        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        num_chunks = self.store.replace_document(doc_id, chunks)

        return {
            "doc_id": doc_id,
            "chunks_created": num_chunks,
            "chunk_size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "store_stats": self.store.stats(),
        }

    # ---------- ask ----------

    def ask(self, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
        """
        Answer a question using document retrieval.
        """
        results = self.store.search(question, top_k=top_k)

        if not results:
            return {
                "question": question,
                "answer": "No relevant policy content found. Has a policy been uploaded yet?",
                "citations": [],
            }

        answer = self._generate_answer(question, results)

        return {
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "score": round(r["score"], 4),
                    "text": r["text"],
                }
                for r in results
            ],
        }

    @staticmethod
    def _generate_answer(question: str, results: List[Dict[str, Any]]) -> str:
        """
        Generates an extractive answer from retrieved passages.
        """
        top = results[0]
        return (
            f"Most relevant passage (score={top['score']:.3f}, doc={top['doc_id']}):\n"
            f"{top['text']}"
        )

