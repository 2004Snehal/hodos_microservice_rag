"""
retriever.py — Retrieval over one document's tree, with optional query
expansion and reranking. ProbabilisticMultiHop and eval/run_eval.py call
`query()` / `linked_leaves()` on this exactly as on the tree itself.

QUERY EXPANSION (query_expansion=True)
Questions are often worded in terms the document never uses: "quick ratio"
vs a balance sheet's "cash and cash equivalents ... total current
liabilities" (FinanceBench misses averaged 28% question-word overlap with
their gold page, several 0%). The LLM rewrites the question into the words
the source document would use; the dense query is the question plus that
rewrite. One LLM call per distinct query, cached.

TABLE CAP (table_cap=N)
Every leaf of a table starts with the same caption / section prefix, so a
query that matches the prefix can fill the whole result list with one
table (on FinanceBench 10-Ks, 28% of top-20 slots were such repeats). At
most N leaves per table are kept; the freed slots go to the next leaves of
a larger candidate pool.

RERANKING (reranker="<fastembed cross-encoder>")
The bi-encoder ranks the top `candidates` leaves; a cross-encoder reads
each (question, leaf) pair and reorders them. Leaves keep their tree
probability (multi-hop weighs evidence by it) and gain `rerank_score`.
The cross-encoder sees the original question, not the expansion.

LLM CACHE
Expansions and table summaries are cached in a JSON file keyed by model +
prompt + input, so re-ingesting the same document or re-running an eval
doesn't repeat LLM calls.
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.llm.ollama import LLMUnavailable, OllamaClient

EXPANSION_SYSTEM = (
    "You rewrite a question into a search query for the document that answers it. "
    "Write the words that document itself would use: names of the sections, statements, "
    "tables, line items, rules or terms that hold the answer. If the question names a "
    "derived measure (a ratio, margin, rate or limit), list the underlying items it is "
    "computed from. Do not answer the question and do not include numbers."
)
EXPANSION_SCHEMA = {
    "type": "object",
    "properties": {"search_query": {"type": "string", "description": "the rewritten search query, at most 40 words"}},
    "required": ["search_query"],
}


class LlmCache:
    """A small persistent JSON cache for LLM outputs."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self._lock = threading.Lock()
        self._data: Dict[str, str] = {}
        if path is not None and path.exists():
            self._data = json.loads(path.read_text())

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._data))


_cross_encoders: Dict[str, Any] = {}
_cross_lock = threading.Lock()


def _cross_encoder(model: str):
    with _cross_lock:
        if model not in _cross_encoders:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _cross_encoders[model] = TextCrossEncoder(model)
        return _cross_encoders[model]


class Retriever:
    def __init__(
        self,
        tree,
        llm: Optional[OllamaClient] = None,
        query_expansion: bool = False,
        reranker: Optional[str] = None,
        candidates: int = 50,
        cache: Optional[LlmCache] = None,
        table_cap: Optional[int] = None,
    ):
        self.tree = tree
        self.llm = llm
        self.query_expansion = query_expansion
        self.reranker = reranker
        self.candidates = candidates
        self.cache = cache or LlmCache(None)
        self.table_cap = table_cap
        self.last_expansion: Optional[str] = None

    def expand(self, text: str) -> str:
        """The question plus the LLM's rewrite in the document's words; the question alone without an LLM."""
        if not self.query_expansion or self.llm is None or not self.llm.available():
            return text
        key = LlmCache.key("expand", self.llm.model, EXPANSION_SYSTEM, text)
        rewrite = self.cache.get(key)
        if rewrite is None:
            try:
                result = self.llm.json(EXPANSION_SYSTEM, f"Question: {text}", EXPANSION_SCHEMA, max_tokens=120)
            except LLMUnavailable:
                return text
            rewrite = str(result.data.get("search_query", "")).strip()
            self.cache.put(key, rewrite)
        self.last_expansion = rewrite
        return f"{text} {rewrite}".strip()

    def query(self, text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        query = self.expand(text)
        widen = self.reranker or self.table_cap
        pool = self.tree.query(query, top_k=max(self.candidates, top_k * 4) if widen else top_k)
        if self.reranker:
            scores = list(_cross_encoder(self.reranker).rerank(text, [item["text"] for item in pool]))
            for item, score in zip(pool, scores):
                item["rerank_score"] = round(float(score), 4)
            pool = sorted(pool, key=lambda item: -item["rerank_score"])
        if self.table_cap:
            pool = self._cap_tables(pool)
        return pool[:top_k]

    def _cap_tables(self, pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        table_of = getattr(self.tree, "table_of", {})
        kept, seen = [], {}
        for item in pool:
            key = table_of.get(item["chunk_id"])
            if key is not None:
                if seen.get(key, 0) >= self.table_cap:
                    continue
                seen[key] = seen.get(key, 0) + 1
            kept.append(item)
        return kept

    def linked_leaves(self, node_id: str):
        return self.tree.linked_leaves(node_id)
