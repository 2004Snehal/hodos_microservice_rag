"""
pipeline.py — Integrated pipeline connecting caching + RAPTOR + multi-hop.

WHAT THIS WIRES TOGETHER
  1. RaptorTree (app.advanced.raptor.raptor) — builds the tree for a
     policy document, using GMM+BIC clustering and centroid-based
     extractive summarization.
  2. RaptorSearchAdapter — translates RaptorTree query results to the
     interface expected by MultiHopRetriever.
  3. MultiHopRetriever (app.advanced.multihop.multihop) — runs its
     iterative hop loop driven by TF-IDF term-coverage scoring
     and Rocchio-style query expansion.
  4. TupleCache (app.advanced.caching.cache) — checked before running
     retrieval, written after a judgment is resolved.
  5. rule_extraction.resolve_rule (app.core.rule_extraction) — resolves
     the accumulated multi-hop evidence into a structured judgment.
"""

from pathlib import Path
from typing import List, Dict, Any

from app.core.chunking import chunk_text
from app.core.rule_extraction import resolve_rule
from app.advanced.raptor.raptor import RaptorTree
from app.advanced.multihop.multihop import MultiHopRetriever
from app.advanced.caching.cache import TupleCache


class RaptorSearchAdapter:
    """
    Adapts RaptorTree's `.query(text, top_k)` to the `.search(query, top_k)`
    interface expected by MultiHopRetriever.
    """

    def __init__(self, tree: RaptorTree):
        self.tree = tree

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raw_results = self.tree.query(query, top_k=top_k)
        return [
            {
                "chunk_id": r["node_id"],
                "doc_id": r["doc_id"],
                "text": r["text"],
                "score": r["score"],
                "level": r["level"],
                "is_summary": r["is_summary"],
            }
            for r in raw_results
        ]


class AdvancedRagPipeline:
    """
    Combines RAPTOR-backed multi-hop retrieval with a tuple-keyed cache.
    """

    def __init__(self, cache_dir: Path, max_hops: int = 3, top_k_per_hop: int = 3):
        self.cache = TupleCache(cache_dir)
        self.trees: Dict[str, RaptorTree] = {}
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop

    def build_tree_for_document(self, doc_id: str, text: str) -> Dict[str, Any]:
        """Chunk + build a RAPTOR tree for one policy document."""
        chunks = chunk_text(text, chunk_size=150, overlap=20)
        tree = RaptorTree(max_levels=2, min_cluster_size=3)
        tree.build(doc_id, chunks)
        self.trees[doc_id] = tree
        return {"doc_id": doc_id, "leaf_chunks": len(chunks), "total_nodes": len(tree.nodes)}

    def ask(
        self,
        question: str,
        doc_id: str,
        category: str,
        grade: str,
        location: str,
        version: str,
        category_keywords: List[str],
    ) -> Dict[str, Any]:
        """
        Cache-first, RAPTOR+multi-hop-backed ask. Returns a resolved
        judgment, either served from cache or freshly computed.
        """
        key = self.cache.make_key(
            category=category, grade=grade, location=location, version=version
        )
        cached = self.cache.get(key)
        if cached is not None:
            return {"source": "cache", **cached["judgment"]}

        if doc_id not in self.trees:
            raise ValueError(
                f"No RAPTOR tree built for doc_id='{doc_id}' — call build_tree_for_document first."
            )

        adapter = RaptorSearchAdapter(self.trees[doc_id])
        hop_retriever = MultiHopRetriever(
            adapter, max_hops=self.max_hops, top_k_per_hop=self.top_k_per_hop
        )
        hop_result = hop_retriever.run(question)

        judgment = self._resolve_judgment(hop_result.evidence, category, category_keywords)
        judgment["hops_taken"] = len(hop_result.hops)
        judgment["stopped_reason"] = hop_result.stopped_reason
        judgment["final_coverage"] = hop_result.hops[-1].coverage if hop_result.hops else 0.0

        self.cache.set(key, judgment, version=version)
        return {"source": "fresh", **judgment}

    @staticmethod
    def _resolve_judgment(
        evidence: List[Dict[str, Any]], category: str, category_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Resolves accumulated multi-hop evidence into a structured judgment.
        """
        combined_text = " ".join(e["text"] for e in evidence)
        resolved = resolve_rule(combined_text, category_keywords)

        return {
            "category": category,
            "rule_type": resolved.rule_type,
            "limit_amount": resolved.limit_amount,
            "matched_window": resolved.matched_window,
            "cited_chunk_ids": [e["chunk_id"] for e in evidence],
            "confidence": 0.6 if resolved.rule_type != "unresolved" else 0.0,
        }

