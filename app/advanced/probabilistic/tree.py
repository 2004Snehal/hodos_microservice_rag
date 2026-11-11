"""
tree.py — Probabilistic RAPTOR tree.

DIFFERENCES FROM app/advanced/raptor/raptor.py
  1. Dense embeddings (bge-small) instead of per-document TF-IDF, so
     clustering and retrieval work on meaning, not word overlap.
  2. Soft clustering is kept soft. A node joins EVERY cluster whose GMM
     posterior is ≥ `membership_threshold` (the RAPTOR paper's own rule), and
     each parent→child edge stores that posterior as a weight. The old tree
     took the argmax, so every node had exactly one parent.
  3. Small documents still get a tree. UMAP needs more points than its
     n_neighbors, so below that we reduce with PCA instead, and fit a
     diagonal-covariance GMM (far fewer parameters) so BIC can actually
     prefer K > 1 on a handful of points.
  4. Summaries are written by the local LLM (qwen3:4b via Ollama), told to
     keep every amount, grade, location and condition verbatim. Falls back
     to extractive summaries if Ollama isn't running.
  5. Retrieval returns a probability distribution, not just scores — see
     `query()`.

RETRIEVAL AS A DISTRIBUTION OVER LEAVES
  P(node | q) = softmax(cos(q, node) / temperature) over ALL nodes (collapsed tree)
  Each summary node then passes its probability down to its children, level
  by level, so all probability mass ends on leaves. A parent splits its mass
  by membership weight AND by how well each child matches the query:
      P(child | parent, q) ∝ w(parent→child) · exp(cos(q, child) / temperature)
      P(leaf | q) = P_direct(leaf | q) + Σ_parents P(parent | q) · P(leaf | parent, q)
  The leaf probabilities sum to 1. Splitting by weight alone would hand a
  broad root summary's mass equally to every cluster, so a one-leaf cluster
  would get as much as a ten-leaf one regardless of relevance. A leaf that's only weakly similar to the
  query but sits under a strongly matching summary still gets mass — that's
  RAPTOR's "summary matched, pull the source leaves" principle, done
  probabilistically rather than as a fixed rule.
"""

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from app.advanced.raptor.extractive_summarizer import summarize_extractive
from app.core.dense_embeddings import Embedder
from app.llm.ollama import LLMUnavailable, OllamaClient

SUMMARY_SYSTEM_PROMPT = (
    "You summarize sections of a company policy for a retrieval index. "
    "Write one dense paragraph of at most 120 words. Keep every amount, currency, "
    "limit, grade, location, time period, exception and approval requirement exactly "
    "as written. Do not add information that is not in the text."
)
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string", "description": "the summary paragraph"}},
    "required": ["summary"],
}


@dataclass
class TreeNode:
    node_id: str
    level: int  # 0 = leaf
    text: str
    doc_id: str
    embedding: np.ndarray
    children: Dict[str, float] = field(default_factory=dict)  # child_id -> membership weight
    parents: Dict[str, float] = field(default_factory=dict)  # parent_id -> membership weight


class ProbabilisticRaptorTree:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        llm: Optional[OllamaClient] = None,
        max_levels: int = 3,
        min_cluster_size: int = 3,
        max_clusters: int = 8,
        membership_threshold: float = 0.25,
        umap_n_neighbors: int = 10,
        reduced_dims: int = 8,
        temperature: float = 0.05,
        cluster_size: Optional[int] = None,
    ):
        self.embedder = embedder or Embedder()
        self.llm = llm
        self.max_levels = max_levels
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.membership_threshold = membership_threshold
        self.umap_n_neighbors = umap_n_neighbors
        self.reduced_dims = reduced_dims
        self.temperature = temperature
        # None: at most max_clusters per level (8). A number: allow about
        # n / cluster_size clusters, so a 2,000-leaf 10-K isn't squeezed into 8
        # clusters of ~250 leaves whose summaries are too broad to route by
        # (and too long for the summarizer's context window).
        self.cluster_size = cluster_size

        self.nodes: Dict[str, TreeNode] = {}
        self.build_log: List[Dict[str, Any]] = []
        self.summary_counts = {"llm": 0, "extractive": 0, "rejected_number_mismatch": 0}
        self.doc_hash: Optional[str] = None
        # leaf id -> leaf ids it's linked to: a sentence citing "Table 14.3" <-> that
        # table's leaves. Set by the pipeline after build; used by multi-hop.
        self.links: Dict[str, List[str]] = {}
        # leaf id -> table key, for leaves that are part of a table (set by the pipeline)
        self.table_of: Dict[str, int] = {}

    # ------------------------------------------------------------------ build

    def build(self, doc_id: str, leaf_chunks: List[str]) -> None:
        if not leaf_chunks:
            raise ValueError("leaf_chunks must be non-empty")

        leaf_vectors = self.embedder.embed_many(leaf_chunks)
        level_nodes = []
        for i, (text, vector) in enumerate(zip(leaf_chunks, leaf_vectors)):
            node = TreeNode(f"{doc_id}::L0::{i}", 0, text, doc_id, vector)
            self.nodes[node.node_id] = node
            level_nodes.append(node)

        level = 0
        while level < self.max_levels and len(level_nodes) >= 2:
            if len(level_nodes) < self.min_cluster_size:
                memberships = np.ones((len(level_nodes), 1))  # too few to cluster: one root
                method = "root"
            else:
                reduced, method = self._reduce(np.stack([n.embedding for n in level_nodes]))
                memberships = self._soft_cluster(reduced)

            k = memberships.shape[1]
            self.build_log.append(
                {"level": level, "nodes": len(level_nodes), "reduction": method, "clusters": k}
            )

            parents = []
            for cluster in range(k):
                members = [
                    (node, float(memberships[i, cluster]))
                    for i, node in enumerate(level_nodes)
                    if memberships[i, cluster] > 0
                ]
                if not members:
                    continue
                summary = self._summarize([node.text for node, _ in members])
                parent = TreeNode(
                    node_id=f"{doc_id}::L{level + 1}::{cluster}",
                    level=level + 1,
                    text=summary,
                    doc_id=doc_id,
                    embedding=self.embedder.embed(summary),
                )
                for node, weight in members:
                    parent.children[node.node_id] = weight
                    node.parents[parent.node_id] = weight
                self.nodes[parent.node_id] = parent
                parents.append(parent)

            level += 1
            if len(parents) <= 1:
                break  # reached a single root
            level_nodes = parents

    def _reduce(self, vectors: np.ndarray):
        n = vectors.shape[0]
        dims = min(self.reduced_dims, n - 2)
        if dims < 1:
            return vectors, "none"
        if n > self.umap_n_neighbors + 1:
            try:
                import umap

                reducer = umap.UMAP(
                    n_neighbors=self.umap_n_neighbors, n_components=dims,
                    metric="cosine", random_state=42,
                )
                return reducer.fit_transform(vectors), "umap"
            except ImportError:
                pass
        return PCA(n_components=dims, random_state=42).fit_transform(vectors), "pca"

    def _soft_cluster(self, points: np.ndarray) -> np.ndarray:
        """
        Pick K by BIC, then return an (n_nodes, K) membership matrix: the GMM
        posterior where it's ≥ membership_threshold, else 0. Every node keeps
        at least its most likely cluster.
        """
        n = points.shape[0]
        covariance = "diag" if n < 20 else "full"
        best, best_bic = None, float("inf")
        k_max = min(max(self.max_clusters, n // self.cluster_size if self.cluster_size else 0), n // 2)
        if k_max <= 12:
            k_values = range(1, k_max + 1)
        else:  # geometric grid keeps the BIC search to ~12 fits for large K
            k_values = sorted({int(round(k)) for k in np.geomspace(1, k_max, 12)})
        for k in k_values:
            try:
                gmm = GaussianMixture(
                    n_components=k, covariance_type=covariance, random_state=42, reg_covar=1e-4
                ).fit(points)
            except ValueError:
                continue
            bic = gmm.bic(points)
            if bic < best_bic:
                best, best_bic = gmm, bic
        if best is None or best.n_components == 1:
            return np.ones((n, 1))

        posterior = best.predict_proba(points)
        memberships = np.where(posterior >= self.membership_threshold, posterior, 0.0)
        top = posterior.argmax(axis=1)
        memberships[np.arange(n), top] = posterior[np.arange(n), top]
        return memberships[:, memberships.sum(axis=0) > 0]  # drop empty clusters

    def _summarize(self, texts: List[str]) -> str:
        """
        LLM summary, rejected if it contains any number that isn't in the
        source text (qwen3:4b was seen turning "Rs. 6000" into "Rs. 60.00").
        Summaries only steer retrieval — rules are always read from leaves —
        but a corrupted number in a summary would still pull the wrong
        leaves, so it falls back to extractive in that case.
        """
        if self.llm is not None and self.llm.available():
            try:
                result = self.llm.json(SUMMARY_SYSTEM_PROMPT, "\n\n".join(texts), SUMMARY_SCHEMA, max_tokens=300)
                summary = str(result.data.get("summary", "")).strip()
                if summary and _numbers(summary) <= _numbers(" ".join(texts)):
                    self.summary_counts["llm"] += 1
                    return summary
                if summary:
                    self.summary_counts["rejected_number_mismatch"] += 1
            except LLMUnavailable:
                pass
        self.summary_counts["extractive"] += 1
        return summarize_extractive(texts)

    # ------------------------------------------------------------------ query

    def query(self, text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Return the top_k leaves by P(leaf | query), each with its probability
        and the summary nodes that contributed mass to it.
        """
        if not self.nodes:
            raise RuntimeError("Tree has not been built yet — call .build() first.")

        ids = list(self.nodes)
        sims = np.stack([self.nodes[i].embedding for i in ids]) @ self.embedder.embed_query(text)
        logits = sims / self.temperature
        likelihood = np.exp(logits - logits.max())
        probs = likelihood / likelihood.sum()
        child_likelihood = dict(zip(ids, likelihood.tolist()))

        direct = dict(zip(ids, probs.tolist()))
        mass = dict(direct)
        via: Dict[str, Dict[str, float]] = {i: {} for i in ids}
        # Push mass down from the top level to the leaves.
        for node in sorted(self.nodes.values(), key=lambda n: -n.level):
            if node.level == 0 or not node.children:
                continue
            scores = {c: w * child_likelihood[c] for c, w in node.children.items()}
            total = sum(scores.values()) or 1.0
            for child_id, score in scores.items():
                fraction = score / total
                mass[child_id] += mass[node.node_id] * fraction
                # Attribution: credit each summary only with its OWN direct
                # probability, so direct + Σ via == the leaf's final mass.
                via[child_id][node.node_id] = via[child_id].get(node.node_id, 0.0) + direct[node.node_id] * fraction
                for ancestor, amount in via[node.node_id].items():
                    via[child_id][ancestor] = via[child_id].get(ancestor, 0.0) + amount * fraction
            mass[node.node_id] = 0.0

        leaves = sorted(
            (n for n in self.nodes.values() if n.level == 0), key=lambda n: -mass[n.node_id]
        )[:top_k]
        return [
            {
                "chunk_id": leaf.node_id,
                "doc_id": leaf.doc_id,
                "text": leaf.text,
                "probability": round(mass[leaf.node_id], 4),
                "direct_probability": round(direct[leaf.node_id], 4),
                "similarity": round(float(sims[ids.index(leaf.node_id)]), 4),
                "via_summaries": {k: round(v, 4) for k, v in via[leaf.node_id].items() if v >= 1e-4},
            }
            for leaf in leaves
        ]

    # ------------------------------------------------------------------ info / persistence

    def structure(self) -> Dict[str, Any]:
        levels: Dict[int, int] = {}
        multi_parent = 0
        for node in self.nodes.values():
            levels[node.level] = levels.get(node.level, 0) + 1
            multi_parent += len(node.parents) > 1
        return {
            "nodes_per_level": dict(sorted(levels.items())),
            "nodes_with_multiple_parents": multi_parent,
            "summaries": self.summary_counts,
            "build_log": self.build_log,
        }

    def linked_leaves(self, node_id: str) -> List[TreeNode]:
        return [self.nodes[i] for i in self.links.get(node_id, []) if i in self.nodes]

    def leaves(self) -> List[TreeNode]:
        return [n for n in self.nodes.values() if n.level == 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"nodes": self.nodes, "build_log": self.build_log,
                         "summaries": self.summary_counts, "doc_hash": self.doc_hash,
                         "links": self.links, "table_of": self.table_of}, f)

    def load(self, path: Path) -> "ProbabilisticRaptorTree":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.nodes, self.build_log, self.summary_counts = state["nodes"], state["build_log"], state["summaries"]
        self.doc_hash = state.get("doc_hash")
        self.links = state.get("links", {})
        self.table_of = state.get("table_of", {})
        return self


def _numbers(text: str) -> set:
    """Every number in `text`, normalized ("1,500" / "1500.00" -> "1500")."""
    found = set()
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        value = float(raw.replace(",", ""))
        found.add(f"{value:g}")
    return found
