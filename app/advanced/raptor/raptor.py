"""
raptor.py — RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) module.

Implements:
  - Clustering: Gaussian Mixture Models (soft clustering) with BIC model selection.
  - Dimensionality reduction: UMAP before clustering.
  - Tree construction: Recursive hierarchical tree building.
  - Collapsed-tree query: Retrieval across all tree levels simultaneously.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

from app.core.embeddings import TfidfEmbedder
from app.advanced.raptor.extractive_summarizer import summarize_extractive


@dataclass
class RaptorNode:
    node_id: str
    level: int  # 0 = leaf
    text: str
    doc_id: str
    child_ids: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class RaptorTree:
    def __init__(
        self,
        max_levels: int = 3,
        min_cluster_size: int = 4,
        bic_candidate_ks: Optional[List[int]] = None,
        umap_n_neighbors: int = 10,
        umap_n_components: int = 10,
        use_umap: bool = True,
    ):
        """
        Args:
            max_levels: recursion cap (leaf level = 0; stop building upward
                        past this even if nodes could still be clustered further)
            min_cluster_size: below this many nodes at a level, stop
                              clustering — treat remaining nodes as their own
                              level and recurse no further
            bic_candidate_ks: candidate cluster counts to sweep with BIC at
                              each level (paper: BIC selects the K that
                              minimizes it, not a fixed K). Defaults to a
                              small sweep scaled to node count if not given.
            umap_n_neighbors: UMAP's n_neighbors parameter — the paper varies
                              this across a global pass and a local pass
                              (§2.3.2 of the tech doc); this module uses one
                              fixed value per clustering call as a
                              documented simplification.
            umap_n_components: target dimensionality after UMAP reduction.
            use_umap: if False, or if `umap-learn` isn't installed, clusters
                      directly on raw embeddings instead — this changes
                      clustering quality (the paper's own stated reason for
                      using UMAP is that distance metrics behave poorly in
                      raw high-dimensional space) but keeps the module
                      runnable without the extra dependency.
        """
        self.max_levels = max_levels
        self.min_cluster_size = min_cluster_size
        self.bic_candidate_ks = bic_candidate_ks
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.use_umap = use_umap and _UMAP_AVAILABLE

        self.nodes: Dict[str, RaptorNode] = {}
        self.embedder = TfidfEmbedder()
        self._built = False

    def build(self, doc_id: str, leaf_chunks: List[str]) -> None:
        """Build the full tree from leaf chunks up to the root(s)."""
        if not leaf_chunks:
            raise ValueError("leaf_chunks must be non-empty")

        # Fit the embedder on leaves once; all higher-level summary
        # embeddings reuse this same fitted vectorizer/vocabulary, per the
        # paper's requirement that leaf and summary nodes share one
        # embedding space (tech doc §2.2).
        leaf_embeddings = self.embedder.fit_transform(leaf_chunks)

        current_level_nodes: List[RaptorNode] = []
        for i, text in enumerate(leaf_chunks):
            node = RaptorNode(
                node_id=f"{doc_id}::L0::{i}",
                level=0,
                text=text,
                doc_id=doc_id,
                embedding=leaf_embeddings[i],
            )
            self.nodes[node.node_id] = node
            current_level_nodes.append(node)

        level = 0
        while (
            level < self.max_levels
            and len(current_level_nodes) >= self.min_cluster_size
        ):
            embeddings = np.stack([n.embedding for n in current_level_nodes])
            reduced = self._maybe_reduce_dimensionality(embeddings)

            k = self._select_k_via_bic(reduced)
            if k is None or k >= len(current_level_nodes):
                break  # BIC found no meaningful further grouping

            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(reduced)

            # Soft clustering: a node's cluster membership is whichever
            # component has highest posterior probability, but the model
            # itself is soft (mixture, not hard partition) — this module
            # takes the argmax per node for tree construction, since a tree
            # needs a definite parent-child structure, while still fitting
            # the GMM as a genuinely soft model per the paper's rationale.
            responsibilities = gmm.predict_proba(reduced)
            labels = np.argmax(responsibilities, axis=1)

            next_level_nodes: List[RaptorNode] = []
            for cluster_id in range(k):
                members = [
                    n for n, lbl in zip(current_level_nodes, labels) if lbl == cluster_id
                ]
                if not members:
                    continue

                summary_text = self._summarize_cluster([m.text for m in members])
                summary_node = RaptorNode(
                    node_id=f"{doc_id}::L{level + 1}::{cluster_id}",
                    level=level + 1,
                    text=summary_text,
                    doc_id=doc_id,
                    child_ids=[m.node_id for m in members],
                )
                # Re-embed the summary using the SAME fitted vocabulary as
                # the leaves, so leaf and summary nodes remain comparable.
                summary_node.embedding = self.embedder.transform([summary_text])[0]

                self.nodes[summary_node.node_id] = summary_node
                next_level_nodes.append(summary_node)

            current_level_nodes = next_level_nodes
            level += 1

        self._built = True

    def _maybe_reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Single-pass UMAP reduction (see module docstring for the gap versus
        the paper's two-pass global/local approach). Falls back to raw
        embeddings if UMAP isn't available/enabled, or if there are too few
        points for UMAP to run meaningfully.
        """
        n_samples = embeddings.shape[0]
        if not self.use_umap or n_samples <= self.umap_n_neighbors:
            return embeddings

        n_components = min(self.umap_n_components, n_samples - 2, embeddings.shape[1])
        if n_components < 2:
            return embeddings

        reducer = umap.UMAP(
            n_neighbors=min(self.umap_n_neighbors, n_samples - 1),
            n_components=n_components,
            random_state=42,
        )
        return reducer.fit_transform(embeddings)

    def _select_k_via_bic(self, embeddings: np.ndarray) -> Optional[int]:
        """
        Sweep candidate cluster counts and return the K that minimizes BIC,
        per the paper's model-selection method (tech doc §2.3.4). Returns
        None if no candidate K is valid (e.g. too few points to cluster).
        """
        n_samples = embeddings.shape[0]
        if n_samples < 2:
            return None

        candidates = self.bic_candidate_ks or list(
            range(1, min(8, n_samples) + 1)
        )
        candidates = [k for k in candidates if 1 <= k < n_samples]
        if not candidates:
            return None

        best_k, best_bic = None, float("inf")
        for k in candidates:
            try:
                gmm = GaussianMixture(n_components=k, random_state=42)
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                if bic < best_bic:
                    best_bic, best_k = bic, k
            except ValueError:
                continue  # e.g. k too large relative to sample/feature count

        return best_k

    def query(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Collapsed-tree retrieval: search across every node at every level
        simultaneously and return the top_k by cosine similarity, whatever
        level they're at. This is the strategy the paper's own controlled
        comparison found superior to layer-by-layer tree traversal (tech
        doc §2.6.2) — implemented here in the EMBEDDING space used for
        storage/retrieval, not the UMAP-reduced space used only internally
        for clustering.
        """
        if not self._built:
            raise RuntimeError("Tree has not been built yet — call .build() first.")

        all_nodes = list(self.nodes.values())
        matrix = np.stack([n.embedding for n in all_nodes])
        query_vec = self.embedder.transform([text])

        sims = cosine_similarity(query_vec, matrix)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]

        return [
            {
                "node_id": all_nodes[i].node_id,
                "level": all_nodes[i].level,
                "text": all_nodes[i].text,
                "doc_id": all_nodes[i].doc_id,
                "score": float(sims[i]),
                "is_summary": all_nodes[i].level > 0,
                "child_ids": all_nodes[i].child_ids,
            }
            for i in top_idx
            if sims[i] > 0
        ]

    def get_source_leaves(self, node_id: str) -> List[RaptorNode]:
        """
        Leaf pull-through helper: given any node (leaf or summary), return
        every leaf (level 0) node underneath it. Mirrors the "always fetch
        source leaves alongside a matched summary" principle (tech doc
        §2.4.5, §2.6.4 — non-leaf nodes are a substantial share of retrieved
        evidence in the paper's own measurements, so this isn't a rare-case
        safeguard) — left as a helper for you to call from your own wiring,
        not called automatically here.
        """
        node = self.nodes.get(node_id)
        if node is None:
            return []
        if node.level == 0:
            return [node]

        leaves = []
        for child_id in node.child_ids:
            leaves.extend(self.get_source_leaves(child_id))
        return leaves

    # ---------- Cluster summarization ----------

    @staticmethod
    def _summarize_cluster(member_texts: List[str]) -> str:
        """
        Centroid-based extractive summarization across cluster member texts.
        """
        return summarize_extractive(member_texts)
