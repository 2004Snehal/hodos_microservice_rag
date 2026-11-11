"""
multihop.py — Probabilistic multi-hop retrieval over a ProbabilisticRaptorTree.

Each hop:
  1. Retrieve: the tree returns a distribution P(leaf | query_t) over leaves,
     plus leaves linked to them ("Table 14.3" mention <-> table), which
     inherit the probability of the leaf that linked them.
  2. Relevance: for each NEW leaf, the LLM's token probability that the
     passage is relevant to the ORIGINAL question:  r(leaf) = P(Yes | leaf, q).
     A leaf's evidence weight combines both signals and accumulates across
     hops with noisy-OR, so a passage found by two different queries
     becomes more credible than one found once:
         e(leaf) = 1 - Π_hops (1 - P(leaf | query_t) ^ α · r(leaf))
     (α < 1 flattens the retrieval distribution, which is sharp by design.)
  3. Sufficiency: P(sufficient) = P(Yes | evidence set, q), where the
     evidence set is the leaves with r ≥ relevance_floor.
  4. Stop when P(sufficient) ≥ sufficiency_threshold, when the hop budget
     is spent, or when the LLM's follow-up query adds nothing new.
     Otherwise the LLM states what's missing and writes a follow-up query.

Without Ollama the loop degrades to a single retrieval hop with relevance
taken from the retrieval distribution alone and P(sufficient) unknown (None).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.advanced.probabilistic.tree import ProbabilisticRaptorTree
from app.llm.ollama import LLMUnavailable, OllamaClient

GENERAL_RULES_NOTE = (
    "A policy rule that does not name a grade, location or situation applies to "
    "everyone, everywhere — it still answers questions about a specific grade or city."
)
RELEVANCE_SYSTEM = (
    "You check whether a passage from a company policy contains information that "
    "helps answer a question about the same expense topic — the rule itself, an "
    "exception, a condition, or a cross-reference that applies to it. "
    + GENERAL_RULES_NOTE
)
SUFFICIENCY_SYSTEM = (
    "You judge whether the evidence contains enough to answer the question, "
    "including any exceptions or conditions that apply to the specific grade, "
    "location or situation in the question. " + GENERAL_RULES_NOTE
)
FOLLOW_UP_SYSTEM = (
    "You plan the next search over a company policy. Given a question and the "
    "evidence found so far, state what information is still missing and write one "
    "short search query that would find it. Do not repeat earlier queries."
)
FOLLOW_UP_SCHEMA = {
    "type": "object",
    "properties": {
        "missing": {"type": "string", "description": "what is still missing, in one sentence"},
        "follow_up_query": {"type": "string", "description": "a short search query, or empty if nothing is missing"},
    },
    "required": ["missing", "follow_up_query"],
}


@dataclass
class Hop:
    number: int
    query: str
    retrieved: List[Dict[str, Any]]
    p_sufficient: Optional[float]
    missing: str = ""


@dataclass
class MultiHopResult:
    question: str
    hops: List[Hop] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # sorted by weight, desc
    p_sufficient: Optional[float] = None
    stopped_reason: str = ""
    llm_used: bool = False


class ProbabilisticMultiHop:
    def __init__(
        self,
        tree: ProbabilisticRaptorTree,
        llm: Optional[OllamaClient] = None,
        max_hops: int = 3,
        top_k_per_hop: int = 4,
        sufficiency_threshold: float = 0.8,
        relevance_floor: float = 0.3,
        retrieval_alpha: float = 0.25,
    ):
        self.tree = tree
        self.llm = llm if llm is not None and llm.available() else None
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop
        self.sufficiency_threshold = sufficiency_threshold
        self.relevance_floor = relevance_floor
        self.retrieval_alpha = retrieval_alpha

    def run(self, question: str) -> MultiHopResult:
        result = MultiHopResult(question=question, llm_used=self.llm is not None)
        evidence: Dict[str, Dict[str, Any]] = {}
        queries = [question]

        for hop_number in range(1, self.max_hops + 1):
            query = queries[-1]
            retrieved = self._with_links(self.tree.query(query, top_k=self.top_k_per_hop))

            for item in retrieved:
                entry = evidence.get(item["chunk_id"])
                if entry is None:
                    entry = {
                        "chunk_id": item["chunk_id"],
                        "text": item["text"],
                        "relevance": self._relevance(question, item),
                        "weight": 0.0,
                        "found_in_hops": [],
                    }
                    evidence[item["chunk_id"]] = entry
                support = (item["probability"] ** self.retrieval_alpha) * entry["relevance"]
                entry["weight"] = 1 - (1 - entry["weight"]) * (1 - support)  # noisy-OR across hops
                entry["found_in_hops"].append(hop_number)

            relevant = self._relevant(evidence)
            hop = Hop(hop_number, query, retrieved, self._sufficiency(question, relevant))
            result.hops.append(hop)

            if hop.p_sufficient is None:
                result.stopped_reason = "no_llm_single_hop"
                break
            if hop.p_sufficient >= self.sufficiency_threshold:
                result.stopped_reason = "sufficient"
                break
            if hop_number == self.max_hops:
                result.stopped_reason = "max_hops_reached"
                break

            missing, follow_up = self._follow_up(question, relevant, queries)
            hop.missing = missing
            if not follow_up or follow_up.strip().lower() in {q.strip().lower() for q in queries}:
                result.stopped_reason = "no_new_follow_up_query"
                break
            queries.append(follow_up)

        result.evidence = sorted(evidence.values(), key=lambda e: -e["weight"])
        for entry in result.evidence:
            entry["relevance"] = round(entry["relevance"], 4)
            entry["weight"] = round(entry["weight"], 4)
        result.p_sufficient = result.hops[-1].p_sufficient if result.hops else None
        return result

    def _with_links(self, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add leaves linked to a retrieved one (a sentence citing "Table 14.3" <->
        the table), with the retrieved leaf's probability. The LLM relevance
        check then decides whether they count as evidence.
        """
        seen = {item["chunk_id"] for item in retrieved}
        linked = []
        for item in retrieved:
            for node in self.tree.linked_leaves(item["chunk_id"]):
                if node.node_id not in seen:
                    seen.add(node.node_id)
                    linked.append({"chunk_id": node.node_id, "doc_id": node.doc_id, "text": node.text,
                                   "probability": item["probability"], "via_link": item["chunk_id"]})
        return retrieved + linked

    # ---------- LLM judgments ----------

    def _relevance(self, question: str, item: Dict[str, Any]) -> float:
        if self.llm is None:
            return 1.0  # retrieval probability alone carries the weight
        try:
            return self.llm.probability_yes(
                RELEVANCE_SYSTEM, f"Passage:\n{item['text']}\n\nQuestion: {question}"
            )
        except LLMUnavailable:
            return 1.0

    def _sufficiency(self, question: str, relevant: List[Dict[str, Any]]) -> Optional[float]:
        if self.llm is None:
            return None
        if not relevant:
            return 0.0
        try:
            return self.llm.probability_yes(
                SUFFICIENCY_SYSTEM, f"Evidence:\n{_format(relevant)}\n\nQuestion: {question}"
            )
        except LLMUnavailable:
            return None

    def _follow_up(self, question: str, relevant: List[Dict[str, Any]], queries: List[str]):
        try:
            result = self.llm.json(
                FOLLOW_UP_SYSTEM,
                f"Question: {question}\n\nEvidence so far:\n{_format(relevant) or '(none)'}\n\n"
                f"Earlier queries: {queries}",
                FOLLOW_UP_SCHEMA,
                max_tokens=120,
            )
        except LLMUnavailable:
            return "", ""
        return str(result.data.get("missing", "")), str(result.data.get("follow_up_query", "")).strip()

    def _relevant(self, evidence: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            (e for e in evidence.values() if e["relevance"] >= self.relevance_floor),
            key=lambda e: -e["weight"],
        )


def _format(evidence: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{e['chunk_id']}] {e['text']}" for e in evidence)
