"""
multihop.py — STANDALONE multi-hop retrieval module.

STATUS: NOT INTEGRATED into the served API. Nothing in app/main.py imports
this file directly (app/advanced/integrated/pipeline.py does, as an
experimental integration — see README_RAPTOR_MULTIHOP.md).

WHAT IT DOES
Implements an iterative retrieval loop (IRCoT / ReAct-flavored):
  1. Retrieve against the current query.
  2. Check whether the retrieved evidence sufficiently covers the
     question's own terms (real TF-IDF term-coverage scoring — see below).
  3. If not, reformulate the query via Rocchio-style pseudo-relevance
     feedback: add the evidence's own most distinctive terms that the
     original query didn't contain.
  4. Stop at max_hops or when coverage crosses the threshold.

This depends only on a `retriever` object exposing a `.search(query, top_k)`
method returning dicts with a `chunk_id` and `text` key — e.g.
app.core.vectorstore.SimpleVectorStore satisfies this directly, and
app.advanced.raptor.raptor.RaptorTree satisfies it via
app.advanced.integrated.pipeline.RaptorSearchAdapter.

WHAT'S REAL VS. WHAT'S STILL A CEILING SHORT OF LLM REASONING
REAL: `_reason_over_evidence` now does genuine information-retrieval work —
term-coverage scoring (does the accumulated evidence contain the question's
own key terms) and Rocchio-style query expansion (a real pseudo-relevance-
feedback technique: assume the retrieved evidence is relevant, and expand
the next query with its most distinctive terms). This is a real upgrade
from the previous version, which stopped after a fixed evidence count
regardless of whether that evidence actually related to the question at
all.

STILL A CEILING, NOT REASONING: term coverage measures LEXICAL overlap, not
semantic sufficiency — evidence can contain all the right words while still
not actually answering the question (e.g. mentioning "meal" and "Mumbai"
and "L3" without ever stating the cap), and this method would report high
coverage anyway. It also can't recognize when a question has been fully
answered by ONE piece of evidence that uses different words entirely
(low lexical overlap, high semantic relevance) — TF-IDF coverage is blind
to synonymy. A real LLM reasoning step (see docs/HODOS_RAPTOR_MULTIHOP_TECHNICAL_DOCS.md
§3.4) is the actual fix for both of these; this module's upgrade closes the
gap between "no real signal at all" and "a real, principled, but
lexical-only signal" — it does not close the gap all the way to genuine
reasoning.

HOW YOU'D WIRE IT IN (left for you to do, beyond what
app/advanced/integrated/pipeline.py already demonstrates)
  from app.advanced.multihop.multihop import MultiHopRetriever
  hop_retriever = MultiHopRetriever(pipeline.store)
  result = hop_retriever.run(question)
  # then feed result["evidence"] into your generation step instead of
  # pipeline.store.search(...)'s single-pass result.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class HopRecord:
    hop_number: int
    query: str
    retrieved: List[Dict[str, Any]]
    sufficient: bool
    coverage: float
    next_query: str = ""


@dataclass
class MultiHopResult:
    original_question: str
    hops: List[HopRecord] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = ""


class MultiHopRetriever:
    def __init__(
        self,
        retriever,
        max_hops: int = 3,
        top_k_per_hop: int = 3,
        coverage_threshold: float = 0.6,
        expansion_terms: int = 5,
    ):
        """
        Args:
            retriever: any object with .search(query: str, top_k: int) -> list[dict],
                       each dict containing at least "chunk_id" and "text"
            max_hops: hard cap on hop count, independent of the sufficiency check
            top_k_per_hop: chunks retrieved per hop
            coverage_threshold: fraction of the ORIGINAL question's distinct
                                 TF-IDF terms that must appear (with nonzero
                                 weight) somewhere in the accumulated
                                 evidence before stopping. 0.6 is a starting
                                 point, not a value derived from any
                                 external benchmark — tune against your own
                                 eval set.
            expansion_terms: how many of the evidence's own top-weighted,
                              not-already-in-the-query terms to append when
                              reformulating (Rocchio-style expansion).
        """
        self.retriever = retriever
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop
        self.coverage_threshold = coverage_threshold
        self.expansion_terms = expansion_terms

    def run(self, question: str) -> MultiHopResult:
        result = MultiHopResult(original_question=question)
        current_query = question
        seen_chunk_ids = set()

        for hop_num in range(1, self.max_hops + 1):
            retrieved = self.retriever.search(current_query, top_k=self.top_k_per_hop)

            new_evidence = [r for r in retrieved if r.get("chunk_id") not in seen_chunk_ids]
            for r in new_evidence:
                seen_chunk_ids.add(r["chunk_id"])
            result.evidence.extend(new_evidence)

            sufficient, coverage, next_query = self._reason_over_evidence(
                question, result.evidence
            )

            result.hops.append(
                HopRecord(
                    hop_number=hop_num,
                    query=current_query,
                    retrieved=retrieved,
                    sufficient=sufficient,
                    coverage=coverage,
                    next_query=next_query,
                )
            )

            if sufficient:
                result.stopped_reason = "coverage_threshold_met"
                break

            if not next_query or next_query == current_query:
                result.stopped_reason = "no_new_terms_to_expand_query_with"
                break

            current_query = next_query
        else:
            result.stopped_reason = "max_hops_reached"

        return result

    # ---------- Real (non-LLM) reasoning step: TF-IDF coverage + Rocchio expansion ----------

    def _reason_over_evidence(
        self, question: str, evidence: List[Dict[str, Any]]
    ) -> Tuple[bool, float, str]:
        """
        Returns (sufficient, coverage_score, next_query).

        Method:
          1. Fit a local TF-IDF vectorizer on [question] + [each evidence text].
             This is a fresh, ephemeral vectorizer per call — not the corpus-wide
             embedding space used for retrieval — since this is measuring
             term overlap between the question and ITS OWN evidence, not
             doing similarity search.
          2. `coverage` = fraction of the question's distinct terms that have
             nonzero TF-IDF weight somewhere in the combined evidence.
          3. If coverage >= threshold, stop.
          4. Otherwise, reformulate: take the evidence's own highest-weighted
             terms that are NOT already in the question, and append the top
             `expansion_terms` of them. This is Rocchio-style pseudo-relevance
             feedback — a real, established query expansion technique, not
             an invented heuristic.
        """
        if not evidence:
            return False, 0.0, question

        texts = [e["text"] for e in evidence]

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform([question] + texts)
        except ValueError:
            # Empty vocabulary after stopword removal (e.g. question and
            # evidence share no meaningful terms at all) — can't measure
            # coverage meaningfully; treat as insufficient with no
            # expansion available, which will fall through to the "no new
            # query" stop condition.
            return False, 0.0, question

        feature_names = vectorizer.get_feature_names_out()
        question_vec = matrix[0].toarray()[0]
        evidence_sum = np.asarray(matrix[1:].sum(axis=0)).ravel()

        question_term_idxs = np.nonzero(question_vec)[0]
        if len(question_term_idxs) == 0:
            return False, 0.0, question

        covered = np.count_nonzero(evidence_sum[question_term_idxs] > 0)
        coverage = covered / len(question_term_idxs)

        if coverage >= self.coverage_threshold:
            return True, coverage, ""

        question_term_set = set(question_term_idxs.tolist())
        candidate_idxs = [
            i for i in np.argsort(evidence_sum)[::-1]
            if i not in question_term_set and evidence_sum[i] > 0
        ]
        top_terms = [feature_names[i] for i in candidate_idxs[: self.expansion_terms]]

        if not top_terms:
            return False, coverage, question  # nothing new to expand with — caller stops

        next_query = f"{question} {' '.join(top_terms)}"
        return False, coverage, next_query
