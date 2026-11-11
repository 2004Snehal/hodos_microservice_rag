"""
pipeline.py — Probabilistic RAPTOR + multi-hop RAG pipeline (served at /v2).

  ingest:  text -> sentence-aware chunks + self-describing table leaves, linked to
           sentences that cite "Table N" -> ProbabilisticRaptorTree (saved to disk)
           text -> tables -> structured rows in TableRuleStore (SQLite)
  ask:     TableRuleStore exact match (category, grade, location) -> answer
           directly, no retrieval or LLM
           -> SemanticCache (L1 memory -> L2 Redis exact -> L3 Redis vector)
           -> ProbabilisticMultiHop (retrieval distribution, LLM relevance &
              sufficiency probabilities, LLM follow-up queries)
           -> LLM rule extraction with a probability for the rule type
           -> cross-check against the deterministic rule engine
           -> cache

CONFIDENCE
  confidence = P(rule_type | evidence) × P(sufficient)
  (for no_rule_found: P(no_rule_found) × (1 − P(sufficient)) — the evidence
  NOT answering the question is what supports "there is no rule")
Both factors are read from qwen3:4b's token probabilities, not from a
number the model writes about itself. P(rule_type) is the model's
probability for the rule type it chose; P(sufficient) is its probability
that the evidence fully answers the question. Without Ollama, the pipeline
falls back to the regex rule engine and reports confidence as None.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.advanced.caching.semantic_cache import DEFAULT_REDIS_URL, SemanticCache, hash_text
from app.advanced.probabilistic.multihop import ProbabilisticMultiHop
from app.advanced.probabilistic.retriever import LlmCache, Retriever
from app.advanced.probabilistic.tree import ProbabilisticRaptorTree, _numbers
from app.core.chunking import Leaf, chunk_document, chunk_text
from app.core.dense_embeddings import EMBEDDING_MODEL, Embedder, count_tokens
from app.core.rule_store import TableMatch, TableRuleStore
from app.core.table_rules import extract_table_rules
from app.core.rule_extraction import resolve_rule
from app.llm.ollama import LLMUnavailable, OllamaClient

REVIEW_THRESHOLD = 0.6  # below this confidence, status is needs_review

TABLE_SUMMARY_SYSTEM = (
    "You describe a table from a document for a search index. In one sentence of at most "
    "35 words, say what the table is and what questions or measures it is used for (for "
    "example liquidity ratios, margins, per-day allowance limits). Use the document's "
    "terms. Do not include numbers."
)
TABLE_SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string", "description": "one sentence, no numbers"}},
    "required": ["summary"],
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "probabilistic"
# Chunk sizes in bge-small tokens (count_tokens), not characters. bge-small
# truncates at 512 tokens; CHUNK_CEILING keeps every leaf well inside that.
# 256-token leaves were measured worse (9/14 vs 12/14 rules right on a
# 930-token test policy with qwen3:4b): a section-sized leaf mixes several
# rules, and the model takes the wrong rule type ("pre-approval" for a
# self-drive car labels the mileage rate requires_preapproval).
CHUNK_SIZE = 128
CHUNK_OVERLAP = 24
CHUNK_MIN = 32
CHUNK_CEILING = 400

RULE_TYPES = ["cap", "waived", "requires_preapproval", "no_rule_found"]
PERIODS = ["per_day", "per_night", "per_trip", "per_month", "per_year", "one_time", "not_applicable"]

EXTRACTION_SYSTEM = (
    "You answer questions about a company expense policy using ONLY the numbered "
    "evidence passages. Decide which rule applies to the exact grade, location and "
    "situation in the question. If an exception in the evidence applies, the "
    "exception wins. A rule that does not name a grade, location or situation "
    "applies to everyone. Cite the passage ids you used. If the evidence does not "
    "state a rule for the question, use no_rule_found."
)
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "rule_type": {"type": "string", "enum": RULE_TYPES},
        "limit_amount": {"type": ["number", "null"], "description": "the cap amount, or null"},
        "currency": {"type": ["string", "null"], "description": "e.g. INR, or null"},
        "period": {"type": "string", "enum": PERIODS},
        "conditions": {
            "type": "array", "items": {"type": "string"},
            "description": "conditions or exceptions that apply, quoted from the evidence",
        },
        "cited_ids": {"type": "array", "items": {"type": "string"}, "description": "passage ids used"},
        "answer": {"type": "string", "description": "one or two sentence answer to the question"},
    },
    "required": ["rule_type", "limit_amount", "currency", "period", "conditions", "cited_ids", "answer"],
}

# Old engine's labels -> this pipeline's labels, for the cross-check.
_ENGINE_TO_RULE = {"daily_cap": "cap", "waived": "waived",
                   "requires_preapproval": "requires_preapproval", "unresolved": "no_rule_found"}


class ProbabilisticRagPipeline:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        llm: Optional[OllamaClient] = None,
        cache: Optional[SemanticCache] = None,
        redis_url: Optional[str] = DEFAULT_REDIS_URL,
        max_hops: int = 3,
        top_k_per_hop: int = 4,
        sufficiency_threshold: float = 0.8,
        chunk_params: Optional[Dict[str, Any]] = None,
        table_aware: bool = True,
        raptor: bool = True,
        raptor_cluster_size: Optional[int] = None,
        table_summaries: str = "off",
        query_expansion: bool = False,
        reranker: Optional[str] = None,
        table_cap: Optional[int] = None,
        embedding_model: str = EMBEDDING_MODEL,
        llm_cache_path: Optional[Path] = None,
    ):
        self.data_dir = data_dir
        # chunk_text keyword arguments; eval/run_eval.py passes others to compare.
        self.chunk_params = chunk_params or dict(
            chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, min_chunk=CHUNK_MIN,
            max_chunk=CHUNK_CEILING, length=count_tokens,
        )
        # Tables as self-describing leaves, linked to sentences that cite them
        # (app.core.chunking.chunk_document). False chunks tables as prose.
        self.table_aware = table_aware
        # Retrieval options, compared in eval/run_eval.py:
        #   raptor=False      flat retrieval over leaves (no summary levels)
        #   table_summaries   "off" | "rows" (caption + column names + row labels)
        #                     | "llm" (one LLM sentence on what the table is for),
        #                     prepended to every leaf of the table
        #   query_expansion / reranker   see app/advanced/probabilistic/retriever.py
        self.raptor = raptor
        self.raptor_cluster_size = raptor_cluster_size
        if table_summaries not in ("off", "rows", "llm"):
            raise ValueError("table_summaries must be off, rows or llm")
        self.table_summaries = table_summaries
        self.query_expansion = query_expansion
        self.reranker = reranker
        self.table_cap = table_cap
        self.llm_cache = LlmCache(llm_cache_path or data_dir / "llm_cache.json")
        self.trees_dir = data_dir / "trees"
        self.embedder = Embedder(embedding_model)
        self.llm = llm or OllamaClient()
        self.cache = cache or SemanticCache(data_dir / "cache_log", redis_url=redis_url, embedder=self.embedder)
        self.max_hops = max_hops
        self.top_k_per_hop = top_k_per_hop
        self.sufficiency_threshold = sufficiency_threshold
        self.table_rules = TableRuleStore(data_dir / "table_rules.db")
        self.trees: Dict[str, ProbabilisticRaptorTree] = {}
        self.doc_hashes: Dict[str, str] = {}
        self._load_saved_trees()

    # ---------------------------------------------------------------- ingest

    def ingest(self, doc_id: str, text: str) -> Dict[str, Any]:
        doc_hash = hash_text(text)
        invalidated = 0
        if self.doc_hashes.get(doc_id) not in (None, doc_hash):
            invalidated = self.cache.invalidate_doc(doc_id)

        if self.table_aware:
            leaves = chunk_document(text, **self.chunk_params)
        else:
            leaves = [Leaf(t) for t in chunk_text(text, **self.chunk_params)]
        summaries = self._table_summaries(leaves)
        chunks = [f"{summaries[leaf.table_key]}\n{leaf.text}" if leaf.table_key in summaries else leaf.text
                  for leaf in leaves]
        tree = ProbabilisticRaptorTree(embedder=self.embedder, llm=self.llm,
                                       max_levels=3 if self.raptor else 0, cluster_size=self.raptor_cluster_size)
        tree.build(doc_id, chunks)
        tree.doc_hash = doc_hash
        tree.links = _table_links(doc_id, leaves)
        tree.table_of = {f"{doc_id}::L0::{i}": leaf.table_key for i, leaf in enumerate(leaves)
                         if leaf.table_key is not None}
        tree.save(self.trees_dir / f"{_safe(doc_id)}.pkl")

        tables = extract_table_rules(text)
        self.table_rules.replace(doc_id, doc_hash, tables)

        self.trees[doc_id] = tree
        self.doc_hashes[doc_id] = doc_hash
        return {
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "leaf_chunks": len(chunks),
            "table_leaves": sum(leaf.kind == "table" for leaf in leaves),
            "table_links": sum(len(v) for v in tree.links.values()) // 2,
            "table_summaries": {"mode": self.table_summaries, "tables": len(summaries)},
            "llm_available": self.llm.available(),
            "cache_entries_invalidated": invalidated,
            **tree.structure(),
            "structured_rules": {
                "tables": tables.tables,
                "rows": [r.public() for r in tables.rules],
                "skipped": tables.skipped,
                "inferred": tables.inferred,
                "city_tiers": tables.city_tiers,
            },
        }

    def _table_summaries(self, leaves) -> Dict[int, str]:
        """table_key -> one summary line, per self.table_summaries."""
        if self.table_summaries == "off":
            return {}
        tables = {}
        for leaf in leaves:
            if leaf.table_key is not None and leaf.table_key not in tables:
                tables[leaf.table_key] = leaf.table_info
        out = {}
        for key, info in tables.items():
            what = info.get("caption") or info.get("section") or ""
            columns = ", ".join(c for c in info.get("header", []) if c)
            rows = ", ".join(info.get("row_labels", [])[:25])
            line = f"Table summary: {what}. Columns: {columns}. Rows: {rows}."
            if self.table_summaries == "llm" and self.llm.available():
                key_ = LlmCache.key("table_summary", self.llm.model, TABLE_SUMMARY_SYSTEM, line)
                described = self.llm_cache.get(key_)
                if described is None:
                    try:
                        result = self.llm.json(TABLE_SUMMARY_SYSTEM, line, TABLE_SUMMARY_SCHEMA, max_tokens=80)
                        described = str(result.data.get("summary", "")).strip()
                        self.llm_cache.put(key_, described)
                    except LLMUnavailable:
                        described = ""
                if described:
                    line = f"Table summary: {described} Columns: {columns}. Rows: {rows}."
            out[key] = line
        return out

    def retriever(self, doc_id: str) -> Retriever:
        return Retriever(self.trees[doc_id], self.llm, query_expansion=self.query_expansion,
                         reranker=self.reranker, cache=self.llm_cache, table_cap=self.table_cap)

    # ---------------------------------------------------------------- ask

    def ask(
        self,
        question: str,
        doc_id: str,
        category: str = "any",
        grade: str = "any",
        location: str = "any",
        version: str = "v1",
        use_cache: bool = True,
        use_structured: bool = True,
    ) -> Dict[str, Any]:
        if doc_id not in self.trees:
            raise KeyError(f"No policy ingested with doc_id='{doc_id}'.")

        table_lookup = None
        if use_structured:
            match = self.table_rules.lookup(doc_id, category, grade, location)
            if match.hit:
                return _structured_judgment(question, match)
            table_lookup = match.summary()

        scope = {"doc_id": doc_id, "doc_hash": self.doc_hashes[doc_id], "category": category,
                 "grade": grade, "location": location, "version": version}
        if use_cache:
            hit = self.cache.lookup(question, scope)
            if hit is not None:
                return {**hit.judgment, "source": "cache", "cache_layer": hit.layer,
                        "cache_similarity": round(hit.similarity, 4), "cached_question": hit.matched_question,
                        "table_lookup": table_lookup}

        retrieval = ProbabilisticMultiHop(
            self.retriever(doc_id), self.llm, max_hops=self.max_hops,
            top_k_per_hop=self.top_k_per_hop, sufficiency_threshold=self.sufficiency_threshold,
        ).run(question)

        judgment = self._judge(question, category, retrieval)
        judgment["retrieval"] = {
            "llm_used": retrieval.llm_used,
            "hops": [
                {"hop": h.number, "query": h.query,
                 "p_sufficient": None if h.p_sufficient is None else round(h.p_sufficient, 4),
                 "missing": h.missing,
                 "retrieved": [{"chunk_id": r["chunk_id"], "probability": r["probability"]} for r in h.retrieved]}
                for h in retrieval.hops
            ],
            "stopped_reason": retrieval.stopped_reason,
            "evidence": retrieval.evidence,
        }
        if use_cache:
            self.cache.store(question, scope, judgment)
        return {**judgment, "source": "fresh", "table_lookup": table_lookup}

    def _judge(self, question: str, category: str, retrieval) -> Dict[str, Any]:
        evidence = [e for e in retrieval.evidence if e["relevance"] >= 0.3] or retrieval.evidence[:2]
        evidence_text = " ".join(e["text"] for e in evidence)
        engine = resolve_rule(evidence_text, _category_keywords(category, question))
        engine_rule = _ENGINE_TO_RULE.get(engine.rule_type, "no_rule_found")

        if not retrieval.llm_used:
            return {
                "question": question, "status": "needs_review", "method": "rule_engine_fallback", "rule_type": engine_rule,
                "limit_amount": engine.limit_amount, "currency": "INR" if engine.limit_amount else None,
                "period": "not_applicable", "conditions": [], "answer": engine.matched_window or "",
                "cited_ids": [e["chunk_id"] for e in evidence],
                "confidence": None, "rule_type_distribution": None, "p_sufficient": None,
                "rule_engine": {"rule_type": engine_rule, "limit_amount": engine.limit_amount, "agrees": True},
                "warnings": ["Ollama unavailable — deterministic rule engine used, no confidence estimate."],
            }

        try:
            result = self.llm.json(
                EXTRACTION_SYSTEM,
                "Evidence:\n" + "\n".join(f"[{e['chunk_id']}] {e['text']}" for e in evidence)
                + f"\n\nQuestion: {question}",
                EXTRACTION_SCHEMA,
                max_tokens=400,
            )
        except LLMUnavailable as exc:
            retrieval.llm_used = False
            judgment = self._judge(question, category, retrieval)
            judgment["warnings"].append(str(exc))
            return judgment

        data = result.data
        distribution = {k: round(v, 4) for k, v in result.distributions["rule_type"].items()}
        p_rule = distribution.get(data["rule_type"], 0.0)
        p_sufficient = retrieval.p_sufficient if retrieval.p_sufficient is not None else 1.0
        valid_ids = {e["chunk_id"] for e in retrieval.evidence}

        warnings: List[str] = []
        numbers_ok = True
        limit = data.get("limit_amount")
        if limit is not None and f"{float(limit):g}" not in _numbers(evidence_text):
            numbers_ok = False
            warnings.append(f"limit_amount {limit} does not appear in the evidence text.")

        # Small models garble numbers in free text ("6000" -> "600.00") even
        # when the structured field is right; never return an answer whose
        # numbers aren't in the evidence or the question.
        answer = str(data.get("answer", ""))
        if not _numbers(answer) <= _numbers(evidence_text) | _numbers(question):
            warnings.append(f"Model answer contained numbers not in the evidence; replaced with a templated answer. Original: {answer!r}")
            answer = _templated_answer(data)
        cited = [c for c in data.get("cited_ids", []) if c in valid_ids]
        if len(cited) < len(data.get("cited_ids", [])):
            warnings.append("Model cited passage ids that were not in the evidence; they were dropped.")

        agrees = engine_rule == data["rule_type"] and (
            data["rule_type"] != "cap" or engine.limit_amount == (None if limit is None else float(limit))
        )
        if not agrees:
            warnings.append(
                f"Rule engine disagrees: it found {engine_rule}"
                + (f" {engine.limit_amount}" if engine.limit_amount is not None else "")
                + ". The regex engine cannot read exceptions or wording like '1500 rupees', so check the citation."
            )

        if data["rule_type"] == "no_rule_found":
            confidence = p_rule * (1 - p_sufficient)
        else:
            confidence = p_rule * p_sufficient
        status = "ok" if numbers_ok and confidence >= REVIEW_THRESHOLD else "needs_review"

        return {
            "question": question,
            "status": status,
            "method": "llm",
            "model": self.llm.model,
            "rule_type": data["rule_type"],
            "limit_amount": limit,
            "currency": data.get("currency"),
            "period": data.get("period"),
            "conditions": data.get("conditions", []),
            "answer": answer,
            "cited_ids": cited,
            "citations": [{"chunk_id": e["chunk_id"], "text": e["text"]} for e in evidence if e["chunk_id"] in cited],
            "confidence": round(confidence, 4),
            "rule_type_distribution": distribution,
            "p_sufficient": None if retrieval.p_sufficient is None else round(retrieval.p_sufficient, 4),
            "rule_engine": {"rule_type": engine_rule, "limit_amount": engine.limit_amount, "agrees": agrees},
            "warnings": warnings,
        }

    # ---------------------------------------------------------------- persistence / info

    def _load_saved_trees(self) -> None:
        if not self.trees_dir.exists():
            return
        for path in self.trees_dir.glob("*.pkl"):
            tree = ProbabilisticRaptorTree(embedder=self.embedder, llm=self.llm).load(path)
            leaves = tree.leaves()
            if not leaves:
                continue
            doc_id = leaves[0].doc_id
            self.trees[doc_id] = tree
            self.doc_hashes[doc_id] = tree.doc_hash or hash_text(doc_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "documents": sorted(self.trees),
            "llm": {"model": self.llm.model, "available": self.llm.available(refresh=True)},
            "cache": self.cache.stats(),
            "structured_rules": self.table_rules.stats(),
        }


def _structured_judgment(question: str, match: TableMatch) -> Dict[str, Any]:
    """A table row answers the request exactly: no retrieval, no LLM, confidence 1."""
    row = match.rules[0]
    data = {"rule_type": row.rule_type, "limit_amount": row.cap, "currency": row.currency,
            "period": row.unit or "not_applicable", "conditions": []}
    return {
        "question": question,
        "status": "ok",
        "method": "structured_table",
        **data,
        "answer": _templated_answer(data),
        "cited_ids": [r.ref for r in match.rules],
        "citations": [{"chunk_id": r.ref, "text": r.source} for r in match.rules],
        "confidence": 1.0,
        "rule_type_distribution": None,
        "p_sufficient": None,
        "structured_row": row.public(),
        "warnings": [],
        "source": "structured",
    }


def _table_links(doc_id: str, leaves) -> Dict[str, List[str]]:
    """Both ways: each leaf mentioning "Table N" <-> every leaf of table N."""
    ids = [f"{doc_id}::L0::{i}" for i in range(len(leaves))]
    tables: Dict[str, List[str]] = {}
    for node_id, leaf in zip(ids, leaves):
        if leaf.kind == "table":
            for table_id in leaf.table_ids:
                tables.setdefault(table_id, []).append(node_id)
    links: Dict[str, List[str]] = {}
    for node_id, leaf in zip(ids, leaves):
        for table_id in (leaf.mentions if leaf.kind == "text" else []):
            for table_node in tables.get(table_id, []):
                links.setdefault(node_id, []).append(table_node)
                links.setdefault(table_node, []).append(node_id)
    return links


def _templated_answer(data: Dict[str, Any]) -> str:
    rule = data["rule_type"]
    if rule == "cap" and data.get("limit_amount") is not None:
        period = (data.get("period") or "").replace("_", " ").replace("not applicable", "").strip()
        text = f"Capped at {data.get('currency') or ''} {float(data['limit_amount']):g} {period}".replace("  ", " ").strip() + "."
    elif rule == "waived":
        text = "The limit is waived."
    elif rule == "requires_preapproval":
        text = "Pre-approval is required."
    else:
        return "The policy evidence does not state a rule for this question."
    if data.get("conditions"):
        text += " Conditions: " + "; ".join(data["conditions"])
    return text


def _category_keywords(category: str, question: str) -> List[str]:
    from app.core.bill_parser import CATEGORY_KEYWORDS

    if category in CATEGORY_KEYWORDS:
        return CATEGORY_KEYWORDS[category]
    lowered = question.lower()
    for keywords in CATEGORY_KEYWORDS.values():
        if any(k in lowered for k in keywords):
            return keywords
    return [w for w in lowered.split() if len(w) > 3]


def _safe(doc_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_id)
