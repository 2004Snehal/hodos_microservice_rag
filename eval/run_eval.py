"""
run_eval.py — Evaluate the /v2 pipeline on a gold dataset and store every step.

  .venv/bin/python -m eval.run_eval                                  # all datasets, production config
  .venv/bin/python -m eval.run_eval -d company_xx_travel_india -c tok128 -c tok256
  .venv/bin/python -m eval.run_eval --no-llm                         # chunking + retrieval only

STEPS, per (dataset, config)
  1. chunk      — leaf count, token / character sizes, the chunks themselves
  2. tree       — RAPTOR build time and structure
  3. retrieval  — tree.query(question) top-k; does the top-k contain every
                  gold evidence phrase, or (datasets with "pages") a leaf on a
                  gold evidence page (recall@1/2/4), rank of the first hit,
                  context tokens sent
  4. judgment   — pipeline.ask() with qwen3:4b (cache off): rule_type /
                  limit_amount / answer_contains / answer_numbers vs gold, confidence, status,
                  hops, evidence recall over all hops, latency.
                  "confidently wrong" = wrong answer the pipeline still
                  marked status=ok — the case an audit can't catch.

OUTPUT
  metrics/eval_runs/<UTC time>_<label>.json  — everything above, per question
  metrics/eval_history.jsonl                 — one summary line per (dataset, config),
                                               appended, never rewritten
"""

import argparse
import json
import platform
import re
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from app.advanced.probabilistic import pipeline as v2
from app.core.dense_embeddings import EMBEDDING_MODEL, count_tokens

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "eval" / "datasets"
RUNS_DIR = ROOT / "metrics" / "eval_runs"
HISTORY = ROOT / "metrics" / "eval_history.jsonl"

PRODUCTION = dict(chunk_size=v2.CHUNK_SIZE, overlap=v2.CHUNK_OVERLAP,
                  min_chunk=v2.CHUNK_MIN, max_chunk=v2.CHUNK_CEILING, unit="tokens")
LLM_CACHE = ROOT / "eval" / "cache" / "llm_cache.json"  # query expansions + table summaries

# Retrieval experiments (FinanceBench stress test, 2026-10-28). Flat = no RAPTOR
# levels. Each adds one change to the one before it.
_R = dict(chunk_size=128, overlap=24, min_chunk=32, max_chunk=400, unit="tokens", structured=False)
CONFIGS_RETRIEVAL = {
    "r0_baseline": {**_R, "table_aware": False, "raptor": True},
    "r1_tables": {**_R, "table_aware": True, "raptor": True},
    "r2_flat": {**_R, "table_aware": True, "raptor": False},
    "r2b_raptor15": {**_R, "table_aware": True, "raptor": True, "raptor_cluster_size": 15},
    "r3_rowsum": {**_R, "table_aware": True, "raptor": False, "table_summaries": "rows"},
    "r1c_cap2": {**_R, "table_aware": True, "raptor": False, "table_cap": 2},
    "r3c_rowsum_cap2": {**_R, "table_aware": True, "raptor": False, "table_summaries": "rows", "table_cap": 2},
    # table-aware chain (with the per-table cap)
    "r4_llmsum": {**_R, "table_aware": True, "raptor": False, "table_cap": 2, "table_summaries": "llm"},
    "r5_expand": {**_R, "table_aware": True, "raptor": False, "table_cap": 2, "table_summaries": "llm",
                  "query_expansion": True},
    "r6_rerank": {**_R, "table_aware": True, "raptor": False, "table_cap": 2, "table_summaries": "llm",
                  "query_expansion": True, "reranker": "BAAI/bge-reranker-base"},
    "r7_bgebase": {**_R, "table_aware": True, "raptor": False, "table_cap": 2, "table_summaries": "llm",
                   "query_expansion": True, "reranker": "BAAI/bge-reranker-base",
                   "embedding_model": "BAAI/bge-base-en-v1.5"},
    # plain chain (tables chunked as prose)
    "p2_flat": {**_R, "table_aware": False, "raptor": False},
    "p5_expand": {**_R, "table_aware": False, "raptor": False, "query_expansion": True},
    "p6_rerank": {**_R, "table_aware": False, "raptor": False, "query_expansion": True,
                  "reranker": "BAAI/bge-reranker-base"},
    "p7_bgebase": {**_R, "table_aware": False, "raptor": False, "query_expansion": True,
                   "reranker": "BAAI/bge-reranker-base", "embedding_model": "BAAI/bge-base-en-v1.5"},
}

CONFIGS = {
    "production": {**PRODUCTION, "structured": True, "table_aware": True},
    "tok128_tables": dict(chunk_size=128, overlap=24, min_chunk=32, max_chunk=400, unit="tokens",
                          structured=False, table_aware=True),
    "tok128": dict(chunk_size=128, overlap=24, min_chunk=32, max_chunk=400, unit="tokens", structured=False),
    "tok256": dict(chunk_size=256, overlap=48, min_chunk=40, max_chunk=400, unit="tokens", structured=False),
    "tok64": dict(chunk_size=64, overlap=16, min_chunk=16, max_chunk=400, unit="tokens", structured=False),
    "chars250": dict(chunk_size=250, overlap=80, min_chunk=0, max_chunk=None, unit="chars", structured=False),
    **CONFIGS_RETRIEVAL,
}
TOP_KS = (1, 2, 4, 10, 20)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _has_all(text: str, phrases: List[str]) -> bool:
    text = _norm(text)
    return all(_norm(p) in text for p in phrases)


def _chunk_params(config: Dict[str, Any]) -> Dict[str, Any]:
    params = {k: config[k] for k in ("chunk_size", "overlap", "min_chunk", "max_chunk")}
    params["length"] = count_tokens if config["unit"] == "tokens" else len
    return params


def _environment(use_llm: bool) -> Dict[str, Any]:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    llm = v2.OllamaClient()
    return {
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": llm.model,
        "llm_available": llm.available() if use_llm else None,
        "max_hops": 3, "top_k_per_hop": 4, "sufficiency_threshold": 0.8,
        "review_threshold": v2.REVIEW_THRESHOLD,
    }


def _answer_numbers(text: str) -> List[float]:
    return [float(n.replace(",", "")) for n in
            re.findall(r"(?<![A-Za-z\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text or "")]


def _judge_item(item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    expected = item["expected"]
    checks = {}
    if "rule_type" in expected:
        checks["rule_type"] = result.get("rule_type") == expected["rule_type"]
    if expected.get("limit_amount") is not None:
        got = result.get("limit_amount")
        checks["limit_amount"] = got is not None and float(got) == float(expected["limit_amount"])
    if expected.get("answer_contains"):
        checks["answer_contains"] = expected["answer_contains"] in str(result.get("answer", ""))
    if expected.get("answer_numbers"):
        # Every gold number must appear in the answer or limit_amount, within 1%.
        got = _answer_numbers(str(result.get("answer", "")))
        if result.get("limit_amount") is not None:
            got.append(float(result["limit_amount"]))
        checks["answer_numbers"] = all(any(abs(g - x) <= max(abs(g) * 0.01, 0.005) for x in got)
                                       for g in expected["answer_numbers"])
    return checks


def _leaf_pages(leaves, pages: List[str]) -> Dict[str, List[int]]:
    """
    Pages each leaf lies on, by word coverage: the pages holding ≥70% of the
    leaf's words (or the best page, if it holds ≥50%). Word-based rather than
    substring-based, so a table leaf whose rows were rewritten as
    "Column: value" sentences still maps to its page.
    """
    words = lambda t: set(re.findall(r"[a-z0-9]{2,}", t.lower()))
    page_words = [words(p) for p in pages]
    found = {}
    for leaf in leaves:
        w = words(leaf.text)
        if not w:
            found[leaf.node_id] = []
            continue
        coverage = [len(w & pw) / len(w) for pw in page_words]
        hits = [i for i, c in enumerate(coverage) if c >= 0.7]
        best = max(range(len(pages)), key=coverage.__getitem__) if pages else None
        found[leaf.node_id] = hits or ([best] if best is not None and coverage[best] >= 0.5 else [])
    return found


def _ingest(pipe, doc_id: str, text: str) -> Dict[str, Any]:
    started = time.time()
    ingest = pipe.ingest(doc_id, text)
    ingest_s = time.time() - started
    leaves = sorted(pipe.trees[doc_id].leaves(), key=lambda n: int(n.node_id.rsplit("::", 1)[1]))
    leaf_tokens = [count_tokens(n.text) for n in leaves]
    return {
        "chunk": {
            "leaves": len(leaves),
            "tokens": {"min": min(leaf_tokens), "median": statistics.median(leaf_tokens), "max": max(leaf_tokens),
                       "over_512": sum(t > 512 for t in leaf_tokens)},
            "chars": {"min": min(len(n.text) for n in leaves), "max": max(len(n.text) for n in leaves)},
            "chunks": [{"id": n.node_id, "tokens": t, "text": n.text} for n, t in zip(leaves, leaf_tokens)],
        },
        "tree": {"ingest_seconds": round(ingest_s, 2), **{k: ingest[k] for k in
                 ("nodes_per_level", "nodes_with_multiple_parents", "summaries", "build_log")},
                 "structured_rows": len(ingest["structured_rules"]["rows"])},
        "_leaves": leaves,
        "_leaf_tokens": leaf_tokens,
    }


def _gold_numbers(answer: str) -> set:
    """Amounts in a gold answer worth matching: ≥3 significant digits, not years."""
    found = set()
    for raw in re.findall(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])", answer or ""):
        if re.fullmatch(r"(19|20)\d\d", raw) or len(re.sub(r"\D", "", raw).lstrip("0")) < 3:
            continue
        found.add(f"{float(raw.replace(',', '')):g}")
    return found


def _text_numbers(text: str) -> set:
    return {f"{float(n.replace(',', '')):g}"
            for n in re.findall(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])", text)}


def evaluate(dataset: Dict[str, Any], name: str, config: Dict[str, Any], use_llm: bool,
             judge: bool = True) -> Dict[str, Any]:
    # A dataset has one "policy", or several "policies" {doc_id: path} with items naming their doc_id.
    policies = dataset.get("policies") or {"eval-doc": dataset["policy"]}
    llm = v2.OllamaClient() if use_llm else v2.OllamaClient(url="http://127.0.0.1:9")  # unreachable = no LLM
    pipe = v2.ProbabilisticRagPipeline(
        data_dir=Path(tempfile.mkdtemp(prefix="hodos-eval-")), llm=llm, redis_url=None,
        chunk_params=_chunk_params(config), table_aware=config.get("table_aware", False),
        raptor=config.get("raptor", True), raptor_cluster_size=config.get("raptor_cluster_size"),
        table_summaries=config.get("table_summaries", "off"),
        query_expansion=config.get("query_expansion", False), reranker=config.get("reranker"),
        table_cap=config.get("table_cap"),
        embedding_model=config.get("embedding_model", v2.EMBEDDING_MODEL), llm_cache_path=LLM_CACHE,
    )

    # 1-2. chunk + tree, per document
    docs, leaf_pages = {}, {}
    for doc_id, path in policies.items():
        docs[doc_id] = _ingest(pipe, doc_id, (ROOT / path).read_text())
        if dataset.get("pages", {}).get(doc_id):
            pages = json.loads((ROOT / dataset["pages"][doc_id]).read_text())
            leaf_pages.update(_leaf_pages(docs[doc_id]["_leaves"], pages))
        print(f"  ingested {doc_id}: {docs[doc_id]['chunk']['leaves']} leaves, "
              f"{docs[doc_id]['tree']['ingest_seconds']}s", flush=True)

    # 3-4. per question
    questions = []
    for item in dataset["items"]:
        doc_id = item.get("doc_id", "eval-doc")
        tree = pipe.retriever(doc_id)
        record: Dict[str, Any] = {"id": item["id"], "doc_id": doc_id, "type": item["type"], "question": item["question"]}
        phrases = item.get("evidence_phrases", [])
        gold_pages = set(item.get("evidence_pages", []))
        if phrases or gold_pages:
            started = time.time()
            ranked = tree.query(item["question"], top_k=max(TOP_KS))
            query_seconds = time.time() - started
            gold_numbers = _gold_numbers(item.get("gold_answer", "")) if gold_pages else set()

            def lenient(top) -> bool:
                """Page hit, or a retrieved leaf holding a gold-answer amount (10-Ks repeat figures)."""
                if hit(top):
                    return True
                return bool(gold_numbers) and any(gold_numbers & _text_numbers(r["text"]) for r in top)

            def hit(top) -> bool:
                if phrases:
                    return _has_all(" ".join(r["text"] for r in top), phrases)
                return any(gold_pages & set(leaf_pages.get(r["chunk_id"], [])) for r in top)

            first = next((i + 1 for i in range(len(ranked)) if hit(ranked[:i + 1])), None)
            linked = [{"chunk_id": n.node_id, "text": n.text} for r in ranked for n in tree.linked_leaves(r["chunk_id"])]
            record["retrieval"] = {
                "scored_by": "phrases" if phrases else "pages",
                "top_ids": [r["chunk_id"] for r in ranked],
                "top_pages": [leaf_pages.get(r["chunk_id"]) for r in ranked] if gold_pages else None,
                "gold_pages": sorted(gold_pages) or None,
                "probabilities": [r["probability"] for r in ranked],
                **{f"recall@{k}": hit(ranked[:k]) for k in TOP_KS},
                **({f"lenient@{k}": lenient(ranked[:k]) for k in TOP_KS} if gold_pages else {}),
                "expansion": tree.last_expansion,
                "query_seconds": round(query_seconds, 2),
                "recall@4+links": hit(ranked + linked),
                "linked_ids": [l["chunk_id"] for l in linked],
                "rank_all_phrases": first,
                "context_tokens@4": sum(count_tokens(r["text"]) for r in ranked[:4]),
            }
        if use_llm and judge and "expected" in item:
            started = time.time()
            result = pipe.ask(item["question"], doc_id, category=item.get("category", "any"),
                              grade=item.get("grade", "any"), location=item.get("location", "any"),
                              use_cache=False, use_structured=config["structured"])
            checks = _judge_item(item, result)
            evidence = (result.get("retrieval") or {}).get("evidence", [])
            if phrases and evidence:
                evidence_recall = _has_all(" ".join(e["text"] for e in evidence), phrases)
            elif gold_pages and evidence:
                evidence_recall = any(gold_pages & set(leaf_pages.get(e["chunk_id"], [])) for e in evidence)
            else:
                evidence_recall = None
            record["judgment"] = {
                "expected": item["expected"],
                "gold_answer": item.get("gold_answer"),
                "got": {k: result.get(k) for k in ("rule_type", "limit_amount", "period", "answer", "status",
                                                   "method", "confidence", "p_sufficient", "cited_ids")},
                "checks": checks,
                "correct": all(checks.values()),
                "rule_type_distribution": result.get("rule_type_distribution"),
                "rule_engine": result.get("rule_engine"),
                "hops": [{"query": h["query"], "p_sufficient": h["p_sufficient"], "missing": h["missing"]}
                         for h in (result.get("retrieval") or {}).get("hops", [])],
                "stopped_reason": (result.get("retrieval") or {}).get("stopped_reason"),
                "evidence_recall": evidence_recall,
                "warnings": result.get("warnings", []),
                "seconds": round(time.time() - started, 2),
            }
        questions.append(record)
        mark = record.get("judgment", {}).get("correct")
        print(f"  {name:10} {item['id']:24} recall@4={record.get('retrieval', {}).get('recall@4')} "
              f"correct={mark}", flush=True)

    leaf_tokens = [t for d in docs.values() for t in d.pop("_leaf_tokens")]
    for d in docs.values():
        d.pop("_leaves")
    totals = {"leaves": len(leaf_tokens), "leaf_tokens_median": statistics.median(leaf_tokens),
              "leaf_tokens_max": max(leaf_tokens),
              "ingest_seconds": round(sum(d["tree"]["ingest_seconds"] for d in docs.values()), 2)}
    steps = ({"chunk": docs["eval-doc"]["chunk"], "tree": docs["eval-doc"]["tree"]} if list(docs) == ["eval-doc"]
             else {"documents": docs})
    return {"dataset": dataset["name"], "config_name": name, "config": config,
            "steps": steps, "questions": questions, "summary": _summarize(questions, totals)}


def _summarize(questions, totals) -> Dict[str, Any]:
    retrieved = [q["retrieval"] for q in questions if "retrieval" in q]
    judged = [q["judgment"] for q in questions if "judgment" in q]
    summary: Dict[str, Any] = {
        **totals,
        "retrieval_questions": len(retrieved),
        **{f"recall@{k}": round(sum(r[f"recall@{k}"] for r in retrieved) / len(retrieved), 4)
           for k in TOP_KS if retrieved},
        "recall@4+links": round(sum(r.get("recall@4+links", r["recall@4"]) for r in retrieved) / len(retrieved), 4)
                          if retrieved else None,
        "mrr_all_phrases": round(sum(1 / r["rank_all_phrases"] for r in retrieved if r["rank_all_phrases"])
                                 / len(retrieved), 4) if retrieved else None,
        "context_tokens@4_mean": round(statistics.mean(r["context_tokens@4"] for r in retrieved), 1) if retrieved else None,
        "query_seconds_mean": round(statistics.mean(r.get("query_seconds", 0) for r in retrieved), 2) if retrieved else None,
    }
    lenient_rows = [r for r in retrieved if "lenient@4" in r]
    if lenient_rows:
        summary.update({f"lenient@{k}": round(sum(r[f"lenient@{k}"] for r in lenient_rows) / len(lenient_rows), 4)
                        for k in TOP_KS})
    types = sorted({q["type"] for q in questions if "retrieval" in q})
    if len(types) > 1:
        summary["by_type"] = {
            t: {"n": len(rs), **{f"recall@{k}": round(sum(r[f"recall@{k}"] for r in rs) / len(rs), 4) for k in (4, 20)},
                **({"lenient@4": round(sum(r["lenient@4"] for r in rs) / len(rs), 4)} if "lenient@4" in rs[0] else {})}
            for t in types for rs in [[q["retrieval"] for q in questions if q["type"] == t and "retrieval" in q]]
        }
    if judged:
        correct = [j for j in judged if j["correct"]]
        wrong = [j for j in judged if not j["correct"]]
        conf = lambda js: round(statistics.mean(j["got"]["confidence"] or 0 for j in js), 4) if js else None
        rule_checked = [j for j in judged if "rule_type" in j["checks"]]
        summary.update({
            "judged_questions": len(judged),
            "accuracy": round(len(correct) / len(judged), 4),
            "rule_type_accuracy": round(sum(j["checks"]["rule_type"] for j in rule_checked) / len(rule_checked), 4)
                                  if rule_checked else None,
            "limit_accuracy": round(statistics.mean(j["checks"]["limit_amount"] for j in judged
                                                    if "limit_amount" in j["checks"]), 4)
                              if any("limit_amount" in j["checks"] for j in judged) else None,
            "evidence_recall": round(statistics.mean(j["evidence_recall"] for j in judged
                                                     if j["evidence_recall"] is not None), 4)
                               if any(j["evidence_recall"] is not None for j in judged) else None,
            "status_ok": sum(j["got"]["status"] == "ok" for j in judged),
            "confidently_wrong": sum(j["got"]["status"] == "ok" for j in wrong),
            "mean_confidence_correct": conf(correct),
            "mean_confidence_wrong": conf(wrong),
            "mean_hops": round(statistics.mean(len(j["hops"]) or 0 for j in judged), 2),
            "structured_answers": sum(j["got"]["method"] == "structured_table" for j in judged),
            "mean_seconds": round(statistics.mean(j["seconds"] for j in judged), 2),
            "wrong_ids": [q["id"] for q in questions if "judgment" in q and not q["judgment"]["correct"]],
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset", action="append", help="dataset name in eval/datasets (default: all)")
    parser.add_argument("-c", "--config", action="append", choices=sorted(CONFIGS), help="default: production")
    parser.add_argument("--no-llm", action="store_true", help="steps 1-3 only, no LLM anywhere")
    parser.add_argument("--no-judge", action="store_true",
                        help="steps 1-3 only, but keep the LLM for query expansion / table summaries / RAPTOR summaries")
    parser.add_argument("--label", default="", help="added to the run file name")
    parser.add_argument("--note", default="", help="free-text note stored with the run")
    args = parser.parse_args()

    names = args.dataset or sorted(p.stem for p in DATASETS_DIR.glob("*.json"))
    configs = args.config or ["production"]
    use_llm = not args.no_llm
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ") + (f"_{args.label}" if args.label else "")
    run = {"run_id": run_id, "started_at": started_at.isoformat(), "note": args.note,
           "environment": _environment(use_llm), "results": []}

    for dataset_name in names:
        dataset = json.loads((DATASETS_DIR / f"{dataset_name}.json").read_text())
        for config_name in configs:
            print(f"== {dataset_name} / {config_name}", flush=True)
            result = evaluate(dataset, config_name, CONFIGS[config_name], use_llm, judge=not args.no_judge)
            run["results"].append(result)
            print("   " + json.dumps(result["summary"]), flush=True)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_path = RUNS_DIR / f"{run_id}.json"
    run_path.write_text(json.dumps(run, indent=1, default=str))
    with open(HISTORY, "a") as f:
        for r in run["results"]:
            f.write(json.dumps({"run_id": run_id, "started_at": run["started_at"], "note": args.note,
                                "git_commit": run["environment"]["git_commit"],
                                "git_dirty": run["environment"]["git_dirty"],
                                "llm_model": run["environment"]["llm_model"] if use_llm else None,
                                "dataset": r["dataset"], "config_name": r["config_name"],
                                "config": r["config"], **r["summary"],
                                "details": str(run_path.relative_to(ROOT))}, default=str) + "\n")
    print(f"\nsaved {run_path.relative_to(ROOT)} and appended {HISTORY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
