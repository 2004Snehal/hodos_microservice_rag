# Hodos RAG — Experiment & Decision Log

As of 2026-11-04. Covers all work since commit `d885987` ("gitignore added").

## 1. Overview

Hodos is a RAG microservice that reads travel and expense policies, extracts caps
and rules, and audits employee bills against them. Since the last commit it gained
a fully local `/v2` pipeline, table-aware chunking, structured table rules, Docling
extraction and a bill router. A new eval harness measured these over 14 runs on
2026-10-27. A ConditionalQA evaluation started on 2026-11-04.

| Item | State |
| --- | --- |
| Last commit | `d885987` (`origin/main` is one behind, at `e04aa2b`) |
| Pipeline work | Uncommitted: 11 files modified (+911 / −127), ~20 new files and folders |
| Active branch | `eval/conditionalqa`, created from `main` on 2026-11-04 |
| Eval record | `metrics/eval_history.jsonl` (one line per dataset × config), `metrics/eval_runs/*.json` (full runs) |

## 2. Pipeline changes

| Area | What changed | Where |
| --- | --- | --- |
| `/v2` pipeline | Local stack: qwen3:4b via Ollama, bge-small embeddings via fastembed, Redis semantic cache. Soft-clustered RAPTOR tree plus multi-hop retrieval. `confidence` = P(rule type) × P(evidence sufficient). Below 0.6, or a number not in the cited text, returns `needs_review`. Endpoints: `/v2/policy`, `/v2/policy/file`, `/v2/ask`, `/v2/audit`, `/v2/health`. | `app/advanced/probabilistic/` |
| Chunking | Rewritten (+407 lines). pysbd sentence splitting; splits after currency abbreviations (`INR. 1000`) are rejoined. `/v2` measures in tokens: 128 target, 24 overlap, 400 ceiling, tail under 32 merged. Each table becomes its own chunks, rows written as `Column: value` sentences. Text mentioning "Table N" is linked to that table; multi-hop follows the link. | `app/core/chunking.py` |
| Structured table rules | Policy tables parsed into rows (category, grade, city tier, cap, unit) in SQLite. An exact category / grade / location match answers directly (`confidence: 1.0`, no LLM). Anything less falls through to retrieval. | `app/core/table_rules.py`, `app/core/rule_store.py` |
| Docling extraction | PDFs converted by a docling-serve sidecar, Tesseract fallback. Numbers cross-checked against Tesseract. Repairs the docling#4151 hyphen merge (`7-10` → `710`); flags other unverified numbers. | `app/core/docling_client.py`, `app/core/document_extraction.py` |
| Bill router | ResNet-50 (ONNX) features + logistic-regression head classify each page: digital, printed or handwritten. Handwritten → vision model (qwen3-vl). Printed → Tesseract, then vision model if OCR confidence < 60. | `app/core/bill_router/`, `scripts/train_bill_router.py` |
| UI and infra | Test console at `/ui`, Dockerfile, docker-compose with a `docling` service, `.env.example`, Postman collections, smoke test. README updated. | `app/static/`, `postman/`, `scripts/` |

## 3. Evaluation harness and datasets

`eval/run_eval.py` runs four steps per dataset × config:

1. **Chunk**: leaf count and sizes.
2. **Tree**: RAPTOR build time and structure.
3. **Retrieval**: recall@1/2/4/10/20 of gold evidence phrases, MRR, context tokens.
4. **Judgment**: `pipeline.ask()` with qwen3:4b. Scores `rule_type` and
   `limit_amount` accuracy, confidence, and `confidently_wrong` (wrong answer
   still marked `status: ok` — the case an audit cannot catch).

`eval/run_table_eval.py` scores table extraction: table recall/precision, cell
accuracy, rule recall, lookup accuracy.

| Dataset | Questions | Source |
| --- | --- | --- |
| `company_xx_travel_india` | 24 | Image-only scanned policy PDF; gold read from page images, not OCR |
| `company_xx_travel_india_docling_tesseract` | 24 | Same PDF, Docling + Tesseract extraction |
| `synthetic_travel_v1` | 22 | Synthetic travel policy |
| `financebench_subset` | 40 | FinanceBench (SEC filings) |
| `fb_part_text`, `fb_part_docling` | 25 | FinanceBench part, text layer vs Docling |
| `company_xx_travel_india_tables` | — | Gold tables and rules for the extraction eval |
| `conditionalqa_dev_subset` | 68 | ConditionalQA dev, 15 gov.uk pages (added 2026-11-04) |

## 4. Experiments and results (2026-10-27)

### 4.1 Chunk size study

Run `20261027T163548Z_chunk_study_llm`.

| Dataset | Config | recall@1 | recall@4 | MRR | Judgment accuracy | Confidently wrong |
| --- | --- | --- | --- | --- | --- | --- |
| company PDF | chars250 | 0.583 | 0.917 | 0.719 | 0.556 | 3 / 18 |
| company PDF | tok128 (production) | 0.750 | 0.917 | 0.813 | 0.500 | 4 / 18 |
| company PDF | tok256 | 0.750 | 0.958 | 0.830 | 0.556 | 6 / 18 |
| synthetic | chars250 | 0.818 | 0.955 | 0.879 | 0.929 | 1 / 14 |
| synthetic | tok128 (production) | 0.818 | 1.000 | 0.902 | 0.714 | 2 / 14 |
| synthetic | tok256 | 0.773 | 1.000 | 0.860 | 0.571 | 3 / 14 |

tok64 was retrieval-only: company recall@1 0.542, synthetic 0.864.
Finding: token chunks improve retrieval on the real PDF, but judgment on the real
PDF stays at 0.50–0.56 for every size. On synthetic, chars250 judged best.

### 4.2 Table extraction (company PDF)

Run `20261027T181204Z_table_extraction` (final of four iterations of the parser).

| Extractor | Time | Table recall | Rule exact recall | Lookup accuracy |
| --- | --- | --- | --- | --- |
| Plain Tesseract text | — | 0.00 | 0.00 | 0.00 |
| Docling, default OCR | 94.1 s | 0.67 | 1.00 | 1.00 |
| Docling + Tesseract OCR | 17.6 s | 1.00 | 0.25 | 0.33 |

Parser fixes across the four runs raised Docling-default rule recall from 0.0 to 1.0.

### 4.3 Table-aware chunking A/B

Run `20261027T184018Z_table_chunking_ab`, tok128 vs tok128_tables.

| Dataset | Metric | tok128 | tok128_tables |
| --- | --- | --- | --- |
| company PDF (Docling) | Judgment accuracy | 0.500 | 0.556 |
| company PDF (Docling) | Limit accuracy | 0.750 | 0.875 |
| company PDF (Docling) | Confidently wrong | 3 | 2 |
| FinanceBench part (Docling) | recall@4 | 0.571 | 0.429 |

### 4.4 FinanceBench

Run `20261027T175331Z_financebench`, production config, 40 questions:
recall@1 0.10, recall@4 0.325, MRR 0.192. Only 7 questions were judgeable;
accuracy 0 / 7, confidently wrong 6 / 7.

### 4.5 Retriever chain

Runs `20261027T191559Z_retr_steps_r0_r3`, `20261027T193610Z_retr_cap`,
`20261027T194549Z_retr_steps_r4_r7`. Retrieval only, no judgment. Each step adds
one change. r = table-aware chain, p = tables chunked as prose.

| Config | Change | Company PDF recall@1 | Company PDF recall@4 | FB-Docling recall@4 |
| --- | --- | --- | --- | --- |
| r0_baseline | RAPTOR, no table chunks | 0.708 | 0.958 | 0.52 |
| r1_tables | + table-aware chunks | 0.708 | 0.958 | 0.48 |
| r2_flat | no RAPTOR levels | 0.708 | 0.958 | 0.48 |
| r3_rowsum | + row summaries | 0.750 | 0.958 | 0.36 |
| r4_llmsum | LLM table summaries, cap 2 | 0.750 | 0.958 | 0.44 |
| r5_expand | + query expansion | 0.708 | 1.000 | 0.44 |
| r6_rerank | + bge-reranker-base | 0.792 | 0.958 | 0.44 |
| r7_bgebase | + bge-base embeddings | 0.792 | 0.958 | 0.32 |
| p2_flat | prose tables, flat | 0.708 | 0.958 | 0.52 |
| p5_expand | + query expansion | 0.708 | 0.917 | 0.56 |
| p6_rerank | + reranker | 0.833 | 0.917 | 0.48 |
| p7_bgebase | + bge-base | 0.833 | 0.917 | 0.40 |

Synthetic stayed at recall@1 0.818 / recall@4 1.0 for r0–r3.
Reranker query time ≈ 2.0 s vs 0.02 s without.

### 4.6 Bill router

`app/core/bill_router/weights/router_metrics.json`. 480 synthetic training pages,
0 real pages. On 120 pages in held-out fonts: page accuracy 0.983, AUC 1.0,
tile accuracy 0.935. 2 handwritten pages read as printed, 0 printed read as
handwritten, 1 uncertain page.

## 5. Decisions and rationale

| Decision | Why | Evidence |
| --- | --- | --- |
| `/v2` chunks in tokens, 128 / 24 | Best recall@1 on the real PDF (0.75 vs 0.58 for chars250) without tok256's extra confidently-wrong answers | 4.1 |
| Exact-match table rules before retrieval | Caps live in tables; an exact row lookup cannot hallucinate, and judgment via retrieval stayed at 0.50–0.56 | 4.1, 4.2 |
| Docling for policy PDFs, Tesseract cross-check | Plain Tesseract recovers no tables; Docling-default gets every rule. Cross-check catches number merges | 4.2 |
| Table-aware chunks on for travel policies | Raised limit accuracy 0.75 → 0.875 and cut confidently-wrong 3 → 2 on the company PDF | 4.3 |
| Reranker and bge-base not adopted by default | Reranker +0.08–0.13 recall@1 but −0.04 recall@4 and ~100× query latency; bge-base adds nothing over bge-small | 4.5 |
| FinanceBench demoted to a retrieval stress test only (2026-11-04) | Different task (10-K numeric reasoning vs policy-rule lookup); 33 of 40 questions don't fit the rule schema; its signal is opposite to the travel PDF's | 4.4, 4.5 |
| ConditionalQA adopted as the next benchmark (2026-11-04) | Short policy-like gov.uk pages, answers that hold only under conditions, unanswerable questions, hand-labelled gold evidence | 6 |
| Branching policy (2026-11-04) | One branch per major change or evaluation line, not per experiment, so history shows when and why each decision was made | — |

## 6. ConditionalQA evaluation (in progress)

Dataset: ConditionalQA dev (Sun et al., ACL 2022), built by
`eval/build_conditionalqa_subset.py`. 15 gov.uk pages, 68 questions:
37 unconditional, 27 conditional, 4 unanswerable.

Gold fields: `evidences` (scored as evidence phrases for recall@k / MRR),
`answers` as `[answer, [conditions]]`, `not_answerable`. The test split has no
public answers, so dev is used.

Scoring limits: retrieval is scored on all 68 questions. Judgment is scored only
on the 4 unanswerable ones; the other 64 have free-text answers and the judge
only checks `rule_type` / `limit_amount`.

Runs launched on branch `eval/conditionalqa`, existing pipeline unchanged:
- `cqa_chunk_study`: production, chars250, tok64, tok128, tok256.
- `cqa_retr_steps`: r0–r7, p2–p7.

Runs `20261104T120911Z_cqa_chunk_study` and `20261104T132509Z_cqa_retr_steps`,
completed 2026-11-05. 64 retrieval questions (27 conditional, 37 unconditional),
4 judged.

Metric caveat: a retrieval hit needs ALL gold evidence phrases in the top k
(`_has_all` in `eval/run_eval.py`). ConditionalQA questions have a median of 5
evidence sentences (max 17; only 8 of 64 have one). Recall@k here is strict and
favours larger chunks; it is not comparable to the travel-policy numbers.

Chunk study:

| Config | Leaves | recall@1 | recall@4 | recall@10 | recall@4+links | MRR | Ctx tokens@4 | recall@4 conditional | recall@4 unconditional |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tok256 | 220 | 0.328 | 0.563 | 0.906 | 1.000 | 0.486 | 592 | 0.370 | 0.703 |
| tok128 | 363 | 0.156 | 0.406 | 0.781 | 0.938 | 0.330 | 331 | 0.185 | 0.568 |
| production (tok128 + tables + structured) | 371 | 0.156 | 0.391 | 0.750 | 0.891 | 0.320 | 333 | 0.148 | 0.568 |
| chars250 | 736 | 0.156 | 0.250 | 0.344 | 0.578 | 0.225 | 162 | 0.111 | 0.351 |
| tok64 | 641 | 0.156 | 0.234 | 0.344 | 0.609 | 0.228 | 174 | 0.111 | 0.324 |

Retriever chain:

| Config | recall@1 | recall@4 | recall@10 | MRR | Query s | recall@4 conditional |
| --- | --- | --- | --- | --- | --- | --- |
| r0_baseline | 0.156 | 0.406 | 0.781 | 0.330 | 0.03 | 0.185 |
| r1_tables | 0.156 | 0.391 | 0.750 | 0.320 | 0.02 | 0.148 |
| r2_flat | 0.156 | 0.375 | 0.719 | 0.313 | 0.02 | 0.148 |
| r3_rowsum | 0.156 | 0.375 | 0.703 | 0.313 | 0.02 | 0.148 |
| r4_llmsum | 0.156 | 0.375 | 0.719 | 0.313 | 0.02 | 0.148 |
| r5_expand | 0.156 | 0.391 | 0.766 | 0.319 | 1.09 | 0.185 |
| r6_rerank | 0.188 | 0.344 | 0.750 | 0.323 | 7.60 | 0.111 |
| r7_bgebase | 0.188 | 0.344 | 0.750 | 0.323 | 1.89 | 0.111 |
| p2_flat | 0.156 | 0.391 | 0.750 | 0.319 | 0.02 | 0.185 |
| p5_expand | 0.156 | 0.406 | 0.813 | 0.326 | 0.03 | 0.222 |
| p6_rerank | 0.188 | 0.375 | 0.781 | 0.334 | 1.22 | 0.148 |
| p7_bgebase | 0.188 | 0.375 | 0.781 | 0.334 | 1.28 | 0.148 |

Unanswerable judgment (4 questions, one question = 0.25): accuracy 0.25–0.75
across configs; tok128 / tok256 / production / r0 reach 0.75 with 1 confidently
wrong; chars250 and tok64 have 0 confidently wrong.

Findings:
- Conditional questions retrieve far worse than unconditional ones in every
  config (recall@4 0.11–0.37 vs 0.32–0.70): the condition sentence is often
  not in the top k.
- No retriever step beats r0_baseline on recall@4. Reranker raises recall@1
  by one question (0.156 → 0.188) but lowers recall@4 and costs 1.2–7.6 s per query.
- Table-aware chunking does nothing useful here (gov.uk pages have few tables).
- tok256 leads, but the all-phrases metric rewards bigger chunks; a
  per-phrase recall is needed before changing the production chunk size.

## 7. Open gaps and next steps

- No judgment eval after the r4–r7 retriever changes; their effect on answer accuracy is unknown.
- Datasets are small (22–24 questions); one question moves a score by ~4 points.
- Confidently-wrong answers remain the main risk on the real PDF (2–6 per 18).
- Docling-default is accurate but slow (94 s); the fast Docling + Tesseract setup misses most rules.
- Bill router has never seen a real bill.
- The judge needs a free-text answer mode to score ConditionalQA answers.
- Pipeline code is uncommitted; a baseline commit is needed for results to trace to code.
