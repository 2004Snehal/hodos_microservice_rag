# Hodos RAG Microservice — RAPTOR & Multi-Hop Architecture

An enterprise-grade Retrieval-Augmented Generation (RAG) microservice built with **RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)**, **Multi-Hop Retrieval**, **Tuple-Keyed Caching**, and **Automated Document Scanning & Policy Audit**.

---

## Key Architectural Highlights

### 1. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
- **Hierarchical Tree Structuring**: Builds multi-level abstraction trees over documents. Leaf nodes contain granular text chunks, while upper-level parent nodes capture higher-level summary representations.
- **Gaussian Mixture Models (GMM) & BIC Selection**: Soft-clustering of document embeddings using GMM, with dynamic component count ($K$) selection optimized via Bayesian Information Criterion (BIC).
- **Extractive & Centroid Summarization**: Generates representative summary nodes for each cluster using sentence-level TF-IDF centroid scoring.
- **Collapsed-Tree Retrieval**: Simultaneously queries across all levels of the tree hierarchy (leaf and summary nodes) to capture both granular details and broad context.

### 2. Multi-Hop Retrieval Pipeline
- **Iterative Evidence Gathering**: Executes sequential retrieval hops to collect multi-clause policy rules across document sections.
- **TF-IDF Term-Coverage Scoring & Rocchio Expansion**: Dynamically checks evidence sufficiency and expands queries based on relevant pseudo-feedback terms.
- **RaptorSearchAdapter**: Seamlessly bridges hierarchical tree nodes with iterative multi-hop retrieval loops.

### 3. Automated Document Scanning & Policy Audit (`POST /audit`)
- **Dual-Path Extraction**: Direct PDF text-layer extraction for digital PDFs, falling back to offline Tesseract OCR for image-based documents.
- **Line Item & Category Parsing**: Parses itemized claims, currency amounts, dates, and vendors from uploaded bills.
- **Policy Compliance Verdicts**: Matches parsed claims against policy caps, generating automated `pass` / `fail` / `waived` verdicts with exact citations.

### 4. Tuple-Keyed Caching (`TupleCache`)
- **Key Normalization**: Hashes (category, grade, location, version) tuples into normalized SHA-256 cache keys.
- **Audit Log Trail**: Persists an append-only JSONL judgment log for complete audit traceability.

---

## Directory Structure

```
hodos-rag-microservice/
├── app/
│   ├── main.py                  # FastAPI application (/policy, /ask, /audit, /health, /v2/*, /ui)
│   ├── static/index.html        # Browser test console served at /ui
│   ├── core/                    # Core RAG, Document Extraction, Audit & Rule Engine
│   │   ├── simple_rag.py        # Vectorstore retrieval & answer generation
│   │   ├── document_extraction.py # PDF / OCR document reader
│   │   ├── bill_parser.py       # Bill line-item & expense parser
│   │   ├── audit_pipeline.py    # Policy compliance verification engine
│   │   ├── rule_extraction.py   # Pattern-matching policy rule engine
│   │   ├── chunking.py          # Sentence-aware chunking (pysbd + abbreviation merge)
│   │   ├── table_rules.py       # Parses policy tables into structured rule rows
│   │   ├── rule_store.py        # SQLite store + exact-match lookup for table rows
│   │   └── vectorstore.py       # Persistent vector database
│   ├── advanced/                # Advanced RAPTOR & Multi-Hop Architecture
│   │   ├── raptor/              # RaptorTree hierarchy & GMM/BIC clustering
│   │   ├── multihop/            # Iterative MultiHopRetriever
│   │   ├── caching/             # TupleCache & append-only audit logging
│   │   ├── integrated/          # Integrated RAPTOR + Multi-Hop + Cache Pipeline
│   │   └── probabilistic/       # /v2 pipeline: probabilistic RAPTOR, multi-hop, LLM judge
│   └── models/                  # Pydantic schemas
├── docs/                        # Technical documentation & architecture specifications
├── eval/                        # Gold datasets, policy sources, run_eval.py harness
├── metrics/                     # RAGAS benchmark metrics, eval_runs/ + eval_history.jsonl
└── scripts/                     # Demonstration & evaluation scripts
```

---

## Quickstart

### Installation & Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependency for OCR (Linux)
sudo apt-get install -y tesseract-ocr
```

### Running the Microservice API

```bash
# Start FastAPI service
uvicorn app.main:app --reload --port 8000
```

### Test UI

Open http://localhost:8000/ui. It's a single static page
(`app/static/index.html`, no build step) served by FastAPI, so it calls the API
from the same origin and needs no CORS setup. It has tabs for Policy, Ask, Audit
and Health, plus a v1/v2 switch. Results are rendered as answers, citations,
verdict tables, retrieval hops and parsed table rows, with the raw JSON
underneath. It's for manual testing only: no auth and no persistence.

### API Endpoints

1. **Ingest Policy Document (`POST /policy`)**
   ```bash
   curl -X POST http://localhost:8000/policy \
     -H "Content-Type: application/json" \
     -d '{
       "doc_id": "travel-policy-2026",
       "text": "Meal expenses are limited to Rs. 1500 per day. Transport expenses via taxi are limited to Rs. 1000 per day."
     }'
   ```

2. **Query Policy (`POST /ask`)**
   ```bash
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{
       "question": "What is the meal cap for travel?",
       "top_k": 3
     }'
   ```

3. **Audit Bill Document (`POST /audit`)**
   ```bash
   curl -X POST http://localhost:8000/audit \
     -F "file=@sample_bill.pdf"
   ```

---

## Chunking

`app/core/chunking.py` is used by all three pipelines. v1 (500/100) and
integrated (300/100) measure chunk size / overlap in **characters**. /v2
measures in **bge-small tokens**: target 128, overlap 24, ceiling 400 (bge-small
truncates at 512), and a final chunk under 32 tokens is merged into the
previous one. A chunk only
ends at a paragraph, sentence, clause (`; : ,`) or word boundary, never inside a
number. Overlap is made of whole trailing sentences. Once a chunk is half full,
it ends at the next paragraph break.

Sentence splitting uses **pysbd** (`clean=False`), which handles numbered lists
(`1. Meals ... 2. Lodging`), "No." as an answer, and lowercase sentence starts.
pysbd doesn't know policy/currency abbreviations, and on its own it splits
`capped at INR. 1000 per day` into `...INR.` / `1000 per day.`. So a split is
rejoined when the previous sentence ends in one of `_ABBREVIATIONS` (`rs`,
`inr`, `approx`, `max`, …) and the next one starts with a digit or a lowercase
letter.

Known gap: newlines inside a paragraph are collapsed, so tables become one line
in chunks. Table content is answered by the structured table rules (below),
not by the chunks.

---

## /v2 — Probabilistic RAPTOR + Multi-Hop (local LLM)

Runs fully locally: `qwen3:4b` through Ollama, bge-small embeddings through
fastembed, and Redis for the shared semantic cache. Code lives in
`app/advanced/probabilistic/`.

```bash
brew install ollama redis && brew services start ollama && brew services start redis
ollama pull qwen3:4b
uvicorn app.main:app --port 8000
python scripts/demo_probabilistic.py
```

| Endpoint | What it does |
|---|---|
| `POST /v2/policy` `{doc_id, text}` | Sentence-aware chunks → soft-clustered RAPTOR tree with LLM summaries |
| `POST /v2/ask` `{question, doc_id, category?, grade?, location?, version?, use_cache?}` | Semantic cache → multi-hop retrieval → structured rule with `confidence` and `status` |
| `POST /v2/audit` (multipart: `file`, `doc_id`, `grade?`, `location?`, `days?`) | Bill OCR/parsing, each category checked against `/v2` judgments |
| `GET /v2/health` | LLM, cache backend and ingested documents |

`confidence` = P(rule type) × P(evidence sufficient), both read from the
model's token probabilities. Results below 0.6 confidence, or whose numbers
don't appear in the cited policy text, come back as `status: needs_review`
(and audit verdicts as `needs_review`) instead of a hard pass/fail.

Config: `OLLAMA_URL`, `HODOS_LLM_MODEL`, `HODOS_REDIS_URL`,
`HODOS_SEMANTIC_THRESHOLD`, `HODOS_CACHE_TTL_SECONDS`.

### Structured table rules

`/v2/policy` also reads tables in the policy text (markdown pipes, tabs, or
columns separated by 2+ spaces) into rows stored in SQLite
(`app/data/probabilistic/table_rules.db`), separate from the trees:

```
{category: meal, grade: L3–L4, city_tier: metro, cap: 1800, unit: per_day}
```

Both long tables (`Category | Grade | City tier | Cap | Unit`) and wide ones
(`Grade | Metro | Non-metro`, with the category and unit taken from the
caption line above, or the nearest `#` heading for the category) are read.
Lines like `Metro cities: Mumbai, Delhi` and `Non-metro: all other cities`, and
table row labels like `Metros + Top cities: (Mumbai, Pune)` / `Rest of the
cities`, map `location` to a tier. The header row is the row with no amounts
and some role word, wherever it sits (some PDFs print it under the data). A
header naming two categories ("Own Accommodation Allowance (inclusive of
Meal Allowance)") becomes its own category. A lodging cap with no unit in a
hotel/room table is read as `per_night` and listed under `inferred` in the
ingest response; any other cap with no unit is skipped.

`/v2/ask` and `/v2/audit` check these rows first. When one row (or several that
agree) matches the request's `category`, `grade` and `location` exactly, the
answer comes straight from the row: `source: structured`,
`method: structured_table`, `confidence: 1.0`, with no retrieval or LLM call.
Anything less than an exact match falls through to the cache and multi-hop path,
and the reason is returned in `table_lookup`. Examples: grade or location left as
`any` when the table varies by it, rows that disagree, or a cell such as
"1800, more with approval". The ingest response lists the parsed rows and the
skipped cells. Pass `use_structured: false` to skip the lookup.

Code: `app/core/table_rules.py` (parsing) and `app/core/rule_store.py` (store and matching).

### Policy PDFs: `POST /v2/policy/file`

```bash
curl -X POST http://localhost:8000/v2/policy/file \
  -F "file=@policy.pdf" -F "doc_id=travel-policy-2026" -F "extractor=auto"
```

`extractor=auto` converts the file with Docling through a **docling-serve**
sidecar (the `docling` service in `docker-compose.yml`; locally
`.venv-docling/bin/docling-serve run --port 5001`, `HODOS_DOCLING_URL`), so
tables arrive as pipe tables, and falls back to Tesseract if it's down.
`docling` / `tesseract` force one. The Docling text is cross-checked against the
Tesseract / text-layer read: a number Docling merged across a line-break
hyphen (`7-10` → `710`, [docling#4151](https://github.com/docling-project/docling/issues/4151))
is restored, and any other number missing from the Tesseract read is listed in
`extraction.number_check.unverified_numbers` with a warning.

Tables are chunked by `app/core/chunking.py:chunk_document`: each table becomes
its own leaves, rows written as `Column: value` sentences, prefixed with the
table's caption, section path and any sentence elsewhere that mentions
"Table N"; such sentences are linked to the table's leaves, and multi-hop
retrieval follows those links.

---

## Bill routing: digital vs printed vs handwritten

Both `/audit` and `/v2/audit` read bills through `app/core/document_extraction.py`:

```
PDF page with a text layer ──────────────────────────────► direct text
image / scanned page ──► CNN router: P(handwritten)
      handwritten ─────────────────────────────────────────► VLM (Ollama)  [no VLM: Tesseract + warning]
      printed / uncertain ──► Tesseract ──► confidence < 60 ► VLM
```

- **Router** (`app/core/bill_router/`): ImageNet ResNet-50 (ONNX via fastembed, no
  PyTorch) features on 224×224 ink-dense tiles + a logistic-regression head; tile
  log-odds are averaged per page. ~0.2–0.4 s per page on CPU.
- **Training**: `python scripts/train_bill_router.py [--real-dir DIR]` — synthetic
  bills in macOS handwriting vs printed fonts, with identical scan degradations for
  both classes; some fonts are held out to measure generalization. Put real bills in
  `DIR/handwritten/` and `DIR/printed/` to train and evaluate on real data.
  Metrics are written to `app/core/bill_router/weights/router_metrics.json`.
- **VLM**: `ollama pull qwen3-vl:4b` (or set `HODOS_VLM_MODEL`). It returns
  structured line items that feed the same bill parser as OCR output.
- Every audit response includes `page_routes` (route, method, `p_handwritten`,
  OCR confidence, notes).

---

## Running Advanced RAPTOR Pipeline Directly

You can execute the RAPTOR + Multi-Hop + Cache integrated pipeline programmatically:

```bash
python scripts/demo_advanced.py
```

```python
from app.advanced.integrated.pipeline import AdvancedRagPipeline
from pathlib import Path

pipeline = AdvancedRagPipeline(cache_dir=Path("/tmp/hodos_cache"))
pipeline.build_tree_for_document("travel-policy-2026", policy_text)

result = pipeline.ask(
    question="what is the meal cap for L3 employees in Mumbai?",
    doc_id="travel-policy-2026",
    category="meal",
    grade="L3",
    location="Mumbai",
    version="v1",
    category_keywords=["meal", "food", "dining"],
)

print(result)
```

---

## Evaluation

`eval/run_eval.py` scores the /v2 pipeline against gold datasets and stores
every step of every run.

```bash
.venv/bin/python -m eval.run_eval                                   # all datasets, production config
.venv/bin/python -m eval.run_eval -d company_xx_travel_india -c tok128 -c tok256
.venv/bin/python -m eval.run_eval --no-llm                          # chunking + retrieval only
```

| Path | Contents |
|---|---|
| `eval/policies/` | Policy sources. `company_xx_travel_india.pdf` is an image-only scan; `.ocr.txt` is its Tesseract text from `app/core/document_extraction.py`, which is what gets ingested. |
| `eval/datasets/*.json` | Gold questions: evidence phrases (retrieval) and expected `rule_type` / `limit_amount` (judgment). Gold for the PDF was read from the page images, not the OCR. |
| `metrics/eval_runs/<time>_<label>.json` | Full run: environment (git commit, models), per-config chunk/tree stats, and per-question retrieval ranks, LLM answer, confidence, hops and warnings. |
| `metrics/eval_history.jsonl` | One summary line per (dataset, config), appended by every run. |

Steps per config: **chunk** (leaf sizes) → **tree** (RAPTOR build) →
**retrieval** (recall@1/2/4 of the gold phrases, MRR) → **judgment**
(accuracy, confidence, `confidently_wrong` = wrong but `status: ok`).

---

## Technical Specifications & Documentation

- [RAPTOR & Multi-Hop Technical Spec](docs/HODOS_RAPTOR_MULTIHOP_TECHNICAL_DOCS.md)
- [System Architecture Overview](docs/ARCHITECTURE.md)
- [RAGAS Benchmark Metrics](docs/benchmarks.md)
