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
│   ├── main.py                  # FastAPI application (/policy, /ask, /audit, /health)
│   ├── core/                    # Core RAG, Document Extraction, Audit & Rule Engine
│   │   ├── simple_rag.py        # Vectorstore retrieval & answer generation
│   │   ├── document_extraction.py # PDF / OCR document reader
│   │   ├── bill_parser.py       # Bill line-item & expense parser
│   │   ├── audit_pipeline.py    # Policy compliance verification engine
│   │   ├── rule_extraction.py   # Pattern-matching policy rule engine
│   │   ├── chunking.py          # Fixed-size overlap chunking
│   │   └── vectorstore.py       # Persistent vector database
│   ├── advanced/                # Advanced RAPTOR & Multi-Hop Architecture
│   │   ├── raptor/              # RaptorTree hierarchy & GMM/BIC clustering
│   │   ├── multihop/            # Iterative MultiHopRetriever
│   │   ├── caching/             # TupleCache & append-only audit logging
│   │   └── integrated/          # Integrated RAPTOR + Multi-Hop + Cache Pipeline
│   └── models/                  # Pydantic schemas
├── docs/                        # Technical documentation & architecture specifications
├── metrics/                     # RAGAS benchmark metrics & comparison analysis
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

## Technical Specifications & Documentation

- [RAPTOR & Multi-Hop Technical Spec](docs/HODOS_RAPTOR_MULTIHOP_TECHNICAL_DOCS.md)
- [System Architecture Overview](docs/ARCHITECTURE.md)
- [RAGAS Benchmark Metrics](docs/benchmarks.md)
