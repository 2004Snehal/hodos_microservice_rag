# System Architecture

## Architecture Overview

The **Hodos RAG Microservice** provides an end-to-end platform for policy document ingestion, retrieval-augmented question answering, automated document scanning, and compliance auditing.

The system integrates four primary technical pillars:
1. **Core RAG Service** (`POST /policy`, `POST /ask`)
2. **Document Audit & Extraction Engine** (`POST /audit`)
3. **RAPTOR Hierarchical Tree Indexing** (`app/advanced/raptor/`)
4. **Multi-Hop Retrieval & Tuple Caching** (`app/advanced/multihop/`, `app/advanced/caching/`)

---

## 1. Core RAG Pipeline (`POST /policy`, `POST /ask`)

```
POST /policy                          POST /ask
      │                                   │
      ▼                                   ▼
Policy Text                          Question Text
      │                                   │
      ▼                                   ▼
Chunking Engine                     TF-IDF Vectorizer
(Fixed-size overlap)                      │
      │                                   ▼
      ▼                          Cosine Similarity Scan
Vector Database                           │
(JSON + NumPy store)                      ▼
                                 Extractive Answer & Citations
```

- **Chunking**: Uses fixed character windowing (`chunk_size=500`, `overlap=100`) to partition policies into retrievable chunks.
- **Embeddings**: Uses TF-IDF vectorization for lightweight, fast vector generation.
- **Vector Database**: Persists chunk texts, NumPy embedding matrices, and vectorizer models under `app/data/vector_store/`.

---

## 2. Document Audit & Scanning Engine (`POST /audit`)

```
Uploaded Bill (PDF / Image)
            │
            ▼
Document Extraction (`document_extraction.py`)
  - PDF Text Layer Detection -> Direct Extraction
  - Rasterized Pages / Images -> Tesseract OCR
            │
            ▼
Expense Line-Item Parsing (`bill_parser.py`)
  - Regex amount extraction & keyword category classification
  - Metadata identification (Vendor, Date)
            │
            ▼
Audit Execution (`audit_pipeline.py`)
  - Group expense totals by category
  - Retrieve target policy chunk from Vector Database
  - Compare claimed total vs. policy limit
            │
            ▼
Audit Result (Verdict, Claims, Policy Citations)
```

- **Dual Extraction**: Inspects PDF pages for native text layers for instant extraction, routing image pages to offline Tesseract OCR.
- **Line Item Parsing**: Identifies categories (`meal`, `lodging`, `transport`) and amounts from raw invoice text.
- **Rule Verification**: Compares extracted claimed totals against policy thresholds and returns structured verdicts (`pass`, `fail`, `waived`).

---

## 3. RAPTOR & Multi-Hop Architecture

```
app/
├── advanced/
│   ├── raptor/
│   │   ├── raptor.py               # RaptorTree (GMM + BIC clustering)
│   │   └── extractive_summarizer.py # Centroid-based summary generator
│   ├── multihop/
│   │   └── multihop.py             # MultiHopRetriever (iterative hop loop)
│   ├── caching/
│   │   └── cache.py                # TupleCache (normalized tuple caching)
│   └── integrated/
│       └── pipeline.py             # AdvancedRagPipeline & RaptorSearchAdapter
```

### RAPTOR Indexing
- Constructs a recursive tree hierarchy over policy documents.
- Performs Gaussian Mixture Model (GMM) soft clustering at each tree depth level.
- Selects cluster count ($K$) dynamically using Bayesian Information Criterion (BIC).
- Generates summary nodes using sentence-level TF-IDF centroid scoring.

### Multi-Hop Retrieval
- Iteratively expands queries to locate multi-clause policy terms across different sections.
- Evaluates evidence sufficiency using TF-IDF term coverage and Rocchio query expansion.

---

## 4. API Reference

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/policy` | POST | `{ "doc_id": "...", "text": "..." }` | Chunking & ingestion metrics |
| `/ask` | POST | `{ "question": "...", "top_k": 4 }` | Answer text & citations |
| `/audit` | POST | Multipart file (`.pdf`, `.png`, `.jpg`) | Per-category verdicts & policy citations |
| `/health` | GET | None | Microservice status & store statistics |
