# RAPTOR + Multi-Hop Advanced Architecture

This document details the design and implementation of the **RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)**, **Multi-Hop Retrieval**, and **Tuple-Keyed Caching** system inside `app/advanced/`.

---

## Architectural Overview

The advanced RAG pipeline combines hierarchical tree indexing with multi-step iterative retrieval:

```
                          User Query
                              │
                              ▼
                     ┌──────────────────┐
                     │    TupleCache    │
                     └────────┬─────────┘
                              │
                    Cache Hit │ Cache Miss
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Return Judgment    MultiHopRetriever
                                        │
                                        ▼
                               RaptorSearchAdapter
                                        │
                                        ▼
                                   RaptorTree
                           (GMM + BIC + Summaries)
                                        │
                                        ▼
                                Rule Extraction Engine
                                        │
                                        ▼
                               Write Cache & Return
```

---

## Module Breakdown

### 1. `app/advanced/raptor/raptor.py` (`RaptorTree`)
- **GMM Soft Clustering**: Clusters document embeddings using Gaussian Mixture Models, enabling soft membership across multiple semantic concepts.
- **BIC-Based Cluster Selection**: Evaluates candidate cluster counts ($K$) and selects the optimal model minimizing the Bayesian Information Criterion (BIC).
- **Centroid-Based Summarization**: Summarizes cluster contents via sentence-level TF-IDF similarity to the cluster centroid (`extractive_summarizer.py`).
- **Collapsed-Tree Search**: Performs similarity search across all tree depth levels simultaneously, retrieving both granular leaf passages and high-level summaries.

### 2. `app/advanced/multihop/multihop.py` (`MultiHopRetriever`)
- **Iterative Hop Loop**: Executes multi-stage retrieval to collect context across scattered policy sections.
- **Sufficiency Check & Rocchio Expansion**: Tracks TF-IDF term coverage of the question keywords in gathered evidence, applying Rocchio-style query expansion on subsequent hops.

### 3. `app/advanced/caching/cache.py` (`TupleCache`)
- **Normalized Key Generation**: Hashes `(category, grade, location, version)` tuples into deterministic SHA-256 keys.
- **Audit Log Persistence**: Appends every computed verdict to a persistent `judgment_log.jsonl` audit file.

### 4. `app/advanced/integrated/pipeline.py` (`AdvancedRagPipeline`)
- **Integrated Pipeline**: Connects `TupleCache`, `RaptorTree`, `RaptorSearchAdapter`, `MultiHopRetriever`, and `rule_extraction` into a unified execution flow.

---

## Running the Advanced Pipeline

Execute the demonstration script to verify end-to-end functionality:

```bash
python scripts/demo_advanced.py
```
