# RAG Evaluation & Benchmarks

This directory details the benchmark evaluation methodology and metrics comparing retrieval configurations across the system stages.

Evaluation metrics are recorded in `metrics/ragas_benchmark_results.json`.

---

## Evaluation Metrics

| Metric | Measurement Focus | Significance |
|---|---|---|
| `context_recall` | Completeness of retrieved context chunks required for accurate answers | Evaluates retrieval coverage across multi-clause policy questions. |
| `context_precision` | Ratio of relevant chunks relative to total retrieved context | Measures signal-to-noise ratio across multi-hop iterations. |
| `faithfulness` | Groundedness of generated answers against retrieved context | Ensures generated answers strictly adhere to retrieved policy clauses. |
| `answer_relevancy` | Direct alignment of the response to the target user question | Evaluates response clarity and intent satisfaction. |

---

## Architecture Stage Comparison

```
Test Question Set
       │
       ├─► Stage 1: Simple RAG (Single-pass retrieval)
       ├─► Stage 2: Multi-Hop RAG (Iterative query expansion)
       ├─► Stage 3: Multi-Hop + RAPTOR Tree (Hierarchical tree retrieval)
       └─► Stage 4: Multi-Hop + RAPTOR + Tuple Cache (Cached latency optimization)
```

---

## Benchmarking Setup

1. **Test Set Construction**: Frozen test suite covering single-clause lookups, multi-clause syntheses, cross-reference resolutions, and thematic summaries.
2. **Execution**: Evaluate each pipeline stage using RAGAS evaluation metrics (`context_recall`, `context_precision`, `faithfulness`, `answer_relevancy`).
3. **Results Tracking**: Benchmark metrics recorded under `metrics/ragas_benchmark_results.json`.
