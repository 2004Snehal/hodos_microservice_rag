# Hodos RAG: RAPTOR + Multi-Hop — Technical Documentation

**Scope**: This document specifies both the high-level architecture and the
low-level implementation details for combining RAPTOR (tree-organized
retrieval) with multi-hop reasoning in the Hodos system. Every parameter
attributed to the original RAPTOR paper (Sarthi et al., ICLR 2024) is cited
directly from the uploaded source PDF, not from memory — where a value is a
Hodos-specific engineering decision rather than something the paper
specifies, it's marked as such explicitly.

---

# PART 1 — HIGH-LEVEL ARCHITECTURE

## 1.1 What problem this solves

Two distinct retrieval failure modes, addressed by two distinct mechanisms:

| Failure mode | Mechanism that addresses it |
|---|---|
| An answer requires chaining facts across documents/sections that don't share vocabulary (general rule → grade override → regional exception) | **Multi-hop retrieval** — iterative retrieve-reason-reformulate loop |
| An answer requires synthesizing across levels of abstraction (a precise clause vs. a whole document's theme) — "most existing methods retrieve only short contiguous chunks... limiting holistic understanding of the overall document context" (RAPTOR paper, Abstract) | **RAPTOR** — recursive clustering + summarization into a tree, queried across all levels at once |

Combined: multi-hop decides *what* to look for across sequential steps;
RAPTOR decides *at what level of abstraction* to look for it within each
step's retrieval call.

## 1.2 System diagram (high level)

```
                         ┌─────────────────────────┐
                         │   Policy corpus (docs)   │
                         └────────────┬─────────────┘
                                      │ (build time — rare, on doc change)
                                      v
                    ┌──────────────────────────────────┐
                    │   RAPTOR TREE CONSTRUCTION        │
                    │   chunk → embed → cluster →       │
                    │   summarize → recurse             │
                    └────────────────┬───────────────────┘
                                     │ persisted tree
                                     v
  Question ──> ┌───────────────────────────────────────────┐
                │           MULTI-HOP REASONING LOOP          │
                │  hop 1: query tree (collapsed retrieval) │
                │  hop 2: reformulate query using hop 1     │
                │         evidence, query tree again        │
                │  ...repeat until sufficiency or max_hops   │
                └───────────────────┬─────────────────────────┘
                                    │ accumulated evidence
                                    v
                         ┌───────────────────┐
                         │  Cache check/write │  (keyed by resolved
                         │  (tuple-keyed)      │   scenario, not raw query)
                         └─────────┬───────────┘
                                   v
                         ┌───────────────────┐
                         │  Answer synthesis   │
                         │  (LLM, cited)       │
                         └───────────────────┘
```

## 1.3 Design principles carried through both layers

1. **Retrieval quality and cost are separate axes.** RAPTOR/multi-hop
   improve *what gets retrieved*; caching improves *how often you pay to
   retrieve it again*. Never conflate a cache hit rate with a retrieval
   quality metric (see `docs/benchmarks.md` pattern established earlier).
2. **Provenance is non-negotiable.** Every answer must cite the specific
   tree node(s) — leaf or summary — used to produce it, plus which layer
   they came from. This is what makes a summary node's potential
   hallucination (see §2.4.5) auditable after the fact rather than silently
   trusted.
3. **The tree is built rarely; it is queried constantly.** Document-driven
   build events and claim-driven query events are separate lifecycles
   with separate triggers — conflating them (rebuilding on every query)
   defeats the entire cost model.

---

# PART 2 — LOW-LEVEL COMPONENT SPECIFICATION

## 2.1 Leaf-level chunking

**Paper's specification** (§3, Methods): "segmenting the retrieval corpus
into short, contiguous texts of length 100 [tokens]... If a sentence
exceeds the 100-token limit, we move the entire sentence to the next
chunk, rather than cutting it mid-sentence."

**Implementation requirements:**
- Chunk target: **100 tokens** (paper's value — not characters; tokenize
  first, then window over tokens).
- **Never split a sentence mid-way.** If adding the next sentence would
  exceed 100 tokens, close the current chunk short and start a new chunk
  with that full sentence. This is a hard rule, not a soft preference —
  the paper attributes RAPTOR's coherence partly to this.
- No overlap is specified in the paper for leaf chunking (unlike Hodos's
  earlier basic-RAG overlap chunking, which is a different, simpler
  strategy used for the non-RAPTOR path). Do not carry the overlap
  parameter over into RAPTOR's leaf layer unless you have a specific reason
  to.

**Threshold table entry:**
| Parameter | Value | Source |
|---|---|---|
| Leaf chunk target length | 100 tokens | Paper §3 |
| Mid-sentence split allowed | No | Paper §3 |
| Overlap between leaf chunks | Not specified/used | Paper §3 (silent — treat as 0) |

## 2.2 Embedding model

**Paper's specification** (§3): SBERT, specifically
**`multi-qa-mpnet-base-cos-v1`** (Reimers & Gurevych, 2019) for all leaf and
summary node embeddings. "All nodes" are embedded with SBERT — including
DPR/BM25 baseline comparisons in Tables 1–2, but SBERT is the one RAPTOR
uses in its main pipeline: "Since RAPTOR with SBERT has the best
performance, we use it in all subsequent experiments" (§4).

**Implementation requirement:** leaf embeddings and summary-node embeddings
**must share the same embedding space** (same model, same fitted
parameters where applicable) — this is what makes a leaf and a summary node
directly comparable in collapsed-tree retrieval (§2.6). If you substitute a
different embedding model (e.g., an API-based embedding service instead of
a locally run SBERT model), the requirement carries over unchanged: one
consistent embedding space across every tree level.

**Threshold table entry:**
| Parameter | Value | Source |
|---|---|---|
| Embedding model | SBERT `multi-qa-mpnet-base-cos-v1` | Paper §3 |
| Embedding consistency requirement | Same model/space for leaf AND summary nodes | Paper §3, Fig. 1 |

## 2.3 Clustering algorithm

This is the most parameter-dense part of RAPTOR. Specify each piece
precisely — a partial implementation (e.g., GMM without the UMAP
dimensionality reduction, or without BIC-based cluster count selection) is
not "RAPTOR-lite," it's a materially different clustering behavior.

### 2.3.1 Why not raw high-dimensional clustering

**Paper's reasoning** (§3, Clustering Algorithm): "The high dimensionality
of vector embeddings presents a challenge for traditional GMMs, as distance
metrics may behave poorly when used to measure similarity in
high-dimensional spaces (Aggarwal et al., 2001)."

### 2.3.2 Dimensionality reduction: UMAP

**Paper's specification:** Uniform Manifold Approximation and Projection
(McInnes et al., 2018) is applied before clustering. The key parameter is
`n_neighbors`, which "determines the balance between the preservation of
local and global structures."

**Two-step application** (this is a specific, load-bearing detail — do not
skip it): "Our algorithm varies `n_neighbors` to create a hierarchical
clustering structure: it first identifies global clusters and then
performs local clustering within these global clusters."

Implementation shape:
```
1. Run UMAP with a LARGER n_neighbors  -> reveals GLOBAL cluster structure
2. Within each global cluster, run UMAP again with a SMALLER n_neighbors
   -> reveals LOCAL cluster structure within that global grouping
3. Cluster (GMM, see below) is applied on the locally-reduced embeddings
```
The paper does not give exact numeric values for the large/small
`n_neighbors` pair — this is a tuning parameter you must set per corpus
size. **Hodos-specific recommendation** (not from the paper): start with
`n_neighbors ≈ sqrt(N)` for the global pass and roughly a third to a
quarter of that for the local pass, where N is the number of nodes at the
current tree level, then tune against your own retrieval eval set.

### 2.3.3 Clustering: Gaussian Mixture Models, soft clustering

**Paper's specification:** GMMs, not k-means or hard clustering. Critically,
this is **soft clustering — a node CAN belong to multiple clusters**: "nodes
can belong to multiple clusters without requiring a fixed number of
clusters. This flexibility is essential because individual text segments
often contain information relevant to various topics, thereby warranting
their inclusion in multiple summaries."

This is a meaningful departure from a naive KMeans implementation (hard,
single-cluster assignment) — **a leaf chunk discussing both "meal
reimbursement" and "international travel exceptions" should be eligible to
appear in both the meal-cluster's summary and the international-travel
cluster's summary**, not forced into exactly one.

The math (paper's own notation):
```
P(x|k) = N(x; μ_k, Σ_k)                    — likelihood under Gaussian k
P(x)   = Σ_{k=1}^{K} π_k · N(x; μ_k, Σ_k)   — overall mixture, π_k = mixture weight
```

### 2.3.4 Selecting the number of clusters: BIC

**Paper's specification:** Bayesian Information Criterion, not a fixed or
manually chosen K:
```
BIC = ln(N)·k − 2·ln(L̂)
```
where N = number of text segments at this level, k = number of model
parameters (a function of embedding dimensionality × number of clusters),
L̂ = maximized likelihood. BIC "penalizes model complexity but also rewards
goodness of fit" — sweep candidate K values, compute BIC for each, select
the K that minimizes it. Then fit final GMM parameters (means, covariances,
mixture weights) via Expectation-Maximization.

### 2.3.5 Recursive re-clustering within a cluster (token overflow handling)

**Paper's specification** (§3): "Should a local cluster's combined context
ever exceed the summarization model's token threshold, our algorithm
recursively applies clustering within the cluster, ensuring that the
context remains within the token threshold."

This is a distinct recursion from the main tree-building recursion (§2.5)
— it's a **width-control mechanism at a single level**, triggered purely by
token budget, not by the main termination condition.

**Threshold table entry:**
| Parameter | Value | Source |
|---|---|---|
| Clustering algorithm | Gaussian Mixture Model (soft, multi-membership) | Paper §3 |
| Dimensionality reduction | UMAP, two-pass (global n_neighbors, then local n_neighbors within each global cluster) | Paper §3 |
| Cluster count selection | BIC, minimized over candidate K | Paper §3 |
| Parameter estimation | Expectation-Maximization | Paper §3 |
| Re-cluster trigger | Cluster's combined token count exceeds summarization model's token limit | Paper §3 |
| Global/local n_neighbors exact values | Not specified — tune per corpus (Hodos recommendation: start global ≈ √N, local ≈ global/3–4) | Not in paper |

## 2.4 Summarization (parent node creation)

### 2.4.1 Model used

**Paper's specification:** `gpt-3.5-turbo` for all cluster summarization in
the main experiments (§3, Model-Based Summarization).

### 2.4.2 Prompt (verbatim from paper, Appendix D, Table 11)

| Role | Content |
|---|---|
| system | You are a Summarizing Text Portal |
| user | Write a summary of the following, including as many key details as possible: {context}: |

Use this as the literal starting point for the summarization prompt if you
want to reproduce the paper's reported behavior/quality — deviating from it
(e.g., adding structure requirements) changes summary characteristics from
what the paper's hallucination/compression measurements were taken against.

### 2.4.3 Compression ratio — the target to sanity-check against

**Paper's measured statistics** (Appendix C, Table 10), averaged across all
three of their eval datasets:
| Metric | Value |
|---|---|
| Avg. summary length | 131 tokens |
| Avg. child node length | 85.6 tokens |
| Avg. # child nodes per parent | 6.7 |
| Avg. compression ratio | 0.28 (i.e., ~72% compression) |

**Use this as a health check on your own summarizer**, not a hard
constraint to enforce: if your cluster summaries are coming out
dramatically longer than ~28% of their combined child length, the
summarization prompt or model may not be compressing effectively, and
downstream tree levels will bloat token cost faster than the paper's
reported linear scaling (§2.7).

### 2.4.4 Cluster size — the number to expect

Table 10 also reports **avg. 6.7 child nodes per parent** across datasets
(range 5.7–6.8 by dataset) — this is a useful sanity range for your BIC-selected
cluster sizes (§2.3.4); clusters wildly outside this range across many
nodes may indicate a UMAP/BIC parameter mismatch for your corpus.

### 2.4.5 Hallucination rate — measured, not assumed

**Paper's finding** (Appendix E): a hand-annotated sample of 150 nodes
across 40 stories found **4% (6 nodes) contained a hallucination** —
typically the model "adding minor information possibly from its training
data... or incorrectly extrapolating." Critically: **"hallucinations did
not propagate to higher layers"** and **"had no discernible impact on
question-answering tasks"** in their evaluation.

**What this means for Hodos, precisely:**
- A ~4% per-summary-node hallucination rate is the empirically observed
  baseline for `gpt-3.5-turbo` summarization under this exact prompt — not
  zero, and not something to design around as if it were negligible in a
  compliance-sensitive setting like expense audit, even though the paper
  found no propagation or QA-task impact in their own literary-text
  evaluation.
- This is exactly why Hodos's leaf pull-through requirement (§2.6.3) exists:
  even if hallucination doesn't propagate to parent summaries structurally
  (per the paper's finding), a summary node used as retrieval evidence
  should still have its source leaves available for the reasoning step to
  cross-check against, particularly for a domain (financial/compliance
  audit) with a much lower error tolerance than literary QA.

**Threshold table entry:**
| Parameter | Value | Source |
|---|---|---|
| Summarization model | gpt-3.5-turbo | Paper §3 |
| Summarization prompt | See Table above (Appendix D) | Paper Appendix D |
| Target compression ratio | ~0.28 (health-check, not hard constraint) | Paper Appendix C |
| Expected child nodes per parent | ~5.7–6.8 (dataset-dependent) | Paper Appendix C |
| Observed hallucination rate | 4% (6/150 sampled nodes) | Paper Appendix E |
| Hallucination propagation to parents | Not observed in paper's sample | Paper Appendix E |

## 2.5 Tree construction — recursion and termination

**Paper's specification** (§3): "the cycle of embedding, clustering, and
summarization continues until further clustering becomes infeasible,
resulting in a structured, multi-layered tree representation."

**Practical termination conditions** (paper does not give one universal
numeric floor; infer from BIC + re-clustering behavior):
- Stop recursing upward when BIC selects K=1 for the current level's node
  set (i.e., no meaningful further grouping exists — everything remaining
  is one cluster, which becomes the root).
- A level with very few nodes (below whatever minimum your BIC sweep can
  meaningfully evaluate) should be treated as the final level rather than
  forced through another clustering pass — this mirrors the general
  principle (not the paper's exact wording, but consistent with its
  "until further clustering becomes infeasible" framing).

**Threshold table entry:**
| Parameter | Value | Source |
|---|---|---|
| Recursion termination | "Until further clustering becomes infeasible" (BIC selects K=1, or too few nodes remain to cluster meaningfully) | Paper §3 (qualitative); numeric floor is an implementation choice |

## 2.6 Retrieval: tree traversal vs. collapsed tree

### 2.6.1 Two strategies, paper tested both

**Tree traversal** (Algorithm 1, Appendix F): start at root layer, compute
cosine similarity to query, take top-k, descend to those nodes' children,
repeat for `d` layers, concatenate `S_1 ∪ S_2 ∪ ... ∪ S_d`.

**Collapsed tree** (Algorithm 2, Appendix F): flatten every node from every
layer into one set, compute cosine similarity against the query for all of
them simultaneously, take top nodes **by token budget** (not fixed k) until
a `max_tokens` threshold is reached.

### 2.6.2 The paper's own conclusion: use collapsed tree

**Empirical result** (§3, Querying, and Fig. 3): "The collapsed tree
approach consistently performs better... offering greater flexibility than
tree traversal... it retrieves information that is at the correct level of
granularity for a given question. In comparison, while using tree traversal
with the same values of d and k, the ratio of nodes from each level of the
tree will be constant. So, the ratio of higher-order thematic information
to granular details will remain the same regardless of the question."

**This is the load-bearing finding that justifies Hodos's design choice**
(established in earlier conversation) to use collapsed-tree-style retrieval
for Hop 1 rather than layer-by-layer descent — it's not an arbitrary
simplification, it's the strategy the paper's own controlled comparison
favored.

### 2.6.3 Collapsed tree's specific parameters used in the paper's main results

**Paper's specification** (§3, end of Querying section): "we use the
collapsed tree with **2000 maximum tokens, which approximately equates to
retrieving the top-20 nodes**." For UnifiedQA specifically (max context
length 512 tokens), they instead cap at **400 tokens** of retrieved context
to leave room for the question and model overhead. "We provide the same
amount of tokens of context to RAPTOR and to the baselines" — i.e., the
token budget is held constant across compared methods, not tuned
per-method.

**Threshold table entry:**
| Parameter | Value | Source |
|---|---|---|
| Preferred retrieval strategy | Collapsed tree (not layer-by-layer traversal) | Paper §3, empirically justified |
| Token budget (general/GPT-based reader) | 2000 tokens (≈ top-20 nodes) | Paper §3 |
| Token budget (UnifiedQA reader, 512 max context) | 400 tokens | Paper §3 |
| Selection rule within budget | Add top-scoring nodes by cosine similarity until token budget is reached, not a fixed node count | Paper Algorithm 2 (Appendix F) |
| Efficiency note at scale | Collapsed tree requires similarity search over ALL nodes — paper notes FAISS as the mitigation for cost at scale (Johnson et al., 2019) | Paper §3, footnote after Fig. 3 |

### 2.6.4 Which layers actually get used — don't assume it's mostly leaves

**Paper's finding** (Appendix I.2, Table 18): across datasets and
retrievers, **18.5%–57% of retrieved nodes in collapsed-tree retrieval come
from non-leaf layers.** With DPR as the retriever, NarrativeQA saw 57.36%
of retrieved nodes come from non-leaf layers — meaning more than half the
evidence used for the literary comprehension task was summary text, not
original passages.

**Layer contribution breakdown with SBERT** (Table 20, the retriever RAPTOR
actually uses in its main results):
| Dataset | Layer 0 (leaf) | Layer 1 | Layer 2 | Layer 3 |
|---|---|---|---|---|
| NarrativeQA | 63.22% | 31.51% | 4.85% | 0.42% |
| QuALITY | 75.59% | 22.78% | 1.63% | — |
| QASPER | 81.51% | 17.84% | 0.65% | — |

**Practical implication for Hodos:** do not assume the leaf layer alone
would suffice, or that summary layers are a rarely-hit fallback — with
SBERT, roughly a fifth to a third of retrieved evidence typically comes
from Layer 1 alone. This is direct empirical support for the leaf
pull-through requirement (bring the underlying leaves along whenever a
summary node is retrieved) — a meaningful fraction of your answers will be
resting partly on summary text.

### 2.6.5 Full-tree search beats any single-layer restriction

**Paper's finding** (§4.1, Table 8, for one QuALITY story): restricting
retrieval to specific layers underperforms full-tree collapsed retrieval:

| Layers queried | Start: Layer 0 (leaf) | Start: Layer 1 | Start: Layer 2 |
|---|---|---|---|
| 1 layer | 57.9 | 57.8 | 57.9 |
| 2 layers | — | 52.6 | 63.15 |
| 3 layers (full tree) | — | — | **73.68** |

Querying all 3 layers together (73.68 accuracy) clearly beats any
single-layer-only or two-layer subset shown here. **Threshold implication:
do not artificially restrict retrieval to "just leaves" or "just the top
layer" for cost reasons without expecting a real accuracy cost** — this
table is the direct evidence for that tradeoff, for this specific story;
the paper reports it varies by story (see Tables 15–17 in Appendix I for
additional stories with different patterns, e.g. Table 16 where 1-layer
leaf-only retrieval (94.7) actually beat 2-layer (89.4) for a different
story) — so this is a real, story-dependent tradeoff, not a universal rule
that "more layers always wins." Validate against your own eval set rather
than assuming either direction.

## 2.7 Multi-hop reasoning loop over the tree

The paper itself does not implement or evaluate multi-hop retrieval — its
qualitative Cinderella study (§3, Qualitative Study; Appendix G) uses
**thematic, multi-hop QUESTIONS** as a way to illustrate RAPTOR's retrieval
quality, not a multi-hop retrieval ALGORITHM. This distinction matters:
RAPTOR's collapsed-tree retrieval answers a multi-hop-shaped question in a
**single retrieval call** by returning nodes at whatever mix of granularity
the question needs — it does not iterate.

**What the paper's qualitative result actually demonstrates** (Table 12/13,
§G): for "How does Cinderella find a happy ending?" — a question requiring
synthesis across the story's beginning, middle, and end — RAPTOR's single
collapsed-tree retrieval pulled evidence spanning the full arc (fairy
godmother's help → the ball → losing the slipper → the prince's search →
the ending), while DPR's single retrieval pulled only densely-clustered
leaf text about the initial transformation, missing the ending entirely.
GPT-4 given DPR's context stated it couldn't determine how the story ended;
given RAPTOR's context it answered correctly.

**Implication for the Hodos design:** RAPTOR's tree is what makes a
**single hop** capable of answering a thematically multi-hop-shaped
question — this is a different mechanism from **iterative** multi-hop
retrieval, where the *reasoning necessarily depends on what a prior
retrieval call returned* (e.g., "find the vendor mentioned in this clause,
then check that vendor's approval status" — a genuine sequential
dependency, not just breadth of synthesis). Hodos's multi-hop loop (Hop
1→2→3→4, as established in earlier design work) is doing the latter; each
hop's tree query internally benefits from RAPTOR's collapsed retrieval
(per §2.6), but the hops themselves remain sequential because later hops
genuinely need earlier hops' resolved outputs (a resolved vendor identity,
a resolved category limit), not just "more of the same document at a
different zoom level."

**Threshold table entry:**
| Parameter | Value | Source |
|---|---|---|
| RAPTOR's own multi-hop handling | Single collapsed-tree retrieval per query, not iterative | Paper §3, §G (inferred from methodology — paper does not implement iterative hops) |
| When Hodos's multi-hop loop is still needed on top of RAPTOR | When a later step's query genuinely depends on an earlier step's resolved output (sequential dependency), not merely broader synthesis within one document | Hodos design decision, not from paper |

## 2.8 Caching layer — what to cache, keyed how, with what thresholds

This section is a Hodos-specific engineering layer; the RAPTOR paper does
not address caching at all. It's included here because it's the piece that
determines whether RAPTOR's real cost (recursive summarization LLM calls at
build time, collapsed-tree similarity search over every node at query time)
gets paid once per distinct scenario or once per request.

### 2.8.0 Two separate lifecycles — do not conflate them

This is the single most common misunderstanding of this design, so it's
stated first, explicitly:

| | Tree build (§2.5) | Cache lookup (this section) |
|---|---|---|
| Trigger | A policy DOCUMENT changes | A CLAIM arrives |
| Frequency | Rare — event-driven off document edits | Constant — every claim, every line-item group |
| Cost | Expensive (recursive LLM summarization) | Cheap (hash lookup) |
| What a "new" input means | New/changed leaf text — triggers a rebuild | A tuple never queried before — triggers a cache MISS, which queries the ALREADY-BUILT tree; it never triggers a tree rebuild |

**A new `group_key` tuple is a cache miss, never a missing tree.** The tree
already covers every policy scenario present in the current documents; an
unqueried tuple just means nobody has asked about that specific
combination yet. Conflating these two lifecycles — e.g., rebuilding the
tree whenever a new tuple shows up — would make the entire caching layer
pointless, since the expensive step you're trying to amortize would fire
on ordinary claim traffic instead of only on real policy changes.

### 2.8.1 What gets cached (the "memory" unit)

**Cache the RESOLVED JUDGMENT from a completed retrieval + reasoning
pass, not the raw retrieved nodes and not the raw LLM completion text.**
Concretely:
```json
{
  "cache_key": "hash(category, grade, location, effective_date_period)",
  "applicable_clause_text": "...",
  "rule_type": "daily_cap | requires_preapproval | waived | ...",
  "limit_amount": 1500.0,
  "cited_node_ids": ["doc::L0::12", "doc::L1::3"],
  "tree_version": "tree-build-2026-07-01T00:00Z",
  "reasoning_model_version": "hop1-reasoner-v2.3",
  "confidence": 0.91,
  "cached_at": "...",
  "hit_count": 0
}
```
Why not cache raw retrieved chunks: the retrieval step is cheap relative to
the reasoning step (an LLM call resolving what those chunks *mean* for this
category/grade/location combination) — caching the resolved output skips
both the retrieval AND the reasoning cost on a hit, whereas caching only
retrieved chunks would still require a fresh reasoning call every time.

`tree_version` and `reasoning_model_version` are what tie a cached judgment
to a specific state of the corpus and the reasoning logic that produced it
— these are the fields invalidation (§2.8.3) actually keys off, and what
let an auditor confirm months later exactly which tree/model state
justified a given historical verdict, independent of what the tree looks
like today.

### 2.8.2 Cache key construction — what to normalize

| Field | Included in key? | Why |
|---|---|---|
| Category (meal/lodging/transport/...) | Yes, normalized to a fixed enum | Raw text varies ("meal"/"MEALS"/"food") |
| Employee grade | Yes | Directly determines which override clause applies |
| Location | Yes, normalized to a canonical code | Raw text varies ("Mumbai"/"Bombay") |
| Exact claim date | **No — resolved to `effective_date_period` cohort instead** | Two dates in the same policy window should hit the same cache entry; a date straddling a policy change must NOT |
| Raw question text | No | Different phrasings of the same underlying scenario should still hit the same cache entry |

Construction is order-independent (sort fields before hashing) so
`make_key(category=X, grade=Y)` and `make_key(grade=Y, category=X)`
collide to the same key.

### 2.8.3 Cache invalidation thresholds

| Trigger | Scope invalidated | Rationale |
|---|---|---|
| Tree rebuild/promotion event for a cohort (primary) | Every cache entry tagged with the superseded `tree_version` for that cohort | Ties invalidation to the actual event that made the cached judgment stale, not to a guess about how long it should remain valid |
| Wall-clock TTL, e.g. 24h (secondary safety net) | Any entry older than the TTL, regardless of version match | Backstop only, in case an invalidation event is missed — should rarely be the mechanism that actually fires |
| Manual invalidation | Single key, on human-reviewer override | For a judgment found wrong due to a reasoning error (not a policy change) — tagged distinctly so retrieval-quality monitoring (§3.4.4) can separate "stale cache" from "wrong from the start" |
| `reasoning_model_version` mismatch | Entries computed under a prior reasoning model/prompt version, if that logic is deliberately updated | Prevents a fixed reasoning bug from leaving stale (still-wrong) judgments live in cache indefinitely |

### 2.8.4 Durable history — the cache is disposable, the log is not

The fast key-value cache (whatever backing store — Redis, an in-memory
dict, etc.) holds only the CURRENT judgment per key and can be safely
flushed at any time. A **separate, append-only judgment log** persists
every judgment ever computed, keyed by `(cache_key, computed_at)`, and is
never overwritten:

```
Fast cache            — one current entry per key, overwritten on invalidation + recompute
Append-only log        — every computation ever made, indexed by (cache_key, computed_at)
```

Every claim's audit record cites the specific `(cache_key, computed_at)`
judgment-log row it used — not just "served from cache." This is what
makes a cache-hit claim's audit trail exactly as complete as a
fresh-computation claim's: if the fast cache were wiped entirely, no past
audit record breaks, because none of them actually depend on the live
cache still holding that entry.

### 2.8.5 Concurrency: duplicate computation, not incorrect computation

If two claims hit the same new `group_key` at nearly the same moment, both
may miss and both may trigger a fresh Hop 1 computation — this produces
**duplicate work, not an incorrect result**, provided Hop 1 reasoning runs
at low/zero temperature (two concurrent computations against the same tree
state should converge to the same judgment). Two computations for the same
key that disagree is itself a signal worth logging — either the reasoning
isn't as deterministic as intended, or retrieval is unstable — rather than
silently letting the second write clobber the first. A per-key lock/lease
on first miss would eliminate the duplicate cost, but is an optimization,
not a correctness requirement.

### 2.8.6 What NOT to cache

- Do not cache at the raw-question level (see §2.8.2) — this fragments the
  cache across phrasings of the same scenario and tanks the hit rate for no
  benefit.
- Do not cache partial/low-confidence judgments as if they were resolved —
  a judgment below your confidence threshold should not populate the
  cache, since a cache hit is assumed equivalent to a fresh high-confidence
  computation; caching an uncertain result would silently serve that
  uncertainty to every future claim matching the same tuple.

## 2.9 Threshold master table — every numeric parameter, one place

| # | Parameter | Value | Source | Notes |
|---|---|---|---|---|
| 1 | Leaf chunk target size | 100 tokens | Paper §3 | Never split mid-sentence |
| 2 | Leaf chunk overlap | 0 (not used) | Paper §3 (silent) | Different from Hodos's simpler non-RAPTOR overlap-chunking path |
| 3 | Embedding model | SBERT `multi-qa-mpnet-base-cos-v1` | Paper §3 | Must be consistent across all tree levels |
| 4 | Clustering method | GMM, soft/multi-membership | Paper §3 | Not KMeans/hard clustering |
| 5 | Dimensionality reduction | UMAP, two-pass (global then local n_neighbors) | Paper §3 | Exact n_neighbors values not given in paper — tune per corpus |
| 6 | Cluster count selection | BIC (minimize) | Paper §3 | Not a fixed K |
| 7 | Re-cluster trigger | Cluster's combined text exceeds summarizer's token limit | Paper §3 | Width-control recursion, distinct from tree-depth recursion |
| 8 | Summarization model | gpt-3.5-turbo | Paper §3 | |
| 9 | Summarization prompt | Verbatim in §2.4.2 above | Paper Appendix D | |
| 10 | Target compression ratio | ~0.28 | Paper Appendix C | Health-check, not hard constraint |
| 11 | Expected child nodes/parent | ~5.7–6.8 | Paper Appendix C | Sanity range for BIC output |
| 12 | Observed hallucination rate | 4% (6/150 sampled) | Paper Appendix E | Did not propagate to parents in paper's sample; still cross-check via leaf pull-through for compliance-sensitive domains |
| 13 | Preferred query strategy | Collapsed tree | Paper §3, empirical | Outperformed tree traversal in paper's own comparison |
| 14 | Token budget, GPT-class reader | 2000 tokens (≈ top-20 nodes) | Paper §3 | |
| 15 | Token budget, UnifiedQA reader | 400 tokens | Paper §3 | Reader has 512 max context |
| 16 | Node selection within budget | Add by cosine similarity score until token budget reached | Paper Algorithm 2 | Not a fixed node count |
| 17 | Non-leaf node share of retrieved evidence | 18.5%–57% depending on dataset/retriever | Paper Appendix I.2, Table 18 | Justifies NOT treating summary layers as rare fallback |
| 18 | Full-tree vs. restricted-layer retrieval | Full tree generally best, but story-dependent (Tables 8, 15–17) | Paper §4.1, Appendix I.1 | Validate on your own eval set, don't assume universally |
| 19 | Multi-hop iteration | Not implemented by RAPTOR itself — needed only when a later step has a genuine sequential dependency on an earlier step's output | Hodos design decision | |
| 20 | Cache key composition | (category, grade, location, effective_date_period) — normalized, NOT raw question text or exact date | Hodos design decision | |
| 21 | Cache invalidation | Primary: tree-rebuild event, scoped to affected cohort. Secondary: wall-clock TTL (e.g. 24h) as backstop only | Hodos design decision | |
| 22 | Token/build-time scaling | Linear in document length, both token expenditure and wall-clock build time | Paper Appendix A, Figs. 5–6 | Measured on an M1 Mac, 16GB RAM, up to 78,000-token documents |

---

## 2.10 Worked numeric example — putting the thresholds together

A policy document of ~30,000 tokens (roughly QASPER-length per the paper's
own scaling study, Fig. 5):

1. **Leaf layer**: ~300 chunks of 100 tokens each (sentence-boundary
   adjusted, so actual count varies slightly).
2. **Clustering pass 1**: UMAP (global n_neighbors) reveals broad thematic
   clusters across the 300 leaves; BIC selects K; expect roughly
   300/6.7 ≈ 45 clusters (using the paper's average cluster size as a rough
   guide, not a guarantee for your specific corpus).
3. **Summarization pass 1**: ~45 parent nodes, each ~131 tokens (paper's
   average summary length), each with `child_ids` pointing back to its
   ~6–7 leaf members. Any cluster whose combined child token count exceeds
   the summarizer's limit triggers a recursive sub-clustering pass before
   summarizing (§2.5).
4. **Clustering pass 2** on the ~45 level-1 summaries: BIC likely selects a
   much smaller K, perhaps 6–8 clusters, given the sharply reduced node
   count.
5. **Summarization pass 2**: ~6–8 level-2 summary nodes.
6. **Termination**: clustering the ~6–8 level-2 nodes likely yields BIC
   K=1 — a single root summary, or the recursion stops with these as the
   top layer if BIC doesn't cleanly collapse further.
7. **Query time**: a question retrieves via collapsed tree across all
   ~300 + 45 + ~7 ≈ 352 nodes at once, taking the top-scoring nodes up to
   the 2000-token budget (≈ top-20 nodes) — per §2.6.4's finding, expect
   roughly a fifth to a third of those 20 nodes to be non-leaf summaries,
   not all leaves.
8. **Cache**: the resolved judgment from that query (not the raw 20 nodes)
   gets stored under a key built from the claim's normalized
   category/grade/location/date-cohort tuple — the next claim matching
   that same tuple skips steps 7 and any reasoning call entirely.

---

# PART 3 — INGESTION, CLASSIFICATION, CONFIDENCE & PRECEDENCE

This part was previously missing from this document — it covers components
established earlier in the Hodos design that sit **upstream and downstream**
of the RAPTOR+multi-hop retrieval core (Parts 1–2), but are just as load-bearing.
None of this comes from the RAPTOR paper — it's Hodos-specific architecture.

## 3.1 Document classification: handwritten vs. printed

### 3.1.1 Intended design: CNN-based classifier

The original Hodos ingestion design calls for a **CNN-based image
classifier** as the first step of document processing, upstream of any
text extraction:

```
Bill image
    |
    v
CNN classifier (binary: handwritten vs. printed)
    |
    +-- printed      --> OCR path (e.g. Marker, or another OCR engine
    |                      that preserves table structure)
    |
    +-- handwritten  --> VLLM path (a vision-language model reads the
                           image directly, since OCR engines built for
                           printed/typed text perform poorly on genuine
                           handwriting)
```

**Why a CNN specifically, and why binary:** a lightweight image classifier
(not a full VLLM) is cheap to run at the front of the pipeline on every
single uploaded bill — this is a routing decision, not the extraction
itself, so it should be fast and inexpensive relative to what it's routing
into. A CNN trained on labeled handwritten-vs-printed document image
patches is the standard architecture for this kind of binary image
classification task — nothing exotic is needed here (a ResNet-style or
even a small custom CNN is sufficient; this is not a task requiring a large
vision transformer).

**Output contract:** the classifier should emit not just a binary label but
a **confidence score**, since the routing decision downstream should be
threshold-gated, not a bare argmax — see §3.1.3.

### 3.1.2 Why this component is absent from the shipped microservice

The CNN classifier and the VLLM handwriting path were **not implemented**
in the working microservice built earlier in this conversation, for a
concrete, tested reason: both require model weights that this specific
sandboxed environment cannot download (no Hugging Face access on the
network allowlist), and installing `marker-pdf` directly was attempted and
failed first on disk space, before the model-download problem even
surfaced. This is a real environment constraint, not a design decision to
drop the classifier.

**What was substituted instead, as a documented, honest stand-in:** a
heuristic that checks whether a PDF page has an extractable text layer at
all, routing to direct text extraction if so and to Tesseract OCR
(fully offline, no model download) if not. This heuristic answers a
different, narrower question than the CNN classifier would (see §3.1.4)
and should be understood as a fallback appropriate for offline/sandboxed
deployment, not a replacement for the intended design in an environment
that DOES have model-download access.

### 3.1.3 Threshold: classifier confidence gating

| Parameter | Value | Status |
|---|---|---|
| Classifier confidence threshold for routing to OCR path | Not numerically fixed in Hodos design — should be tuned against a labeled validation set of bill images, targeting a false-negative rate on "handwritten" low enough that genuinely handwritten bills rarely get routed to OCR (which would produce poor-quality extraction silently) | Design intent, not yet parameterized |
| Below-threshold behavior | A low-confidence classification should NOT default silently to either path — route to the more conservative/expensive path (VLLM) or flag for human review, rather than guessing | Design intent |
| Classifier confidence propagation | The classifier's confidence score should be carried downstream alongside `extraction_path` and `extraction_confidence` (§3.1.4) — a low-confidence classification followed by a low-confidence extraction is a stronger signal to route to human review than either alone | Design intent |

### 3.1.4 What the substitute heuristic does NOT do — be precise about the gap

The text-layer-presence heuristic answers "is this page already
machine-readable text, or just pixels" — it does **not** answer "is this
handwritten or printed," which is a different question entirely:
- A scanned printed receipt (no text layer, but the content is typed/printed)
  and a scanned handwritten note both fall into the "no text layer" bucket
  and both get routed to the same OCR engine under the substitute heuristic.
- The CNN classifier's actual job — distinguishing handwriting from print
  *within* an image-only page, so that handwriting specifically routes to a
  VLLM instead of an OCR engine ill-suited to it — is not performed at all
  by the substitute. Tesseract runs on both cases identically, and its
  accuracy on genuine handwriting is materially worse than on printed text.

This gap was directly observed, not hypothetical: testing the substitute
against a synthetic "scanned-style" bill image showed OCR misreadings (a
date digit misread, a word's first letter misread) — the kind of error a
real handwriting-aware path is specifically meant to reduce.

## 3.2 Claim structuring and line-item grouping

### 3.2.1 Structuring output contract

Extracted text (from whichever path in §3.1 produced it) is parsed into a
structured claim object — **not one amount per bill, but a bill-level
envelope plus a `line_items` array**, since a single bill commonly contains
multiple reimbursable categories (a restaurant bill with several dishes, a
hotel folio with room + tax + minibar).

```json
{
  "claim_id": "...",
  "employee_id": "...",
  "employee_grade": "...",
  "trip_id": "...",
  "vendor": "...",
  "date": "...",
  "location": "...",
  "extraction_path": "ocr | vllm",
  "extraction_confidence": 0.0,
  "extraction_status": "ok | extraction_failed",
  "line_items": [
    { "item_id": "...", "category": "...", "amount": 0.0, "currency": "..." }
  ]
}
```

### 3.2.2 The `extraction_failed` gate — an independent threshold from confidence

**This is distinct from a low `extraction_confidence` score, and both
gates must exist:**

| State | Meaning | Routing |
|---|---|---|
| `extraction_status: extraction_failed` | Output does not conform to the structured schema at all (missing required fields, malformed rows, empty extraction) | Bypass the verification engine entirely — straight to human review. Running any hop against null/malformed fields produces a verdict that LOOKS normal but is reasoning over garbage. |
| `extraction_status: ok`, but low `extraction_confidence` | Output conforms to the schema, but the classifier/OCR/VLLM path had low confidence in its own read | Proceeds into verification, but this confidence value is carried forward as an independent input to the final aggregation gate (§3.4.3) — a claim can have perfectly-formed data that's still likely wrong |

### 3.2.3 Grouping line items before retrieval

Before any hop runs, line items are grouped by the tuple that actually
determines what needs to be retrieved:

```
group_key = (category, employee_grade, location, effective_date_period)
```

This is what makes a 5-item bill typically require far fewer than 5
retrieval calls — items sharing a `group_key` share the same applicable
policy judgment. `effective_date_period` resolves the claim's raw date to
whichever policy cohort is active (see §3.3.2 on version partitioning) —
this is the field that prevents two claims filed a day apart, one before
and one after a policy change, from incorrectly sharing a cached judgment.

## 3.3 RAPTOR tree — version safety (missing from Part 2)

This belongs alongside Part 2's tree-construction detail: the RAPTOR paper
has no concept of a document corpus that changes over time with
version-sensitive rules (its datasets are static). Hodos's policy corpus
is not static — policies get amended — so this is a Hodos-specific
addition to the tree-build process in §2.5.

### 3.3.1 The risk: version-mixing in a summary node

If leaf chunks from two different policy versions (e.g., pre- and
post-amendment text covering the same category) end up in the same
cluster, the resulting summary node conflates rules that were never
simultaneously in effect — a genuinely dangerous failure mode for an audit
system, since the summary text itself won't visibly flag that it's mixing
eras.

### 3.3.2 The fix: partition before clustering, not detect after

**Leaf chunks are partitioned into effective-date cohorts BEFORE the
clustering step in §2.3 runs** — clustering, summarization, and all
recursive levels then operate strictly within one cohort. A cluster can
never span two policy versions, because the candidate leaves available to
clustering never include another cohort's leaves. This is a structural
prevention, not a scan-after-the-fact check.

| Parameter | Value |
|---|---|
| Version partitioning point | Before UMAP/clustering (§2.3), not after summarization |
| Scope of partitioning | Every tree level (leaf through root) inherits the cohort boundary — recursion stays within-cohort at every level |
| Secondary safety net | A query-time metadata filter on `effective_date_period` as a second, independent check — catches an ingestion-time mistagging bug even though partitioning should prevent the issue structurally |

## 3.4 Confidence, precedence, and the "confident no"

This is the aggregation logic across the multi-hop pipeline's outputs —
also entirely absent from Part 2, which only covered RAPTOR's own retrieval
mechanics.

### 3.4.1 Per-hop outputs are not just pass/fail

Each hop (policy retrieval, vendor verification, precedent check,
itinerary cross-check) returns a structured verdict with an attached
confidence, not a bare boolean:
```json
{ "verdict": "pass | fail | needs_review", "confidence": 0.0, "cited_passage_ids": [...] }
```

### 3.4.2 The precedence rule — "confident no" dominates

**Aggregation is a fixed precedence, not a vote or a weighted average
across hops:**

1. **Any hop returning a confident `fail`** (confidence above that hop's
   own threshold) makes the claim/item `rejected` — **regardless of how
   confidently the other hops passed.** A confirmed vendor-not-approved or
   a confirmed policy-limit breach must not be diluted by three unrelated
   hops passing cleanly. This is the "confident no" rule: one strong
   rejection signal overrides any number of weak or unrelated positive
   signals.
2. If no hop returns a confident fail, but any hop returns `needs_review`
   or a below-threshold result, the aggregate verdict is
   `needs_human_review`.
3. Only if every hop returns a confident `pass` is the claim eligible for
   `approved`.

**Why not average confidences:** averaging would let three passing hops
mathematically outweigh one failing hop, which is exactly backwards for an
audit system — a single confirmed violation should never be washed out by
unrelated confirmations elsewhere.

### 3.4.3 Extraction confidence as an independent gate, not folded into hop confidence

**A perfectly-executed set of hop verdicts is not sufficient for approval
if the underlying extraction was low-confidence.** This is a distinct
failure mode from any single hop failing: all four hops can pass cleanly
while still auditing a claim whose extracted amount doesn't actually match
the receipt, if the OCR/VLLM extraction itself was uncertain. Extraction
confidence (§3.2.2) is therefore checked as its own gate at final
aggregation, separate from the four hops' own confidences — not blended
into their average, and not treated as "one more hop."

| Gate | What it checks | Independent of |
|---|---|---|
| Per-hop confidence + precedence (§3.4.2) | Did the retrieved evidence support the claimed verdict | Extraction quality |
| Extraction confidence gate (§3.4.3) | Was the underlying OCR/VLLM read of the source document itself reliable | Hop verdict correctness |
| Retrieval-quality metric (§3.4.4) | Did retrieval find the RIGHT evidence at all | Whether the eventual verdict happened to be correct anyway |

### 3.4.4 Confidence calibration vs. retrieval-quality monitoring — two different questions

These must be tracked as **separate metrics**, not one:

- **Confidence calibration** (does a hop's reported confidence predict
  reviewer agreement with the verdict) is learned from human-review
  outcomes over time. This answers: "when this hop says 90% confident, is
  it actually right 90% of the time?"
- **Retrieval-quality monitoring** (did Hop 1's retrieval step find the
  right evidence at all) is a distinct signal — a policy cluster that
  systematically surfaces the wrong clause produces confident, consistent,
  WRONG verdicts. A calibration model alone will just learn to distrust
  that confidence level over time without ever surfacing the actual
  fixable defect (a bad cluster, a stale node). Human-review disagreement
  should be tagged explicitly as "retrieval failure" vs. "reasoning
  failure" at the point of review, precisely so these two metrics don't
  collapse into one undifferentiated "accuracy went down" signal.

### 3.4.5 Threshold table additions for Part 3

| # | Parameter | Value | Status |
|---|---|---|---|
| 23 | Handwriting classifier | CNN, binary (handwritten/printed), confidence-scored | Design intent — not implemented in the shipped microservice due to environment model-download constraints (§3.1.2) |
| 24 | Classifier confidence threshold | Not numerically fixed — tune against a labeled validation set; below-threshold routes conservatively, never silently defaults | Design intent |
| 25 | `extraction_failed` gate | Independent from `extraction_confidence` — malformed/unparseable output bypasses verification entirely | Design decision |
| 26 | Grouping tuple | (category, employee_grade, location, effective_date_period) | Design decision |
| 27 | Version partitioning point | Before clustering (structural prevention), not after summarization (detection) | Design decision, layered onto §2.3/§2.5 |
| 28 | Aggregation precedence | Confident fail on ANY hop dominates regardless of other hops' confidence ("confident no" rule) — not an average | Design decision |
| 29 | Extraction confidence gate | Independent of hop-level confidence and precedence; checked separately at final aggregation | Design decision |
| 30 | Confidence calibration vs. retrieval-quality monitoring | Tracked as two separate metrics, never merged | Design decision |

---

## Appendix: source attribution

Every value in §2.1–2.7 and the corresponding rows of the master threshold
table (§2.9, rows 1–19) is drawn directly from the uploaded PDF: *RAPTOR:
Recursive Abstractive Processing for Tree-Organized Retrieval*, Sarthi,
Abdullah, Tuli, Khanna, Goldie, and Manning, Stanford University, published
as a conference paper at ICLR 2024 (arXiv:2401.18059v1). Rows 20–22 of the
threshold table, and all of §2.8 (caching) and §2.7's multi-hop framing,
are Hodos-specific engineering decisions layered on top of the paper's
retrieval mechanism, not claims from the paper itself.

All of Part 3 (§3.1–§3.4, threshold rows 23–30) — the CNN handwriting
classifier, claim structuring and grouping, version partitioning as applied
to a non-static corpus, aggregation precedence (the "confident no" rule),
and the confidence-calibration/retrieval-quality split — reflects
Hodos-specific design work established earlier in this project's
conversation history, not the RAPTOR paper. It's included in this document
because this is meant to be the complete Hodos RAPTOR+multi-hop reference,
not a paper summary — omitting it, as the first version of this document
did, would leave out load-bearing parts of the actual system.
