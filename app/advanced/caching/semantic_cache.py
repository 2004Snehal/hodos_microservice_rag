"""
semantic_cache.py — Multi-layer semantic cache for resolved policy judgments.

LAYERS (checked in order, cheapest first)
  L1  in-process LRU, exact match           ~microseconds, per process
  L2  Redis key/value, exact match          ~sub-ms, shared across processes
  L3  Redis vector search (KNN), semantic   ~ms, catches paraphrases
      ("daily food limit for L3 in Mumbai" ≈ "meal cap for L3 in Mumbai")

A hit in a lower layer is written back into the layers above it.

SCOPE
Every entry is bound to a scope:
  (doc_id, doc_hash, category, grade, location, version)
Semantic matching only ever compares questions inside the SAME scope, so a
paraphrase can't leak an answer across documents, grades or locations.
`doc_hash` is a hash of the policy text the tree was built from, so editing
a policy automatically stops old answers from matching — no manual
version bump needed. The exact-match key is hash(scope + normalized question).

BACKENDS
  RedisBackend     — Redis 8+ (or Redis Stack) with the search module, which
                     ships with `brew install redis`. Uses an HNSW cosine
                     index over HASH keys.
  InMemoryBackend  — fallback when Redis isn't reachable; same behavior,
                     numpy brute-force KNN, not shared across processes.

AUDIT LOG
Every freshly computed judgment is appended to an append-only JSONL log, as
the old TupleCache did — this survives cache flushes and invalidations.

EMBEDDINGS
Uses fastembed (BAAI/bge-small-en-v1.5, 384-dim, local ONNX — no API calls).
The model is downloaded once (~70MB) on first use. Measured on policy
questions: paraphrases score ~0.87-0.88 cosine, different questions in the
same category ≤ 0.79 — hence the 0.85 default threshold.
"""

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.dense_embeddings import Embedder, EMBEDDING_DIM

DEFAULT_REDIS_URL = os.getenv("HODOS_REDIS_URL", "redis://localhost:6379/0")
DEFAULT_THRESHOLD = float(os.getenv("HODOS_SEMANTIC_THRESHOLD", "0.85"))
DEFAULT_TTL_SECONDS = int(os.getenv("HODOS_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))

SCOPE_FIELDS = ("doc_id", "doc_hash", "category", "grade", "location", "version")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _normalize_question(question: str) -> str:
    return re.sub(r"[^\w\s]", "", _normalize(question)).strip()


@dataclass
class CacheHit:
    judgment: Dict[str, Any]
    layer: str  # "l1_exact" | "l2_exact" | "l3_semantic"
    similarity: float
    matched_question: str


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class InMemoryBackend:
    name = "memory"

    def __init__(self):
        self._exact: Dict[str, Dict[str, Any]] = {}
        self._entries: List[Dict[str, Any]] = []

    def get_exact(self, key: str) -> Optional[Dict[str, Any]]:
        return self._exact.get(key)

    def put(self, key: str, entry: Dict[str, Any], vector: np.ndarray) -> None:
        self._exact[key] = entry
        self._entries = [e for e in self._entries if e["key"] != key]
        self._entries.append({"key": key, "entry": entry, "vector": vector})

    def search(self, scope_id: str, vector: np.ndarray) -> Optional[tuple]:
        candidates = [e for e in self._entries if e["entry"]["scope_id"] == scope_id]
        if not candidates:
            return None
        sims = np.stack([c["vector"] for c in candidates]) @ vector
        best = int(np.argmax(sims))
        return candidates[best]["entry"], float(sims[best])

    def invalidate(self, field: str, value: str) -> int:
        doomed = {k for k, e in self._exact.items() if e["scope"].get(field) == value}
        for k in doomed:
            del self._exact[k]
        self._entries = [e for e in self._entries if e["key"] not in doomed]
        return len(doomed)

    def count(self) -> int:
        return len(self._exact)

    def clear(self) -> None:
        self._exact.clear()
        self._entries.clear()


class RedisBackend:
    name = "redis"
    INDEX = "hodos:semcache:idx"
    PREFIX = "hodos:semcache:"

    def __init__(self, url: str = DEFAULT_REDIS_URL, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        import redis
        from redis.commands.search.field import TagField, TextField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType

        self.client = redis.Redis.from_url(url)
        self.client.ping()
        self.ttl = ttl_seconds
        try:
            self.client.ft(self.INDEX).info()
        except redis.ResponseError:
            self.client.ft(self.INDEX).create_index(
                [
                    TagField("scope_id"),
                    *[TagField(f) for f in SCOPE_FIELDS],
                    TextField("question"),
                    VectorField(
                        "vector", "HNSW",
                        {"TYPE": "FLOAT32", "DIM": EMBEDDING_DIM, "DISTANCE_METRIC": "COSINE"},
                    ),
                ],
                definition=IndexDefinition(prefix=[self.PREFIX], index_type=IndexType.HASH),
            )

    def get_exact(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self.client.hget(self.PREFIX + key, "entry")
        return json.loads(raw) if raw else None

    def put(self, key: str, entry: Dict[str, Any], vector: np.ndarray) -> None:
        redis_key = self.PREFIX + key
        mapping = {
            "entry": json.dumps(entry),
            "scope_id": entry["scope_id"],
            "question": entry["question"],
            "vector": vector.astype(np.float32).tobytes(),
            **{f: _tag_safe(entry["scope"][f]) for f in SCOPE_FIELDS},
        }
        pipe = self.client.pipeline()
        pipe.hset(redis_key, mapping=mapping)
        if self.ttl:
            pipe.expire(redis_key, self.ttl)
        pipe.execute()

    def search(self, scope_id: str, vector: np.ndarray) -> Optional[tuple]:
        from redis.commands.search.query import Query

        query = (
            Query(f"(@scope_id:{{{scope_id}}})=>[KNN 1 @vector $vec AS distance]")
            .sort_by("distance")
            .return_fields("entry", "distance")
            .dialect(2)
        )
        result = self.client.ft(self.INDEX).search(
            query, query_params={"vec": vector.astype(np.float32).tobytes()}
        )
        if not result.docs:
            return None
        doc = result.docs[0]
        return json.loads(doc.entry), 1.0 - float(doc.distance)

    def invalidate(self, field: str, value: str) -> int:
        from redis.commands.search.query import Query

        removed = 0
        while True:
            result = self.client.ft(self.INDEX).search(
                Query(f"@{field}:{{{_tag_safe(value)}}}").no_content().paging(0, 500).dialect(2)
            )
            if not result.docs:
                return removed
            removed += self.client.delete(*[d.id for d in result.docs])

    def count(self) -> int:
        return int(self.client.ft(self.INDEX).info()["num_docs"])

    def clear(self) -> None:
        keys = list(self.client.scan_iter(match=self.PREFIX + "*", count=500))
        if keys:
            self.client.delete(*keys)


def _tag_safe(value: str) -> str:
    """Redis TAG values can't contain punctuation unescaped; use a stable token."""
    return re.sub(r"[^a-z0-9_]", "_", _normalize(value))


# ---------------------------------------------------------------------------
# Multi-layer cache
# ---------------------------------------------------------------------------

class SemanticCache:
    def __init__(
        self,
        log_dir: Path,
        redis_url: Optional[str] = DEFAULT_REDIS_URL,
        threshold: float = DEFAULT_THRESHOLD,
        l1_size: int = 1024,
        embedder: Optional[Embedder] = None,
    ):
        self.threshold = threshold
        self.embedder = embedder or Embedder()
        self.l1: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.l1_size = l1_size
        self.stats_counter = {"l1_exact": 0, "l2_exact": 0, "l3_semantic": 0, "miss": 0}

        self.backend = InMemoryBackend()
        self.backend_error: Optional[str] = None
        if redis_url:
            try:
                self.backend = RedisBackend(redis_url)
            except Exception as exc:  # unreachable Redis, no search module, client missing
                self.backend_error = f"{type(exc).__name__}: {exc}"

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "judgment_log.jsonl"

    # ---------- keys ----------

    @staticmethod
    def scope_id(scope: Dict[str, str]) -> str:
        normalized = "|".join(f"{f}={_normalize(scope[f])}" for f in SCOPE_FIELDS)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    @classmethod
    def exact_key(cls, question: str, scope: Dict[str, str]) -> str:
        raw = f"{cls.scope_id(scope)}|{_normalize_question(question)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ---------- read / write ----------

    def lookup(self, question: str, scope: Dict[str, str]) -> Optional[CacheHit]:
        key = self.exact_key(question, scope)

        entry = self.l1.get(key)
        if entry is not None:
            self.l1.move_to_end(key)
            return self._hit(entry, "l1_exact", 1.0)

        entry = self.backend.get_exact(key)
        if entry is not None:
            self._l1_put(key, entry)
            return self._hit(entry, "l2_exact", 1.0)

        found = self.backend.search(self.scope_id(scope), self.embedder.embed(question))
        if found is not None:
            entry, similarity = found
            if similarity >= self.threshold:
                self._l1_put(key, entry)
                return self._hit(entry, "l3_semantic", similarity)

        self.stats_counter["miss"] += 1
        return None

    def store(self, question: str, scope: Dict[str, str], judgment: Dict[str, Any]) -> None:
        key = self.exact_key(question, scope)
        entry = {
            "cache_key": key,
            "scope_id": self.scope_id(scope),
            "scope": {f: str(scope[f]) for f in SCOPE_FIELDS},
            "question": question,
            "judgment": judgment,
            "cached_at": time.time(),
        }
        self.backend.put(key, entry, self.embedder.embed(question))
        self._l1_put(key, entry)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ---------- invalidation ----------

    def invalidate_doc(self, doc_id: str) -> int:
        return self._invalidate("doc_id", doc_id)

    def invalidate_version(self, version: str) -> int:
        return self._invalidate("version", version)

    def _invalidate(self, field: str, value: str) -> int:
        target = _normalize(value)
        for k in [k for k, e in self.l1.items() if _normalize(e["scope"][field]) == target]:
            del self.l1[k]
        return self.backend.invalidate(field, value)

    def clear(self) -> None:
        self.l1.clear()
        self.backend.clear()

    # ---------- audit / stats ----------

    def history(self, question: str, scope: Dict[str, str]) -> List[Dict[str, Any]]:
        key = self.exact_key(question, scope)
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            return [r for r in map(json.loads, f) if r["cache_key"] == key]

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.name,
            "backend_error": self.backend_error,
            "entries": self.backend.count(),
            "l1_entries": len(self.l1),
            "threshold": self.threshold,
            "hits": dict(self.stats_counter),
        }

    # ---------- internals ----------

    def _hit(self, entry: Dict[str, Any], layer: str, similarity: float) -> CacheHit:
        self.stats_counter[layer] += 1
        return CacheHit(entry["judgment"], layer, similarity, entry["question"])

    def _l1_put(self, key: str, entry: Dict[str, Any]) -> None:
        self.l1[key] = entry
        self.l1.move_to_end(key)
        while len(self.l1) > self.l1_size:
            self.l1.popitem(last=False)
