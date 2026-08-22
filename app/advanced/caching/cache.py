"""
cache.py — Retrieval-result cache, keyed by a normalized tuple.

WHAT IT DOES
  - A normalized cache_key is derived from a tuple (category, scope, sub_scope, version)
  - The cached VALUE is a full structured judgment, including source chunks
  - A separate, append-only judgment LOG persists every computed judgment for auditing
  - Invalidation is explicit and scoped (`invalidate(version=...)`)
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List


class TupleCache:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.cache_path = self.storage_dir / "cache.json"
        self.log_path = self.storage_dir / "judgment_log.jsonl"  # append-only

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ---------- key construction ----------

    @staticmethod
    def make_key(**fields: str) -> str:
        """
        Build a normalized cache key from arbitrary named fields, e.g.:
            make_key(category="meal", scope="L3", sub_scope="Mumbai", version="policy-v3")
        Field order in the call doesn't matter — keys are sorted before hashing,
        so make_key(a=1, b=2) and make_key(b=2, a=1) produce the same key.
        """
        normalized = "|".join(
            f"{k}={str(v).strip().lower()}" for k, v in sorted(fields.items())
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ---------- reads/writes ----------

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        entry["hit_count"] = entry.get("hit_count", 0) + 1
        self._persist_cache()
        return entry

    def set(self, key: str, judgment: Dict[str, Any], version: str) -> Dict[str, Any]:
        entry = {
            "cache_key": key,
            "judgment": judgment,
            "version": version,
            "cached_at": time.time(),
            "hit_count": 0,
        }
        self._cache[key] = entry
        self._append_log(entry)
        self._persist_cache()
        return entry

    def invalidate(self, version: str) -> int:
        """Drop every cache entry computed under `version`. Returns count removed."""
        to_remove = [k for k, v in self._cache.items() if v["version"] == version]
        for k in to_remove:
            del self._cache[k]
        self._persist_cache()
        return len(to_remove)

    def invalidate_key(self, key: str, reason: str = "manual") -> bool:
        """Force-invalidate a single key (e.g. a human reviewer flags a bad judgment)."""
        if key in self._cache:
            del self._cache[key]
            self._persist_cache()
            return True
        return False

    def history_for_key(self, key: str) -> List[Dict[str, Any]]:
        """
        Read every judgment ever computed for this key from the append-only
        log — this is what survives cache flushes/invalidations, and what an
        audit trail would cite rather than the live cache value.
        """
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, "r") as f:
            for line in f:
                record = json.loads(line)
                if record["cache_key"] == key:
                    records.append(record)
        return records

    def stats(self) -> Dict[str, Any]:
        return {
            "total_keys": len(self._cache),
            "total_hits": sum(v.get("hit_count", 0) for v in self._cache.values()),
            "versions_present": sorted({v["version"] for v in self._cache.values()}),
        }

    # ---------- persistence ----------

    def _load(self) -> None:
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                self._cache = json.load(f)

    def _persist_cache(self) -> None:
        with open(self.cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _append_log(self, entry: Dict[str, Any]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
