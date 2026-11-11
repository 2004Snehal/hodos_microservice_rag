"""
ollama.py — Minimal client for a local Ollama server, with token-level
probabilities.

All calls use:
  - think=False + a JSON schema (`format`). qwen3:4b is a thinking model; with
    thinking on, a single judgment takes 40-70s. With thinking off and a
    schema it answers directly in ~0.4-1.5s. Without a schema it ignores
    think=False and starts reasoning in the content, so a schema is always set.
  - temperature 0, so the same input gives the same output.
  - logprobs, so enum fields come back with a probability distribution
    rather than just a choice.

HOW ENUM PROBABILITIES ARE READ
Ollama reports top_logprobs from the model's distribution BEFORE the JSON
grammar mask is applied. At the token where an enum value starts, we sum
the probability of every top token that is a prefix of each option
(case-insensitive) and renormalize over the options. Example: for
{"verdict": "Yes"|"No"} the model's raw top tokens might be
"Yes" 0.60, "yes" 0.15, "No" 0.20 → P(Yes) = 0.75 / 0.95 = 0.79.
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("HODOS_LLM_MODEL", "qwen3:4b")
DEFAULT_TIMEOUT = float(os.getenv("HODOS_LLM_TIMEOUT", "120"))


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class JsonResult:
    data: Dict[str, Any]
    # field name -> {option: probability}, for every enum field in the schema
    distributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    duration_s: float = 0.0


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT,
        top_logprobs: int = 20,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.top_logprobs = top_logprobs
        self._available: Optional[bool] = None

    # ---------- health ----------

    def available(self, refresh: bool = False) -> bool:
        if self._available is None or refresh:
            try:
                tags = requests.get(f"{self.url}/api/tags", timeout=3).json()
                names = {m["name"] for m in tags.get("models", [])}
                self._available = self.model in names or f"{self.model}:latest" in names
            except requests.RequestException:
                self._available = False
        return self._available

    # ---------- calls ----------

    def json(self, system: str, user: str, schema: Dict[str, Any], max_tokens: int = 512) -> JsonResult:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "options": {"temperature": 0, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": f"{system}\n\n{_schema_instructions(schema)}"},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailable(f"Ollama call failed: {exc}") from exc

        body = response.json()
        content = body["message"]["content"]
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"Model returned invalid JSON: {content[:200]!r}") from exc

        enums = {
            name: spec["enum"]
            for name, spec in schema.get("properties", {}).items()
            if isinstance(spec, dict) and "enum" in spec
        }
        distributions = {
            name: _enum_distribution(body.get("logprobs") or [], name, options, data.get(name))
            for name, options in enums.items()
        }
        return JsonResult(data, distributions, body.get("total_duration", 0) / 1e9)

    def probability_yes(self, system: str, user: str) -> float:
        """P(Yes) for a binary judgment, from token probabilities."""
        schema = {
            "type": "object",
            "properties": {"verdict": {"type": "string", "enum": ["Yes", "No"]}},
            "required": ["verdict"],
        }
        result = self.json(
            system,
            user,
            schema,
            max_tokens=16,
        )
        return result.distributions["verdict"]["Yes"]


def _schema_instructions(schema: Dict[str, Any]) -> str:
    """
    Spell out the schema, and every enum's allowed values, in the prompt.
    Grammar-constrained decoding guarantees valid output either way, but the
    probabilities we read are from BEFORE the grammar mask — if the model
    hasn't been told the options, its probability mass sits on unrelated
    tokens and the distribution is meaningless.
    """
    lines = ["Respond only with a JSON object with these fields:"]
    for name, spec in schema.get("properties", {}).items():
        if "enum" in spec:
            options = " | ".join(f'"{o}"' for o in spec["enum"])
            lines.append(f"- {name}: one of {options}")
        else:
            kind = spec.get("type", "any")
            kind = " or ".join(kind) if isinstance(kind, list) else kind
            note = f" — {spec['description']}" if "description" in spec else ""
            lines.append(f"- {name}: {kind}{note}")
    return "\n".join(lines)


def _normalize_token(token: str) -> str:
    return token.strip().strip('"').lower()


def _enum_distribution(
    logprobs: List[Dict[str, Any]], field_name: str, options: List[str], chosen: Optional[str]
) -> Dict[str, float]:
    """
    Find the token where `field_name`'s value starts and turn its top
    alternatives into a distribution over `options`. Falls back to a point
    mass on the chosen option if logprobs are missing.
    """
    point_mass = {o: (1.0 if o == chosen else 0.0) for o in options}
    if not logprobs:
        return point_mass

    text = ""
    key_marker = f'"{field_name}"'
    for i, entry in enumerate(logprobs):
        text += entry["token"]
        if key_marker not in text:
            continue
        after_key = text.split(key_marker, 1)[1]
        if '"' not in after_key.split(":", 1)[-1]:
            continue  # haven't reached the opening quote of the value yet
        # The value starts at the first token after the opening quote that
        # carries letters; scan forward from here.
        for value_entry in logprobs[i:]:
            if _normalize_token(value_entry["token"]):
                return _distribution_from_top(value_entry, options) or point_mass
        break
    return point_mass


def _distribution_from_top(entry: Dict[str, Any], options: List[str]) -> Optional[Dict[str, float]]:
    """
    Probability mass on tokens that don't start any option (the model wanted
    to say something else) is spread uniformly across the options — it
    represents uncertainty, so it should lower the top option's probability
    rather than be renormalized away.
    """
    mass = {o: 0.0 for o in options}
    candidates = entry.get("top_logprobs") or [entry]
    for candidate in candidates:
        token = _normalize_token(candidate["token"])
        if not token:
            continue
        matches = [o for o in options if o.lower().startswith(token)]
        if not matches:
            continue
        p = math.exp(candidate["logprob"])
        for o in matches:  # a token shared by several options splits its mass
            mass[o] += p / len(matches)
    matched = sum(mass.values())
    if matched <= 0:
        return None
    residual = max(0.0, 1.0 - matched) / len(options)
    return {o: m + residual for o, m in mass.items()}
