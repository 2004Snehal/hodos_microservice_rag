"""
vlm.py — Vision-language model hook for bills Tesseract can't read
(handwritten, or printed but low OCR confidence).

Default backend: a local VLM through Ollama (HODOS_VLM_MODEL, default
qwen3-vl:4b — `ollama pull qwen3-vl:4b`). Anything with the same
`available()` / `transcribe()` shape can be swapped in (a hosted VLM API,
a different local model).

The model returns structured line items (JSON schema, thinking off,
temperature 0). They are rendered back into plain "description amount"
lines so the existing bill parser categorizes them exactly like OCR output.
"""

import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from app.llm.ollama import DEFAULT_OLLAMA_URL

DEFAULT_VLM_MODEL = os.getenv("HODOS_VLM_MODEL", "qwen3-vl:4b")
MAX_IMAGE_SIDE = 1600  # downscale huge scans; VLM cost grows with pixels

TRANSCRIBE_PROMPT = (
    "This is a photo or scan of a bill or receipt, possibly handwritten. Read it "
    "carefully. Return the vendor name, the date as written, and every line item "
    "with its amount exactly as written (digits only, no currency symbol). Include "
    "the total as an item named 'Total' if present. Do not invent items. If a value "
    "is unreadable, use null."
)
TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "amount": {"type": ["number", "null"]},
                },
                "required": ["description", "amount"],
            },
        },
    },
    "required": ["vendor", "date", "items"],
}


@dataclass
class VlmTranscription:
    text: str
    vendor: Optional[str]
    date: Optional[str]
    items: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""


class OllamaVLM:
    def __init__(self, model: str = DEFAULT_VLM_MODEL, url: str = DEFAULT_OLLAMA_URL, timeout: float = 300):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is None:
            try:
                tags = requests.get(f"{self.url}/api/tags", timeout=3).json()
                names = {m["name"] for m in tags.get("models", [])}
                self._available = self.model in names or f"{self.model}:latest" in names
            except requests.RequestException:
                self._available = False
        return self._available

    def transcribe(self, image: Image.Image) -> VlmTranscription:
        image = image.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": TRANSCRIBE_SCHEMA,
            "options": {"temperature": 0},
            "messages": [{
                "role": "user",
                "content": TRANSCRIBE_PROMPT,
                "images": [base64.b64encode(buffer.getvalue()).decode()],
            }],
        }
        response = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = json.loads(response.json()["message"]["content"])

        lines = [line for line in (data.get("vendor"), f"Date: {data['date']}" if data.get("date") else None) if line]
        for item in data.get("items", []):
            amount = item.get("amount")
            amount_text = f"Rs. {amount:g}" if isinstance(amount, (int, float)) else "(unreadable)"
            lines.append(f"{item.get('description', '').strip()} {amount_text}")
        return VlmTranscription("\n".join(lines), data.get("vendor"), data.get("date"), data.get("items", []), self.model)
