"""
router.py — CNN router: is this bill image handwritten or printed?

MODEL
  Feature extractor: ResNet-50 (ImageNet-pretrained CNN, ONNX via fastembed,
  ~100MB, downloaded once, no PyTorch needed) → 2048-dim features per tile.
  Head: logistic regression trained by scripts/train_bill_router.py on
  synthetic bills (app/core/bill_router/synth.py) plus any real labelled
  bills you add.

WHY TILES
Resizing a whole page to the CNN's 224×224 input shrinks the text until
stroke shape — the thing that actually separates handwriting from print —
is gone. Instead the page is normalized to a fixed width and cut into
224×224 tiles; the tiles with the most ink are classified and their
log-odds averaged into one page-level P(handwritten).

DECISION
  P(handwritten) ≥ HANDWRITTEN_THRESHOLD  -> "handwritten"
  P(handwritten) ≤ PRINTED_THRESHOLD      -> "printed"
  otherwise                               -> "uncertain"
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageOps

CNN_MODEL = "Qdrant/resnet50-onnx"
HEAD_PATH = Path(os.getenv("HODOS_ROUTER_HEAD", Path(__file__).parent / "weights" / "router_head.pkl"))
HANDWRITTEN_THRESHOLD = float(os.getenv("HODOS_ROUTER_HANDWRITTEN_THRESHOLD", "0.65"))
PRINTED_THRESHOLD = float(os.getenv("HODOS_ROUTER_PRINTED_THRESHOLD", "0.35"))

PAGE_WIDTH = 896  # 4 tiles across
TILE = 224
MAX_TILES = 16


@dataclass
class RouteDecision:
    label: str  # "handwritten" | "printed" | "uncertain"
    p_handwritten: float
    tiles_used: int
    model: str


class CnnFeatureExtractor:
    _model = None

    def features(self, tiles: List[Image.Image]) -> np.ndarray:
        if CnnFeatureExtractor._model is None:
            from fastembed import ImageEmbedding
            CnnFeatureExtractor._model = ImageEmbedding(CNN_MODEL)
        return np.asarray(list(CnnFeatureExtractor._model.embed(tiles)), dtype=np.float32)


def page_tiles(image: Image.Image, max_tiles: int = MAX_TILES) -> List[Image.Image]:
    """Grayscale, normalize width, cut into tiles, keep the most ink-dense ones."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    scale = PAGE_WIDTH / gray.width
    gray = gray.resize((PAGE_WIDTH, max(TILE, int(gray.height * scale))), Image.BILINEAR)
    pixels = np.asarray(gray, dtype=np.float32)
    ink_level = np.median(pixels) - 60

    candidates = []
    for top in range(0, gray.height - TILE + 1, TILE // 2):
        for left in range(0, PAGE_WIDTH - TILE + 1, TILE // 2):
            ink = float((pixels[top:top + TILE, left:left + TILE] < ink_level).mean())
            if 0.01 <= ink <= 0.5:
                candidates.append((ink, left, top))
    if not candidates:  # blank-ish or tiny image: fall back to the whole page
        return [gray.convert("RGB").resize((TILE, TILE))]
    candidates.sort(reverse=True)
    return [gray.crop((l, t, l + TILE, t + TILE)).convert("RGB") for _, l, t in candidates[:max_tiles]]


class HandwritingRouter:
    def __init__(self, head_path: Path = HEAD_PATH):
        self.head_path = head_path
        self.extractor = CnnFeatureExtractor()
        self._head = None

    def available(self) -> bool:
        return self.head_path.exists()

    def _load(self):
        if self._head is None:
            with open(self.head_path, "rb") as f:
                self._head = pickle.load(f)
        return self._head

    def tile_probabilities(self, tiles: List[Image.Image]) -> np.ndarray:
        head = self._load()
        feats = head["scaler"].transform(self.extractor.features(tiles))
        return head["classifier"].predict_proba(feats)[:, 1]

    def route(self, image: Image.Image) -> Optional[RouteDecision]:
        if not self.available():
            return None
        tiles = page_tiles(image)
        probs = np.clip(self.tile_probabilities(tiles), 1e-4, 1 - 1e-4)
        mean_logit = float(np.mean(np.log(probs / (1 - probs))))
        p = 1 / (1 + np.exp(-mean_logit))
        if p >= HANDWRITTEN_THRESHOLD:
            label = "handwritten"
        elif p <= PRINTED_THRESHOLD:
            label = "printed"
        else:
            label = "uncertain"
        return RouteDecision(label, round(p, 4), len(tiles), CNN_MODEL)
