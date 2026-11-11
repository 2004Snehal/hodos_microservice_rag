"""
train_bill_router.py — Train the handwritten/printed router head.

  python scripts/train_bill_router.py                      # synthetic only
  python scripts/train_bill_router.py --real-dir data/bills # + real samples

--real-dir must contain `handwritten/` and `printed/` subfolders of images
(png/jpg/jpeg/tif/bmp) or PDFs. Real samples are always used for training
and, with --real-test-fraction, a slice is held out for evaluation — real
bills are what tells you how the router does in practice; the synthetic
held-out-font score only tells you it generalizes beyond its training fonts.

Writes app/core/bill_router/weights/router_head.pkl and router_metrics.json.
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.bill_router.router import CNN_MODEL, CnnFeatureExtractor, HEAD_PATH, page_tiles  # noqa: E402
from app.core.bill_router.synth import available_fonts, render_bill  # noqa: E402

HELD_OUT = {
    "handwritten": ["Chalkduster.ttf", "Savoye LET.ttc", "Comic Sans MS.ttf"],
    "printed": ["Georgia.ttf", "Futura.ttc", "Andale Mono.ttf", "Tahoma.ttf"],
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_real(real_dir: Path):
    import pymupdf
    from PIL import Image

    samples = []
    for label, kind in ((1, "handwritten"), (0, "printed")):
        for path in sorted((real_dir / kind).glob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append((Image.open(path).convert("RGB"), label, f"real:{path.name}"))
            elif path.suffix.lower() == ".pdf":
                doc = pymupdf.open(str(path))
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    samples.append((Image.frombytes("RGB", (pix.width, pix.height), pix.samples), label, f"real:{path.name}"))
    return samples


def synthetic(pages_per_class: int, seed: int):
    train, test = [], []
    for label, kind in ((1, "handwritten"), (0, "printed")):
        fonts = available_fonts(kind)
        held = [f for f in fonts if f in HELD_OUT[kind]]
        used = [f for f in fonts if f not in held]
        for i in range(pages_per_class):
            font = used[i % len(used)]
            train.append((render_bill(kind, font, seed + i * 7 + label), label, font))
        for i in range(max(10, pages_per_class // 4)):
            font = held[i % len(held)]
            test.append((render_bill(kind, font, seed + 100_000 + i * 7 + label), label, font))
    return train, test


def featurize(samples, extractor):
    feats, labels, groups = [], [], []
    for page_index, (image, label, _) in enumerate(samples):
        tiles = page_tiles(image)
        f = extractor.features(tiles)
        feats.append(f)
        labels.extend([label] * len(f))
        groups.extend([page_index] * len(f))
    return np.vstack(feats), np.array(labels), np.array(groups)


def page_scores(probs, groups):
    probs = np.clip(probs, 1e-4, 1 - 1e-4)
    logits = np.log(probs / (1 - probs))
    return np.array([1 / (1 + np.exp(-logits[groups == g].mean())) for g in np.unique(groups)])


def evaluate(name, scaler, clf, feats, labels, groups):
    tile_probs = clf.predict_proba(scaler.transform(feats))[:, 1]
    page_labels = np.array([labels[groups == g][0] for g in np.unique(groups)])
    page_probs = page_scores(tile_probs, groups)
    page_pred = (page_probs >= 0.5).astype(int)
    result = {
        "pages": int(len(page_labels)),
        "tile_accuracy": round(float(accuracy_score(labels, tile_probs >= 0.5)), 4),
        "page_accuracy": round(float(accuracy_score(page_labels, page_pred)), 4),
        "page_auc": round(float(roc_auc_score(page_labels, page_probs)), 4) if len(set(page_labels)) > 1 else None,
        "confusion_matrix [[printed→printed, printed→hw],[hw→printed, hw→hw]]": confusion_matrix(page_labels, page_pred, labels=[0, 1]).tolist(),
        "uncertain_pages (0.35<p<0.65)": int(((page_probs > 0.35) & (page_probs < 0.65)).sum()),
    }
    print(f"[{name}] {json.dumps(result)}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages-per-class", type=int, default=200)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--real-test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    t0 = time.time()
    train, test = synthetic(args.pages_per_class, args.seed)
    real_train, real_test = [], []
    if args.real_dir:
        real = load_real(args.real_dir)
        random.Random(args.seed).shuffle(real)
        cut = int(len(real) * args.real_test_fraction)
        real_test, real_train = real[:cut], real[cut:]
        print(f"real samples: {len(real_train)} train, {len(real_test)} test")
    print(f"rendered {len(train)} train / {len(test)} held-out-font synthetic pages in {time.time() - t0:.0f}s")

    extractor = CnnFeatureExtractor()
    t0 = time.time()
    x_train, y_train, _ = featurize(train + real_train, extractor)
    x_test, y_test, g_test = featurize(test, extractor)
    print(f"CNN features: {len(x_train)} train tiles, {len(x_test)} test tiles in {time.time() - t0:.0f}s")

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(scaler.transform(x_train), y_train)

    metrics = {
        "cnn": CNN_MODEL,
        "head": "logistic_regression(C=0.05, balanced)",
        "train_pages": len(train) + len(real_train),
        "real_train_pages": len(real_train),
        "held_out_fonts": HELD_OUT,
        "synthetic_held_out_fonts": evaluate("synthetic, unseen fonts", scaler, clf, x_test, y_test, g_test),
    }
    if real_test:
        xr, yr, gr = featurize(real_test, extractor)
        metrics["real_held_out"] = evaluate("real, held out", scaler, clf, xr, yr, gr)

    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HEAD_PATH, "wb") as f:
        pickle.dump({"scaler": scaler, "classifier": clf, "metrics": metrics}, f)
    (HEAD_PATH.parent / "router_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {HEAD_PATH}")


if __name__ == "__main__":
    main()
