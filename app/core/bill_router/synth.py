"""
synth.py — Synthetic bill images for bootstrapping the handwritten/printed router.

Real labelled bills are the right training data; this exists so the router
works before you have any. Each image is a bill-like layout (vendor, date,
line items with amounts, total) rendered either:
  printed      — a regular/monospace font, aligned columns, uniform glyphs
  handwritten  — a handwriting-style font drawn character by character with
                 random size, rotation, baseline drift and spacing, in pen ink

Both classes then get the SAME scan degradations (rotation, blur, noise,
paper tint, JPEG compression, contrast) so the classifier can't separate
them by image quality — only by the shape of the writing.

Fonts are macOS system fonts. Training holds some fonts out per class (see
scripts/train_bill_router.py) to measure generalization to unseen writing.
"""

import io
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts")]

HANDWRITTEN_FONTS = [
    "Bradley Hand Bold.ttf", "Noteworthy.ttc", "MarkerFelt.ttc", "Chalkboard.ttc",
    "ChalkboardSE.ttc", "SnellRoundhand.ttc", "Apple Chancery.ttf", "Brush Script.ttf",
    "Savoye LET.ttc", "Chalkduster.ttf", "Comic Sans MS.ttf",
]
PRINTED_FONTS = [
    "Helvetica.ttc", "HelveticaNeue.ttc", "Courier.ttc", "Menlo.ttc", "Monaco.ttf", "Arial.ttf",
    "Courier New.ttf", "Andale Mono.ttf", "Georgia.ttf", "Verdana.ttf", "Avenir.ttc",
    "Optima.ttc", "Palatino.ttc", "Futura.ttc", "Charter.ttc", "Geneva.ttf", "SFNSMono.ttf",
    "Times.ttc", "Tahoma.ttf", "Trebuchet MS.ttf",
]

ITEMS = [
    "Dinner", "Lunch", "Breakfast", "Coffee", "Tea", "Paneer tikka", "Veg thali", "Naan",
    "Hotel room", "Room stay", "Lodging", "Taxi fare", "Cab to airport", "Uber ride",
    "Auto fare", "Parking", "Water bottle", "Snacks", "Laundry", "Service charge", "GST",
    "Train ticket", "Bus fare", "Room service", "Mineral water", "Toll",
]
VENDORS = [
    "Hotel Sea View", "Taj Restaurant", "City Cabs", "Cafe Mocha", "Sharma Dhaba",
    "Green Leaf Inn", "Metro Taxi Service", "Blue Bay Lodge", "Annapurna Bhojnalaya",
]


def find_font(name: str) -> Optional[Path]:
    for directory in FONT_DIRS:
        path = directory / name
        if path.exists():
            return path
    return None


def available_fonts(kind: str) -> List[str]:
    names = HANDWRITTEN_FONTS if kind == "handwritten" else PRINTED_FONTS
    return [n for n in names if find_font(n)]


def _bill_lines(rng: random.Random) -> List[Tuple[str, str]]:
    lines = [(rng.choice(VENDORS), ""),
             (f"Date: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/2026", "")]
    total = 0
    for _ in range(rng.randint(3, 8)):
        amount = rng.choice([rng.randint(20, 999), rng.randint(1000, 12000)])
        total += amount
        qty = f"{rng.randint(1, 4)} x " if rng.random() < 0.3 else ""
        prefix = rng.choice(["Rs. ", "Rs.", "INR ", ""])  # no ₹: several fonts lack the glyph
        lines.append((f"{qty}{rng.choice(ITEMS)}", f"{prefix}{amount:,}" if rng.random() < 0.6 else f"{prefix}{amount}"))
    lines.append(("Total", f"Rs. {total:,}"))
    return lines


def _draw_printed(draw, font_path, lines, width, rng):
    size = rng.randint(22, 34)
    font = ImageFont.truetype(str(font_path), size)
    y = rng.randint(30, 80)
    margin = rng.randint(40, 90)
    for left, right in lines:
        draw.text((margin, y), left, font=font, fill=(20, 20, 20))
        if right:
            w = draw.textlength(right, font=font)
            draw.text((width - margin - w, y), right, font=font, fill=(20, 20, 20))
        y += int(size * rng.uniform(1.3, 1.8))
    return y


def _draw_handwritten(img, font_path, lines, width, rng):
    base = rng.randint(28, 42)
    ink = rng.choice([(20, 30, 120), (15, 15, 15), (30, 40, 90), (60, 20, 20)])
    slant = rng.uniform(-6, 6)
    y = rng.randint(30, 80)
    margin = rng.randint(30, 90)
    for left, right in lines:
        x = margin + rng.randint(-10, 10)
        for text, anchor_right in ((left, False), (right, True)):
            if not text:
                continue
            if anchor_right:
                x = width - margin - int(len(text) * base * 0.55) + rng.randint(-25, 25)
            baseline = y
            for ch in text:
                size = max(12, int(base * rng.uniform(0.85, 1.15)))
                font = ImageFont.truetype(str(font_path), size)
                glyph = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
                ImageDraw.Draw(glyph).text((size // 2, size // 3), ch, font=font, fill=ink + (255,))
                glyph = glyph.rotate(slant + rng.uniform(-8, 8), resample=Image.BICUBIC)
                baseline += rng.uniform(-1.5, 1.5)
                img.paste(glyph, (int(x - size // 2), int(baseline - size // 3)), glyph)
                x += font.getlength(ch) * rng.uniform(0.9, 1.15) if ch != " " else size * rng.uniform(0.3, 0.6)
        y += int(base * rng.uniform(1.4, 2.0))
    return y


def _degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    arr = np.asarray(img).astype(np.float32)
    tint = np.array([rng.uniform(235, 255), rng.uniform(230, 252), rng.uniform(215, 250)])
    arr = arr / 255.0 * tint  # paper colour
    contrast = rng.uniform(0.6, 1.0)
    arr = 255 - (255 - arr) * contrast
    arr += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 10), arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.2)))
    if rng.random() < 0.25:  # low-resolution phone photo / fax: lose detail, then upsample
        small = rng.uniform(0.2, 0.45)
        w, h = img.size
        img = img.resize((max(1, int(w * small)), max(1, int(h * small))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.2, 2.2)))
    if rng.random() < 0.7:
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=rng.randint(35, 90))
        img = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    scale = rng.uniform(0.3, 0.6) if rng.random() < 0.2 else rng.uniform(0.6, 1.2)
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)


def render_bill(kind: str, font_name: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    font_path = find_font(font_name)
    if font_path is None:
        raise FileNotFoundError(font_name)
    width, height = rng.randint(800, 1100), 1500
    img = Image.new("RGB", (width, height), (255, 255, 255))
    lines = _bill_lines(rng)
    # Ruled paper on BOTH classes at the same rate — otherwise "has ruled
    # lines" becomes a shortcut feature for "handwritten".
    if rng.random() < 0.35:
        d = ImageDraw.Draw(img)
        for y in range(60, height, rng.randint(45, 70)):
            d.line([(0, y), (width, y)], fill=(170, 190, 230), width=1)
    if kind == "handwritten":
        bottom = _draw_handwritten(img, font_path, lines, width, rng)
    else:
        bottom = _draw_printed(ImageDraw.Draw(img), font_path, lines, width, rng)
    img = img.crop((0, 0, width, min(height, bottom + rng.randint(40, 120))))
    return _degrade(img, rng)
