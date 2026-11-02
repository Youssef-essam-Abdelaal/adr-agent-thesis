"""
Cheap image operations: no model calls.

The pixel diff and screen hash feed Layer 1, the Monitor's free checks. The red
circle, the blue change box and the zoomed crops build the evidence that the judge
and the Diagnostician look at (a red circle on the click as in HiViG; the changed
region boxed, as in evidence-first reflection; a crop-and-zoom around the click
for grounding).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def diff_info(before: Image.Image, after: Image.Image, scale: int = 1, thresh: int = 24) -> dict:
    """Changed pixels at CSS scale (a gray value that moved by more than thresh), plus the changed region's box in
    image pixels."""
    w, h = before.size
    small = (max(1, w // scale), max(1, h // scale))
    a = np.asarray(before.convert("L").resize(small), dtype=np.int16)
    b = np.asarray(after.convert("L").resize(small), dtype=np.int16)
    if a.shape != b.shape:
        return {"changed_px": a.size, "bbox": None}
    m = np.abs(a - b) > thresh
    n = int(m.sum())
    bbox = None
    if n:
        ys, xs = np.where(m)
        bbox = [int(xs.min()) * scale, int(ys.min()) * scale, int(xs.max() + 1) * scale, int(ys.max() + 1) * scale]
    return {"changed_px": n, "bbox": bbox}


def screen_hash(img: Image.Image) -> str:
    """Average hash, 16x16. Equal hashes = visually the same screen."""
    g = np.asarray(img.convert("L").resize((16, 16)), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return "".join("1" if v else "0" for v in bits)


def mark_click(img: Image.Image, x: float, y: float) -> Image.Image:
    out = img.copy()
    d = ImageDraw.Draw(out)
    r = max(8, out.size[0] // 40)
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=3)
    d.line([x - r - 4, y, x + r + 4, y], fill=(255, 0, 0), width=1)
    d.line([x, y - r - 4, x, y + r + 4], fill=(255, 0, 0), width=1)
    return out


def box(img: Image.Image, bbox, color=(0, 160, 255)) -> Image.Image:
    out = img.copy()
    if bbox:
        ImageDraw.Draw(out).rectangle(bbox, outline=color, width=3)
    return out


def crop_zoom(img: Image.Image, x: float, y: float, half: int = 120, zoom: int = 2) -> Image.Image:
    w, h = img.size
    l, t = int(max(0, x - half)), int(max(0, y - half))
    r, b = int(min(w, x + half)), int(min(h, y + half))
    c = img.crop((l, t, r, b))
    return c.resize((c.size[0] * zoom, c.size[1] * zoom), Image.LANCZOS)


def thumb(img: Image.Image, side: int = 360) -> Image.Image:
    out = img.copy()
    out.thumbnail((side, side))
    return out


def action_point(action: dict) -> Optional[Tuple[float, float]]:
    if action and "x" in action and "y" in action:
        try:
            return float(action["x"]), float(action["y"])
        except (TypeError, ValueError):
            return None
    return None
