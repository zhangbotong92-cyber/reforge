"""Layer 3 — closed-loop scoring.

"Reproducible" only means something if you can measure it.  These metrics are
deliberately model-free (numpy + Pillow): they run anywhere, cost nothing, and
already tell you whether a regeneration is in the right ballpark.

Heavy metrics (CLIP / DINO / ArcFace) plug in through :func:`score_pair`'s
``extra`` argument once you install the optional extras.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # Pillow is a hard dependency for image I/O
    from PIL import Image, ImageFilter
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc

__all__ = [
    "load_gray",
    "psnr",
    "ssim",
    "phash",
    "phash_distance",
    "edge_map",
    "edge_iou",
    "palette",
    "palette_distance",
    "score_pair",
]


def load_gray(path: str | Path, size: Optional[int] = None) -> np.ndarray:
    with Image.open(path) as img:
        gray = img.convert("L")
        if size:
            gray = gray.resize((size, size), Image.LANCZOS)
        return np.asarray(gray, dtype=np.float64)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((255.0**2) / mse)


def _uniform(x: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    padded = np.pad(x, pad, mode="reflect")
    view = np.lib.stride_tricks.sliding_window_view(padded, (win, win))
    return view.mean(axis=(2, 3))


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7) -> float:
    """Mean SSIM with a uniform window (no scipy / skimage dependency)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if min(a.shape) < win:
        win = max(3, min(a.shape) // 2 * 2 + 1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _uniform(a, win), _uniform(b, win)
    mu_a2, mu_b2, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b
    sig_a2 = _uniform(a * a, win) - mu_a2
    sig_b2 = _uniform(b * b, win) - mu_b2
    sig_ab = _uniform(a * b, win) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    return float(np.mean(num / den))


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    return np.cos(np.pi * (2 * x + 1) * k / (2 * n))


def phash(path: str | Path, hash_size: int = 8, highfreq: int = 4) -> int:
    """Perceptual hash: 64-bit DCT hash, robust to resize/recompression."""
    size = hash_size * highfreq
    gray = load_gray(path, size=size)
    d = _dct_matrix(size)
    coeffs = d @ gray @ d.T
    low = coeffs[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    bits = 0
    for i, v in enumerate(low):
        if v > med:
            bits |= 1 << i
    return bits


def phash_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def edge_map(path: str | Path, threshold: int = 40, max_side: int = 512) -> np.ndarray:
    with Image.open(path) as img:
        gray = img.convert("L")
        gray.thumbnail((max_side, max_side))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        arr = np.asarray(edges, dtype=np.float64)
    return (arr > threshold).astype(np.uint8)


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def edge_iou(a_path: str | Path, b_path: str | Path, threshold: int = 40) -> float:
    a, b = edge_map(a_path, threshold), edge_map(b_path, threshold)
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]
    a, b = _dilate(a), _dilate(b)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union


def palette(path: str | Path, k: int = 5) -> Sequence[Tuple[int, int, int]]:
    with Image.open(path) as img:
        small = img.convert("RGB")
        small.thumbnail((256, 256))
        quant = small.quantize(colors=k, method=Image.MEDIANCUT)
        pal = quant.getpalette() or []
        counts = sorted(quant.getcolors() or [], reverse=True)
    colors: list[Tuple[int, int, int]] = []
    for _count, idx in counts[:k]:
        r, g, b = pal[idx * 3 : idx * 3 + 3]
        colors.append((r, g, b))
    return colors


def palette_distance(a_path: str | Path, b_path: str | Path, k: int = 5) -> float:
    pa, pb = palette(a_path, k), palette(b_path, k)
    if not pa or not pb:
        return 1.0
    total = 0.0
    for ca in pa:
        best = min(sum((x - y) ** 2 for x, y in zip(ca, cb)) ** 0.5 for cb in pb)
        total += best
    return total / (len(pa) * 441.67)  # normalise by max RGB distance (sqrt(3)*255)


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def score_pair(
    target: str | Path,
    candidate: str | Path,
    weights: Optional[Dict[str, float]] = None,
    size: int = 512,
) -> Dict[str, Any]:
    """Compare a regeneration against the original. Returns raw metrics + composite."""
    ta, ca = load_gray(target, size=size), load_gray(candidate, size=size)
    raw = {
        "psnr": psnr(ta, ca),
        "ssim": ssim(ta, ca),
        "phash": phash_distance(phash(target), phash(candidate)),
        "edge_iou": edge_iou(target, candidate),
        "palette_distance": palette_distance(target, candidate),
    }
    raw["phash_similarity"] = 1.0 - raw["phash"] / 64.0

    w = weights or {
        "ssim": 0.35,
        "edge_iou": 0.25,
        "phash_similarity": 0.2,
        "palette_similarity": 0.15,
        "psnr": 0.05,
    }
    parts = {
        "ssim": raw["ssim"],
        "edge_iou": raw["edge_iou"],
        "phash_similarity": raw["phash_similarity"],
        "palette_similarity": 1.0 - raw["palette_distance"],
        "psnr": _norm(raw["psnr"], 10.0, 40.0),
    }
    total_w = sum(w.values()) or 1.0
    composite = sum(parts[k] * w.get(k, 0.0) for k in parts) / total_w

    raw["composite"] = round(composite, 4)
    raw["grade"] = _grade(composite)
    raw["parts"] = {k: round(v, 4) for k, v in parts.items()}
    return raw


def _grade(score: float) -> str:
    if score >= 0.90:
        return "L3-pixel"
    if score >= 0.75:
        return "L2-structural"
    if score >= 0.55:
        return "L1-semantic"
    return "L0-drift"
