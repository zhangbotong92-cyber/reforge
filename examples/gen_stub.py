"""A deliberately dumb generator, for demonstrating the closed loop on any laptop.

It does not run a model. It deterministically renders a synthetic image seeded by
the prompt, so `reforge refine` can be exercised end-to-end (and in CI) without a
GPU. Swap it for your real generator command and nothing else changes.

    reforge refine target.png --generator-kind command \
      --generator "python examples/gen_stub.py --prompt-file {prompt_file} --out {out} --seed {seed}"

Contract: read --prompt-file, write an image to --out.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def render(prompt: str, seed: int, size: int = 512) -> Image.Image:
    digest = hashlib.sha256(f"{prompt}|{seed}".encode("utf-8")).digest()
    base = np.frombuffer(digest[:9], dtype=np.uint8).astype(np.int32)

    y, x = np.mgrid[0:size, 0:size]
    r = ((np.sin(x / (18 + base[0] % 40)) * 50 + 120 + base[0]) % 255).astype(np.uint8)
    g = ((np.cos(y / (22 + base[1] % 40)) * 50 + 110 + base[1]) % 255).astype(np.uint8)
    b = (((x // 4 + y // 4) * (1 + base[2] % 5)) % 255).astype(np.uint8)
    arr = np.dstack([r, g, b])

    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = 120 + base[3] % 260, 120 + base[4] % 260
    rad = 60 + base[5] % 90
    draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=tuple(60 + (c % 170) for c in base[6:9]))
    draw.text((18, size - 26), f"stub generator | seed {seed}", fill=(255, 255, 255))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(prompt, args.seed, args.size).save(out, "PNG")
    print(f"stub: wrote {out} from {len(prompt)} chars of prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
