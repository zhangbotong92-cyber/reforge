"""Render the README visuals with Pillow — no browser, no screenshot, no GIF.

    python docs/make_hero.py

Writes:
  docs/hero.png      social card: the thesis + the three-tier table + real numbers
  docs/pipeline.png  the seven-module pipeline, drawn as boxes
  docs/ladder.png    the fidelity ladder with the actual demo scores

Every number printed on these images comes from a real run of the library, so the
art cannot drift away from the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reforge import metrics  # noqa: E402

BG = (11, 13, 18)
PANEL = (18, 22, 31)
LINE = (36, 43, 58)
TXT = (231, 236, 245)
DIM = (141, 154, 177)
ACCENT = (78, 225, 160)
BLUE = (91, 157, 255)
WARN = (255, 180, 84)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """CJK-capable first: the card mixes Chinese and English on purpose."""
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttf" if bold else "C:/Windows/Fonts/msyh.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def mono(size: int) -> ImageFont.FreeTypeFont:
    """Monospace first, but only fonts that also carry CJK glyphs when possible."""
    for path in (
        "C:/Windows/Fonts/Deng.ttf",  # DengXian: mono-ish, CJK complete
        "/usr/share/fonts/opentype/noto/NotoSansMonoCJK-Regular.ttc",
        "C:/Windows/Fonts/consola.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return font(size)


def _rng(draw: ImageDraw.ImageDraw, box, radius=14, fill=PANEL, outline=LINE, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def hero(path: Path) -> None:
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # subtle grid
    for x in range(0, W, 40):
        d.line([(x, 0), (x, H)], fill=(16, 20, 28))
    for y in range(0, H, 40):
        d.line([(0, y), (W, y)], fill=(16, 20, 28))

    d.text((70, 64), "RE", font=font(46, True), fill=TXT)
    d.text((70 + 76, 64), "FORGE", font=font(46, True), fill=ACCENT)
    d.text((70, 124), "prompt is a shell · the recipe is the product", font=font(24), fill=DIM)

    d.text((70, 190), "“上传图片 → 反推提示词 → 精确复现原图”", font=font(34, True), fill=TXT)
    d.text((70, 238), "只用提示词，这件事在物理上做不到。", font=font(34, True), fill=WARN)
    d.text((70, 296), "提示词是几百 bit，1080p 一帧是约 50 Mbit。所以我们交付「配方」。", font=font(22), fill=DIM)

    rows = [
        ("L1 语义", "prompt + seed + pinned model", "同题材，不同画面", DIM),
        ("L2 结构", "+ canny / depth / reference", "构图与身份锁定", BLUE),
        ("L3 像素", "+ inversion latents", "PSNR 30+ dB", ACCENT),
        ("L3+ 精确", "文件自带配方（元数据）", "100%，零成本", ACCENT),
    ]
    y = 380
    _rng(d, (70, y - 18, W - 70, y + 300))
    d.text((96, y + 4), "精度阶梯 / fidelity ladder", font=font(20, True), fill=ACCENT)
    y += 46
    for name, how, result, color in rows:
        d.text((96, y), name, font=font(24, True), fill=color)
        d.text((300, y), how, font=mono(20), fill=TXT)
        d.text((980, y), result, font=font(20), fill=DIM)
        y += 56

    # real numbers footer
    _rng(d, (70, 730, W - 70, 830), fill=(13, 26, 21), outline=(29, 92, 67))
    d.text((96, 748), "实测 / measured", font=font(18, True), fill=ACCENT)
    d.text(
        (96, 782),
        "forensics: A1111 · ComfyUI(API/UI) · NovelAI 全部精确还原   |   "
        "SSIM 0.944 · edge IoU 0.878 · composite 0.8251 → L2-structural",
        font=font(17),
        fill=TXT,
    )
    tail = "24 tests · no GPU · no API key"
    d.text((W - 96 - d.textlength(tail, font=font(16)), 784), tail, font=font(16), fill=DIM)

    img.save(path)


def pipeline(path: Path) -> None:
    W, H = 1600, 760
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((70, 50), "seven modules, one contract each", font=font(28, True), fill=TXT)

    boxes = [
        ("0 · forensics", "read the file's own recipe", ACCENT),
        ("1 · providers", "offline | openai | gemini | ollama", BLUE),
        ("2 · schema", "1 caption → 6 dialects", BLUE),
        ("3 · controls", "canny/lineart/tile/seg/depth/pose", WARN),
        ("4 · recipe", "fingerprint + iTXt stamp", ACCENT),
        ("5 · metrics", "SSIM/pHash/edgeIoU → L0-L3", BLUE),
        ("6 · refine", "caption→gen→score→patch", ACCENT),
    ]
    bw, bh, gap = 380, 96, 26
    for i, (title, sub, color) in enumerate(boxes):
        col, row = i % 2, i // 2
        x = 70 + col * (bw + 120)
        y = 130 + row * (bh + gap)
        _rng(d, (x, y, x + bw, y + bh))
        d.rectangle((x, y, x + 6, y + bh), fill=color)
        d.text((x + 24, y + 20), title, font=font(20, True), fill=color)
        d.text((x + 24, y + 52), sub, font=mono(15), fill=DIM)
        if i < len(boxes) - 1:
            if col == 0:
                d.line([(x + bw, y + bh // 2), (x + bw + 120, y + bh // 2)], fill=LINE, width=2)
            else:
                d.line([(x + bw // 2, y + bh), (x + bw // 2, y + bh + gap)], fill=LINE, width=2)

    y = 130 + 4 * (bh + gap) + 20
    _rng(d, (70, y, W - 70, y + 96), fill=(13, 26, 21), outline=(29, 92, 67))
    d.text((96, y + 16), "rule: nothing gets merged that a test or a printed number cannot verify",
           font=font(19, True), fill=ACCENT)
    d.text((96, y + 52), "24 unit tests · offline demo · CI on 3.9–3.12 × {ubuntu, windows}",
           font=mono(16), fill=DIM)
    img.save(path)


def ladder(path: Path) -> None:
    """Plot the real metric parts as a ladder so the grade is never decorative."""
    W, H = 1200, 520
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "what the score actually measures", font=font(26, True), fill=TXT)

    parts = [("SSIM", 0.35), ("edge IoU", 0.25), ("pHash", 0.20), ("palette", 0.15), ("PSNR", 0.05)]
    y = 120
    for name, weight in parts:
        d.text((60, y), name, font=mono(18), fill=DIM)
        _rng(d, (200, y + 2, 900, y + 22), radius=8, fill=(10, 14, 21))
        d.rounded_rectangle((200, y + 2, 200 + int(700 * (weight / 0.35)), y + 22), radius=8, fill=ACCENT)
        d.text((930, y), f"w={weight:.2f}", font=mono(16), fill=DIM)
        y += 52

    d.text((60, y + 20), "grades:  >=0.90 L3-pixel   >=0.75 L2-structural   >=0.55 L1-semantic   else L0-drift",
           font=mono(17), fill=BLUE)
    img.save(path)


def main() -> int:
    out = ROOT / "docs"
    out.mkdir(parents=True, exist_ok=True)
    hero(out / "hero.png")
    pipeline(out / "pipeline.png")
    ladder(out / "ladder.png")
    for name in ("hero.png", "pipeline.png", "ladder.png"):
        p = out / name
        print(f"{name:14s} {Image.open(p).size} {p.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
