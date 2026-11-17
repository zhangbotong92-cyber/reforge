"""Generate synthetic fixtures so the repo is testable without any model.

    python examples/make_sample.py out/

Produces:
  a1111.png      -- WebUI `parameters` chunk (exact recipe recoverable)
  comfyui.png    -- ComfyUI API graph in the `prompt` chunk
  comfyui_ui.png -- ComfyUI UI-format `workflow` chunk (low-confidence path)
  novelai.png    -- NovelAI `Comment` JSON
  plain.jpg      -- nothing embedded (caption path)
  alt.png        -- visually similar variant, for scoring demos
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo


def _canvas(seed: int = 7, shift: int = 0, size: int = 512) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    r = (np.sin(x / 40.0 + shift) * 60 + 140).astype(np.uint8)
    g = (np.cos(y / 55.0) * 50 + 120).astype(np.uint8)
    b = ((x + y) % 255).astype(np.uint8)
    arr = np.dstack([r, g, b])
    arr = np.clip(arr + rng.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(img)
    d.ellipse((150 + shift, 140, 360 + shift, 350), fill=(240, 200, 160), outline=(40, 40, 40), width=3)
    d.rectangle((40, 400, 470, 470), fill=(30, 30, 40))
    d.text((50, 428), "REFORGE fixture", fill=(250, 250, 250))
    return img


A1111_PARAMS = (
    "a cinematic portrait of an astronaut standing in a neon rainstorm, "
    "volumetric light, 85mm, shallow depth of field, ultra detailed\n"
    "Negative prompt: blurry, lowres, watermark, extra fingers, jpeg artifacts\n"
    'Steps: 28, Sampler: DPM++ 2M Karras, CFG scale: 6.5, Seed: 1234567890, '
    'Size: 512x512, Model hash: 31e35c80fc, Model: sd_xl_base_1.0, '
    'Denoising strength: 0.45, Clip skip: 2, '
    'Lora hashes: "film_grain: 8f2b1c, neon_glow: 44aa90", Version: v1.10.1'
)

COMFY_GRAPH = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-dev-fp8.safetensors"}},
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "a lone lighthouse on a cliff at dusk, long exposure, muted teal palette", "clip": ["11", 0]},
    },
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "text, watermark, oversaturated", "clip": ["11", 0]}},
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 768, "height": 1024, "batch_size": 1},
    },
    "27": {"class_type": "LoraLoader", "inputs": {"lora_name": "add_detail.safetensors", "strength_model": 0.7}},
    "28": {"class_type": "ControlNetApplyAdvanced", "inputs": {"strength": 0.65}},
    "30": {"class_type": "IPAdapterAdvanced", "inputs": {"weight": 0.55}},
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 884422,
            "steps": 30,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
}

COMFY_UI = {
    "last_node_id": 9,
    "last_link_id": 9,
    "nodes": [
        {"id": 3, "type": "KSampler", "widgets_values": [777001, "randomize", 25, 4.0, "dpmpp_2m", "karras", 1.0]},
        {"id": 4, "type": "CheckpointLoaderSimple", "widgets_values": ["dreamshaper_8.safetensors"]},
        {"id": 6, "type": "CLIPTextEncode", "widgets_values": ["a cozy bookshop interior, warm window light, film photo"]},
        {"id": 7, "type": "CLIPTextEncode", "widgets_values": ["blurry, cgi"]},
        {"id": 5, "type": "EmptyLatentImage", "widgets_values": [832, 1216, 1]},
    ],
    "links": [],
}

NOVELAI = {
    "prompt": "masterpiece, best quality, 1girl, silver hair, city night, bokeh, cinematic",
    "uc": "lowres, bad anatomy, bad hands, signature",
    "steps": 28,
    "scale": 5.0,
    "seed": 42424242,
    "sampler": "k_euler_ancestral",
    "width": 832,
    "height": 1216,
    "software": "novelai",
}


def main(out_dir: str = "examples/out") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = _canvas()

    meta = PngInfo()
    meta.add_text("parameters", A1111_PARAMS)
    img.save(out / "a1111.png", "PNG", pnginfo=meta)

    meta = PngInfo()
    meta.add_itxt("prompt", json.dumps(COMFY_GRAPH))
    meta.add_itxt("workflow", json.dumps({"nodes": []}))
    _canvas(seed=11, shift=30).save(out / "comfyui.png", "PNG", pnginfo=meta)

    meta = PngInfo()
    meta.add_text("workflow", json.dumps(COMFY_UI))
    _canvas(seed=13).save(out / "comfyui_ui.png", "PNG", pnginfo=meta)

    meta = PngInfo()
    meta.add_text("Comment", json.dumps(NOVELAI))
    meta.add_text("Software", "NovelAI")
    _canvas(seed=17).save(out / "novelai.png", "PNG", pnginfo=meta)

    img.save(out / "plain.jpg", "JPEG", quality=92)
    _canvas(seed=7, shift=4).save(out / "alt.png", "PNG")

    print(f"fixtures written to {out.resolve()}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:18s} {f.stat().st_size/1024:8.1f} KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/out")
