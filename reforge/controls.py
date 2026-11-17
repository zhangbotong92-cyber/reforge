"""Layer 3 — control signals, extracted for real.

Text cannot carry geometry. This module turns an image (or a frame) into the
tensors a generator needs to lock layout and identity:

    edges / canny     clean line structure              (cv2, always available)
    lineart           thickened contour art             (cv2, always available)
    tile              luminance tile for tiling/upscale (cv2, always available)
    seg               foreground mask                   (GrabCut, approximate)
    depth             relative depth                    (DCP proxy, approximate)
    pose              skeleton keypoints                (model required)
    flow              temporal motion                   (model required)
    camera            camera trajectory                 (model required)

Three backends, in order of honesty:

    builtin  -- classical CV, zero model files, marked ``approximate`` when it is
    onnx     -- any Depth-Anything / DWPose / SAM ONNX you drop in the model dir
    command  -- your own CLI: REFORGE_CTL_DEPTH="python run.py --img {image} --out {out}"

Nothing here pretends to be a neural network it does not have. ``doctor()``
prints the capability matrix, and ``extract()`` raises ``MissingBackend`` with
the exact upstream command to run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

try:  # opencv is optional; builtin falls back to numpy-only paths
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

MODEL_DIR = Path(os.environ.get("REFORGE_MODEL_DIR", Path.home() / ".cache" / "reforge" / "models"))

KINDS = ("canny", "lineart", "tile", "seg", "depth", "pose", "flow", "camera")

# where each signal comes from, per backend
BACKENDS: Dict[str, Dict[str, Any]] = {
    "canny": {"impl": "builtin", "model": "cv2.Canny", "note": "structure lock, weight 0.6-0.75"},
    "lineart": {"impl": "builtin", "model": "cv2.Canny + dilate", "note": "clean line art for illustration"},
    "tile": {"impl": "builtin", "model": "Lab L channel", "note": "detail-preserving upscale / colour lock"},
    "seg": {
        "impl": "builtin",
        "model": "cv2.GrabCut",
        "note": "foreground mask; approximate, good for subject isolation",
        "approximate": True,
    },
    "depth": {
        "impl": "builtin|onnx|command",
        "model": "DCP dark-channel proxy (He et al. 2009) / Depth-Anything-V2",
        "url": "https://github.com/DepthAnything/Depth-Anything-V2",
        "note": "builtin is a proxy; put depth.onnx in the model dir for real depth",
        "approximate": True,
    },
    "pose": {
        "impl": "onnx|command",
        "model": "DWPose / RTMPose",
        "url": "https://github.com/IDEA-Research/DWPose",
        "note": "needs a keypoint model: pose.onnx, or REFORGE_CTL_POSE=<command template>",
    },
    "flow": {
        "impl": "onnx|command",
        "model": "RAFT",
        "url": "https://github.com/princeton-vl/RAFT",
        "note": "video temporal control; needs two frames",
    },
    "camera": {
        "impl": "command",
        "model": "VGGT / MegaSaM",
        "url": "https://github.com/facebookresearch/vggt",
        "note": "camera trajectory JSON for ReCamMaster-style re-rendering",
    },
}


class MissingBackend(RuntimeError):
    """Raised when a signal needs a model or tool that is not installed."""


# ---------------------------------------------------------------- builtin extraction


def edges(src: str | Path, dst: str | Path, threshold: int = 40, max_side: int = 1024) -> Path:
    """Canny-like edge map, white on black. Uses cv2 when present, PIL otherwise."""
    dst = Path(dst)
    with Image.open(src) as img:
        gray = img.convert("L")
        gray.thumbnail((max_side, max_side))
        arr = np.asarray(gray, dtype=np.uint8)

    if cv2 is not None:
        mask = cv2.Canny(arr, threshold, threshold * 2, L2gradient=True)
    else:  # numpy Sobel fallback
        from PIL import ImageFilter

        mask = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.uint8)
        mask = np.where(mask > threshold, 255, 0).astype(np.uint8)

    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, "L").save(dst)
    return dst


def lineart(src: str | Path, dst: str | Path, threshold: int = 30, thickness: int = 2) -> Path:
    """Contour art: edges, cleaned and thickened."""
    tmp_edges = Path(dst).with_name(Path(dst).stem + ".raw.png")
    edges(src, tmp_edges, threshold=threshold)
    arr = np.asarray(Image.open(tmp_edges).convert("L"))
    if cv2 is not None:
        kernel = np.ones((thickness, thickness), np.uint8)
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
        arr = 255 - arr  # black lines on white, as ControlNet lineart expects
    tmp_edges.unlink(missing_ok=True)
    Image.fromarray(arr, "L").save(dst)
    return Path(dst)


def tile(src: str | Path, dst: str | Path) -> Path:
    """Luminance tile (Lab-L): colour-agnostic detail reference."""
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        if cv2 is not None:
            lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)
            l_chan = lab[:, :, 0]
        else:
            l_chan = np.asarray(rgb.convert("L"))
    Image.fromarray(l_chan.astype(np.uint8), "L").save(dst)
    return Path(dst)


def seg(src: str | Path, dst: str | Path, iterations: int = 3) -> Path:
    """Foreground mask via GrabCut. Approximate by construction."""
    if cv2 is None:
        raise MissingBackend("seg needs opencv: pip install opencv-python-headless")
    with Image.open(src) as img:
        rgb = np.asarray(img.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (
        int(w * 0.08),
        int(h * 0.08),
        int(w * 0.84),
        int(h * 0.84),
    )
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iterations, cv2.GC_INIT_WITH_RECT)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    Image.fromarray(out, "L").save(dst)
    return Path(dst)


def depth(src: str | Path, dst: str | Path, radius: int = 15, max_side: int = 768) -> Path:
    """Approximate relative depth from the dark-channel prior.

    ``depth = 1 - dark_channel`` after a local-max normalisation: hazy (far)
    regions have a bright dark channel, so they land dark in the output. This is
    a documented classical cue, **not** a learned depth model — the recipe marks
    it ``approximate`` so nothing downstream over-claims.
    """
    with Image.open(src) as img:
        rgb = np.asarray(img.convert("RGB").resize((min(img.width, max_side), min(img.height, max_side))), dtype=np.float32) / 255.0

    dark = rgb.min(axis=2)
    if cv2 is not None:
        kernel = np.ones((radius, radius), np.uint8)
        dark = cv2.erode(dark, kernel)
        dark = cv2.GaussianBlur(dark, (0, 0), radius / 3.0)
    else:  # crude min-pool + blur
        from PIL import ImageFilter

        dark = np.asarray(
            Image.fromarray((dark * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(min(radius, 9))),
            dtype=np.float32,
        ) / 255.0

    lo, hi = float(dark.min()), float(dark.max())
    norm = (dark - lo) / (hi - lo + 1e-6)
    out = ((1.0 - norm) * 255).astype(np.uint8)
    Image.fromarray(out, "L").save(dst)
    return Path(dst)


# ---------------------------------------------------------------- onnx backend


def _model_path(kind: str) -> Optional[Path]:
    for suffix in (".onnx",):
        p = MODEL_DIR / f"{kind}{suffix}"
        if p.exists():
            return p
    return None


def _model_config(kind: str) -> Dict[str, Any]:
    cfg_path = MODEL_DIR / f"{kind}.json"
    defaults = {"input": None, "output": None, "size": [518, 518], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    if cfg_path.exists():
        try:
            defaults.update(json.loads(cfg_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return defaults


def onnx_depth(src: str | Path, dst: str | Path, kind: str = "depth") -> Path:
    """Run a user-supplied ONNX model (Depth-Anything / DWPose / SAM style)."""
    model = _model_path(kind)
    if model is None:
        raise MissingBackend(
            f"no {kind}.onnx in {MODEL_DIR}. Drop a model there (optional {kind}.json for shape/mean/std), "
            f"or set REFORGE_CTL_{kind.upper()} to a command template. "
            f"Depth: {BACKENDS['depth']['url']}  Pose: {BACKENDS['pose']['url']}"
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise MissingBackend("onnxruntime missing: pip install onnxruntime") from exc

    cfg = _model_config(kind)
    size = int(cfg["size"][0]), int(cfg["size"][1])
    with Image.open(src) as img:
        resized = img.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
    tensor = arr.transpose(2, 0, 1)[None, ...]

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = cfg.get("input") or sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: tensor})
    pred = np.asarray(outputs[0]).squeeze()

    if kind == "pose" and pred.ndim == 2:  # (N, K) keypoints -> draw skeleton
        return _draw_pose(pred, dst, src)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6) * 255.0
    with Image.open(src) as img:
        out = Image.fromarray(pred.astype(np.uint8), "L").resize(img.size, Image.BILINEAR)
    out.save(dst)
    return Path(dst)


def _draw_pose(points: np.ndarray, dst: str | Path, src: str | Path) -> Path:
    """Render a sparse keypoint set onto a black canvas (control-image style)."""
    with Image.open(src) as img:
        w, h = img.size
    canvas = np.zeros((h, w), np.uint8)
    for x, y in points[:30]:
        cx, cy = int(x * w), int(y * h)
        if 0 <= cx < w and 0 <= cy < h:
            if cv2 is not None:
                cv2.circle(canvas, (cx, cy), 4, 255, -1)
            else:  # pragma: no cover
                canvas[max(0, cy - 4) : cy + 4, max(0, cx - 4) : cx + 4] = 255
    Image.fromarray(canvas, "L").save(dst)
    return Path(dst)


# ---------------------------------------------------------------- command backend


def command_extract(kind: str, src: str | Path, dst: str | Path) -> Path:
    """Escape hatch: let the user's own CLI produce the map."""
    template = os.environ.get(f"REFORGE_CTL_{kind.upper()}")
    if not template:
        raise MissingBackend(
            f"no {kind} backend. Set REFORGE_CTL_{kind.upper()}='python run.py --img {{image}} --out {{out}}' "
            f"or place {kind}.onnx in {MODEL_DIR}."
        )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    argv = [tok.format(image=str(src), out=str(dst), kind=kind) for tok in template.split()]
    proc = subprocess.run(argv, capture_output=True, timeout=900)
    if proc.returncode != 0:
        raise MissingBackend(f"{kind} command failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[-400:]}")
    if not dst.exists():
        raise MissingBackend(f"{kind} command exited 0 but wrote nothing to {dst}")
    return dst


# ---------------------------------------------------------------- dispatcher


def available_backend(kind: str) -> str:
    """Which backend would serve this kind right now."""
    if kind in ("canny", "lineart", "tile"):
        return "builtin"
    if kind == "seg":
        return "builtin" if cv2 is not None else "none"
    if os.environ.get(f"REFORGE_CTL_{kind.upper()}"):
        return "command"
    if _model_path(kind):
        return "onnx"
    if kind == "depth":
        return "builtin"
    return "none"


def extract(kind: str, src: str | Path, dst: str | Path, backend: str = "auto", **options: Any) -> Dict[str, Any]:
    """Extract one control signal. Returns metadata for the recipe entry."""
    backend = available_backend(kind) if backend == "auto" else backend
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if backend == "none":
        raise MissingBackend(
            f"'{kind}' unavailable. Place {kind}.onnx in {MODEL_DIR} "
            f"({BACKENDS.get(kind, {}).get('url', 'see UPSTREAM.md')}) "
            f"or set REFORGE_CTL_{kind.upper()}."
        )

    if backend == "builtin":
        fn = {"canny": edges, "edges": edges, "lineart": lineart, "tile": tile, "seg": seg, "depth": depth}.get(kind)
        if fn is None:
            raise MissingBackend(f"'{kind}' has no builtin implementation")
        fn(src, dst, **{k: v for k, v in options.items() if k in ("threshold", "thickness", "iterations", "radius")})
        approximate = bool(BACKENDS.get(kind, {}).get("approximate"))
        method = BACKENDS.get(kind, {}).get("model", "cv2")
    elif backend == "onnx":
        onnx_depth(src, dst, kind=kind)
        approximate, method = False, f"{kind}.onnx"
    elif backend == "command":
        command_extract(kind, src, dst)
        approximate, method = False, f"command: REFORGE_CTL_{kind.upper()}"
    else:
        raise MissingBackend(f"unknown backend {backend!r}")

    return {"kind": kind, "path": str(dst), "backend": backend, "method": method, "approximate": approximate}


def extract_all(src: str | Path, outdir: str | Path, kinds: Optional[List[str]] = None, backend: str = "auto") -> Dict[str, Any]:
    """Extract every available control; report what was skipped and why."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    wanted = kinds or ["canny", "lineart", "tile", "seg", "depth"]
    done: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for kind in wanted:
        try:
            done.append(extract(kind, src, outdir / f"{kind}.png", backend=backend))
        except MissingBackend as exc:
            skipped.append({"kind": kind, "reason": str(exc)})
    return {"controls": done, "skipped": skipped, "outdir": str(outdir)}


# ---------------------------------------------------------------- media probe


def palette(src: str | Path, k: int = 5) -> List[Dict[str, Any]]:
    with Image.open(src) as img:
        small = img.convert("RGB")
        small.thumbnail((256, 256))
        quant = small.quantize(colors=k, method=Image.MEDIANCUT)
        pal = quant.getpalette() or []
        counts = quant.getcolors() or []
    total = sum(c for c, _ in counts) or 1
    out = []
    for count, idx in sorted(counts, reverse=True)[:k]:
        rgb = tuple(pal[idx * 3 : idx * 3 + 3])
        out.append({"hex": "#%02x%02x%02x" % rgb, "rgb": list(rgb), "share": round(count / total, 3)})
    return out


def probe(path: str | Path) -> Dict[str, Any]:
    """Media probe: images via Pillow, video via ffprobe when present."""
    p = Path(path)
    info: Dict[str, Any] = {"path": str(p), "bytes": p.stat().st_size}
    try:
        with Image.open(p) as img:
            info.update({"kind": "image", "width": img.width, "height": img.height, "mode": img.mode})
            return info
    except Exception:  # noqa: BLE001 - may be a video
        pass
    if shutil.which("ffprobe"):
        try:
            raw = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,codec_name",
                    "-show_entries", "format=duration",
                    "-of", "json", str(p),
                ],
                capture_output=True, check=True, timeout=30,
            ).stdout
            blob = json.loads(raw)
            stream = (blob.get("streams") or [{}])[0]
            info.update(
                {
                    "kind": "video",
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "fps": stream.get("avg_frame_rate"),
                    "frames": stream.get("nb_frames"),
                    "codec": stream.get("codec_name"),
                    "duration": (blob.get("format") or {}).get("duration"),
                }
            )
            return info
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
    info["kind"] = "unknown"
    info["note"] = "install ffmpeg to probe video containers"
    return info


def keyframes(video: str | Path, outdir: str | Path, count: int = 8) -> List[Path]:
    """Evenly sample frames; needs ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise MissingBackend(
            "ffmpeg not found. Install it, or pass --frames-dir with pre-extracted frames."
        )
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    duration = float(info.get("duration") or 0) or 1.0
    fps = count / duration
    pattern = str(out / "frame_%03d.png")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={max(fps, 0.01):.4f}", "-frames:v", str(count), pattern],
        check=True,
        timeout=300,
    )
    return sorted(out.glob("frame_*.png"))


def scenes(video: str | Path, outdir: str | Path, threshold: float = 0.3) -> List[Path]:
    """Shot-boundary detection via ffmpeg's scene score."""
    if not shutil.which("ffmpeg"):
        raise MissingBackend("ffmpeg not found: required for shot detection.")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(video),
            "-filter:v", f"select='gt(scene,{threshold})',showinfo",
            "-vsync", "vfr", str(out / "shot_%03d.png"),
        ],
        check=True,
        timeout=600,
    )
    return sorted(out.glob("shot_*.png"))


# ---------------------------------------------------------------- capability matrix


def doctor() -> Dict[str, Any]:
    """What can actually run here, and what is one download away."""
    rows = {}
    for kind in KINDS:
        backend = available_backend(kind)
        row: Dict[str, Any] = {"backend": backend, "available": backend != "none"}
        if backend == "builtin":
            row["method"] = BACKENDS.get(kind, {}).get("model", "builtin")
            row["approximate"] = bool(BACKENDS.get(kind, {}).get("approximate"))
        elif backend == "onnx":
            row["method"] = str(_model_path(kind))
        elif backend == "command":
            row["method"] = f"REFORGE_CTL_{kind.upper()}"
        else:
            row["install"] = BACKENDS.get(kind, {}).get("url")
            row["hint"] = BACKENDS.get(kind, {}).get("note")
        rows[kind] = row
    return {
        "opencv": bool(cv2),
        "onnxruntime": _has("onnxruntime"),
        "torch": _has("torch"),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "model_dir": str(MODEL_DIR),
        "kinds": rows,
    }


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- execution plan


def plan(recipe: Dict[str, Any], runtime: str = "diffusers") -> Dict[str, Any]:
    """Turn a recipe into an actionable pipeline description.

    ``runtime`` is one of ``diffusers`` | ``comfyui`` | ``api``.
    Nothing here executes a model: it is the hand-off document that a worker
    (or a human) follows, and it is what makes the repo usable before any GPU
    is attached.
    """
    controls = recipe.get("controls") or []
    refs = recipe.get("refs") or []
    fidelity = recipe.get("fidelity", "L1-semantic")
    sampling = recipe.get("sampling") or {}
    base_model = (recipe.get("models") or [{}])[0].get("name") or "stabilityai/stable-diffusion-xl-base-1.0"

    steps: List[Dict[str, Any]] = []

    if recipe.get("init", {}).get("latents"):
        steps.append(
            {
                "op": "load_latents",
                "path": recipe["init"]["latents"],
                "method": recipe["init"].get("method", "rf-solver"),
                "note": "inversion latents pin the output to the original pixels",
            }
        )
    if recipe.get("init", {}).get("image"):
        steps.append({"op": "img2img", "image": recipe["init"]["image"], "denoise": recipe["init"].get("denoising", 0.35)})

    for c in controls:
        steps.append(
            {
                "op": "apply_control",
                "kind": c.get("kind"),
                "model": c.get("model") or c.get("method"),
                "path": c.get("path"),
                "weight": c.get("weight"),
                "source": c.get("source"),
                "approximate": c.get("approximate", False),
            }
        )
    for r in refs:
        steps.append({"op": "apply_reference", "path": r.get("path"), "weight": r.get("weight"), "role": r.get("role")})

    for a in recipe.get("adapters", []):
        steps.append({"op": "load_adapter", "kind": a.get("kind", "lora"), "name": a.get("name"), "strength": a.get("strength", 0.8)})

    steps.append(
        {
            "op": "sample",
            "model": base_model,
            "prompt": recipe.get("prompt"),
            "negative": recipe.get("negative"),
            "seed": sampling.get("seed"),
            "steps": sampling.get("steps") or 28,
            "cfg": sampling.get("cfg") or 6.0,
            "sampler": sampling.get("sampler") or "euler",
            "size": [recipe.get("source", {}).get("width"), recipe.get("source", {}).get("height")],
        }
    )

    return {
        "runtime": runtime,
        "fidelity_target": fidelity,
        "fingerprint": recipe.get("fingerprint"),
        "steps": steps,
        "gaps": gaps(recipe),
    }


def gaps(recipe: Dict[str, Any]) -> List[str]:
    out = []
    if not recipe.get("sampling", {}).get("seed"):
        out.append("seed: output will vary between runs")
    if not recipe.get("controls") and not recipe.get("refs"):
        out.append("no structural control: only semantic similarity is achievable")
    if any(c.get("approximate") for c in recipe.get("controls") or []):
        out.append("an approximate control is in use: install an ONNX model for real depth/pose")
    if not recipe.get("models"):
        out.append("no checkpoint pinned: results drift across model updates")
    if recipe.get("shots"):
        unanchored = [s["index"] for s in recipe["shots"] if not s.get("first_frame")]
        if unanchored:
            out.append(f"shots {unanchored} have no first-frame anchor: identity will drift")
    return out


_RUNTIME_HINTS = {
    "diffusers": "QwenImageEditPlusPipeline / FluxKontextPipeline for edits; StableDiffusionControlNetPipeline for L2.",
    "comfyui": "LoadImage -> ControlNetApplyAdvanced -> KSampler; use IPAdapter Advanced for refs; VHS for video.",
    "api": "Send the original image + the 'banana' dialect prompt: instruction-editing models hold identity best.",
}


def describe(runtime: str) -> str:
    return _RUNTIME_HINTS.get(runtime, "")
