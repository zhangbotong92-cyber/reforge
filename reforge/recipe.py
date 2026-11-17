"""Layer 4 — the recipe: the thing that actually makes reproduction possible.

A prompt alone is a lossy description. A recipe is a *build artifact*: prompt +
init/denoise + seed + pinned checkpoints + control signals + reference hashes +
optional inversion latents, plus the fidelity tier it claims.

The killer feature is :func:`embed_into_png` — write the recipe back into the
file (iTXt ``reforge:recipe``). Then the file carries its own reconstruction
instructions, so anyone, on any stack, can rebuild it exactly. That is the only
honest answer to "the same prompt must reproduce the same image".
"""

from __future__ import annotations

import hashlib
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "reforge.recipe/1"
EMBED_KEY = "reforge:recipe"
PNG_SIG = b"\x89PNG\r\n\x1a\n"

CONTROL_KINDS = {
    "canny": ("controlnet-canny", 0.65),
    "depth": ("controlnet-depth", 0.60),
    "pose": ("dwpose-openpose", 0.55),
    "seg": ("sam-seg", 0.50),
    "lineart": ("controlnet-lineart", 0.60),
    "tile": ("controlnet-tile", 0.45),
    "ref": ("ip-adapter|reference-only", 0.55),
    "flow": ("raft-flow", 0.40),
}


def _sha256(path: str | Path, limit: int = 32 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        remaining = limit
        while remaining > 0:
            chunk = fh.read(min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def new_recipe(**fields: Any) -> Dict[str, Any]:
    recipe: Dict[str, Any] = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fidelity": "L1-semantic",
        "source": {},
        "tool": None,
        "prompt": "",
        "negative": "",
        "init": {},          # image-to-image / denoise / inversion
        "sampling": {},      # steps / cfg / sampler / scheduler / seed
        "models": [],        # [{"role": "base", "name": ..., "hash": ...}]
        "adapters": [],      # LoRA / LyCORIS / IP-Adapter weights
        "controls": [],      # [{"kind": "depth", "path": ..., "weight": ...}]
        "refs": [],          # reference images for identity/style
        "camera": {},        # video: camera trajectory / fov
        "shots": [],         # video: per-shot sub-recipes
        "caption": None,     # full canonical caption (for re-compilation)
        "metrics": None,     # last verification report
        "notes": [],
    }
    recipe.update(fields)
    return recipe


def from_inspection(report: Dict[str, Any], caption: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a recipe from a forensic report (+ optional caption)."""
    rec = report.get("recovered", {})
    src = report.get("path", "")
    recipe = new_recipe(
        tool=report.get("tool"),
        source={
            "path": str(src),
            "format": report.get("format"),
            "width": report.get("width"),
            "height": report.get("height"),
            "sha256": _sha256(src) if src and Path(src).exists() else None,
        },
        prompt=rec.get("prompt", "") or "",
        negative=rec.get("negative", "") or "",
        caption=caption,
    )
    recipe["sampling"] = {
        "seed": rec.get("seed"),
        "steps": rec.get("steps"),
        "cfg": rec.get("cfg"),
        "sampler": rec.get("sampler"),
        "scheduler": rec.get("scheduler"),
        "denoising": rec.get("denoising"),
        "clip_skip": rec.get("clip_skip"),
    }
    if rec.get("model") or rec.get("model_hash"):
        recipe["models"].append(
            {"role": "base", "name": rec.get("model"), "hash": rec.get("model_hash"), "pinned": bool(rec.get("model_hash"))}
        )
    for lora in rec.get("loras") or []:
        recipe["adapters"].append({"kind": "lora", **lora})
    for name in rec.get("controls") or []:
        recipe["controls"].append({"kind": _guess_control_kind(name), "source": name, "weight": _default_weight(name)})
    if rec.get("tool_version"):
        recipe["notes"].append(f"written by {recipe['tool']} {rec['tool_version']}")
    recipe["fidelity"] = fidelity_level(recipe)
    return recipe


def from_caption(caption: Dict[str, Any], report: Optional[Dict[str, Any]] = None, prompt: Optional[str] = None) -> Dict[str, Any]:
    report = report or {}
    rec = report.get("recovered", {})
    recipe = new_recipe(
        prompt=prompt or rec.get("prompt", "") or caption.get("summary", ""),
        negative=rec.get("negative", "") or ", ".join(caption.get("negative", [])),
        caption=caption,
        source={
            "path": report.get("path"),
            "format": report.get("format"),
            "width": report.get("width"),
            "height": report.get("height"),
            "sha256": _sha256(report["path"]) if report.get("path") and Path(report["path"]).exists() else None,
        },
        tool=report.get("tool"),
    )
    recipe["sampling"] = {"seed": rec.get("seed"), "steps": rec.get("steps"), "cfg": rec.get("cfg")}
    if rec.get("model"):
        recipe["models"].append({"role": "base", "name": rec["model"], "hash": rec.get("model_hash")})
    recipe["fidelity"] = fidelity_level(recipe)
    recipe["notes"].append("prompt is a visual caption; add controls or init latents to raise fidelity")
    return recipe


def _guess_control_kind(node_name: str) -> str:
    low = node_name.lower()
    for kind in ("canny", "depth", "pose", "seg", "lineart", "tile", "flow"):
        if kind in low:
            return kind
    if "ipadapter" in low or "reference" in low:
        return "ref"
    return "control"


def _default_weight(node_name: str) -> Optional[float]:
    kind = _guess_control_kind(node_name)
    return CONTROL_KINDS.get(kind, (None, None))[1]


def fidelity_level(recipe: Dict[str, Any]) -> str:
    """Claim a fidelity tier from what the recipe actually contains."""
    init = recipe.get("init") or {}
    if init.get("latents"):
        return "L3-pixel"
    if recipe.get("controls") or recipe.get("refs") or init.get("image"):
        return "L2-structural"
    if recipe.get("sampling", {}).get("seed") is not None and recipe.get("models"):
        return "L1-reproducible"
    return "L1-semantic"


def add_control(recipe: Dict[str, Any], kind: str, path: str, weight: Optional[float] = None, source: str = "extracted") -> Dict[str, Any]:
    model, default_weight = CONTROL_KINDS.get(kind, (kind, 0.5))
    recipe["controls"].append(
        {
            "kind": kind,
            "model": model,
            "path": str(path),
            "weight": default_weight if weight is None else float(weight),
            "source": source,
        }
    )
    recipe["fidelity"] = fidelity_level(recipe)
    return recipe


def add_ref(recipe: Dict[str, Any], path: str, weight: float = 0.6, role: str = "identity") -> Dict[str, Any]:
    recipe["refs"].append(
        {"path": str(path), "weight": float(weight), "role": role, "sha256": _sha256(path) if Path(path).exists() else None}
    )
    recipe["fidelity"] = fidelity_level(recipe)
    return recipe


def add_inversion(recipe: Dict[str, Any], method: str, latents: str, steps: int = 20) -> Dict[str, Any]:
    recipe["init"] = {"method": method, "latents": str(latents), "inv_steps": int(steps)}
    recipe["fidelity"] = fidelity_level(recipe)
    return recipe


def to_dict(recipe: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(recipe, ensure_ascii=False, default=str))
    out["fingerprint"] = fingerprint(recipe)
    return out


def fingerprint(recipe: Dict[str, Any]) -> str:
    """Stable hash of the parts that decide the output pixels."""
    key = {
        "prompt": recipe.get("prompt"),
        "negative": recipe.get("negative"),
        "sampling": recipe.get("sampling"),
        "models": [{k: m.get(k) for k in ("role", "name", "hash")} for m in recipe.get("models", [])],
        "adapters": [a.get("name") for a in recipe.get("adapters", [])],
        "controls": sorted(c.get("kind", "") for c in recipe.get("controls", [])),
        "init": {k: recipe.get("init", {}).get(k) for k in ("method", "image", "latents")},
    }
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def validate(recipe: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable gaps; empty means the recipe is complete."""
    problems = []
    if not recipe.get("prompt"):
        problems.append("prompt is empty")
    if recipe.get("schema") != SCHEMA:
        problems.append(f"schema mismatch: {recipe.get('schema')} != {SCHEMA}")
    sampling = recipe.get("sampling") or {}
    if sampling.get("seed") is None:
        problems.append("seed missing: output is not reproducible")
    if not recipe.get("models"):
        problems.append("no base model pinned: output will drift across checkpoints")
    if not any(c.get("weight") for c in recipe.get("controls", [])):
        problems.append("no control signals: layout and identity are not locked")
    return problems


# ---------------------------------------------------------------- sidecar


def sidecar_path(media: str | Path) -> Path:
    p = Path(media)
    return p.with_name(p.name + ".recipe.json")


def write_sidecar(media: str | Path, recipe: Dict[str, Any]) -> Path:
    out = sidecar_path(media)
    out.write_text(json.dumps(to_dict(recipe), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def read_sidecar(media: str | Path) -> Optional[Dict[str, Any]]:
    p = sidecar_path(media)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ---------------------------------------------------------------- embed into PNG


def _chunk(ctype: bytes, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(4, "big")
        + ctype
        + payload
        + (zlib.crc32(ctype + payload) & 0xFFFFFFFF).to_bytes(4, "big")
    )


def itxt_payload(keyword: str, text: str) -> bytes:
    return keyword.encode("latin-1") + b"\x00" + b"\x00\x00" + b"\x00" + b"\x00" + text.encode("utf-8")


def embed_into_png(src: str | Path, dst: str | Path, recipe: Dict[str, Any]) -> Path:
    """Copy ``src`` to ``dst`` and inject the recipe as an iTXt chunk (idempotent)."""
    data = Path(src).read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError(f"{src} is not a PNG; use write_sidecar() instead")

    blob = json.dumps(to_dict(recipe), ensure_ascii=False, separators=(",", ":"))
    new_chunk = _chunk(b"iTXt", itxt_payload(EMBED_KEY, blob))

    out = bytearray(PNG_SIG)
    i, n = 8, len(data)
    injected = False
    while i + 8 <= n:
        length = int.from_bytes(data[i : i + 4], "big")
        ctype = data[i + 4 : i + 8]
        end = i + 8 + length + 4
        if end > n:
            break
        if ctype == b"iTXt":
            payload = data[i + 8 : i + 8 + length]
            if payload.startswith(EMBED_KEY.encode("latin-1") + b"\x00"):
                block = new_chunk
                injected = True
            else:
                block = data[i:end]
        elif ctype == b"IEND":
            block = new_chunk + data[i:end] if not injected else data[i:end]
            injected = True
        else:
            block = data[i:end]
        out += block
        i = end

    dst = Path(dst)
    dst.write_bytes(bytes(out))
    return dst


def read_embedded(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read a recipe stamped into a PNG by :func:`embed_into_png`."""
    from . import forensics

    try:
        texts = forensics.png_text_map(Path(path).read_bytes())
    except ValueError:
        return None
    blob = texts.get(EMBED_KEY)
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- human summary


def summary(recipe: Dict[str, Any]) -> str:
    s = recipe.get("sampling") or {}
    bits = [
        f"fidelity={recipe.get('fidelity')}",
        f"seed={s.get('seed')}",
        f"steps={s.get('steps')}",
        f"cfg={s.get('cfg')}",
        f"model={(recipe.get('models') or [{}])[0].get('name')}",
        f"controls={len(recipe.get('controls') or [])}",
        f"refs={len(recipe.get('refs') or [])}",
    ]
    return " ".join(bits)
