"""Layer 5.5 — video.

A single prompt for a 30-second clip always fails, for a boring structural reason:
diffusion video models hold identity for a few seconds, not for a whole timeline.
So a video recipe is a **list of shot recipes**, each anchored to a first frame, and
each independently reproducible.

Two input paths, because ffmpeg is not always installed:

    analyze("clip.mp4")          -> ffmpeg scene detection + per-shot frame grabs
    analyze(frames_dir="frames/") -> one shot per image (CI-friendly, no ffmpeg)

The output feeds :func:`build_recipe`, which writes ``recipe["shots"]`` and a
per-shot execution plan for Wan2.2-VACE / FLF2V style pipelines.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import controls, schema

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class VideoUnavailable(RuntimeError):
    pass


def _frame_meta(path: Path) -> Dict[str, Any]:
    info = controls.probe(path)
    return {"width": info.get("width"), "height": info.get("height")}


def analyze(
    video: Optional[str | Path] = None,
    frames_dir: Optional[str | Path] = None,
    outdir: Optional[str | Path] = None,
    max_shots: int = 8,
    scene_threshold: float = 0.30,
) -> Dict[str, Any]:
    """Split a video into shots and pick an anchor frame for each."""
    if frames_dir is not None:
        return _analyze_frames(Path(frames_dir), max_shots=max_shots)

    if video is None:
        raise VideoUnavailable("pass a video path or frames_dir=")
    video = Path(video)
    info = controls.probe(video)
    out = Path(outdir or (video.parent / "reforge_shots"))
    out.mkdir(parents=True, exist_ok=True)

    if info.get("kind") != "video":
        raise VideoUnavailable(
            f"{video} is not a readable video ({info.get('kind')}). "
            "Install ffmpeg for container probing, or pass frames_dir= with extracted frames."
        )
    if not shutil.which("ffmpeg"):
        raise VideoUnavailable("ffmpeg is required for scene detection. Or pass frames_dir=.")

    shots = controls.scenes(video, out, threshold=scene_threshold)[:max_shots]
    if not shots:
        # a single continuous shot: sample evenly instead
        shots = controls.keyframes(video, out, count=1)

    duration = float(info.get("duration") or 0)
    per = duration / max(len(shots), 1) if duration else None

    result: Dict[str, Any] = {
        "kind": "video",
        "backend": "ffmpeg",
        "duration": duration,
        "fps": info.get("fps"),
        "width": info.get("width"),
        "height": info.get("height"),
        "shots": [],
        "notes": [],
    }
    for idx, frame in enumerate(shots):
        start = round(idx * per, 3) if per else None
        end = round((idx + 1) * per, 3) if per else None
        result["shots"].append(
            {
                "index": idx,
                "start": start,
                "end": end,
                "keyframe": str(frame),
                "first_frame": str(frame),
                "last_frame": str(shots[idx + 1]) if idx + 1 < len(shots) else None,
                **_frame_meta(frame),
            }
        )
    if not per:
        result["notes"].append("duration unknown: shot timings omitted")
    return result


def _analyze_frames(frames_dir: Path, max_shots: int = 8) -> Dict[str, Any]:
    if not frames_dir.is_dir():
        raise VideoUnavailable(f"{frames_dir} is not a directory of frames")
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        raise VideoUnavailable(f"no image frames in {frames_dir}")
    picked = _evenly(frames, max_shots)
    shots = []
    for idx, frame in enumerate(picked):
        shots.append(
            {
                "index": idx,
                "start": idx,
                "end": idx + 1,
                "keyframe": str(frame),
                "first_frame": str(frame),
                "last_frame": str(picked[idx + 1]) if idx + 1 < len(picked) else None,
                **_frame_meta(frame),
            }
        )
    meta = _frame_meta(picked[0])
    return {
        "kind": "frames",
        "backend": "frames-dir",
        "duration": float(len(frames)),
        "fps": None,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "shots": shots,
        "notes": [f"{len(frames)} frames found, {len(picked)} used as shots; timings are frame indices"],
    }


def _evenly(items: List[Path], k: int) -> List[Path]:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[min(int(i * step), len(items) - 1)] for i in range(k)]


# ---------------------------------------------------------------- scripting


def script(
    analysis: Dict[str, Any],
    captioner: Any,
    dialect: str = "kling",
    hint: str = "",
) -> Dict[str, Any]:
    """Caption every shot and compile it into the target dialect."""
    scripted = []
    for shot in analysis["shots"]:
        cap = captioner.caption(shot["keyframe"], hint=hint)
        cap["width"] = shot.get("width")
        cap["height"] = shot.get("height")
        compiled = schema.compile_prompt(cap, dialect)
        scripted.append({**shot, "caption": cap, "compiled": compiled})
    return {**analysis, "dialect": dialect, "shots": scripted}


def merge_prompts(scripted: Dict[str, Any], max_shots: int = 4) -> str:
    """One-line global description: the first shots' summaries, not a concatenation of everything."""
    bits = []
    for shot in scripted.get("shots", [])[:max_shots]:
        summary = (shot.get("caption") or {}).get("summary")
        if summary:
            bits.append(summary)
    return " | ".join(dict.fromkeys(bits))


# ---------------------------------------------------------------- recipe


def build_recipe(
    scripted: Dict[str, Any],
    title: str = "",
    runtime: str = "diffusers",
    anchor_refs: bool = True,
) -> Dict[str, Any]:
    """Turn a scripted analysis into a video recipe with per-shot anchors."""
    from . import recipe as recipelib

    shots_out: List[Dict[str, Any]] = []
    for shot in scripted.get("shots", []):
        compiled = shot.get("compiled") or {}
        entry: Dict[str, Any] = {
            "index": shot["index"],
            "start": shot.get("start"),
            "end": shot.get("end"),
            "prompt": compiled.get("prompt") or (shot.get("caption") or {}).get("summary", ""),
            "negative": compiled.get("negative", ""),
            "first_frame": shot.get("first_frame"),
            "last_frame": shot.get("last_frame"),
            "keyframe": shot.get("keyframe"),
            "controls": [],
        }
        shots_out.append(entry)

    rec = recipelib.new_recipe(
        prompt=merge_prompts(scripted),
        source={"path": scripted.get("path") or "", "format": "video", "width": scripted.get("width"), "height": scripted.get("height")},
        tool=scripted.get("backend"),
        shots=shots_out,
    )
    rec["camera"] = {
        "source": "per-shot anchor" if anchor_refs else "unknown",
        "note": "recover a real trajectory with VGGT/MegaSaM, then feed ReCamMaster",
    }
    if anchor_refs and shots_out:
        first = Path(shots_out[0]["first_frame"]) if shots_out[0].get("first_frame") else None
        if first and first.exists():
            recipelib.add_ref(rec, str(first), weight=0.7, role="identity")
    rec["video_plan"] = plan_video(rec, runtime=runtime)
    rec["notes"].append(f"{len(shots_out)} shots; each pinned to its own first frame")
    if title:
        rec["notes"].append(f"title: {title}")
    rec["fidelity"] = recipelib.fidelity_level(rec) if shots_out else "L0-drift"
    return rec


def plan_video(recipe: Dict[str, Any], runtime: str = "diffusers") -> Dict[str, Any]:
    """Per-shot execution plan for reference-to-video / first-last-frame pipelines."""
    shots = recipe.get("shots") or []
    steps = []
    for shot in shots:
        steps.append(
            {
                "op": "shot",
                "index": shot["index"],
                "anchor": shot.get("first_frame"),
                "end_anchor": shot.get("last_frame"),
                "prompt": shot.get("prompt"),
                "negative": shot.get("negative"),
                "model": "Wan2.2-VACE" if runtime in ("diffusers", "comfyui") else "reference-to-video API",
                "note": "feed the anchor as first frame; VACE reference mode keeps the subject",
            }
        )
    return {
        "runtime": runtime,
        "shots": len(shots),
        "steps": steps,
        "rules": [
            "one prompt per shot, never one prompt for the timeline",
            "re-anchor identity at every shot boundary",
            "keep shots <= 5s where the model allows it",
            "score each shot separately before stitching",
        ],
        "gaps": controls.gaps(recipe),
    }


# ---------------------------------------------------------------- verification


def score_video(a_dir: str | Path, b_dir: str | Path) -> Dict[str, Any]:
    """Average the image metrics across aligned frames; worst frame is reported too."""
    from . import metrics

    a = sorted(p for p in Path(a_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    b = sorted(p for p in Path(b_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    pairs = list(zip(a, b))
    if not pairs:
        raise VideoUnavailable("no comparable frames")
    scores = [metrics.score_pair(x, y) for x, y in pairs]
    composites = [s["composite"] for s in scores]
    worst = min(scores, key=lambda s: s["composite"])
    best = max(scores, key=lambda s: s["composite"])
    return {
        "frames": len(pairs),
        "mean_composite": round(sum(composites) / len(composites), 4),
        "min_composite": worst["composite"],
        "max_composite": best["composite"],
        "stability": round(1.0 - (max(composites) - min(composites)), 4),
        "grade": metrics._grade(sum(composites) / len(composites)),
    }


def summarize(analysis: Dict[str, Any]) -> str:
    return (
        f"{analysis.get('kind')} via {analysis.get('backend')}: "
        f"{len(analysis.get('shots', []))} shots "
        f"({analysis.get('width')}x{analysis.get('height')}, duration={analysis.get('duration')})"
    )
