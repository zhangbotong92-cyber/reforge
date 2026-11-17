"""Layer 2 — the caption contract.

A free-form paragraph is a dead end: it cannot be diffed, scored, or compiled.
So every captioner in REFORGE must answer in this one JSON shape, and every
downstream consumer (dialect compilers, the refine loop, the UI) reads only it.

    Caption(raw) -> normalize() -> compile_prompt(caption, dialect)

Adding a platform is one function, not a fork.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

SCHEMA = "reforge.caption/1"

STRING_FIELDS = ("summary", "scene", "action", "medium", "style", "lighting", "composition", "mood")
LIST_FIELDS = ("subject", "color", "details", "text_in_image", "tags", "negative")

ALIASES = {
    "prompt": "summary",
    "description": "summary",
    "caption": "summary",
    "negative_prompt": "negative",
    "neg": "negative",
    "objects": "subject",
    "subjects": "subject",
    "colors": "color",
    "colour": "color",
    "colours": "color",
    "details": "details",
    "style": "style",
    "camera": "camera",
    "composition": "composition",
    "lighting": "lighting",
    "light": "lighting",
    "text": "text_in_image",
    "ocr": "text_in_image",
    "motion": "motion",
}

DEFAULT_NEGATIVE = [
    "lowres",
    "blurry",
    "jpeg artifacts",
    "watermark",
    "text artifacts",
    "extra limbs",
    "deformed hands",
    "oversaturated",
]


def empty_caption() -> Dict[str, Any]:
    cap: Dict[str, Any] = {"schema": SCHEMA, "confidence": 0.0, "provider": "unknown", "source": "caption"}
    for f in STRING_FIELDS:
        cap[f] = ""
    for f in LIST_FIELDS:
        cap[f] = []
    cap["camera"] = {"shot": "", "lens": "", "angle": ""}
    cap["motion"] = {"camera_move": "", "subject_move": "", "pacing": ""}
    return cap


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
            elif v is not None:
                out.append(str(v))
        return out
    return [str(value)]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def normalize(raw: Dict[str, Any], provider: str = "unknown", confidence: float = 0.5) -> Dict[str, Any]:
    """Coerce any captioner output into the canonical contract."""
    cap = empty_caption()
    cap["provider"] = provider
    cap["confidence"] = float(confidence)

    for key, value in (raw or {}).items():
        target = ALIASES.get(key, key)
        if target in STRING_FIELDS:
            cap[target] = _as_str(value) or cap[target]
        elif target in LIST_FIELDS:
            cap[target] = _as_list(value) or cap[target]
        elif target == "camera":
            if isinstance(value, dict):
                for k in cap["camera"]:
                    cap["camera"][k] = _as_str(value.get(k, cap["camera"][k]))
            elif value:
                cap["camera"]["shot"] = _as_str(value)
        elif target == "motion":
            if isinstance(value, dict):
                for k in cap["motion"]:
                    cap["motion"][k] = _as_str(value.get(k, cap["motion"][k]))
            elif value:
                cap["motion"]["subject_move"] = _as_str(value)

    if not cap["negative"]:
        cap["negative"] = list(DEFAULT_NEGATIVE)
    if not cap["summary"]:
        cap["summary"] = _synth_summary(cap)
    if not cap["tags"]:
        cap["tags"] = _derive_tags(cap)
    return cap


def _synth_summary(cap: Dict[str, Any]) -> str:
    bits: List[str] = []
    if cap["subject"]:
        bits.append(", ".join(cap["subject"][:4]))
    for field in ("action", "scene", "lighting", "style", "medium"):
        if cap[field]:
            bits.append(cap[field])
    if cap["camera"].get("shot"):
        bits.append(cap["camera"]["shot"])
    return ", ".join(bits)


def _derive_tags(cap: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    tags.extend(cap["subject"][:6])
    for field in ("style", "medium", "lighting", "mood", "composition"):
        if cap[field]:
            tags.extend([t.strip() for t in cap[field].split(",") if t.strip()][:3])
    if cap["camera"].get("shot"):
        tags.append(cap["camera"]["shot"])
    if cap["camera"].get("lens"):
        tags.append(cap["camera"]["lens"])
    tags.extend(cap["color"][:4])
    seen, out = set(), []
    for t in tags:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[:24]


def merge(captions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Arbitrate several captioners: take the longest string, union the lists."""
    captions = [c for c in captions if c]
    if not captions:
        return empty_caption()
    if len(captions) == 1:
        return captions[0]

    out = empty_caption()
    out["provider"] = " + ".join(c.get("provider", "?") for c in captions)
    out["confidence"] = max(float(c.get("confidence", 0)) for c in captions)
    for field in STRING_FIELDS:
        best = max((c.get(field, "") for c in captions), key=len, default="")
        out[field] = best
    for field in LIST_FIELDS:
        seen, merged = set(), []
        for c in captions:
            for item in c.get(field, []):
                if item.lower() not in seen:
                    seen.add(item.lower())
                    merged.append(item)
        out[field] = merged
    for group in ("camera", "motion"):
        for k in out[group]:
            out[group][k] = max((c.get(group, {}).get(k, "") for c in captions), key=len, default="")
    out["source"] = "hybrid"
    return out


# ------------------------------------------------------------------ dialects


def _camera_sentence(cap: Dict[str, Any]) -> str:
    cam = cap.get("camera", {})
    bits = [cam.get(k, "") for k in ("shot", "lens", "angle") if cam.get(k)]
    return ", ".join(bits)


def _motion_sentence(cap: Dict[str, Any]) -> str:
    mo = cap.get("motion", {})
    bits = [mo.get(k, "") for k in ("camera_move", "subject_move", "pacing") if mo.get(k)]
    return ", ".join(bits)


def _sd(cap: Dict[str, Any]) -> Dict[str, Any]:
    tags = list(cap["tags"])
    if cap["summary"] and cap["summary"].lower() not in seen_lower(tags):
        tags.insert(0, cap["summary"])
    prompt = ", ".join(tags)
    return {
        "prompt": prompt,
        "negative": ", ".join(cap["negative"]),
        "params": {"steps": 28, "cfg": 6.5, "sampler": "DPM++ 2M Karras", "width": 1024, "height": 1024},
    }


def _flux(cap: Dict[str, Any]) -> Dict[str, Any]:
    body = cap["summary"]
    extras = [cap["lighting"], cap["composition"], _camera_sentence(cap)]
    sentence = ". ".join([body] + [e for e in extras if e])
    if not sentence.endswith("."):
        sentence += "."
    return {"prompt": sentence, "negative": "", "params": {"guidance": 3.5, "steps": 28, "sampler": "euler"}}


def _mj(cap: Dict[str, Any]) -> Dict[str, Any]:
    tags = ", ".join(cap["tags"][:16])
    ar = "16:9" if _aspect(cap) and _aspect(cap) > 1.2 else ("9:16" if _aspect(cap) and _aspect(cap) < 0.85 else "1:1")
    return {
        "prompt": f"{tags} --ar {ar} --style raw --s 150",
        "negative": ", ".join(cap["negative"][:6]),
        "params": {"ar": ar, "stylize": 150, "version": "v6.1"},
    }


def _jimeng(cap: Dict[str, Any]) -> Dict[str, Any]:
    bits = [cap["summary"], cap["lighting"], cap["style"], _camera_sentence(cap), _motion_sentence(cap)]
    return {
        "prompt": "，".join(b for b in bits if b) or cap["summary"],
        "negative": "、".join(cap["negative"][:8]),
        "params": {"时长": "5s", "比例": "16:9", "运镜": cap["motion"].get("camera_move") or "固定"},
    }


def _kling(cap: Dict[str, Any]) -> Dict[str, Any]:
    motion = _motion_sentence(cap) or "缓慢推近"
    bits = [
        f"主体：{cap['summary']}",
        f"场景：{cap['scene']}" if cap["scene"] else "",
        f"光线：{cap['lighting']}" if cap["lighting"] else "",
        f"镜头：{_camera_sentence(cap)}" if _camera_sentence(cap) else "",
        f"运动：{motion}",
    ]
    return {
        "prompt": "；".join(b for b in bits if b),
        "negative": "、".join(cap["negative"][:8]),
        "params": {"时长": "5s", "模式": "std", "运镜": motion},
    }


def _veo(cap: Dict[str, Any]) -> Dict[str, Any]:
    parts = [
        cap["summary"] or "a single continuous shot",
        f"Shot: {_camera_sentence(cap)}." if _camera_sentence(cap) else "",
        f"Lighting: {cap['lighting']}." if cap["lighting"] else "",
        f"Motion: {_motion_sentence(cap)}." if _motion_sentence(cap) else "",
        f"Palette: {', '.join(cap['color'][:4])}." if cap["color"] else "",
    ]
    return {"prompt": " ".join(p for p in parts if p), "negative": "", "params": {"duration": 8, "resolution": "1080p"}}


def _nano_banana(cap: Dict[str, Any]) -> Dict[str, Any]:
    """Instruction-style editing prompt: keep what matters, change what you must."""
    return {
        "prompt": (
            f"Using the provided image as reference, keep the subject identity, layout and palette unchanged. "
            f"Recreate it as: {cap['summary']}. "
            f"Match {cap['lighting'] or 'the original lighting'}."
        ),
        "negative": "",
        "params": {"mode": "image-edit", "reference_strength": 0.75},
    }


def seen_lower(items: Sequence[str]) -> set:
    return {i.lower().strip() for i in items}


def _aspect(cap: Dict[str, Any]) -> Optional[float]:
    w, h = cap.get("width"), cap.get("height")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)) and h:
        return float(w) / float(h)
    return None


DIALECTS = {
    "sd": _sd,
    "flux": _flux,
    "mj": _mj,
    "midjourney": _mj,
    "jimeng": _jimeng,
    "即梦": _jimeng,
    "kling": _kling,
    "可灵": _kling,
    "veo": _veo,
    "sora": _veo,
    "banana": _nano_banana,
    "nano-banana": _nano_banana,
}


def compile_prompt(caption: Dict[str, Any], dialect: str = "sd") -> Dict[str, Any]:
    """Render the canonical caption into one platform's prompt dialect."""
    fn = DIALECTS.get(dialect.lower())
    if fn is None:
        raise KeyError(f"unknown dialect {dialect!r}; available: {', '.join(sorted(set(DIALECTS)))}")
    out = fn(caption)
    out["dialect"] = dialect
    out["notes"] = _dialect_notes(dialect)
    return out


def _dialect_notes(dialect: str) -> List[str]:
    d = dialect.lower()
    if d in ("sd",):
        return ["Tag order matters: subject -> attributes -> style. Weights like (word:1.2) are accepted."]
    if d in ("flux",):
        return ["FLUX ignores negative prompts; describe absence positively instead."]
    if d in ("mj", "midjourney"):
        return ["Negative prompts only work through --no at the end of the prompt."]
    if d in ("kling", "可灵", "jimeng", "即梦"):
        return ["Camera movement is a first-class parameter; keep it short and physical."]
    return []


def compile_all(caption: Dict[str, Any], dialects: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    names = list(dialects or ["sd", "flux", "mj", "kling", "jimeng", "banana"])
    out = {}
    for name in names:
        try:
            out[name] = compile_prompt(caption, name)
        except KeyError:
            continue
    return out
