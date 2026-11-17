"""Layer 1 — caption backends.

BYOK by design: the repo ships a working offline heuristic captioner and thin
adapters for any OpenAI-compatible endpoint, Gemini, or Ollama. No keys are
bundled, nothing phones home, and every adapter answers in the same contract
(:mod:`reforge.schema`).

    build_captioner("offline")                 # no network, no key
    build_captioner("openai:gpt-4o-mini")      # OPENAI_API_KEY / REFORGE_BASE_URL
    build_captioner("gemini:gemini-2.5-flash") # GEMINI_API_KEY
    build_captioner("ollama:qwen2.5vl")        # local
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import forensics, schema

TIMEOUT = float(os.environ.get("REFORGE_TIMEOUT", "120"))

CAPTION_SPEC = """You are a prompt reverse-engineer for text-to-image and text-to-video models.
You receive ONE image (or one frame of a video) and must describe it so that another
model can REBUILD it as closely as possible.

Rules:
- Only state what is visible. Never invent brands, names, or hidden context.
- Prefer concrete, model-usable language over literary prose.
- Camera language must be physical: shot size, angle, lens, depth of field.
- For video frames, fill motion with plausible camera/subject movement.
- Keep negative short: only the defects a generator would actually produce.

Return STRICT JSON with exactly these keys (omit nothing, use "" or [] when unknown):
{
  "summary": "one sentence, <=200 chars, the global description",
  "subject": ["main subject", "secondary subject"],
  "scene": "location / environment",
  "action": "what is happening",
  "medium": "photo | film still | 3d render | oil painting | anime | ...",
  "style": "art direction, references, era",
  "lighting": "direction, quality, colour temperature",
  "color": ["dominant colour names"],
  "composition": "framing, rule of thirds, symmetry, depth layers",
  "mood": "emotional tone",
  "details": ["notable texture / material / prop details"],
  "text_in_image": ["verbatim strings, or []"],
  "tags": ["12-24 danbooru-style tags, lowercase, comma free"],
  "negative": ["4-10 defect tags to suppress"],
  "camera": {"shot": "medium close-up", "lens": "85mm f/1.8", "angle": "eye level"},
  "motion": {"camera_move": "slow dolly in", "subject_move": "...", "pacing": "..."}
}
Return the JSON object only. No markdown fences, no commentary."""

CRITIQUE_SPEC = """You are auditing a reverse-engineered prompt against the original image.
Compare the caption JSON with the image and report what would make a regeneration drift.

Return STRICT JSON:
{"issues": ["specific mismatch"], "patches": {"field": "replacement value"}, "confidence": 0.0}
`patches` must use the same keys as the caption contract."""


class Captioner:
    """Base adapter. Subclasses implement :meth:`_raw_caption`."""

    name = "base"
    requires_key = True

    def available(self) -> bool:
        return not self.requires_key or bool(self._key())

    def _key(self) -> str:
        return ""

    def caption(self, image: str | Path, hint: str = "") -> Dict[str, Any]:
        raw, confidence = self._raw_caption(Path(image), hint)
        cap = schema.normalize(raw, provider=self.name, confidence=confidence)
        cap["source"] = "caption"
        return cap

    def critique(self, image: str | Path, caption: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._raw_critique(Path(image), caption)
        return {
            "issues": list(raw.get("issues") or []),
            "patches": dict(raw.get("patches") or {}),
            "confidence": float(raw.get("confidence") or 0.0),
            "provider": self.name,
        }

    # -- to implement -------------------------------------------------
    def _raw_caption(self, image: Path, hint: str) -> tuple[Dict[str, Any], float]:
        raise NotImplementedError

    def _raw_critique(self, image: Path, caption: Dict[str, Any]) -> Dict[str, Any]:
        return {"issues": [], "patches": {}, "confidence": 0.0}


# ---------------------------------------------------------------- offline


class OfflineCaptioner(Captioner):
    """Zero-dependency heuristic captioner.

    It cannot see semantics, but it *can* measure structure: aspect, palette,
    contrast, edge density, and — crucially — any parameters already embedded in
    the file. That is often enough to bootstrap a usable starting prompt.
    """

    name = "offline-heuristic"
    requires_key = False

    def _raw_caption(self, image: Path, hint: str) -> tuple[Dict[str, Any], float]:
        from PIL import Image, ImageFilter, ImageStat

        report = forensics.inspect(image)
        with Image.open(image) as img:
            img = img.convert("RGB")
            w, h = img.size
            small = img.copy()
            small.thumbnail((256, 256))
            stat = ImageStat.Stat(small)
            mean = stat.mean
            stddev = stat.stddev
            edges = small.convert("L").filter(ImageFilter.FIND_EDGES)
            edge_mean = ImageStat.Stat(edges).mean[0]
            colors = self._palette(small)

        brightness = sum(mean) / 3.0
        saturation = self._saturation(small)
        aspect = w / h if h else 1.0

        lighting = (
            "high-key, soft even light"
            if brightness > 175
            else "low-key, deep shadows" if brightness < 85 else "balanced natural light"
        )
        if stddev and sum(stddev) / 3.0 > 60:
            lighting += ", high contrast"
        detail = "highly detailed texture" if edge_mean > 26 else ("clean minimal surfaces" if edge_mean < 12 else "moderate detail")
        framing = "wide establishing shot" if aspect > 1.6 else ("vertical portrait framing" if aspect < 0.8 else "square-ish medium shot")
        palette = list(dict.fromkeys(self._color_name(c) for c in colors[:4]))

        recovered = report.get("recovered") or {}
        embedded = recovered.get("prompt")
        summary = embedded or (
            f"{framing} with {', '.join(palette)} dominant palette, {lighting}, {detail}"
        )

        cap: Dict[str, Any] = {
            "summary": summary,
            "subject": [],
            "scene": "",
            "action": "",
            "medium": "photograph" if report.get("format") in ("jpeg", "webp") else "digital image",
            "style": "",
            "lighting": lighting,
            "color": palette,
            "composition": framing,
            "mood": "neutral" if 85 <= brightness <= 175 else ("airy" if brightness > 175 else "moody"),
            "details": [detail],
            "text_in_image": [],
            "tags": [
                framing,
                lighting,
                detail,
                *palette,
                f"{round(brightness/255, 2)} brightness",
                f"{round(saturation, 2)} saturation",
            ],
            "negative": [],
            "camera": {"shot": framing, "lens": "", "angle": "eye level"},
        }
        if embedded:
            cap["style"] = f"embedded parameters from {report.get('tool')}"
            if recovered.get("negative"):
                cap["negative"] = [t.strip() for t in str(recovered["negative"]).split(",") if t.strip()][:10]
        confidence = 0.55 if embedded else 0.2
        return cap, confidence

    @staticmethod
    def _palette(img: Any, k: int = 5) -> List[tuple]:
        quant = img.quantize(colors=k, method=2)
        pal = quant.getpalette() or []
        counts = sorted(quant.getcolors() or [], reverse=True)
        out = []
        for _count, idx in counts[:k]:
            out.append(tuple(pal[idx * 3 : idx * 3 + 3]))
        return out

    @staticmethod
    def _saturation(img: Any) -> float:
        hsv = img.convert("HSV")
        import statistics

        data = list(hsv.getdata(1))
        return round(statistics.mean(data) / 255.0, 3) if data else 0.0

    @staticmethod
    def _color_name(rgb: tuple) -> str:
        r, g, b = [c / 255.0 for c in rgb]
        mx, mn = max(r, g, b), min(r, g, b)
        lightness = (mx + mn) / 2
        if mx - mn < 0.08:
            return "grey" if lightness < 0.75 else "off-white"
        if mx == r:
            hue = 60 * ((g - b) / (mx - mn))
        elif mx == g:
            hue = 60 * (2 + (b - r) / (mx - mn))
        else:
            hue = 60 * (4 + (r - g) / (mx - mn))
        hue %= 360
        base = (
            "red" if hue < 15 or hue >= 345 else
            "orange" if hue < 45 else
            "yellow" if hue < 70 else
            "green" if hue < 165 else
            "cyan" if hue < 195 else
            "blue" if hue < 255 else
            "purple" if hue < 290 else
            "magenta"
        )
        if lightness < 0.25:
            return f"deep {base}"
        if lightness > 0.8:
            return f"pale {base}"
        return base


# ---------------------------------------------------------------- OpenAI-compatible


class OpenAICompatCaptioner(Captioner):
    name = "openai-compatible"
    requires_key = True

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.provider = (provider or "openai").lower()
        self.base_url = (base_url or os.environ.get("REFORGE_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.name = name or f"openai:{model}"

    def _key(self) -> str:
        # provider-specific env first, then the generic ones
        for var in (f"{self.provider.upper()}_API_KEY", "OPENAI_API_KEY", "REFORGE_API_KEY"):
            value = os.environ.get(var)
            if value:
                return value
        return self._api_key or ""

    def _chat(self, image: Path, instruction: str, json_mode: bool = True) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            # reasoning models burn the budget on reasoning_content and return an empty
            # `content` when max_tokens is tight, so this must be generous.
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyse this image and return the JSON."},
                        {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    ],
                },
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = _post_json(f"{self.base_url}/chat/completions", payload, {"Authorization": f"Bearer {self._key()}"})
        choice = data["choices"][0]
        text = choice.get("message", {}).get("content") or ""
        if not text.strip():
            reason = choice.get("finish_reason")
            reasoning = choice.get("message", {}).get("reasoning_content") or ""
            hint = "raise max_tokens" if reason == "length" else "the model returned nothing"
            raise RuntimeError(
                f"{self.name} returned empty content (finish_reason={reason}); {hint}. "
                f"reasoning_chars={len(reasoning)}"
            )
        return _extract_json(text)

    def _raw_caption(self, image: Path, hint: str) -> tuple[Dict[str, Any], float]:
        spec = CAPTION_SPEC if not hint else CAPTION_SPEC + f"\n\nExtra instruction from the user: {hint}"
        return self._chat(image, spec), 0.85

    def _raw_critique(self, image: Path, caption: Dict[str, Any]) -> Dict[str, Any]:
        spec = CRITIQUE_SPEC + "\n\nCaption under audit:\n" + json.dumps(caption, ensure_ascii=False)
        return self._chat(image, spec)


# ---------------------------------------------------------------- Gemini


class GeminiCaptioner(Captioner):
    name = "gemini"
    requires_key = True

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        self.model = model
        self._api_key = api_key
        self.name = f"gemini:{model}"

    def _key(self) -> str:
        return self._api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

    def _gen(self, image: Path, instruction: str) -> Dict[str, Any]:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": instruction},
                        {"inline_data": {"mime_type": _mime(image), "data": base64.b64encode(image.read_bytes()).decode()}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self._key()}"
        data = _post_json(url, payload, {})
        parts = data["candidates"][0]["content"]["parts"]
        return _extract_json("".join(p.get("text", "") for p in parts))

    def _raw_caption(self, image: Path, hint: str) -> tuple[Dict[str, Any], float]:
        spec = CAPTION_SPEC if not hint else CAPTION_SPEC + f"\n\nExtra instruction from the user: {hint}"
        return self._gen(image, spec), 0.85

    def _raw_critique(self, image: Path, caption: Dict[str, Any]) -> Dict[str, Any]:
        return self._gen(image, CRITIQUE_SPEC + "\n\nCaption under audit:\n" + json.dumps(caption, ensure_ascii=False))


# ---------------------------------------------------------------- Ollama


class OllamaCaptioner(Captioner):
    name = "ollama"
    requires_key = False

    def __init__(self, model: str = "qwen2.5vl", host: Optional[str] = None):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        self.name = f"ollama:{model}"

    def _gen(self, image: Path, instruction: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": "Analyse this image.", "images": [base64.b64encode(image.read_bytes()).decode()]},
            ],
        }
        data = _post_json(f"{self.host}/api/chat", payload, {})
        return _extract_json(data["message"]["content"])

    def _raw_caption(self, image: Path, hint: str) -> tuple[Dict[str, Any], float]:
        return self._gen(image, CAPTION_SPEC), 0.7

    def _raw_critique(self, image: Path, caption: Dict[str, Any]) -> Dict[str, Any]:
        return self._gen(image, CRITIQUE_SPEC + "\n\n" + json.dumps(caption, ensure_ascii=False))


# ---------------------------------------------------------------- factory


# OpenAI-compatible endpoints that only differ by base URL. `model` is whatever the
# provider calls it; the adapter does not care.
PROVIDER_BASES = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "siliconflow": "https://api.siliconflow.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
}


def build_captioner(spec: str) -> Captioner:
    """``offline`` | ``<provider>:<model>`` | ``ollama:<model>``.

    Known providers get their base URL filled in, and their key is read from
    ``<PROVIDER>_API_KEY`` first, so ``DEEPSEEK_API_KEY`` + ``deepseek:deepseek-flash``
    is all you need.
    """
    spec = (spec or "offline").strip()
    if spec in ("offline", "none", ""):
        return OfflineCaptioner()
    if ":" not in spec:
        raise ValueError(f"bad captioner spec {spec!r} (expected provider:model)")
    provider, model = spec.split(":", 1)
    provider = provider.lower()

    if provider == "gemini":
        return GeminiCaptioner(model)
    if provider == "ollama":
        return OllamaCaptioner(model)

    base = PROVIDER_BASES.get(provider)
    if base is None:
        raise ValueError(
            f"unknown provider {provider!r}; use one of {', '.join(sorted(PROVIDER_BASES))}, gemini, ollama, or set "
            f"REFORGE_BASE_URL with provider 'openai'"
        )
    return OpenAICompatCaptioner(model, base_url=base, provider=provider, name=f"{provider}:{model}")


# ---------------------------------------------------------------- http helpers


def _mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}.get(
        ext, "image/png"
    )


def _data_url(path: Path) -> str:
    return f"data:{_mime(path)};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"{url} -> HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} unreachable: {exc.reason}") from exc


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError(f"no JSON object in model output: {text[:200]}")
