"""Layer 0 — file forensics.

A surprisingly large share of "AI images" on the internet still carry their own
creation parameters: Stable Diffusion WebUI writes ``parameters`` into a PNG tEXt
chunk, ComfyUI writes ``prompt`` + ``workflow``, NovelAI writes ``Comment``,
InvokeAI writes ``sd-metadata`` / ``invokeai_metadata``, and modern services attach
C2PA manifests.  Recovering those beats *guessing* a prompt by a mile: it is exact,
free, and instant.

Everything here is dependency-light on purpose (stdlib + Pillow for EXIF only).
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPEG_SIG = b"\xff\xd8"
MP4_BOXES = {b"moov", b"udta", b"meta", b"ilst", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"moof", b"traf"}

# ---------------------------------------------------------------- public API


def inspect(path: str | Path) -> Dict[str, Any]:
    """Inspect one media file and return the forensic report.

    The report is JSON-serialisable and stable::

        {
          "path": str, "format": "png|jpeg|webp|mp4|unknown",
          "width": int|None, "height": int|None,
          "has_c2pa": bool, "is_ai_generated": bool,
          "tool": "automatic1111"|"comfyui"|"novelai"|"invokeai"|None,
          "recovered": {...},          # normalized generation parameters
          "raw": {"png:text:parameters": "...", ...},
          "evidence": [{"field": "prompt", "source": ..., "confidence": 0.99}],
          "notes": [...]
        }
    """
    p = Path(path)
    data = p.read_bytes()
    report: Dict[str, Any] = {
        "path": str(p),
        "format": _sniff(data),
        "width": None,
        "height": None,
        "has_c2pa": _has_c2pa(data),
        "is_ai_generated": False,
        "tool": None,
        "recovered": {},
        "raw": {},
        "evidence": [],
        "notes": [],
    }

    fmt = report["format"]
    if fmt == "png":
        _inspect_png(data, report)
    elif fmt in ("jpeg", "webp"):
        _inspect_pillow(p, report)
    elif fmt == "mp4":
        _inspect_mp4(data, report)

    _promote(report)
    return report


# ---------------------------------------------------------------- sniffing


def _sniff(data: bytes) -> str:
    if data.startswith(PNG_SIG):
        return "png"
    if data.startswith(JPEG_SIG):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"skip"):
        return "mp4"
    if data[4:8] == b"ftyp":
        return "mp4"
    return "unknown"


def _has_c2pa(data: bytes) -> bool:
    head = data[: 8 * 1024 * 1024]
    return b"c2pa" in head or b"jumbf" in head


# ---------------------------------------------------------------- PNG


def iter_png_chunks(data: bytes) -> List[Tuple[str, bytes]]:
    """Yield ``(chunk_type, payload)`` for every PNG chunk (CRC stripped)."""
    if not data.startswith(PNG_SIG):
        raise ValueError("not a PNG")
    out: List[Tuple[str, bytes]] = []
    i = 8
    n = len(data)
    while i + 8 <= n:
        length = int.from_bytes(data[i : i + 4], "big")
        ctype = data[i + 4 : i + 8].decode("latin-1")
        start, end = i + 8, i + 8 + length
        if end > n:
            break
        out.append((ctype, data[start:end]))
        i = end + 4  # skip CRC
    return out


def decode_text_chunk(ctype: str, payload: bytes) -> Optional[Tuple[str, str]]:
    """Decode a tEXt / zTXt / iTXt chunk into ``(keyword, text)``."""
    try:
        if ctype == "tEXt":
            key, _, text = payload.partition(b"\x00")
            return key.decode("latin-1"), text.decode("latin-1")
        if ctype == "zTXt":
            key, _, rest = payload.partition(b"\x00")
            if len(rest) < 1:
                return None
            return key.decode("latin-1"), zlib.decompress(rest[1:]).decode("latin-1")
        if ctype == "iTXt":
            key, _, rest = payload.partition(b"\x00")
            if len(rest) < 2:
                return None
            comp_flag, _comp_method = rest[0], rest[1]
            body = rest[2:]
            _lang, _, body = body.partition(b"\x00")
            _trans, _, body = body.partition(b"\x00")
            if comp_flag == 1:
                body = zlib.decompress(body)
            return key.decode("latin-1"), body.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - corrupt chunk must never break the scan
        return None
    return None


def png_text_map(data: bytes) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for ctype, payload in iter_png_chunks(data):
        if ctype in ("tEXt", "zTXt", "iTXt"):
            kv = decode_text_chunk(ctype, payload)
            if kv:
                key, text = kv
                out[key] = text
                out.setdefault(f"{ctype}:{key}", text)
    return out


def _inspect_png(data: bytes, report: Dict[str, Any]) -> None:
    for ctype, payload in iter_png_chunks(data):
        if ctype == "IHDR" and len(payload) >= 8:
            report["width"] = int.from_bytes(payload[0:4], "big")
            report["height"] = int.from_bytes(payload[4:8], "big")
            break

    texts = png_text_map(data)
    report["raw"] = {k: v for k, v in texts.items() if ":" in k}

    # ---- A1111 / Forge / SD.Next
    if "parameters" in texts:
        rec = parse_a1111(texts["parameters"])
        if rec:
            report["tool"] = rec.pop("_tool", "automatic1111")
            _merge(report, rec, "png:tEXt:parameters", 0.99)

    # ---- ComfyUI
    if "prompt" in texts or "workflow" in texts:
        got = False
        if "prompt" in texts:
            rec = parse_comfy_api(texts["prompt"])
            if rec:
                report["tool"] = "comfyui"
                _merge(report, rec, "png:tEXt:prompt(api-graph)", 0.98)
                got = True
        if not got and "workflow" in texts:
            rec = parse_comfy_ui(texts["workflow"])
            if rec:
                report["tool"] = "comfyui"
                _merge(report, rec, "png:tEXt:workflow(ui-graph)", 0.6)
                report["notes"].append(
                    "Only the UI workflow was embedded; positive/negative were inferred by node order."
                )

    # ---- NovelAI
    if "Comment" in texts and "novelai" not in report["raw"]:
        rec = parse_novelai(texts["Comment"])
        if rec:
            report["tool"] = "novelai"
            _merge(report, rec, "png:tEXt:Comment", 0.95)

    # ---- InvokeAI
    for key in ("sd-metadata", "invokeai_metadata"):
        if key in texts:
            rec = parse_invokeai(texts[key])
            if rec:
                report["tool"] = "invokeai"
                _merge(report, rec, f"png:tEXt:{key}", 0.9)

    # ---- plain-text hints
    for key in ("Software", "Description", "parameters", "negative_prompt", "Prompt"):
        if key in texts and key not in report["recovered"]:
            if len(texts[key]) < 4000:
                report["raw"].setdefault(f"png:text:{key}", texts[key])

    if report["raw"]:
        report["is_ai_generated"] = True


# ---------------------------------------------------------------- Pillow path (JPEG/WebP/TIFF)


def _inspect_pillow(path: Path, report: Dict[str, Any]) -> None:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        report["notes"].append("Pillow missing: EXIF/XMP inspection skipped.")
        return

    try:
        with Image.open(path) as img:
            report["width"], report["height"] = img.size
            info = dict(img.info)
            exif = img.getexif()
            for tag, field in ((0x010E, "ImageDescription"), (0x0131, "Software"), (0x0110, "Model"), (0x010F, "Make")):
                try:
                    val = exif.get(tag)
                except Exception:  # noqa: BLE001
                    val = None
                if val:
                    report["raw"][f"exif:{field}"] = _as_text(val)
            try:
                exif_ifd = exif.get_ifd(0x8769)
                if 0x9286 in exif_ifd:  # UserComment
                    report["raw"]["exif:UserComment"] = _clean_user_comment(exif_ifd[0x9286])
            except Exception:  # noqa: BLE001
                pass
            for key in ("parameters", "prompt", "workflow", "Comment", "xmp", "XML:com.adobe.xmp", "adobe"):
                if key in info:
                    report["raw"][f"fmt:{key}"] = _as_text(info[key])[:20000]
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"Pillow could not open the file: {exc}")
        return

    # Reuse the PNG parameter parsers on any text blob we found.
    for key, blob in list(report["raw"].items()):
        text = str(blob)
        if "Steps:" in text and "Negative prompt" in text:
            rec = parse_a1111(text)
            if rec:
                report["tool"] = rec.pop("_tool", "automatic1111")
                _merge(report, rec, key, 0.75)
        m = re.search(r'parameters="(.*?)"\s', text, re.S)
        if m:
            rec = parse_a1111(m.group(1).replace("&quot;", '"').replace("&#10;", "\n"))
            if rec:
                _merge(report, rec, f"{key}#parameters-attr", 0.7)
        if '"class_type"' in text and "KSampler" in text:
            rec = parse_comfy_api(text)
            if rec:
                report["tool"] = "comfyui"
                _merge(report, rec, key, 0.7)

    if report["raw"]:
        report["is_ai_generated"] = True


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _clean_user_comment(value: Any) -> str:
    raw = value if isinstance(value, bytes) else str(value).encode("utf-8", "replace")
    if raw.startswith(b"UNICODE\x00"):
        return raw[8:].decode("utf-16-be", "replace").strip("\x00")
    if raw.startswith(b"ASCII\x00\x00\x00"):
        return raw[8:].decode("ascii", "replace").strip("\x00")
    return raw.decode("utf-8", "replace").strip("\x00")


# ---------------------------------------------------------------- MP4


def _inspect_mp4(data: bytes, report: Dict[str, Any]) -> None:
    strings: List[str] = []
    _walk_boxes(data, 0, len(data), strings, depth=0)
    for s in strings:
        if any(k in s for k in ("Steps:", "Negative prompt", '"prompt"', "model_hash")):
            report["raw"].setdefault("mp4:text", s[:20000])
            rec = parse_a1111(s)
            if rec:
                report["tool"] = rec.pop("_tool", "automatic1111")
                _merge(report, rec, "mp4:text", 0.6)
        elif s.strip() and len(s) < 4000:
            report["raw"].setdefault("mp4:comment", s[:4000])
    if report["has_c2pa"]:
        report["notes"].append(
            "C2PA/JUMBF detected: run `c2patool` to read the signed manifest for model + prompt provenance."
        )
    if report["raw"]:
        report["is_ai_generated"] = True


def _walk_boxes(buf: bytes, start: int, end: int, out: List[str], depth: int) -> None:
    if depth > 8:
        return
    i = start
    while i + 8 <= end:
        size = int.from_bytes(buf[i : i + 4], "big")
        btype = buf[i + 4 : i + 8]
        header = 8
        if size == 1 and i + 16 <= end:
            size = int.from_bytes(buf[i + 8 : i + 16], "big")
            header = 16
        if size <= 0 or i + size > end:
            break
        body_start = i + header
        if btype in MP4_BOXES:
            skip = 4 if btype == b"meta" else 0
            _walk_boxes(buf, body_start + skip, i + size, out, depth + 1)
        elif btype == b"data" and i + 12 <= end:
            payload = buf[i + 12 : i + size]
            _collect_strings(payload, out)
        else:
            _collect_strings(buf[body_start : i + size], out)
        i += size


def _collect_strings(payload: bytes, out: List[str]) -> None:
    text = payload.decode("utf-8", "ignore")
    printable = "".join(ch if 32 <= ord(ch) < 127 or ch in "\n\t" else " " for ch in text)
    printable = re.sub(r"\s{3,}", " ", printable).strip()
    if len(printable) > 8 and sum(c.isalpha() for c in printable) > 8:
        out.append(printable)


# ---------------------------------------------------------------- parameter parsers


_KV_RE = re.compile(r'([A-Za-z][A-Za-z0-9 _.\-/]*?):\s*("(?:[^"\\]|\\.)*"|[^,]*)(?:,|$)')


def parse_a1111(text: str) -> Dict[str, Any]:
    """Parse an Automatic1111 / Forge parameter block into a flat dict."""
    if not text:
        return {}
    text = text.strip()
    negative = ""
    params_line = ""
    params_key = (
        r"\n(?=(?:Steps|Sampler|CFG scale|Seed|Size|Model hash|Model|Version|Denoising strength|"
        r"Clip skip|Schedule type|VAE|Lora hashes|TI hashes|Hires upscale|Hires upscaler)\s*:)"
    )

    idx = text.find("Negative prompt:")
    if idx >= 0:
        prompt = text[:idx].strip()
        rest = text[idx + len("Negative prompt:") :]
        m = re.search(params_key, rest)
        if m:
            negative = rest[: m.start()].strip()
            params_line = rest[m.start() + 1 :]
        else:
            negative = rest.strip()
    else:
        m = re.search(params_key, text)
        if m:
            prompt = text[: m.start()].strip()
            params_line = text[m.start() + 1 :]
        else:
            prompt = text.strip()

    prompt = prompt.rstrip("\n").strip()

    kv: Dict[str, Any] = {}
    for key, value in _KV_RE.findall(params_line or ""):
        value = value.strip().strip('"')
        kv[key.strip()] = value

    out: Dict[str, Any] = {}
    if prompt:
        out["prompt"] = prompt
    if negative:
        out["negative"] = negative
    if "Seed" in kv:
        out["seed"] = _to_int(kv["Seed"])
    if "Steps" in kv:
        out["steps"] = _to_int(kv["Steps"])
    if "CFG scale" in kv:
        out["cfg"] = _to_float(kv["CFG scale"])
    if "Sampler" in kv:
        out["sampler"] = kv["Sampler"]
    if "Model" in kv:
        out["model"] = kv["Model"]
    if "Model hash" in kv:
        out["model_hash"] = kv["Model hash"]
    if "VAE" in kv:
        out["vae"] = kv["VAE"]
    if "Size" in kv and "x" in kv["Size"]:
        w, _, h = kv["Size"].partition("x")
        out["width"], out["height"] = _to_int(w), _to_int(h)
    if "Schedule type" in kv:
        out["scheduler"] = kv["Schedule type"]
    if "Version" in kv:
        out["_tool"] = "automatic1111"
        out["tool_version"] = kv["Version"]
    if "Lora hashes" in kv:
        loras = []
        for part in kv["Lora hashes"].split(","):
            name, _, digest = part.partition(":")
            if name.strip():
                loras.append({"name": name.strip(), "hash": digest.strip()})
        if loras:
            out["loras"] = loras
    if "Denoising strength" in kv:
        out["denoising"] = _to_float(kv["Denoising strength"])
    if "Clip skip" in kv:
        out["clip_skip"] = _to_int(kv["Clip skip"])
    out.setdefault("_tool", "automatic1111")
    return out


_TEXT_NODES = {
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeFlux",
    "BNK_CLIPTextEncodeAdvanced",
    "smZ CLIPTextEncode",
    "Text Multiline",
}
_SAMPLER_NODES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced", "KSampler (Efficient)"}
_LATENT_NODES = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyLatentImagePresets", "EmptyFluxLatentImage"}
_LOADER_KEYS = ("ckpt_name", "unet_name", "model_name", "config_name")


def parse_comfy_api(text: str) -> Dict[str, Any]:
    """Unwrap a ComfyUI *API-format* graph (the ``prompt`` chunk) into parameters."""
    try:
        graph = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(graph, dict) or not graph:
        return {}
    if not all(isinstance(v, dict) and "class_type" in v for v in graph.values()):
        return parse_comfy_ui(text)

    out: Dict[str, Any] = {}

    def node_of(link: Any) -> Optional[Dict[str, Any]]:
        if isinstance(link, (list, tuple)) and link and isinstance(link[0], str):
            n = graph.get(link[0])
            return n if isinstance(n, dict) else None
        return None

    def text_of(link: Any, depth: int = 0) -> str:
        if depth > 6:
            return ""
        node = node_of(link)
        if node is None:
            return str(link) if isinstance(link, str) else ""
        cls = node.get("class_type", "")
        inputs = node.get("inputs") or {}
        if cls in _TEXT_NODES or "text" in inputs:
            raw = inputs.get("text", "")
            if isinstance(raw, str):
                return raw
            return text_of(raw, depth + 1)
        if cls in ("String Literal", "Text", "PrimitiveNode"):
            return str(inputs.get("string", inputs.get("value", "")))
        parts = [text_of(v, depth + 1) for v in inputs.values() if isinstance(v, (list, tuple)) and v and isinstance(v[0], str)]
        return " ".join(p for p in parts if p)

    sampler = next((n for n in graph.values() if n.get("class_type") in _SAMPLER_NODES), None)
    if sampler:
        ins = sampler.get("inputs") or {}
        pos = text_of(ins.get("positive"))
        neg = text_of(ins.get("negative"))
        if pos:
            out["prompt"] = pos.strip()
        if neg:
            out["negative"] = neg.strip()
        for key, field in (("seed", "seed"), ("noise_seed", "seed")):
            if key in ins and isinstance(ins[key], (int, float)):
                out["seed"] = int(ins[key])
                break
        for key, field, caster in (("steps", "steps", _to_int), ("cfg", "cfg", _to_float)):
            if key in ins and isinstance(ins[key], (int, float)):
                out[field] = caster(ins[key])
        for key, field in (("sampler_name", "sampler"), ("scheduler", "scheduler")):
            if isinstance(ins.get(key), str):
                out[field] = ins[key]

        # model / checkpoint
        model_link = ins.get("model")
        seen = set()
        while model_link and isinstance(model_link, (list, tuple)) and model_link[0] not in seen:
            seen.add(model_link[0])
            n = graph.get(model_link[0]) or {}
            for k in _LOADER_KEYS:
                if isinstance((n.get("inputs") or {}).get(k), str):
                    out["model"] = n["inputs"][k]
                    break
            if "model" in out:
                break
            model_link = (n.get("inputs") or {}).get("model")

        # latent size
        latent = ins.get("latent_image")
        ln = node_of(latent)
        if ln:
            li = ln.get("inputs") or {}
            if isinstance(li.get("width"), (int, float)):
                out["width"] = int(li["width"])
            if isinstance(li.get("height"), (int, float)):
                out["height"] = int(li["height"])

    # LoRAs
    loras = []
    for n in graph.values():
        if "LoraLoader" in str(n.get("class_type", "")):
            li = n.get("inputs") or {}
            name = li.get("lora_name")
            if isinstance(name, str):
                loras.append(
                    {
                        "name": name,
                        "strength": li.get("strength_model"),
                    }
                )
    if loras:
        out["loras"] = loras

    controls = sorted({str(n.get("class_type")) for n in graph.values() if _is_control_node(str(n.get("class_type")))})
    if controls:
        out["controls"] = controls

    out["_tool"] = "comfyui"
    return out


def parse_comfy_ui(text: str) -> Dict[str, Any]:
    """Best-effort unwrap of a ComfyUI *UI-format* workflow (lower confidence)."""
    try:
        wf = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    nodes = wf.get("nodes") if isinstance(wf, dict) else None
    if not isinstance(nodes, list):
        return {}

    out: Dict[str, Any] = {}
    texts = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if n.get("type") in ("CLIPTextEncode", "CLIPTextEncodeSDXL") and n.get("widgets_values"):
            val = n["widgets_values"][0]
            if isinstance(val, str) and val.strip():
                texts.append(val.strip())
        if n.get("type") in _SAMPLER_NODES and n.get("widgets_values"):
            wv = n["widgets_values"]
            try:
                out.setdefault("seed", int(wv[0]))
                out.setdefault("steps", int(wv[2]))
                out.setdefault("cfg", float(wv[3]))
                out.setdefault("sampler", str(wv[4]))
                if len(wv) > 5:
                    out.setdefault("scheduler", str(wv[5]))
            except (IndexError, TypeError, ValueError):
                pass
        if n.get("type") in _LATENT_NODES and n.get("widgets_values"):
            wv = n["widgets_values"]
            if len(wv) >= 2 and isinstance(wv[0], (int, float)):
                out.setdefault("width", int(wv[0]))
                out.setdefault("height", int(wv[1]))
        if n.get("type") in ("CheckpointLoaderSimple", "UNETLoader") and n.get("widgets_values"):
            out.setdefault("model", str(n["widgets_values"][0]))

    if texts:
        out["prompt"] = texts[0]
        if len(texts) > 1 and len(texts[1]) < len(texts[0]):
            out["negative"] = texts[1]
        elif len(texts) > 1:
            out["negative"] = texts[1]
    out["_tool"] = "comfyui"
    return out


def _is_control_node(cls: str) -> bool:
    low = cls.lower()
    if any(skip in low for skip in ("sampler", "loader", "save", "preview", "note", "string")):
        return False
    keys = ("controlnet", "ipadapter", "reference", "dwpose", "openpose", "canny", "depth", "segment", " ipadapter")
    return any(k.strip() in low for k in keys)


def parse_novelai(text: str) -> Dict[str, Any]:
    try:
        blob = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(blob, dict):
        return {}
    # NovelAI V4 nests per-character prompts; keep the flat fields.
    out: Dict[str, Any] = {}
    if isinstance(blob.get("prompt"), str):
        out["prompt"] = blob["prompt"]
    if isinstance(blob.get("uc"), str):
        out["negative"] = blob["uc"]
    for key, field, caster in (
        ("steps", "steps", _to_int),
        ("scale", "cfg", _to_float),
        ("seed", "seed", _to_int),
        ("sampler", "sampler", str),
        ("width", "width", _to_int),
        ("height", "height", _to_int),
    ):
        if key in blob:
            out[field] = caster(blob[key])
    if blob.get("software") == "novelai" or "uc" in blob:
        out["_tool"] = "novelai"
    return out


def parse_invokeai(text: str) -> Dict[str, Any]:
    try:
        blob = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(blob, dict):
        return {}
    out: Dict[str, Any] = {}
    node = blob.get("node") if isinstance(blob.get("node"), dict) else {}
    for src in (node, blob):
        if isinstance(src.get("prompt"), str) and "prompt" not in out:
            out["prompt"] = src["prompt"]
        if isinstance(src.get("positive_prompt"), str) and "prompt" not in out:
            out["prompt"] = src["positive_prompt"]
        if isinstance(src.get("negative_prompt"), str):
            out["negative"] = src["negative_prompt"]
        for key, field in (("seed", "seed"), ("steps", "steps"), ("cfg_scale", "cfg"), ("width", "width"), ("height", "height")):
            if key in src and field not in out:
                out[field] = src[key]
        if "model" in src and isinstance(src["model"], dict) and "model_name" in src["model"]:
            out.setdefault("model", src["model"]["model_name"])
    out["_tool"] = "invokeai"
    return out


# ---------------------------------------------------------------- helpers


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _merge(report: Dict[str, Any], rec: Dict[str, Any], source: str, confidence: float) -> None:
    rec = {k: v for k, v in rec.items() if v is not None}
    tool = rec.pop("_tool", None)
    if tool and not report.get("tool"):
        report["tool"] = tool
    for key, value in rec.items():
        if key in ("prompt", "negative") and isinstance(value, str) and not value.strip():
            continue
        if key not in report["recovered"]:
            report["recovered"][key] = value
            report["evidence"].append({"field": key, "source": source, "confidence": confidence})
    if report["recovered"]:
        report["is_ai_generated"] = True


def _promote(report: Dict[str, Any]) -> None:
    """Fill size/model pins from the container when the metadata lacks them."""
    if report["width"] and not report["recovered"].get("width"):
        report["recovered"]["width"] = report["width"]
        report["evidence"].append({"field": "width", "source": "container", "confidence": 1.0})
    if report["height"] and not report["recovered"].get("height"):
        report["recovered"]["height"] = report["height"]
        report["evidence"].append({"field": "height", "source": "container", "confidence": 1.0})
    if report["has_c2pa"]:
        report["notes"].append("C2PA Content Credentials present - signed provenance is recoverable.")
    if not report["raw"] and not report["recovered"].get("prompt"):
        report["notes"].append("No embedded generation metadata: fall back to visual captioning.")


def summarize(report: Dict[str, Any]) -> str:
    """One-line human summary for CLI output."""
    if not report["raw"] and not report["recovered"].get("prompt"):
        return f"{report['format']} {report['width']}x{report['height']} - no embedded recipe (caption needed)"
    tool = report.get("tool") or "unknown-tool"
    fields = ", ".join(sorted(k for k in report["recovered"] if k not in ("width", "height"))) or "raw-only"
    return f"{report['format']} via {tool} - recovered: {fields}"
