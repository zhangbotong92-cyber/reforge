"""Layer 7 — a zero-dependency local Web UI.

No node, no bundler, no framework: ``reforge serve`` starts a stdlib
``ThreadingHTTPServer`` that exposes four JSON endpoints and one HTML page.
The page is a single file, so it also works if you just open it and point it at
any remote deployment of the same API.
"""

from __future__ import annotations

import base64
import json
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from . import __version__, controls, forensics, metrics, providers, recipe as recipelib, schema, video

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 64 * 1024 * 1024
_STATE: Dict[str, Any] = {"provider": "offline"}
_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = f"reforge/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if self.path.startswith("/api/"):
            return  # keep the console clean

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"ok": False, "error": message})

    def _body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ValueError("empty body")
        if length > MAX_BODY:
            raise ValueError(f"payload too large ({length} bytes)")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    # -- routes -----------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(HTTPStatus.NO_CONTENT, b"")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._serve_file(STATIC / "index.html", "text/html; charset=utf-8")
        if path == "/api/health":
            return self._json(200, {"ok": True, "version": __version__, "provider": _STATE["provider"]})
        if path == "/api/doctor":
            return self._json(200, {"ok": True, **controls.doctor()})
        candidate = (STATIC / path.lstrip("/")).resolve()
        if STATIC in candidate.parents and candidate.is_file():
            return self._serve_file(candidate, _ctype(candidate.suffix))
        self._error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        try:
            payload = self._body()
        except Exception as exc:  # noqa: BLE001
            return self._error(400, f"bad request: {exc}")
        try:
            if path == "/api/analyze":
                return self._json(200, api_analyze(payload))
            if path == "/api/embed":
                return self._json(200, api_embed(payload))
            if path == "/api/score":
                return self._json(200, api_score(payload))
            if path == "/api/recipe":
                return self._json(200, api_recipe(payload))
            if path == "/api/controls":
                return self._json(200, api_controls(payload))
            if path == "/api/video":
                return self._json(200, api_video(payload))
        except Exception as exc:  # noqa: BLE001 - surface the message to the UI
            return self._error(500, f"{type(exc).__name__}: {exc}")
        self._error(404, "not found")

    def _serve_file(self, path: Path, ctype: str) -> None:
        if not path.is_file():
            return self._error(404, "missing static asset")
        self._send(200, path.read_bytes(), ctype)


def _ctype(suffix: str) -> str:
    return {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
    }.get(suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------- api


def _decode(payload: Dict[str, Any], key: str = "data") -> bytes:
    blob = payload.get(key) or ""
    if "," in blob[:120] and "base64" in blob[:120]:
        blob = blob.split(",", 1)[1]
    if not blob:
        raise ValueError(f"missing '{key}' (base64)")
    return base64.b64decode(blob)


def _tempfile(blob: bytes, name: str) -> Path:
    suffix = Path(name or "upload.bin").suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="reforge-")
    with open(fd, "wb") as fh:  # type: ignore[arg-type]
        fh.write(blob)
    return Path(tmp)


def api_analyze(payload: Dict[str, Any]) -> Dict[str, Any]:
    blob = _decode(payload)
    name = payload.get("filename") or "upload.png"
    provider_spec = payload.get("provider") or _STATE["provider"]
    tmp = _tempfile(blob, name)
    try:
        report = forensics.inspect(tmp)
        report["path"] = name
        caption = None
        provider_name = "metadata-only"
        if payload.get("caption", True):
            try:
                captioner = providers.build_captioner(provider_spec)
                if captioner.available():
                    caption = captioner.caption(tmp, hint=payload.get("hint") or "")
                    provider_name = captioner.name
                else:
                    caption = providers.OfflineCaptioner().caption(tmp)
                    provider_name = "offline-heuristic"
            except Exception as exc:  # noqa: BLE001 - captioning must never block forensics
                caption = providers.OfflineCaptioner().caption(tmp)
                provider_name = f"offline-heuristic (fallback: {exc})"
        caption = caption or schema.empty_caption()
        caption["width"] = report.get("width")
        caption["height"] = report.get("height")

        if report.get("recovered"):
            rec = recipelib.from_inspection(report, caption=caption)
            if not rec.get("prompt"):
                rec["prompt"] = caption.get("summary", "")
        else:
            rec = recipelib.from_caption(caption, report)
        rec["plan"] = controls.plan(rec, runtime=payload.get("runtime") or "diffusers")
        rec["gaps"] = recipelib.validate(rec)
        rec["palette"] = controls.palette(tmp)

        return {
            "ok": True,
            "provider": provider_name,
            "forensics": report,
            "caption": caption,
            "dialects": schema.compile_all(caption),
            "recipe": recipelib.to_dict(rec),
            "plan": rec["plan"],
            "gaps": rec["gaps"],
        }
    finally:
        tmp.unlink(missing_ok=True)


def api_embed(payload: Dict[str, Any]) -> Dict[str, Any]:
    blob = _decode(payload)
    rec = payload.get("recipe") or {}
    tmp = _tempfile(blob, payload.get("filename") or "upload.png")
    out_dir = Path(tempfile.mkdtemp(prefix="reforge-embed-"))
    try:
        dst = out_dir / "stamped.png"
        recipelib.embed_into_png(tmp, dst, rec)
        return {"ok": True, "filename": "stamped.png", "png": base64.b64encode(dst.read_bytes()).decode()}
    finally:
        tmp.unlink(missing_ok=True)
        for f in out_dir.glob("*"):
            f.unlink(missing_ok=True)
        out_dir.rmdir()


def api_recipe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Re-read a recipe stamped inside a PNG (round-trip check from the UI)."""
    blob = _decode(payload)
    tmp = _tempfile(blob, payload.get("filename") or "upload.png")
    try:
        found = recipelib.read_embedded(tmp)
        return {"ok": True, "recipe": found}
    finally:
        tmp.unlink(missing_ok=True)


def api_score(payload: Dict[str, Any]) -> Dict[str, Any]:
    a = _tempfile(_decode(payload, "a"), "a.png")
    b = _tempfile(_decode(payload, "b"), "b.png")
    try:
        return {"ok": True, "score": metrics.score_pair(a, b)}
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)


def api_controls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract control maps and hand them back as data URLs for the UI gallery."""
    blob = _decode(payload)
    kinds = payload.get("kinds") or ["canny", "lineart", "tile", "seg", "depth"]
    tmp = _tempfile(blob, payload.get("filename") or "upload.png")
    work = Path(tempfile.mkdtemp(prefix="reforge-ctl-"))
    try:
        result = controls.extract_all(tmp, work, kinds=kinds, backend=payload.get("backend") or "auto")
        images = {}
        for c in result["controls"]:
            path = Path(c["path"])
            if path.exists():
                images[c["kind"]] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
        return {"ok": True, "controls": result["controls"], "skipped": result["skipped"], "images": images}
    finally:
        tmp.unlink(missing_ok=True)
        for f in work.glob("*"):
            f.unlink(missing_ok=True)
        work.rmdir()


def api_video(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Shot-split from a list of frames (the UI's practical video path)."""
    frames = payload.get("frames") or []
    if not frames:
        raise ValueError("send frames: [{filename, data}]")
    work = Path(tempfile.mkdtemp(prefix="reforge-video-"))
    try:
        for i, frame in enumerate(frames):
            suffix = Path(frame.get("filename") or "f.png").suffix or ".png"
            (work / f"f{i:03d}{suffix}").write_bytes(_decode(frame))
        analysis = video.analyze(frames_dir=work, max_shots=int(payload.get("max_shots") or 8))
        captioner_name = payload.get("provider") or _STATE["provider"]
        try:
            captioner = providers.build_captioner(captioner_name)
            if not captioner.available():
                captioner = providers.OfflineCaptioner()
        except Exception:  # noqa: BLE001
            captioner = providers.OfflineCaptioner()
        scripted = video.script(analysis, captioner, dialect=payload.get("dialect") or "kling")
        rec = video.build_recipe(scripted, runtime=payload.get("runtime") or "diffusers")
        return {
            "ok": True,
            "summary": video.summarize(analysis),
            "shots": [
                {"index": s["index"], "start": s["start"], "end": s["end"], "prompt": s["compiled"]["prompt"]}
                for s in scripted["shots"]
            ],
            "recipe": recipelib.to_dict(rec),
            "plan": rec["video_plan"],
        }
    finally:
        for f in work.glob("*"):
            f.unlink(missing_ok=True)
        work.rmdir()


# ---------------------------------------------------------------- entrypoint


def serve(host: str = "127.0.0.1", port: int = 8770, provider: str = "offline") -> None:
    _STATE["provider"] = provider or "offline"
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"REFORGE UI  ->  {url}")
    print(f"provider    ->  {_STATE['provider']}  (change with --provider openai:gpt-4o-mini)")
    print("ctrl-c to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
