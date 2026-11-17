"""REFORGE command line.

    reforge inspect  IMG...              forensic report (metadata first)
    reforge caption  IMG                 structured caption + all dialects
    reforge recipe   IMG                 full recipe, optional --embed out.png
    reforge score    A B                 fidelity metrics between two files
    reforge controls IMG                 extract edges / palette / plan
    reforge refine   IMG                 closed loop with a generator command
    reforge serve                        local Web UI
    reforge demo                         end-to-end run on generated fixtures
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__, controls, forensics, invert, metrics, providers, recipe as recipelib, refine as refinelib, schema, video


def _out(text: str = "") -> None:
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:  # pragma: no cover - legacy Windows consoles
        sys.stdout.write(text.encode("ascii", "replace").decode("ascii") + "\n")


def _dump(obj: Any) -> None:
    _out(json.dumps(obj, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- commands


def cmd_inspect(args: argparse.Namespace) -> int:
    rows = []
    for target in args.paths:
        report = forensics.inspect(target)
        rows.append(report)
        if args.json:
            continue
        _out(f"\n=== {target}")
        _out(f"  {forensics.summarize(report)}")
        if report["recovered"]:
            for key, value in report["recovered"].items():
                shown = value if not isinstance(value, str) or len(value) < 160 else value[:160] + "..."
                _out(f"    {key:14s} {shown}")
        for note in report["notes"]:
            _out(f"  ! {note}")
    if args.json:
        _dump(rows)
    return 0


def cmd_caption(args: argparse.Namespace) -> int:
    captioner = providers.build_captioner(args.provider)
    if not captioner.available():
        _out(f"provider {captioner.name} is not configured (missing API key); falling back to offline")
        captioner = providers.OfflineCaptioner()
    report = forensics.inspect(args.path)
    caption = captioner.caption(args.path, hint=args.hint or "")
    caption["width"] = report.get("width")
    caption["height"] = report.get("height")
    dialects = [d.strip() for d in args.dialects.split(",") if d.strip()]
    result = {"forensics": report, "caption": caption, "dialects": schema.compile_all(caption, dialects)}
    if args.json:
        _dump(result)
        return 0
    _out(f"\n=== {args.path}")
    _out(f"  provider: {caption['provider']}  confidence: {caption['confidence']}  source: {caption['source']}")
    _out(f"  summary : {caption['summary']}")
    if caption.get("tags"):
        _out(f"  tags    : {', '.join(caption['tags'])}")
    for name, compiled in result["dialects"].items():
        _out(f"\n  [{name}] {compiled['prompt']}")
        if compiled.get("negative"):
            _out(f"        negative: {compiled['negative']}")
    return 0


def cmd_recipe(args: argparse.Namespace) -> int:
    report = forensics.inspect(args.path)
    caption = None
    if args.caption:
        captioner = providers.build_captioner(args.provider)
        if captioner.available():
            caption = captioner.caption(args.path, hint=args.hint or "")
            caption["width"] = report.get("width")
            caption["height"] = report.get("height")
    rec = recipelib.from_inspection(report, caption=caption) if report.get("recovered") else recipelib.from_caption(
        caption or schema.empty_caption(), report
    )
    if args.controls:
        outdir = Path(args.controls_dir or (Path(args.path).parent / "reforge_controls"))
        kinds = [k.strip() for k in (args.kinds or "canny,lineart,tile,seg,depth").split(",") if k.strip()]
        result = controls.extract_all(args.path, outdir, kinds=kinds, backend=args.backend)
        for c in result["controls"]:
            weight = 0.30 if c.get("approximate") else 0.60
            rec["controls"].append(
                {"kind": c["kind"], "model": c["method"], "path": c["path"], "weight": weight,
                 "source": c["backend"], "approximate": c["approximate"]}
            )
        for s in result["skipped"]:
            rec["notes"].append(f"skipped {s['kind']}: {s['reason'][:160]}")
        recipelib.add_ref(rec, str(args.path), weight=0.55, role="identity")
        rec["palette"] = controls.palette(args.path)
        rec["fidelity"] = recipelib.fidelity_level(rec)

    rec["plan"] = controls.plan(rec, runtime=args.runtime)
    rec["gaps"] = recipelib.validate(rec)

    if args.out:
        Path(args.out).write_text(json.dumps(recipelib.to_dict(rec), ensure_ascii=False, indent=2), encoding="utf-8")
        _out(f"recipe written: {args.out}")
    if args.embed:
        dst = recipelib.embed_into_png(args.path, args.embed, rec)
        _out(f"recipe stamped into: {dst}")
    if args.sidecar:
        side = recipelib.write_sidecar(args.path, rec)
        _out(f"sidecar written: {side}")

    if args.json or not (args.out or args.embed or args.sidecar):
        _dump(recipelib.to_dict(rec))
    else:
        _out(recipelib.summary(rec))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    result = metrics.score_pair(args.target, args.candidate)
    if args.json:
        _dump(result)
        return 0
    _out(f"\ntarget   : {args.target}")
    _out(f"candidate: {args.candidate}")
    for key in ("psnr", "ssim", "phash", "phash_similarity", "edge_iou", "palette_distance", "composite"):
        _out(f"  {key:18s} {result[key]}")
    _out(f"  grade              {result['grade']}")
    return 0


def cmd_controls(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir or (Path(args.path).parent / "reforge_controls"))
    kinds = [k.strip() for k in (args.kinds or "canny,lineart,tile,seg,depth").split(",") if k.strip()]
    result = controls.extract_all(args.path, outdir, kinds=kinds, backend=args.backend)
    info = {
        "probe": controls.probe(args.path),
        "palette": controls.palette(args.path),
        "controls": result["controls"],
        "skipped": result["skipped"],
        "outdir": result["outdir"],
        "doctor": controls.doctor(),
    }
    if args.json:
        _dump(info)
        return 0
    _out(f"\n=== {args.path}")
    for c in info["controls"]:
        flag = " (approximate)" if c.get("approximate") else ""
        _out(f"  [ok]   {c['kind']:9s} {c['backend']:8s} {c['method']}{flag} -> {c['path']}")
    for s in info["skipped"]:
        _out(f"  [skip] {s['kind']:9s} {s['reason'][:110]}")
    _out(f"\n  outdir: {info['outdir']}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    report = controls.doctor()
    if args.json:
        _dump(report)
        return 0
    _out("\n=== capability matrix")
    _out(f"  opencv={report['opencv']}  onnxruntime={report['onnxruntime']}  torch={report['torch']}  ffmpeg={report['ffmpeg']}")
    _out(f"  model dir: {report['model_dir']}")
    for kind, row in report["kinds"].items():
        mark = "ok  " if row["available"] else "MISS"
        approx = " (approximate)" if row.get("approximate") else ""
        extra = row.get("method") or row.get("hint") or row.get("install") or ""
        _out(f"  [{mark}] {kind:8s} {row['backend']:8s} {extra}{approx}")
    return 0


def cmd_invert(args: argparse.Namespace) -> int:
    """L3: turn an image into a latent that samples back to itself."""
    if not args.model:
        # no checkpoint: prove the numerics instead of pretending to invert
        report = invert.self_test(steps=args.steps)
        if args.json:
            _dump(report)
            return 0
        _out("\n=== inversion self-test (no checkpoint needed)")
        _out(f"  constant field : max|err| {report['const_field_max_abs_error']:.2e}  (exact, any step count)")
        _out(f"  taylor (2nd)   : rel err   {report['linear_taylor_rel_error']:.2e}  convergence x{report['linear_taylor_convergence']:.2f} per halving")
        _out(f"  euler  (1st)   : rel err   {report['linear_euler_rel_error']:.2e}")
        _out(f"  ddim algebra   : rel err   {report['ddim_oracle_rel_error']:.2e}  (exact with an oracle eps)")
        _out(f"  ddim predicted : rel err   {report['ddim_approx_rel_error']:.2e}  (the inherent gap null-text fixes)")
        _out("\n  pass --model <hf-id> for a real inversion (needs diffusers + torch + a checkpoint)")
        _out("  e.g. reforge invert img.png --model stabilityai/stable-diffusion-2-1 --prompt \"...\"")
        return 0

    try:
        result = invert.invert_with_diffusers(
            args.model,
            args.path,
            args.prompt or "",
            steps=args.steps,
            guidance=args.guidance,
        )
    except RuntimeError as exc:
        _out(f"error: {exc}")
        return 1

    latent = result.pop("latent")
    out = Path(args.out or Path(args.path).with_suffix(".latents.npz"))
    path = invert.save_latents(out, latent, result)

    if args.recipe:
        rec_path = Path(args.recipe)
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        recipelib.add_inversion(rec, "ddim", str(path), steps=args.steps)
        rec["fidelity"] = recipelib.fidelity_level(rec)
        rec["init"]["model"] = args.model
        rec_path.write_text(json.dumps(recipelib.to_dict(rec), ensure_ascii=False, indent=2), encoding="utf-8")
        _out(f"recipe upgraded to {rec['fidelity']}: {rec_path}")

    if args.json:
        _dump({**result, "latents": str(path)})
        return 0
    _out(f"\n=== inverted with {args.model} on {result.get('device')}")
    _out(f"  latents : {path}")
    if "psnr" in result:
        _out(f"  verify  : {result['psnr']:.2f} dB  -> {'L3 pixel-exact' if result.get('passes_l3') else 'below the 30 dB bar'}")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    captioner = providers.build_captioner(args.provider)
    if not captioner.available():
        _out("provider unavailable; using offline captioner")
        captioner = providers.OfflineCaptioner()

    try:
        analysis = video.analyze(
            video=args.path,
            frames_dir=args.frames_dir,
            outdir=args.shots_dir,
            max_shots=args.max_shots,
            scene_threshold=args.threshold,
        )
    except video.VideoUnavailable as exc:
        _out(f"error: {exc}")
        return 1

    scripted = video.script(analysis, captioner, dialect=args.dialect, hint=args.hint or "")
    rec = video.build_recipe(scripted, runtime=args.runtime)
    rec["plan"] = controls.plan(rec, runtime=args.runtime)

    if args.out:
        Path(args.out).write_text(json.dumps(recipelib.to_dict(rec), ensure_ascii=False, indent=2), encoding="utf-8")
        _out(f"recipe written: {args.out}")

    if args.json:
        _dump(recipelib.to_dict(rec))
        return 0

    _out(f"\n=== {video.summarize(analysis)}")
    for shot in scripted["shots"]:
        _out(f"\n  shot {shot['index']}  [{shot.get('start')}s - {shot.get('end')}s]  {Path(shot['keyframe']).name}")
        _out(f"    {shot['compiled']['prompt'][:180]}")
    _out(f"\n  {len(rec['shots'])} shot recipes, {len(rec.get('refs') or [])} identity anchors")
    for rule in rec["video_plan"]["rules"]:
        _out(f"  - {rule}")
    return 0


def cmd_refine(args: argparse.Namespace) -> int:
    captioner = providers.build_captioner(args.provider)
    if not captioner.available():
        _out("provider unavailable; using offline captioner")
        captioner = providers.OfflineCaptioner()
    generator = refinelib.make_generator(args.generator_kind, args.generator)
    loop = refinelib.RefineLoop(
        target=args.path,
        captioner=captioner,
        generator=generator,
        dialect=args.dialect,
        max_rounds=args.rounds,
        threshold=args.threshold,
        workdir=args.workdir,
    )
    result = loop.run()
    if args.json:
        _dump(result)
    else:
        best = result["best"]
        if best and best.get("score"):
            _out(f"\nbest round {best['round']}: composite={best['score']['composite']} ({best['score']['grade']})")
            _out(f"prompt: {best['prompt']}")
        _out(f"converged: {result['converged']}")
        _out(f"artifacts: {result['workdir']}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server

    server.serve(host=args.host, port=args.port, provider=args.provider)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Prove the whole chain works, offline, in one command."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    try:
        import make_sample  # type: ignore
    except ImportError:
        _out("examples/make_sample.py not found; run from the repo root")
        return 1
    out = Path(args.outdir or "demo_out")
    out.mkdir(parents=True, exist_ok=True)
    make_sample.main(str(out))

    _out("\n[1/7] forensics")
    for name in ("a1111.png", "comfyui.png", "novelai.png", "plain.jpg"):
        _out(f"  {name:14s} {forensics.summarize(forensics.inspect(out / name))}")
    _out("  doctor: " + ", ".join(f"{k}={v['backend']}" for k, v in controls.doctor()["kinds"].items()))

    _out("\n[2/7] recipe + stamp")
    rec = recipelib.from_inspection(forensics.inspect(out / "a1111.png"))
    stamped = recipelib.embed_into_png(out / "a1111.png", out / "a1111.stamped.png", rec)
    back = recipelib.read_embedded(stamped)
    _out(f"  recipe: {recipelib.summary(rec)}")
    _out(f"  stamp round-trip: {'OK' if back and back['fingerprint'] == recipelib.to_dict(rec)['fingerprint'] else 'FAILED'}")

    _out("\n[3/7] offline caption + dialects")
    cap = providers.OfflineCaptioner().caption(out / "plain.jpg")
    for name, compiled in schema.compile_all(cap).items():
        _out(f"  [{name}] {compiled['prompt'][:110]}")

    _out("\n[4/7] fidelity score (near-duplicate should land L2+)")
    score = metrics.score_pair(out / "a1111.png", out / "alt.png")
    _out(f"  composite={score['composite']} grade={score['grade']} ssim={score['ssim']:.3f} edge_iou={score['edge_iou']:.3f}")

    _out("\n[5/7] control signals")
    ctl = controls.extract_all(out / "a1111.png", out / "ctl", kinds=["canny", "lineart", "tile", "seg", "depth", "pose"])
    for c in ctl["controls"]:
        _out(f"  [ok]   {c['kind']:9s} {c['backend']:8s}{' (approximate)' if c['approximate'] else ''}")
    for s in ctl["skipped"]:
        _out(f"  [skip] {s['kind']:9s} {s['reason'][:90]}")

    _out("\n[6/7] video: shot-split + per-shot recipe")
    frames = out / "frames"
    frames.mkdir(exist_ok=True)
    for i, name in enumerate(("a1111.png", "alt.png", "comfyui.png")):
        (frames / f"f{i:02d}.png").write_bytes((out / name).read_bytes())
    analysis = video.analyze(frames_dir=frames, max_shots=3)
    scripted = video.script(analysis, providers.OfflineCaptioner(), dialect="kling")
    vrec = video.build_recipe(scripted)
    _out(f"  {video.summarize(analysis)}")
    for shot in scripted["shots"]:
        _out(f"  shot {shot['index']}: {shot['compiled']['prompt'][:96]}")
    _out(f"  recipe: {len(vrec['shots'])} shots, {len(vrec['refs'])} identity anchors, fidelity={vrec['fidelity']}")

    _out("\n[7/7] inversion numerics (no checkpoint, no GPU)")
    st = invert.self_test(steps=20)
    _out(f"  constant field : max|err| {st['const_field_max_abs_error']:.1e}  (exact)")
    _out(f"  taylor vs euler: {st['linear_taylor_rel_error']:.1e} vs {st['linear_euler_rel_error']:.1e}"
         f"  (order {st['linear_taylor_convergence']:.2f})")
    _out(f"  ddim algebra   : {st['ddim_oracle_rel_error']:.1e}  exact; predicted-eps gap {st['ddim_approx_rel_error']:.2e}")
    _out("  -> `reforge invert img.png --model <hf-id>` turns this into a real L3 latent")

    _out("\ndone.")
    return 0


# ---------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reforge", description="Reverse-engineer any AI image/video into a reproducible recipe.")
    p.add_argument("--version", action="version", version=f"reforge {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="forensic report: what the file already knows")
    s.add_argument("paths", nargs="+")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("caption", help="structured caption + platform dialects")
    s.add_argument("path")
    s.add_argument("--provider", default="offline")
    s.add_argument("--dialects", default="sd,flux,mj,kling,jimeng,banana")
    s.add_argument("--hint", default="")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_caption)

    s = sub.add_parser("recipe", help="build a recipe, optionally stamp it into the file")
    s.add_argument("path")
    s.add_argument("--caption", action="store_true", help="add a visual caption (needs a provider)")
    s.add_argument("--provider", default="offline")
    s.add_argument("--hint", default="")
    s.add_argument("--controls", action="store_true", help="extract control maps (canny/lineart/tile/seg/depth)")
    s.add_argument("--controls-dir", default=None)
    s.add_argument("--kinds", default=None, help="comma list for --controls")
    s.add_argument("--backend", default="auto", choices=["auto", "builtin", "onnx", "command"])
    s.add_argument("--runtime", default="diffusers", choices=["diffusers", "comfyui", "api"])
    s.add_argument("--out", default=None)
    s.add_argument("--embed", default=None, help="stamp the recipe into a new PNG")
    s.add_argument("--sidecar", action="store_true", help="write NAME.recipe.json next to the media")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_recipe)

    s = sub.add_parser("score", help="fidelity metrics between original and regeneration")
    s.add_argument("target")
    s.add_argument("candidate")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("controls", help="extract control maps (canny/lineart/tile/seg/depth/pose)")
    s.add_argument("path")
    s.add_argument("--outdir", default=None)
    s.add_argument("--kinds", default=None, help="comma list, default canny,lineart,tile,seg,depth")
    s.add_argument("--backend", default="auto", choices=["auto", "builtin", "onnx", "command"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_controls)

    s = sub.add_parser("doctor", help="what this machine can actually extract")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("invert", help="L3: invert an image into a latent (or prove the numerics offline)")
    s.add_argument("path", nargs="?", default=None)
    s.add_argument("--model", default=None, help="hf checkpoint, e.g. stabilityai/stable-diffusion-2-1")
    s.add_argument("--prompt", default="")
    s.add_argument("--steps", type=int, default=30)
    s.add_argument("--guidance", type=float, default=5.0)
    s.add_argument("--out", default=None, help="latents .npz path")
    s.add_argument("--recipe", default=None, help="patch this recipe file to L3-pixel")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_invert)

    s = sub.add_parser("video", help="shot-split a video and build per-shot recipes")
    s.add_argument("path", nargs="?", default=None)
    s.add_argument("--frames-dir", default=None, help="use pre-extracted frames (no ffmpeg needed)")
    s.add_argument("--shots-dir", default=None)
    s.add_argument("--max-shots", type=int, default=8)
    s.add_argument("--threshold", type=float, default=0.30)
    s.add_argument("--dialect", default="kling")
    s.add_argument("--provider", default="offline")
    s.add_argument("--hint", default="")
    s.add_argument("--runtime", default="diffusers", choices=["diffusers", "comfyui", "api"])
    s.add_argument("--out", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_video)

    s = sub.add_parser("refine", help="closed loop: caption->generate->score->critique")
    s.add_argument("path")
    s.add_argument("--provider", default="offline")
    s.add_argument("--dialect", default="sd")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--threshold", type=float, default=0.8)
    s.add_argument("--workdir", default=None)
    s.add_argument("--generator-kind", default="null", choices=["null", "manual", "command"])
    s.add_argument("--generator", default=None, help='e.g. "python gen.py --prompt-file {prompt_file} --seed {seed} --out {out}"')
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_refine)

    s = sub.add_parser("serve", help="local Web UI (stdlib http.server)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--provider", default="offline")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("demo", help="offline end-to-end proof")
    s.add_argument("--outdir", default="demo_out")
    s.set_defaults(func=cmd_demo)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - older interpreters
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except BrokenPipeError:  # pragma: no cover
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should fail loudly but readably
        _out(f"error: {exc}")
        if "--traceback" in sys.argv:
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
