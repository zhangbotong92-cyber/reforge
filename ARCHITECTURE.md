# Architecture

REFORGE is seven modules in a line. Each one has one job, one contract, and can be
replaced without touching the others. That is the whole design.

```
                    ┌──────────────────────────────────────────────┐
  upload ──────────▶│ 0. forensics      forensics.inspect(path)    │
                    │    metadata first, guessing later            │
                    └───────────────┬──────────────────────────────┘
                                    │ report {recovered, raw, evidence[]}
                    ┌───────────────▼──────────────────────────────┐
                    │ 1. providers      Captioner.caption(img)     │
                    │    offline | openai:* | gemini:* | ollama:*  │
                    └───────────────┬──────────────────────────────┘
                                    │ caption  (schema.Caption, strict JSON)
                    ┌───────────────▼──────────────────────────────┐
                    │ 2. schema        compile_prompt(cap, "kling")│
                    │    6 dialects, one contract                  │
                    └───────────────┬──────────────────────────────┘
                                    │ {prompt, negative, params}
                    ┌───────────────▼──────────────────────────────┐
                    │ 3. controls      extract_all(img, ctl/)      │
                    │    builtin → onnx → command   + doctor()     │
                    └───────────────┬──────────────────────────────┘
                                    │ controls[] + refs[]
                    ┌───────────────▼──────────────────────────────┐
                    │ 4. recipe        from_* → embed_into_png     │
                    │    the file now carries its own recipe       │
                    └───────────────┬──────────────────────────────┘
                                    │ recipe.json (+ iTXt chunk)
        video ──▶ 5.5 video  analyze → script → shots[] → plan_video
                                    │
                    ┌───────────────▼──────────────────────────────┐
                    │ 5. metrics       score_pair(orig, regen)     │
                    │ 6. refine        caption→gen→score→critique  │
                    └───────────────┬──────────────────────────────┘
                                    │ composite + grade L0..L3
                    ┌───────────────▼──────────────────────────────┐
                    │ 7. server        stdlib HTTP + single-file UI│
                    └──────────────────────────────────────────────┘
```

## Contracts

### `forensics.inspect(path) -> report`

Never raises on bad input; a corrupt chunk must degrade, not explode.

```python
{
  "format": "png|jpeg|webp|mp4|unknown",
  "width": int, "height": int,
  "tool": "automatic1111|comfyui|novelai|invokeai|None",
  "is_ai_generated": bool,
  "has_c2pa": bool,
  "recovered": {...},                       # normalised generation parameters
  "raw": {"png:tEXt:parameters": "..."},    # untouched source blobs
  "evidence": [{"field": "seed", "source": "png:tEXt:parameters", "confidence": 0.99}],
  "notes": ["..."]
}
```

**Rule:** every recovered field carries provenance. If you cannot say *where* a value came
from, it does not belong in `recovered` — it belongs in a caption.

Parsers implemented today: A1111/Forge/SD.Next parameter blocks, ComfyUI API graphs (with
link-following through `positive`/`negative`/`model`/`latent_image`), ComfyUI UI workflows
(low confidence, node order), NovelAI `Comment`, InvokeAI `sd-metadata`, PNG `tEXt/zTXt/iTXt`,
EXIF `UserComment/ImageDescription/Software`, XMP `parameters=` attributes, MP4 `udta/meta/ilst`
strings, C2PA presence.

### `schema.normalize(raw) -> caption`, `schema.compile_prompt(caption, dialect) -> prompt`

One caption shape for every backend. 28 keys, no nesting except `camera` / `motion`.
Adding a platform is **one function** in `DIALECTS`; it must return
`{"prompt", "negative", "params"}` and nothing else.

### `recipe` — a build artifact, not a description

`fingerprint()` hashes only pixel-deciding fields. `validate()` returns the gaps that keep you
from the next tier. `embed_into_png()` is idempotent: stamping twice leaves one `iTXt` chunk.

The fidelity claim is derived, never asserted by the user:

```
init.latents        -> L3-pixel
controls or refs    -> L2-structural
seed + model pins   -> L1-reproducible
otherwise           -> L1-semantic
```

### `metrics.score_pair(a, b) -> report`

Model-free on purpose: numpy + Pillow only, so it runs in CI and on any laptop.
`composite` is a weighted blend of SSIM (0.35), edge IoU (0.25), pHash similarity (0.20),
palette similarity (0.15), PSNR (0.05). Grades: `>=0.90 L3-pixel`, `>=0.75 L2-structural`,
`>=0.55 L1-semantic`, else `L0-drift`.

Heavy metrics (CLIP-I, DINO-I, ArcFace) are *not* included: they would drag in torch and
silently make the repo heavy. `score_pair` is written so a caller can blend its own extras in.

### `controls` — three backends, one honesty rule

```
builtin   cv2 / numpy, ships with the repo         canny, lineart, tile, seg, depth-proxy
onnx      <MODEL_DIR>/<kind>.onnx + <kind>.json    depth, pose, seg (any graph you trust)
command   REFORGE_CTL_<KIND>="cmd {image} {out}"   depth, pose, flow, camera
```

**Honesty rule:** a map produced by a classical approximation carries
`approximate: true`, gets weight 0.30 instead of 0.60, and adds a line to `plan()["gaps"]`.
A recipe can therefore never *silently* claim learned depth. `doctor()` is the public answer to
"can this machine do it", `extract()` raises `MissingBackend` with the upstream URL otherwise.

`plan()` consumes the maps and emits runtime-specific steps plus the gaps, which is what
`refine` shows a human and what a worker executes.

### `video` — shots beat timelines

`analyze()` accepts either a video (ffmpeg scene detection) or a directory of frames
(no ffmpeg, CI-friendly). `script()` captions each shot and compiles it; `build_recipe()` writes
`recipe["shots"]` with a `first_frame` anchor per shot and promotes the first anchor to an
identity `ref`; `plan_video()` emits a per-shot Wan2.2-VACE/FLF2V plan with four anti-drift rules.
`score_video()` averages frame metrics and reports the worst frame plus a stability number.

The invariant: **a shot without a first-frame anchor is a drift bug**, and `controls.gaps()`
says so by index.

### `refine.RefineLoop`

Generator is a callable with one method, adapted from argv templates (not shell-expanded, so paths
with spaces work on Windows). It keeps the best round, stops on threshold, and records every round
(prompt, score, critique) so a failure is inspectable. `NullGenerator` exists so the loop is
runnable — and testable — with no model at all.

### `invert` — the only path to L3

Model-agnostic by design: it takes callables, not pipelines.

```
rf_invert(v_fn, x0)     dx/dt = v(x,t),  t 0 -> 1   (order 1 Euler | order 2 Taylor)
rf_sample(v_fn, x1)     the same field, t 1 -> 0
ddim_invert(eps_fn, x0) deterministic DDIM, eta = 0
ddim_sample(eps_fn, xT) the exact reverse walk
```

Two claims are measured and kept separate, because conflating them is how projects lie:

| Claim | How it is checked | Result |
|---|---|---|
| algebra is right | oracle eps derived from the ground-truth x0 | 2.2e-17 |
| solver is right | constant velocity field has a linear solution | 4.4e-16 |
| order is right | halve the steps, error must drop ~4x | x3.9 |
| approximation is honest | predicted eps vs oracle eps | 2.7e-02, and reported |

`invert_with_diffusers()` is the bridge to real checkpoints: VAE-encode, walk the DDIM
steps, decode, and compute PSNR with a `passes_l3` flag at the 30 dB bar. It raises a clear
`RuntimeError` when `diffusers` is missing — and a test asserts that it does.

## Testing strategy

36 tests, no GPU, no network, no API key:

- **forensics** — every format gets a synthetic fixture from `examples/make_sample.py` and an
  assertion on a *specific field* (never "no exception").
- **controls** — every extractable map is asserted non-constant; `depth` must report
  `approximate`; `pose` must raise with an actionable message; the `command` backend is tested
  with a stub CLI.
- **onnx** — the test exports a ~30 KB model with `torch.onnx.export`, drops it in a temp model
  dir, and runs the real loading path. Skipped when torch/onnx are absent, so CI stays light.
- **video** — frame directories (no ffmpeg), asserting shot count, anchors, and that a recipe with
  shots survives a JSON round-trip.

## Deliberate non-goals

| Not done | Why |
|---|---|
| Bundling a depth/pose model | 100 MB+ of weights. `MODEL_DIR` + `REFORGE_CTL_*` keeps the repo light and the licence question out of it. |
| Vendoring upstream code | Licences differ (some NC). Adapters only, listed in `UPSTREAM.md`. |
| Calling a paid API by default | The default path must work offline, forever. |
| Claiming "exact" without inversion | That would be a lie the metrics immediately expose. |
| Shipping a Gradio dependency in core | `spaces/` is a thin shell; core stays numpy+pillow. |

## Extension points

1. **New captioner** — subclass `providers.Captioner`, implement `_raw_caption`, register in `build_captioner`.
2. **New dialect** — one function in `schema.DIALECTS`.
3. **New metadata format** — one `parse_*` in `forensics.py` + a branch in `_inspect_png`.
4. **New control backend** — implement `onnx_<kind>()` or set `REFORGE_CTL_<KIND>`; add a row to `KINDS`/`BACKENDS`.
5. **New metric** — add to `score_pair`'s `parts` and rebalance `weights`.
6. **New video backend** — one branch in `video.analyze`, keep the `{index,start,end,keyframe,first_frame}` shape.

If an extension needs more than that, the contract is wrong and should be fixed first.
