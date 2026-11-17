# Roadmap

Ordering rule: **nothing gets merged that cannot be verified by a test or a printed number.**
Each item lists the acceptance check so "done" is not a matter of opinion.

## v0.2 — read more, guess less  *(shipped)*

- [x] **XMP `parameters=` attribute** parse for JPEG/WebP. *Accept: fixture round-trip test.*
- [x] **MP4 `udta/ilst` string scan** + prompt-shaped detection. *Accept: write→read on fixture.*
- [x] **C2PA detection** (presence + actionable note). *Accept: `has_c2pa` flips on a marked file.*
- [x] **ComfyUI UI-graph fallback** with explicit low confidence (0.6) instead of pretending.
- [ ] **PNG alpha-LSB (NovelAI "stealth") pnginfo** decode. *Accept: known-seed fixture decodes to the original blob.*
- [ ] **`reforge batch`** over a directory with a CSV summary. *Accept: 100 files, one table, no crash on garbage.*

## v0.3 — structure lock  *(shipped)*

- [x] **Builtin control extraction**: canny, lineart, tile, GrabCut seg, DCP depth proxy — in-process, no model files. *Accept: every map asserted non-constant.*
- [x] **ONNX backend**: `<MODEL_DIR>/<kind>.onnx` + optional `<kind>.json` (shape/mean/std). *Accept: torch-exported model loads and runs in the test suite.*
- [x] **Command backend**: `REFORGE_CTL_<KIND>` template, tested with a stub CLI.
- [x] **`approximate` flag** propagated into weights, `plan()` and `gaps()` so an approximation can never masquerade as learned depth.
- [x] **`reforge doctor`** capability matrix (kind × backend × install hint).
- [x] **`plan()` emits per-control steps** with backend, method and weight.
- [ ] **Auto-weight search**: grid over control weights to maximise `composite` on a held-out pair. *Accept: beats fixed 0.6/0.3 baseline on 20 pairs.*
- [ ] **`plan()` emits real ComfyUI API JSON** instead of prose steps. *Accept: graph loads and reproduces the fixture recipe.*

## v0.4 — inversion (L3 for real)  *(shipped, checkpoint run pending a GPU)*

- [x] **Inversion worker**: DDIM (eta=0) and rectified-flow solvers, 1st and 2nd order, `latents.npz` storage, `--recipe` promotes a recipe to `L3-pixel`. *Accept: constant field exact to 4.4e-16 · Taylor convergence x3.9 per halving · DDIM algebra exact to 2.2e-17 with an oracle eps.*
- [x] **Self-test you can run anywhere**: `reforge invert` proves the numerics in ~1 s with no checkpoint and no GPU, and reports the DDIM predicted-eps gap instead of hiding it.
- [x] **diffusers bridge** (`invert_with_diffusers`): real VAE encode → DDIM inversion → decode → PSNR check with a `passes_l3` boolean at 30 dB. *Accept: clear RuntimeError when diffusers is absent; covered by a test.*
- [x] **Checkpoint PSNR run** on a real model. *Measured on `nota-ai/bk-sdm-tiny` (CPU): 16.40 dB at guidance 1, 14.32 at 3.0, 11.35 at 7.5 — the gap tracks CFG monotonically, exactly as the literature predicts. Table in [docs/inversion.md](docs/inversion.md).*
- [ ] **Reach the 30 dB bar**: null-text refinement (needs backprop through the UNet) or a rectified-flow checkpoint with the already-implemented second-order solver. *Accept: `passes_l3=true` recorded in docs/inversion.md. Blocked on a GPU, not on code.*
- [ ] **Null-text refinement** pass on top of the inversion to close the predicted-eps gap. *Accept: halves `ddim_approx_rel_error` on the self-test.*
- [ ] **PEZ hard-prompt optimiser** as a caption backend. *Accept: raises composite vs. natural-language caption at equal seed. — upstream: [hard-prompts-made-easy](https://github.com/anwoflow/hard-prompts-made-easy)*
- [ ] **Latent catalogue**: content-addressed store keyed by `fingerprint`. *Accept: second run is a cache hit.*

## v0.5 — video  *(shipped, ffmpeg path pending a CI runner with ffmpeg)*

- [x] **Shot splitting** via ffmpeg scene score, plus a **frames-dir** path so it runs with no ffmpeg. *Accept: N frames → N shots, anchors assigned.*
- [x] **Per-shot recipes** with `first_frame` / `last_frame` anchors and an identity ref from shot 0. *Accept: `shots[]` survives JSON round-trip.*
- [x] **Per-shot VACE/FLF2V plan** + four anti-drift rules. *Accept: `plan_video()["steps"]` length == shot count.*
- [x] **Video scoring**: mean / worst-frame / stability. *Accept: identical frame dirs score >0.99 on all three.*
- [ ] **Camera trajectory** control from VGGT/MegaSaM → ReCamMaster. *Accept: pan/tilt/zoom recovers within tolerance. — upstream: [vggt](https://github.com/facebookresearch/vggt)*
- [ ] **Temporal consistency metric** (flicker + identity across shots), VBench-style.

## v0.6 — distribution  *(partially shipped)*

- [x] **Local Web UI** (stdlib server, single-file front end) with controls gallery + shots tab.
- [x] **Gradio Space** in `spaces/` (no GPU, no key).
- [x] **README visuals generated from real numbers** (`docs/make_hero.py`), so the art cannot drift from the code.
- [ ] Publish `reforge.recipe/1` as a JSON Schema file + versioned changelog (`docs/recipe.schema.json`).
- [ ] Embedded-recipe reader inside A1111 / ComfyUI / InvokeAI nodes.
- [ ] Benchmark harness: 200 fixtures, per-backend score table in CI, regressions block merges.

## Explicitly not planned

- Training or fine-tuning a captioner. The contract is the contribution; the model is a commodity.
- A hosted service. `reforge serve` is local by design.
- Anything that needs a licence key to read a file's own metadata.

## How to claim an item

Open an issue titled `roadmap: <item>` with the acceptance check as the definition of done.
First working PR with a test wins; no reservations.
