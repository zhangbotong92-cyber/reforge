# Bill of materials — what was taken from where

REFORGE is an assembly. This file is the receipt: every upstream project, what it is used for,
and what was written here instead of copied. **No third-party source is vendored** — adapters
call APIs or shell out; formats are re-implemented from public specs.

## Formats re-implemented from spec (no dependency)

| Component | Spec / source | Used for |
|---|---|---|
| PNG chunk walker (`tEXt`/`zTXt`/`iTXt`, CRC32, insertion) | PNG spec (W3C), zlib | forensic read + recipe stamping |
| A1111 parameter grammar | Automatic1111 WebUI output format (public) | `parameters` chunk parsing |
| ComfyUI graph traversal | ComfyUI API format (public, JSON) | positive/negative/model/latent unwrap |
| NovelAI `Comment` blob | NovelAI metadata (public) | prompt/uc/seed recovery |
| InvokeAI `sd-metadata` | InvokeAI (public) | prompt recovery |
| MP4 box walker (`udta`/`meta`/`ilst`/`data`) | ISO/IEC 14496-12 | comment + prompt recovery in containers |
| EXIF `UserComment`/`ImageDescription` | EXIF 2.3, Pillow `getexif()` | JPEG/WebP parameter recovery |
| SSIM / PSNR / DCT-pHash / edge IoU | Wang et al. 2004; standard DCT hashing | verification metrics |
| Dark-channel prior depth proxy | He, Sun & Tang, *Single Image Haze Removal* (CVPR 2009) | `controls.depth` builtin approximation, labelled `approximate` |

## Runtime libraries (declared, not vendored)

| Library | Used for | Needed? |
|---|---|---|
| [numpy](https://numpy.org) | everything numeric | yes (core) |
| [Pillow](https://python-pillow.org) | decode/encode, EXIF, palette | yes (core) |
| [opencv-python-headless](https://github.com/opencv/opencv-python) | Canny, morphology, GrabCut, Lab | optional: `seg` needs it, `depth` degrades to numpy |
| [onnxruntime](https://github.com/microsoft/onnxruntime) | the `onnx` control backend | optional |
| [torch](https://pytorch.org) | only in a test, to export a toy ONNX model | optional (test skips) |
| ffmpeg / ffprobe | video probe, scene split, keyframes | optional: `--frames-dir` replaces it |

## ONNX models you can drop into `~/.cache/reforge/models/`

| File | Model | Upstream | Notes |
|---|---|---|---|
| `depth.onnx` (+ `depth.json`) | Depth-Anything-V2 (S/B/L) | [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | export with `torch.onnx.export`, input `1x3xHxW`, ImageNet mean/std |
| `pose.onnx` | DWPose / RTMPose | [DWPose](https://github.com/IDEA-Research/DWPose) | keypoints `(N,K)` are drawn by `_draw_pose` |
| `seg.onnx` | SAM / MobileSAM | [sam2](https://github.com/facebookresearch/sam2) | mask thresholded at 0.5 |

`<kind>.json` is optional: `{"input": "...", "output": "...", "size": [H, W], "mean": [...], "std": [...]}`.
Check licences before redistributing any exported weights — Depth-Anything-V2 large is non-commercial.

## Upstream projects this repo is designed to sit on top of

Not bundled. Each row says *what to install* and *what REFORGE hands it*.

| Project | Repo | Role in the pipeline | License (check repo) |
|---|---|---|---|
| ControlNet | [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) | L2 structure lock (canny/depth/lineart) | Apache-2.0 |
| IP-Adapter | [tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) | reference-image identity/style | Apache-2.0 |
| InstantID | [instantX-research/InstantID](https://github.com/instantX-research/InstantID) | face identity, single image | Apache-2.0 |
| PuLID | [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) | identity without finetuning | Apache-2.0 |
| FLUX.1 Kontext | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | instruction editing with reference consistency | FLUX non-commercial |
| Qwen-Image-Edit | [Qwen/Qwen-Image-Edit-2509](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) | multi-image edit, native ControlNet | Apache-2.0 |
| OmniGen2 | [VectorSpaceLab/OmniGen2](https://github.com/vectorspacelab/omnigen2) | unified edit + in-context generation | Apache-2.0 |
| Depth Anything V2 | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | `controls.BACKENDS["depth"]` | Apache-2.0 (S/B) |
| DWPose | [IDEA-Research/DWPose](https://github.com/IDEA-Research/DWPose) | `controls.BACKENDS["pose"]` | Apache-2.0 |
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | `controls.BACKENDS["seg"]` | Apache-2.0 |
| RAFT | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | `controls.BACKENDS["flow"]` (video) | BSD-3 |
| VGGT / MegaSaM | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) · [mega-sam/mega-sam](https://github.com/mega-sam/mega-sam) | `controls.BACKENDS["camera"]` | see repo |
| Wan2.1 / Wan2.2-VACE | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) · [ali-vilab/VACE](https://github.com/ali-vilab/VACE) | video: reference-to-video, v2v edit | Apache-2.0 |
| Phantom | [Phantom-video/Phantom](https://github.com/Phantom-video/Phantom) | subject-to-video consistency | see repo |
| ReCamMaster | [kwaivgi/recammaster](https://github.com/kwaivgi/recammaster) | camera-trajectory re-rendering | see repo |
| RF-Solver-Edit | [wangjiangshan0725/RF-Solver-Edit](https://github.com/wangjiangshan0725/RF-Solver-Edit) | L3 inversion for rectified-flow models | see repo |
| VBench | [Vchitect/VBench](https://github.com/Vchitect/VBench) | video consistency evaluation | Apache-2.0 |
| JoyCaption | [fancyfeast/joy-caption-alpha-two](https://huggingface.co/spaces/fancyfeast/joy-caption-alpha-two) | optional caption backend | see repo |
| Florence-2 | [anyantudre/florence-2-cookbook](https://github.com/anyantudre/florence-2-cookbook) | optional caption backend | MIT (model) |
| clip-interrogator | [pharmapsychotic/clip-interrogator](https://github.com/pharmapsychotic/clip-interrogator) | reference design for prompt search | see repo |
| c2pa-rs / c2patool | [contentauth/c2pa-rs](https://github.com/contentauth/c2pa-rs) | reading signed provenance | Apache-2.0 / MIT |

## Ideas taken from prior art (credited, not copied)

| Idea | Where it comes from | How it shows up here |
|---|---|---|
| "prompt is not enough, inversion is" | DDIM inversion, Null-text inversion, Negative-Prompt Inversion | the L3 tier + `recipe.init` |
| hard-prompt optimisation (PEZ) | [hard-prompts-made-easy](https://github.com/anwoflow/hard-prompts-made-easy) | roadmap: `refine` with an optimiser backend |
| multi-pass caption + self-review | [prompt-mirror](https://github.com/jiangshang-star/prompt-mirror) | `Captioner.critique` + `RefineLoop` |
| video reverse-prompt with shot splitting | [prompt-lens](https://github.com/raojiacui/prompt-lens) · [video-prompt-reverse](https://github.com/lozhilzh/video-prompt-reverse) | roadmap: `shots[]` in the recipe |
| "recipe travels with the file" | A1111 / ComfyUI embedding practice, C2PA | `recipe.embed_into_png` |

## What was written here

- The three-tier fidelity model and the derived `fidelity` field.
- One caption contract + six dialect compilers that all consume it.
- Provenance-carrying forensics (per-field source + confidence).
- Recipe fingerprinting and `validate()` gap reporting that drives the refine loop.
- **The inversion worker** (`invert.py`): DDIM and rectified-flow solvers in numpy, with a
  two-claim self-test (algebraic exactness against an oracle epsilon, approximation quality
  against a predicted one) so L3 is verifiable without a checkpoint, plus the diffusers bridge.
- Model-free verification metrics and the L0–L3 grading.
- **Three-backend control extraction** (`builtin` → `onnx` → `command`) with an `approximate`
  flag that propagates into weights, plans and gaps, plus `reforge doctor`.
- **The video layer**: shot splitting with a no-ffmpeg frames path, per-shot recipes with
  first-frame anchors, per-shot VACE/FLF2V plans, and video scoring with a stability number.
- `controls.plan()`: recipe → concrete runtime node list.
- Idempotent `iTXt` recipe stamping + round-trip reader.
- The stdlib Web UI (controls gallery + shots tab), the Gradio Space, and the
  numbers-derived README visuals in `docs/make_hero.py`.

If you think something here is mis-credited, open an issue — the receipt gets fixed, not defended.
