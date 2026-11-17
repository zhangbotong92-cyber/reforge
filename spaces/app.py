"""Hugging Face Space: REFORGE demo (offline, no key, no GPU).

Thin Gradio shell over the same library API the CLI and server use — nothing is
reimplemented here, so what you see on the Space is what `reforge demo` prints.

Local run:
    pip install gradio && python spaces/app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from reforge import controls, forensics, metrics, providers, recipe as recipelib, schema  # noqa: E402

CAPTIONERS = {
    "offline (heuristic, no key)": "offline",
    "auto (uses Space secrets if set)": "auto",
}


def _provider(spec: str):
    if spec == "auto":
        import os

        for candidate in ("openai:gpt-4o-mini", "gemini:gemini-2.5-flash", "ollama:qwen2.5vl"):
            cap = providers.build_captioner(candidate)
            if cap.available():
                return cap
        return providers.OfflineCaptioner()
    return providers.build_captioner(spec)


def analyze(image_path: str, provider_spec: str, want_controls: bool, dialect: str):
    if not image_path:
        return "没有文件", "", "", "", ""

    tmp = Path(image_path)
    report = forensics.inspect(tmp)
    captioner = _provider(provider_spec)
    cap = captioner.caption(tmp)
    cap["width"] = report.get("width")
    cap["height"] = report.get("height")

    rec = recipelib.from_inspection(report, caption=cap) if report.get("recovered") else recipelib.from_caption(cap, report)

    if want_controls:
        outdir = Path(tempfile.mkdtemp(prefix="reforge-space-"))
        result = controls.extract_all(tmp, outdir, kinds=["canny", "tile", "seg", "depth"])
        for c in result["controls"]:
            rec["controls"].append(
                {
                    "kind": c["kind"],
                    "model": c["method"],
                    "path": c["path"],
                    "weight": 0.30 if c.get("approximate") else 0.60,
                    "source": c["backend"],
                    "approximate": c["approximate"],
                }
            )
        recipelib.add_ref(rec, str(tmp), weight=0.55, role="identity")
        rec["fidelity"] = recipelib.fidelity_level(rec)

    rec["plan"] = controls.plan(rec)
    rec["gaps"] = recipelib.validate(rec)

    forensic_lines = [forensics.summarize(report)]
    for key, value in (report.get("recovered") or {}).items():
        shown = value if not isinstance(value, str) or len(value) < 200 else value[:200] + "..."
        forensic_lines.append(f"  {key:14s} {shown}")
    for note in report.get("notes", []):
        forensic_lines.append(f"  ! {note}")

    compiled = schema.compile_prompt(cap, dialect)
    prompt_block = f"{compiled['prompt']}\n\nnegative: {compiled['negative'] or '-'}"
    all_dialects = "\n\n".join(
        f"[{name}] {body['prompt']}" for name, body in schema.compile_all(cap).items()
    )
    return (
        "\n".join(forensic_lines),
        prompt_block,
        all_dialects,
        json.dumps(recipelib.to_dict(rec), ensure_ascii=False, indent=2),
        f"fidelity={rec['fidelity']}  gaps={len(rec.get('gaps') or [])}",
    )


def score(original: str, candidate: str):
    if not (original and candidate):
        return "需要两张图"
    s = metrics.score_pair(original, candidate)
    return (
        f"composite **{s['composite']}** · {s['grade']}\n\n"
        f"- SSIM {s['ssim']:.3f} · edge IoU {s['edge_iou']:.3f} · pHash sim {s['phash_similarity']:.3f}\n"
        f"- palette Δ {s['palette_distance']:.3f} · PSNR {s['psnr']:.1f} dB"
    )


INTRO = """
# REFORGE

**反推提示词 = 只能做到语义相似。要做到可复现，必须交付「配方」。**

这里演示三件事：
1. **先读元数据**：A1111 / ComfyUI / NovelAI 的图会自己交代参数，命中即 100% 精确。
2. **读不到才反推**：统一 JSON 契约 → 编译成 6 家平台方言；点一下还能抽出 canny/tile/seg/depth 控制图。
3. **配方随文件走**：`recipe.json` 里有 seed、模型 hash、控制信号，可以盖回 PNG。

全程不需要 GPU，默认后端不需要任何 API Key。
"""

with gr.Blocks(title="REFORGE", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("① 取证 + 反推"):
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="拖入图片")
                provider = gr.Dropdown(
                    choices=list(CAPTIONERS), value=list(CAPTIONERS)[0], label="反推后端"
                )
                dialect = gr.Dropdown(
                    choices=["sd", "flux", "mj", "kling", "jimeng", "banana"], value="kling", label="目标方言"
                )
                want_controls = gr.Checkbox(value=True, label="同时抽取控制图（canny/tile/seg/depth）")
                btn = gr.Button("分析", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                forensic = gr.Textbox(label="取证：这个文件已经知道什么", lines=10)
                prompt = gr.Textbox(label="编译后的提示词", lines=5)

        with gr.Accordion("六家方言", open=False):
            all_d = gr.Textbox(lines=14, show_label=False)
        with gr.Accordion("完整配方 recipe.json", open=False):
            rec_json = gr.Code(language="json", lines=18)

        btn.click(
            analyze,
            inputs=[img, provider, want_controls, dialect],
            outputs=[forensic, prompt, all_d, rec_json, status],
        )

    with gr.Tab("② 复现度校验"):
        gr.Markdown("把重新生成的图放右边，看它离 L3 有多远。")
        with gr.Row():
            a = gr.Image(type="filepath", label="原图")
            b = gr.Image(type="filepath", label="重新生成")
        sbtn = gr.Button("打分", variant="primary")
        out = gr.Markdown()
        sbtn.click(score, [a, b], out)

    with gr.Tab("③ 说明书"):
        gr.Markdown(
            """
| 档位 | 交付 | 结果 |
|---|---|---|
| L1 语义 | prompt + seed + 模型 | 同题材，不同画面 |
| L2 结构 | + canny/深度/参考图 | 构图与身份锁定 |
| L3 像素 | + 反演 latent | PSNR 30+ dB |
| L3+ 精确 | 文件自带配方 | 100% |

**命令**：`reforge inspect / caption / recipe / controls / video / score / refine / serve / doctor`

源码、路线图、拼装清单：见仓库根目录的 `README.md`、`ROADMAP.md`、`UPSTREAM.md`。
            """
        )

if __name__ == "__main__":
    demo.launch()
