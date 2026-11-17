---
title: REFORGE
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: 反推提示词 + 生成可复现配方
---

# REFORGE Space

Gradio shell over the `reforge` library. No GPU, no API key: the default backend is the
offline heuristic captioner, and metadata forensics (`A1111 / ComfyUI / NovelAI`) is pure Python.

- **取证** — upload an AI-generated PNG: if it still carries its parameters, they are recovered exactly.
- **反推** — otherwise a structured caption is produced and compiled into 6 platform dialects.
- **控制图** — canny / lineart / tile / GrabCut seg / DCP depth proxy are extracted in-process.
- **校验** — drop a regeneration next to the original and get PSNR/SSIM/pHash/edge-IoU + an L0–L3 grade.

To enable a real VLM here, add `OPENAI_API_KEY` or `GEMINI_API_KEY` as a Space secret and pick
"auto" in the backend dropdown; nothing else changes.
