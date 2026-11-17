"""Layer 6 — the closed loop.

This is the part everyone skips and then wonders why "reverse prompt" feels like
a toy: you cannot know a prompt is right until you generate with it and compare.
REFORGE ships the loop, not just the captioner.

    target -> caption -> compile -> generate -> score -> critique -> patch -> ...

The generator is a pluggable callable. Anything that turns a prompt file into an
image file will do: a local ComfyUI graph, a diffusers script, a hosted API, even
a human pasting into a web UI (see :class:`ManualGenerator`).
"""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import metrics, schema

PROMPT_FILE = "prompt.txt"
NEGATIVE_FILE = "negative.txt"


class GeneratorError(RuntimeError):
    pass


class Generator:
    """Interface: turn a prompt into an image on disk."""

    name = "generator"

    def generate(self, prompt: str, negative: str, seed: int, out: Path) -> Path:
        raise NotImplementedError

    def available(self) -> bool:
        return True


class NullGenerator(Generator):
    """Dry run: produces nothing, keeps the loop scoring-free but complete."""

    name = "null"

    def generate(self, prompt: str, negative: str, seed: int, out: Path) -> Path:
        raise GeneratorError(
            "No generator configured. Pass --generator \"<command with {prompt_file} and {out}>\" "
            "or --dry-run to inspect the prompts without sampling."
        )


class CommandGenerator(Generator):
    """Any CLI that reads a prompt file and writes an image.

    The template is split argv-style (not shell-expanded), so paths with spaces
    work on Windows, macOS and Linux alike. Prefix with ``shell:`` if you really
    want a shell pipeline and will handle quoting yourself.

    Placeholders: ``{prompt_file}`` ``{negative_file}`` ``{seed}`` ``{out}`` ``{prompt}``
    Example:
        --generator "python gen.py --prompt-file {prompt_file} --seed {seed} --out {out}"
    """

    name = "command"

    def __init__(self, template: str, timeout: int = 1800):
        self.template = template
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.template.strip())

    def generate(self, prompt: str, negative: str, seed: int, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        prompt_file = out.parent / PROMPT_FILE
        negative_file = out.parent / NEGATIVE_FILE
        prompt_file.write_text(prompt, encoding="utf-8")
        negative_file.write_text(negative, encoding="utf-8")

        fields = {
            "prompt_file": str(prompt_file),
            "negative_file": str(negative_file),
            "seed": str(seed),
            "out": str(out),
            "prompt": prompt,
        }

        if self.template.startswith("shell:"):
            cmd = self.template[len("shell:") :].format(
                prompt_file=f'"{prompt_file}"',
                negative_file=f'"{negative_file}"',
                seed=seed,
                out=f'"{out}"',
                prompt=prompt.replace('"', '\\"'),
            )
            argv: Optional[List[str]] = None
        else:
            cmd = None
            argv = [token.format(**fields) for token in shlex.split(self.template, posix=True)]

        proc = subprocess.run(cmd if argv is None else argv, shell=argv is None, capture_output=True, timeout=self.timeout)
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace")[-800:]
            raise GeneratorError(f"generator failed ({proc.returncode}): {tail}")
        if not out.exists():
            raise GeneratorError(f"generator exited 0 but produced no file at {out}")
        return out


class ManualGenerator(Generator):
    """Human-in-the-loop: writes the prompts, waits for you to drop a file in."""

    name = "manual"

    def generate(self, prompt: str, negative: str, seed: int, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / PROMPT_FILE).write_text(prompt, encoding="utf-8")
        (out.parent / NEGATIVE_FILE).write_text(negative, encoding="utf-8")
        raise GeneratorError(
            f"Manual mode: prompt written to {out.parent / PROMPT_FILE}. "
            f"Generate the image yourself, save it as {out}, then re-run with --reuse."
        )


def apply_patches(caption: Dict[str, Any], patches: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a critique's ``patches`` into a caption without breaking its contract."""
    updated = json.loads(json.dumps(caption))
    for key, value in (patches or {}).items():
        if key in schema.STRING_FIELDS:
            updated[key] = schema._as_str(value) or updated.get(key, "")
        elif key in schema.LIST_FIELDS:
            updated[key] = schema._as_list(value) or updated.get(key, [])
        elif key in ("camera", "motion") and isinstance(value, dict):
            updated.setdefault(key, {})
            updated[key].update({k: schema._as_str(v) for k, v in value.items()})
    if not updated.get("summary"):
        updated["summary"] = schema._synth_summary(updated)
    updated["tags"] = schema._derive_tags(updated)
    return updated


class RefineLoop:
    """Caption -> compile -> generate -> score -> critique, until it converges."""

    def __init__(
        self,
        target: str | Path,
        captioner: Any,
        generator: Optional[Generator] = None,
        dialect: str = "sd",
        max_rounds: int = 3,
        threshold: float = 0.80,
        workdir: Optional[str | Path] = None,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.target = Path(target)
        self.captioner = captioner
        self.generator = generator or NullGenerator()
        self.dialect = dialect
        self.max_rounds = max_rounds
        self.threshold = threshold
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="reforge-"))
        self.seed = seed
        self.verbose = verbose

    def run(self, initial_caption: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.workdir.mkdir(parents=True, exist_ok=True)
        history: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        caption = initial_caption or self.captioner.caption(self.target)
        for round_idx in range(self.max_rounds):
            compiled = schema.compile_prompt(caption, self.dialect)
            out = self.workdir / f"round_{round_idx}.png"
            record: Dict[str, Any] = {
                "round": round_idx,
                "dialect": self.dialect,
                "prompt": compiled["prompt"],
                "negative": compiled["negative"],
                "caption": caption,
                "generated": None,
                "score": None,
                "critique": None,
            }
            try:
                self.generator.generate(compiled["prompt"], compiled["negative"], self.seed, out)
                record["generated"] = str(out)
                score = metrics.score_pair(self.target, out)
                record["score"] = score
                if self.verbose:
                    print(f"  round {round_idx}: composite={score['composite']:.3f} ({score['grade']})")
                if best is None or (score["composite"] > best["score"]["composite"]):
                    best = record
                if score["composite"] >= self.threshold:
                    history.append(record)
                    break
                critique = self.captioner.critique(self.target, caption)
                record["critique"] = critique
                caption = apply_patches(caption, critique.get("patches") or {})
            except GeneratorError as exc:
                record["error"] = str(exc)
                history.append(record)
                if self.verbose:
                    print(f"  round {round_idx}: {exc}")
                break
            history.append(record)

        return {
            "target": str(self.target),
            "best": best,
            "history": history,
            "converged": bool(best and best["score"] and best["score"]["composite"] >= self.threshold),
            "workdir": str(self.workdir),
        }


def patch_prompt_with_feedback(base_prompt: str, issues: List[str]) -> str:
    """Offline fallback revision: append the audit findings as explicit constraints."""
    if not issues:
        return base_prompt
    extra = "; ".join(i.strip() for i in issues[:6] if i.strip())
    return f"{base_prompt}, {extra}" if extra else base_prompt


def make_generator(kind: str, template: Optional[str] = None) -> Generator:
    kind = (kind or "null").lower()
    if kind in ("null", "dry-run", "none"):
        return NullGenerator()
    if kind == "command" and template:
        return CommandGenerator(template)
    if kind == "manual":
        return ManualGenerator()
    if kind.startswith("command:"):
        return CommandGenerator(kind.split(":", 1)[1])
    raise GeneratorError(f"unknown generator {kind!r} (use null | manual | command | command:<template>)")
