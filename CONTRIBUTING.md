# Contributing

Short version: bring a fixture, a test, and a number.

## Setup

```bash
git clone https://github.com/OWNER/reforge && cd reforge
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
python -m unittest discover -s tests -v
python -m reforge demo
```

No GPU, no API key, no network needed for the test suite.

## Enable CI

The workflow lives at `docs/ci.yml.example` because publishing tools without the
`workflow` OAuth scope cannot write into `.github/workflows/`. Copy it into place once:

```bash
mkdir -p .github/workflows && cp docs/ci.yml.example .github/workflows/ci.yml
git add .github/workflows/ci.yml && git commit -m "ci: enable workflow"
```

Then push with a token that has the `workflow` scope (classic PAT) — otherwise GitHub
rejects the push even though every other file goes through.

## The three rules

1. **Fixtures or it did not happen.** Any parser change needs a synthetic fixture produced by
   `examples/make_sample.py` (add your format there) and an assertion on a *specific field*.
   A test that only checks "no exception" will be rejected.
2. **Provenance is mandatory.** A value recovered from a file must carry `{field, source, confidence}`
   in `report["evidence"]`. If you cannot name the source, it is a caption, not a recovery.
3. **No new hard dependencies.** `numpy` + `Pillow` is the budget. Everything else goes behind an
   optional extra and must degrade with a readable `MissingBackend` message.

## Recipes for common PRs

### Add a captioner

```python
# reforge/providers.py
class MyCaptioner(Captioner):
    name = "mine"
    requires_key = False

    def _raw_caption(self, image: Path, hint: str) -> tuple[dict, float]:
        raw = ...                      # must include the keys from providers.CAPTION_SPEC
        return raw, 0.7                # your honest confidence
```

Register it in `build_captioner()` and add a test that `schema.normalize()` accepts the output.

### Add a dialect

One function in `reforge/schema.py`:

```python
def _mysite(cap: dict) -> dict:
    return {"prompt": ..., "negative": ..., "params": {...}}
```

Add it to `DIALECTS`. The test `test_every_dialect_compiles` covers it automatically.

### Add a metadata format

1. `forensics.parse_<format>(text) -> dict` returning normalised keys (`prompt`, `seed`, `steps`, `cfg`, `sampler`, `model`, `width`, `height`, `loras`, `controls`).
2. One branch in `_inspect_png` / `_inspect_pillow`.
3. A fixture in `examples/make_sample.py`.
4. An assertion on the round-trip value.

## Style

- Stdlib `unittest`, not pytest, for the core suite (it must run with zero installs).
- No `print()` in library code; return data. CLI is the only place that talks.
- ASCII-only in CLI output (Windows consoles mangle em dashes).
- Type hints everywhere; `from __future__ import annotations` at the top.

## Commits

`module: what changed` — e.g. `forensics: recover NovelAI stealth alpha LSB`.
Squash noise; keep the diff reviewable.

## Reporting a wrong result

Open an issue with: the file (or the fixture that produces it), `reforge inspect --json` output,
and what you expected. Wrong-but-confident is the worst failure mode this project can have, so it
gets priority.
