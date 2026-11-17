"""Tests for scoring, schema compilation, recipes and the embed round-trip."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reforge import metrics, recipe as recipelib, schema  # noqa: E402


def _fixtures(tmp: Path) -> dict:
    sys.path.insert(0, str(ROOT / "examples"))
    import make_sample  # type: ignore

    make_sample.main(str(tmp))
    return {p.name: p for p in tmp.iterdir()}


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reforge-metrics-"))
        cls.files = _fixtures(cls.tmp)

    def test_identity(self):
        a = self.files["a1111.png"]
        gray = metrics.load_gray(a)
        self.assertTrue(math.isinf(metrics.psnr(gray, gray)))
        self.assertAlmostEqual(metrics.ssim(gray, gray), 1.0, places=4)
        self.assertEqual(metrics.phash_distance(metrics.phash(a), metrics.phash(a)), 0)

    def test_similar_beats_different(self):
        target = self.files["a1111.png"]
        near = metrics.score_pair(target, self.files["alt.png"])
        far = metrics.score_pair(target, self.files["comfyui.png"])
        self.assertGreaterEqual(near["composite"], far["composite"])
        self.assertIn(near["grade"], ("L1-semantic", "L2-structural", "L3-pixel"))

    def test_score_shape(self):
        s = metrics.score_pair(self.files["a1111.png"], self.files["alt.png"])
        for key in ("psnr", "ssim", "phash", "edge_iou", "palette_distance", "composite", "grade", "parts"):
            self.assertIn(key, s)
        self.assertGreaterEqual(s["composite"], 0.0)
        self.assertLessEqual(s["composite"], 1.0)


class TestSchema(unittest.TestCase):
    def test_normalize_aliases(self):
        cap = schema.normalize({"prompt": "a dog", "negative_prompt": ["bad"], "objects": ["dog", "ball"]}, "test", 0.9)
        self.assertEqual(cap["summary"], "a dog")
        self.assertEqual(cap["negative"], ["bad"])
        self.assertEqual(cap["subject"], ["dog", "ball"])
        self.assertEqual(cap["schema"], schema.SCHEMA)

    def test_normalize_synthesises(self):
        cap = schema.normalize({"subject": ["a robot"], "lighting": "rim light", "style": "cyberpunk"})
        self.assertTrue(cap["summary"])
        self.assertTrue(cap["tags"])

    def test_every_dialect_compiles(self):
        cap = schema.normalize(
            {
                "summary": "a lighthouse at dusk",
                "subject": ["lighthouse"],
                "lighting": "golden hour",
                "color": ["teal", "amber"],
                "camera": {"shot": "wide", "lens": "24mm", "angle": "low"},
                "motion": {"camera_move": "slow pan right"},
            },
            "test",
            0.8,
        )
        out = schema.compile_all(cap)
        self.assertGreaterEqual(len(out), 6)
        for name, compiled in out.items():
            self.assertTrue(compiled["prompt"], name)

    def test_unknown_dialect(self):
        with self.assertRaises(KeyError):
            schema.compile_prompt(schema.normalize({}), "nope")

    def test_merge(self):
        a = schema.normalize({"summary": "short", "subject": ["cat"]}, "x", 0.4)
        b = schema.normalize({"summary": "a much longer description", "subject": ["cat", "dog"]}, "y", 0.9)
        m = schema.merge([a, b])
        self.assertEqual(m["summary"], "a much longer description")
        self.assertEqual(len(m["subject"]), 2)
        self.assertEqual(m["confidence"], 0.9)


class TestRecipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reforge-recipe-"))
        cls.files = _fixtures(cls.tmp)

    def test_from_inspection_pins_everything(self):
        from reforge import forensics

        rec = recipelib.from_inspection(forensics.inspect(self.files["a1111.png"]))
        self.assertEqual(rec["sampling"]["seed"], 1234567890)
        self.assertEqual(rec["models"][0]["name"], "sd_xl_base_1.0")
        self.assertTrue(rec["models"][0]["pinned"])
        self.assertEqual(rec["fidelity"], "L1-reproducible")
        gaps = recipelib.validate(rec)
        # seed + model are pinned by the metadata; structure is still unlocked
        self.assertEqual(gaps, ["no control signals: layout and identity are not locked"])

    def test_fidelity_ladder(self):
        rec = recipelib.new_recipe(prompt="x")
        self.assertEqual(recipelib.fidelity_level(rec), "L1-semantic")
        recipelib.add_ref(rec, str(self.files["a1111.png"]))
        self.assertEqual(rec["fidelity"], "L2-structural")
        recipelib.add_inversion(rec, "rf-solver", "latents.npz")
        self.assertEqual(rec["fidelity"], "L3-pixel")

    def test_embed_roundtrip(self):
        from reforge import forensics

        rec = recipelib.from_inspection(forensics.inspect(self.files["a1111.png"]))
        out = self.tmp / "stamped.png"
        recipelib.embed_into_png(self.files["a1111.png"], out, rec)
        back = recipelib.read_embedded(out)
        self.assertIsNotNone(back)
        self.assertEqual(back["fingerprint"], recipelib.to_dict(rec)["fingerprint"])
        # the original metadata must survive the stamp
        self.assertIn("astronaut", forensics.inspect(out)["recovered"]["prompt"])

    def test_embed_is_idempotent(self):
        from reforge import forensics

        rec = recipelib.from_inspection(forensics.inspect(self.files["a1111.png"]))
        once = self.tmp / "once.png"
        twice = self.tmp / "twice.png"
        recipelib.embed_into_png(self.files["a1111.png"], once, rec)
        recipelib.embed_into_png(once, twice, rec)
        keys = [c for c in forensics.iter_png_chunks(twice.read_bytes()) if c[0] == "iTXt"]
        self.assertEqual(len(keys), 1)

    def test_sidecar(self):
        rec = recipelib.from_inspection({"recovered": {"prompt": "x"}, "path": "", "format": "png"})
        path = recipelib.write_sidecar(self.files["plain.jpg"], rec)
        self.assertTrue(path.exists())
        self.assertEqual(recipelib.read_sidecar(self.files["plain.jpg"])["prompt"], "x")
        path.unlink()

    def test_plan_has_steps_and_gaps(self):
        from reforge import controls, forensics

        rec = recipelib.from_inspection(forensics.inspect(self.files["a1111.png"]))
        plan = controls.plan(rec)
        self.assertTrue(plan["steps"])
        self.assertEqual(plan["steps"][-1]["op"], "sample")
        self.assertTrue(plan["gaps"])


if __name__ == "__main__":
    unittest.main()
