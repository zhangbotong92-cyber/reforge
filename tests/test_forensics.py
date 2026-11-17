"""Tests for the forensic layer — the part that must never be wrong.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reforge import forensics  # noqa: E402


def _fixtures(tmp: Path) -> dict:
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        import make_sample  # type: ignore
    except ImportError:  # pragma: no cover
        raise unittest.SkipTest("examples/make_sample.py missing")
    make_sample.main(str(tmp))
    return {p.name: p for p in tmp.iterdir()}


class TestA1111(unittest.TestCase):
    def test_parse_parameters(self):
        text = (
            "a cat on a mat\nNegative prompt: blurry, ugly\n"
            "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 42, Size: 512x768, "
            "Model hash: deadbeef, Model: sd15, Lora hashes: \"x: 1, y: 2\", Version: v1.9"
        )
        got = forensics.parse_a1111(text)
        self.assertEqual(got["prompt"], "a cat on a mat")
        self.assertEqual(got["negative"], "blurry, ugly")
        self.assertEqual(got["seed"], 42)
        self.assertEqual(got["steps"], 20)
        self.assertAlmostEqual(got["cfg"], 7.0)
        self.assertEqual((got["width"], got["height"]), (512, 768))
        self.assertEqual(got["model"], "sd15")
        self.assertEqual(len(got["loras"]), 2)

    def test_parse_without_negative(self):
        got = forensics.parse_a1111("solo prompt\nSteps: 5, Sampler: euler, CFG scale: 1, Seed: 1")
        self.assertEqual(got["prompt"], "solo prompt")
        self.assertEqual(got["seed"], 1)

    def test_multiline_prompt(self):
        text = "line one,\nline two\nNegative prompt: bad\nSteps: 1, Sampler: e, CFG scale: 1, Seed: 9"
        got = forensics.parse_a1111(text)
        self.assertIn("line two", got["prompt"])


class TestComfy(unittest.TestCase):
    def test_api_graph_unwrapped(self):
        graph = {
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux.safetensors"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red door"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 640, "height": 480}},
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 7, "steps": 12, "cfg": 2.5, "sampler_name": "euler", "scheduler": "beta",
                    "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
                },
            },
        }
        got = forensics.parse_comfy_api(json.dumps(graph))
        self.assertEqual(got["prompt"], "a red door")
        self.assertEqual(got["negative"], "blurry")
        self.assertEqual(got["seed"], 7)
        self.assertEqual(got["model"], "flux.safetensors")
        self.assertEqual((got["width"], got["height"]), (640, 480))
        self.assertEqual(got["sampler"], "euler")

    def test_ui_graph_fallback(self):
        wf = {"nodes": [{"id": 1, "type": "CLIPTextEncode", "widgets_values": ["hello world"]}]}
        got = forensics.parse_comfy_api(json.dumps(wf))
        self.assertEqual(got["prompt"], "hello world")


class TestContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reforge-test-"))
        cls.files = _fixtures(cls.tmp)

    def test_png_parameters(self):
        rec = forensics.inspect(self.files["a1111.png"])["recovered"]
        self.assertIn("astronaut", rec["prompt"])
        self.assertEqual(rec["seed"], 1234567890)
        self.assertEqual(rec["sampler"], "DPM++ 2M Karras")

    def test_comfyui_png(self):
        report = forensics.inspect(self.files["comfyui.png"])
        self.assertEqual(report["tool"], "comfyui")
        self.assertIn("lighthouse", report["recovered"]["prompt"])
        self.assertIn("IPAdapterAdvanced", report["recovered"]["controls"])

    def test_novelai_png(self):
        rec = forensics.inspect(self.files["novelai.png"])["recovered"]
        self.assertEqual(rec["seed"], 42424242)
        self.assertIn("silver hair", rec["prompt"])

    def test_clean_jpeg(self):
        report = forensics.inspect(self.files["plain.jpg"])
        self.assertFalse(report["raw"])
        self.assertEqual(report["width"], 512)
        self.assertTrue(any("caption" in n for n in report["notes"]))

    def test_chunk_roundtrip_preserves_pixels(self):
        data = self.files["a1111.png"].read_bytes()
        chunks = forensics.iter_png_chunks(data)
        self.assertEqual(chunks[0][0], "IHDR")
        self.assertEqual(chunks[-1][0], "IEND")


if __name__ == "__main__":
    unittest.main()
