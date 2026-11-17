"""Controls, video and the ONNX backend — the parts that touch real media.

The ONNX test exports a tiny model with torch, so the model-loading code path is
exercised for real without downloading 100 MB of weights.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reforge import controls, video  # noqa: E402


def _fixtures(tmp: Path) -> dict:
    sys.path.insert(0, str(ROOT / "examples"))
    import make_sample  # type: ignore

    make_sample.main(str(tmp))
    return {p.name: p for p in tmp.iterdir()}


class TestControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reforge-ctl-"))
        cls.files = _fixtures(cls.tmp)
        cls.img = cls.files["a1111.png"]

    def test_doctor_reports_matrix(self):
        report = controls.doctor()
        self.assertIn("kinds", report)
        for kind in ("canny", "tile", "seg", "depth", "pose"):
            self.assertIn(kind, report["kinds"])
            self.assertIn("available", report["kinds"][kind])
        # canny is cv2/PIL based, must always work
        self.assertTrue(report["kinds"]["canny"]["available"])

    def test_extract_all_returns_maps(self):
        out = self.tmp / "ctl"
        result = controls.extract_all(self.img, out, kinds=["canny", "tile", "depth"])
        kinds = {c["kind"] for c in result["controls"]}
        self.assertEqual(kinds, {"canny", "tile", "depth"})
        for c in result["controls"]:
            self.assertTrue(Path(c["path"]).exists(), c["path"])
            arr = np.asarray(Image.open(c["path"]))
            self.assertGreater(arr.std(), 0.0, f"{c['kind']} is a constant image")

    def test_depth_is_marked_approximate(self):
        result = controls.extract("depth", self.img, self.tmp / "d.png")
        self.assertEqual(result["backend"], "builtin")
        self.assertTrue(result["approximate"], "the DCP proxy must never claim to be a real depth model")

    def test_seg_mask_is_not_degenerate(self):
        out = controls.seg(self.img, self.tmp / "s.png")
        arr = np.asarray(Image.open(out))
        foreground = float((arr > 127).mean())
        self.assertGreater(foreground, 0.01)
        self.assertLess(foreground, 0.99)

    def test_missing_backend_is_actionable(self):
        os.environ.pop("REFORGE_CTL_POSE", None)
        saved = controls.MODEL_DIR
        controls.MODEL_DIR = self.tmp / "empty-models"
        try:
            with self.assertRaises(controls.MissingBackend) as ctx:
                controls.extract("pose", self.img, self.tmp / "p.png")
            self.assertIn("pose.onnx", str(ctx.exception))
        finally:
            controls.MODEL_DIR = saved

    def test_command_backend(self):
        stub = self.tmp / "stub_ctl.py"
        stub.write_text(
            "import sys, numpy as np\n"
            "from PIL import Image\n"
            "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
            "Image.fromarray((np.arange(64*64) % 255).astype('uint8').reshape(64,64), 'L').save(args['--out'])\n",
            encoding="utf-8",
        )
        os.environ["REFORGE_CTL_POSE"] = f"{sys.executable} {stub} --image {{image}} --out {{out}}"
        try:
            result = controls.extract("pose", self.img, self.tmp / "pose_cmd.png")
            self.assertEqual(result["backend"], "command")
            self.assertTrue(Path(result["path"]).exists())
        finally:
            os.environ.pop("REFORGE_CTL_POSE", None)


class TestOnnxBackend(unittest.TestCase):
    """Export a 30 KB model with torch, then load it through the real code path."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reforge-onnx-"))
        cls.files = _fixtures(cls.tmp)

    def test_onnx_depth(self):
        try:
            import onnx  # noqa: F401
            import torch
            import torch.nn as nn
        except ImportError:
            raise unittest.SkipTest("torch+onnx not installed")

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 3, padding=1)

            def forward(self, x):
                return torch.sigmoid(self.conv(x))

        model_dir = self.tmp / "models"
        model_dir.mkdir()
        model_path = model_dir / "depth.onnx"
        try:
            torch.onnx.export(
                Tiny(),
                torch.zeros(1, 3, 64, 64),
                str(model_path),
                input_names=["pixel_values"],
                output_names=["predicted_depth"],
                opset_version=14,
                dynamo=False,
            )
        except Exception as exc:  # noqa: BLE001 - exporter API drift
            raise unittest.SkipTest(f"torch.onnx.export unavailable: {exc}")

        (model_dir / "depth.json").write_text(
            json.dumps({"input": "pixel_values", "output": "predicted_depth", "size": [64, 64]}), encoding="utf-8"
        )

        saved = controls.MODEL_DIR
        controls.MODEL_DIR = model_dir
        try:
            self.assertEqual(controls.available_backend("depth"), "onnx")
            result = controls.extract("depth", self.files["a1111.png"], self.tmp / "onnx_depth.png")
            self.assertEqual(result["backend"], "onnx")
            self.assertFalse(result["approximate"], "a real model must not be flagged approximate")
            arr = np.asarray(Image.open(result["path"]))
            self.assertGreater(arr.std(), 0.0)
        finally:
            controls.MODEL_DIR = saved


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="reforge-video-"))
        cls.files = _fixtures(cls.tmp)
        cls.frames = cls.tmp / "frames"
        cls.frames.mkdir()
        for i, name in enumerate(("a1111.png", "alt.png", "comfyui.png", "novelai.png")):
            (cls.frames / f"f{i:02d}.png").write_bytes(cls.files[name].read_bytes())

    def test_analyze_frames(self):
        analysis = video.analyze(frames_dir=self.frames, max_shots=3)
        self.assertEqual(analysis["backend"], "frames-dir")
        self.assertEqual(len(analysis["shots"]), 3)
        self.assertTrue(Path(analysis["shots"][0]["keyframe"]).exists())
        self.assertEqual(analysis["shots"][0]["index"], 0)

    def test_script_and_recipe(self):
        from reforge import providers

        analysis = video.analyze(frames_dir=self.frames, max_shots=3)
        scripted = video.script(analysis, providers.OfflineCaptioner(), dialect="kling")
        self.assertTrue(all(s["compiled"]["prompt"] for s in scripted["shots"]))

        rec = video.build_recipe(scripted)
        self.assertEqual(len(rec["shots"]), 3)
        self.assertTrue(rec["prompt"])
        self.assertEqual(len(rec["refs"]), 1, "the first shot must be an identity anchor")
        self.assertIn(rec["fidelity"], ("L1-semantic", "L2-structural", "L3-pixel"))
        self.assertEqual(rec["video_plan"]["shots"], 3)
        self.assertTrue(rec["video_plan"]["rules"])

    def test_recipe_serialises(self):
        from reforge import providers, recipe as recipelib

        analysis = video.analyze(frames_dir=self.frames, max_shots=2)
        rec = video.build_recipe(video.script(analysis, providers.OfflineCaptioner()))
        blob = json.dumps(recipelib.to_dict(rec), ensure_ascii=False)
        self.assertIn("shots", blob)
        self.assertIn("fingerprint", blob)

    def test_missing_inputs_are_explained(self):
        with self.assertRaises(video.VideoUnavailable):
            video.analyze()
        with self.assertRaises(video.VideoUnavailable):
            video.analyze(frames_dir=self.tmp / "nope")

    def test_score_video(self):
        analysis = video.analyze(frames_dir=self.frames, max_shots=3)
        # build an identical copy of exactly the selected shots, then compare
        a_dir = self.tmp / "cmp_a"
        b_dir = self.tmp / "cmp_b"
        a_dir.mkdir(exist_ok=True)
        b_dir.mkdir(exist_ok=True)
        for shot in analysis["shots"]:
            src = Path(shot["keyframe"])
            (a_dir / f"{shot['index']:03d}.png").write_bytes(src.read_bytes())
            (b_dir / f"{shot['index']:03d}.png").write_bytes(src.read_bytes())
        result = video.score_video(a_dir, b_dir)
        self.assertEqual(result["frames"], 3)
        self.assertGreater(result["mean_composite"], 0.99)
        self.assertGreater(result["stability"], 0.99)


if __name__ == "__main__":
    unittest.main()
