"""Inversion numerics.

These tests do not need a checkpoint — they assert the solvers against fields with
closed-form solutions, which is the only way to know the L3 tier is real rather
than aspirational.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reforge import invert, recipe as recipelib  # noqa: E402


class TestSolvers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(1)
        cls.x0 = cls.rng.normal(0, 1, (3, 32, 32))

    def test_constant_field_is_exact(self):
        """dx/dt = c has a linear solution: inversion then sampling must be identity."""
        c = self.rng.normal(0, 0.5, self.x0.shape)
        field = lambda x, t: c
        back = invert.rf_sample(field, invert.rf_invert(field, self.x0, steps=37), steps=37)
        self.assertLess(float(np.max(np.abs(back - self.x0))), 1e-10)

    def test_taylor_beats_euler(self):
        def field(x, t):
            return -x

        e1 = invert.rf_sample(field, invert.rf_invert(field, self.x0, steps=20, order=1), steps=20, order=1)
        e2 = invert.rf_sample(field, invert.rf_invert(field, self.x0, steps=20, order=2), steps=20, order=2)
        r1 = np.linalg.norm(e1 - self.x0) / np.linalg.norm(self.x0)
        r2 = np.linalg.norm(e2 - self.x0) / np.linalg.norm(self.x0)
        self.assertLess(r2, r1 / 5, f"taylor {r2:.2e} should be far better than euler {r1:.2e}")

    def test_taylor_is_second_order(self):
        """Halving the step count must cut the error by roughly 4x."""
        def field(x, t):
            return -x

        err = {}
        for steps in (16, 32):
            back = invert.rf_sample(field, invert.rf_invert(field, self.x0, steps=steps, order=2), steps=steps, order=2)
            err[steps] = np.linalg.norm(back - self.x0) / np.linalg.norm(self.x0)
        ratio = err[16] / err[32]
        self.assertGreater(ratio, 3.0, f"expected ~4x convergence, got {ratio:.2f}")
        self.assertLess(ratio, 5.0, f"expected ~4x convergence, got {ratio:.2f}")

    def test_ddim_algebra_is_exact(self):
        """With an oracle eps the DDIM step pair must be the identity."""
        abar = invert.alphas_cumprod(400)
        span = len(abar) - 1
        x0 = self.x0

        def eps_oracle(x, t_norm):
            a = abar[int(round(t_norm * span))]
            return (x - np.sqrt(a) * x0) / np.sqrt(1 - a)

        back = invert.ddim_sample(eps_oracle, invert.ddim_invert(eps_oracle, x0, abar, steps=25), abar, steps=25)
        rel = np.linalg.norm(back - x0) / np.linalg.norm(x0)
        self.assertLess(float(rel), 1e-9, f"DDIM algebra broken: rel={rel:.2e}")

    def test_ddim_with_predicted_eps_is_bounded_but_inexact(self):
        """The known DDIM gap: bounded, non-zero, and documented rather than hidden."""
        abar = invert.alphas_cumprod(400)
        eps = lambda x, t: 0.35 * x
        back = invert.ddim_sample(eps, invert.ddim_invert(eps, self.x0, abar, steps=20), abar, steps=20)
        rel = np.linalg.norm(back - self.x0) / np.linalg.norm(self.x0)
        self.assertGreater(rel, 1e-6, "predicted-eps DDIM should not be exact")
        self.assertLess(rel, 0.25, f"DDIM drift out of bound: {rel:.2e}")

    def test_schedule_shapes(self):
        abar = invert.alphas_cumprod(1000)
        self.assertEqual(len(abar), 1000)
        self.assertTrue(np.all(np.diff(abar) < 0), "alphas_cumprod must decrease monotonically")
        self.assertGreater(abar[0], 0.99)
        self.assertLess(abar[-1], 0.02)


class TestSelfTest(unittest.TestCase):
    def test_reports_both_claims(self):
        r = invert.self_test(steps=20)
        for key in ("const_field_max_abs_error", "linear_taylor_rel_error", "ddim_oracle_rel_error",
                    "ddim_approx_rel_error", "linear_taylor_convergence"):
            self.assertIn(key, r)
        self.assertLess(r["const_field_max_abs_error"], 1e-10)
        self.assertLess(r["ddim_oracle_rel_error"], 1e-9)
        self.assertGreater(r["ddim_approx_rel_error"], r["ddim_oracle_rel_error"])


class TestStorage(unittest.TestCase):
    def test_latent_roundtrip(self):
        tmp = Path(tempfile.mkdtemp(prefix="reforge-lat-"))
        latent = np.random.default_rng(0).normal(0, 1, (4, 64, 64)).astype(np.float32)
        path = invert.save_latents(tmp / "latents.npz", latent, {"method": "rf-solver", "steps": 20, "seed": 7})
        back, meta = invert.load_latents(path)
        np.testing.assert_array_equal(back, latent)
        self.assertEqual(meta["method"], "rf-solver")
        self.assertEqual(meta["seed"], 7)

    def test_psnr_identity_and_drift(self):
        a = np.random.default_rng(2).normal(0, 1, (16, 16))
        self.assertTrue(np.isinf(invert.roundtrip_psnr(a, a)))
        self.assertLess(invert.roundtrip_psnr(a, a + np.random.default_rng(3).normal(0, 3, a.shape)), 30)


class TestRecipeIntegration(unittest.TestCase):
    def test_inversion_promotes_recipe_to_l3(self):
        rec = recipelib.new_recipe(prompt="x", sampling={"seed": 1})
        self.assertEqual(rec["fidelity"], "L1-semantic")
        recipelib.add_inversion(rec, "rf-solver", "latents.npz", steps=20)
        self.assertEqual(rec["fidelity"], "L3-pixel")
        plan = __import__("reforge.controls", fromlist=["plan"]).plan(rec)
        self.assertEqual(plan["steps"][0]["op"], "load_latents")
        self.assertNotIn("seed: output will vary between runs", plan["gaps"])


class TestDiffusersBridge(unittest.TestCase):
    def test_clear_error_without_diffusers(self):
        if invert.diffusers_available():
            self.skipTest("diffusers installed; the bridge needs a checkpoint to run")
        with self.assertRaises(RuntimeError) as ctx:
            invert.invert_with_diffusers("stabilityai/stable-diffusion-2-1", "x.png", "p")
        self.assertIn("diffusers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
