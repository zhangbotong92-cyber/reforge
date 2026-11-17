"""Layer 6 — inversion. This is what turns L2 (structural) into L3 (pixel).

Text cannot pin pixels; an inverted latent can. Given an image and a model that
exposes either a velocity field (rectified flow: FLUX, SD3, Wan) or an epsilon
prediction (DDIM: SD1.5/SDXL), we integrate the *forward* ODE from the data
manifold to noise, store the latent, and integrating back reproduces the frame.

Two solvers, both training-free:

    ddim     deterministic DDIM inversion with an epsilon parameterisation
    rf       rectified-flow inversion, 1st order (Euler) or 2nd order (Taylor),
             the latter is the RF-Solver idea: reuse the previous velocity
             evaluation to cancel the leading truncation error for free

The module is deliberately model-agnostic: it takes plain callables, so the
numerics are testable against fields with known closed-form solutions and the
same code drives `diffusers` when a checkpoint is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray
EpsFn = Callable[[Array, float], Array]     # epsilon prediction at (x, t)
VelFn = Callable[[Array, float], Array]     # velocity field at (x, t)

__all__ = [
    "linear_beta_schedule",
    "alphas_cumprod",
    "ddim_invert",
    "ddim_sample",
    "rf_invert",
    "rf_sample",
    "roundtrip_psnr",
    "self_test",
    "save_latents",
    "load_latents",
    "diffusers_available",
    "invert_with_diffusers",
]


# ---------------------------------------------------------------- schedules


def linear_beta_schedule(steps: int, beta_start: float = 0.00085, beta_end: float = 0.012) -> Array:
    """The SD-style scaled linear schedule (betas are sqrt-scaled)."""
    return np.linspace(beta_start**0.5, beta_end**0.5, steps, dtype=np.float64) ** 2


def alphas_cumprod(steps: int = 1000, schedule: Optional[Array] = None) -> Array:
    betas = linear_beta_schedule(steps) if schedule is None else np.asarray(schedule, dtype=np.float64)
    return np.cumprod(1.0 - betas)


# ---------------------------------------------------------------- DDIM


def ddim_invert(eps_fn: EpsFn, x0: Array, abar: Array, steps: Optional[int] = None) -> Array:
    """Image -> noise. Deterministic DDIM inversion (eta = 0).

    ``abar`` is the cumulative alpha schedule, index 0 = clean, last = noise.
    """
    abar = np.asarray(abar, dtype=np.float64)
    n = len(abar) if steps is None else int(steps)
    idx = np.linspace(0, len(abar) - 1, n).astype(int)
    x = np.asarray(x0, dtype=np.float64).copy()

    for i in range(len(idx) - 1):
        t_cur, t_next = idx[i], idx[i + 1]
        a_c, a_n = abar[t_cur], abar[t_next]
        eps = eps_fn(x, t_cur / max(len(abar) - 1, 1))
        x0_hat = (x - np.sqrt(1 - a_c) * eps) / np.sqrt(a_c)
        x = np.sqrt(a_n) * x0_hat + np.sqrt(1 - a_n) * eps
    return x


def ddim_sample(eps_fn: EpsFn, xT: Array, abar: Array, steps: Optional[int] = None) -> Array:
    """Noise -> image, the exact reverse walk of :func:`ddim_invert`."""
    abar = np.asarray(abar, dtype=np.float64)
    n = len(abar) if steps is None else int(steps)
    idx = np.linspace(0, len(abar) - 1, n).astype(int)[::-1]
    x = np.asarray(xT, dtype=np.float64).copy()

    for i in range(len(idx) - 1):
        t_cur, t_next = idx[i], idx[i + 1]
        a_c, a_n = abar[t_cur], abar[t_next]
        eps = eps_fn(x, t_cur / max(len(abar) - 1, 1))
        x0_hat = (x - np.sqrt(1 - a_c) * eps) / np.sqrt(a_c)
        x = np.sqrt(a_n) * x0_hat + np.sqrt(1 - a_n) * eps
    return x


# ---------------------------------------------------------------- rectified flow


def rf_invert(v_fn: VelFn, x0: Array, steps: int = 50, order: int = 2) -> Array:
    """Data -> noise along dx/dt = v(x, t), t: 0 -> 1.

    ``order=2`` uses the Taylor/Adams-Bashforth hybrid from RF-Solver: the
    previous velocity evaluation is reused, so the second-order term costs no
    extra network pass.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    h = 1.0 / steps
    v_prev: Optional[Array] = None
    for i in range(steps):
        t = i * h
        v = v_fn(x, t)
        if order >= 2 and v_prev is not None:
            x = x + h * v + 0.5 * h * (v - v_prev)
        else:
            x = x + h * v
        v_prev = v
    return x


def rf_sample(v_fn: VelFn, x1: Array, steps: int = 50, order: int = 2) -> Array:
    """Noise -> data along the same field, t: 1 -> 0."""
    x = np.asarray(x1, dtype=np.float64).copy()
    h = 1.0 / steps
    v_prev: Optional[Array] = None
    for i in range(steps):
        t = 1.0 - i * h
        v = v_fn(x, t)
        if order >= 2 and v_prev is not None:
            x = x - h * v - 0.5 * h * (v - v_prev)
        else:
            x = x - h * v
        v_prev = v
    return x


# ---------------------------------------------------------------- measurement


def _to_uint8(a: Array) -> Array:
    """Map an arbitrary-range array to 0..255 for PSNR."""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def roundtrip_psnr(original: Array, recovered: Array) -> float:
    """PSNR between the original and the inverted-then-sampled result."""
    from .metrics import psnr

    return psnr(_to_uint8(original).astype(np.float64), _to_uint8(recovered).astype(np.float64))


def _probe(reference: Array) -> Array:
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, np.asarray(reference).shape)


def self_test(steps: int = 50, seed: int = 0) -> Dict[str, Any]:
    """Prove the solvers on fields whose trajectory is known exactly.

    Two different claims are measured, and they are not the same thing:

    * **algebraic correctness** - with an *oracle* epsilon (computed from the
      ground-truth x0), one DDIM inversion step reproduces the forward marginal
      exactly, so inversion∘sampling must be the identity to machine precision.
      A non-zero ``ddim_oracle_rel_error`` means the formulas are wrong.
    * **approximation quality** - with a *predicted* epsilon, DDIM inversion is
      known to be inexact: the eps is evaluated at the inverted point, not at the
      true one, and the residual compounds. That is precisely the gap null-text
      inversion optimises away, and the reason RF-Solver-style solvers exist.
      ``ddim_approx_rel_error`` reports it instead of hiding it.

    No checkpoint, no GPU, no download: if these numbers were wrong, every L3
    claim in this repo would be wrong too.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, (3, 64, 64))

    # 1) constant velocity: dx/dt = c  ->  exact for any step count
    c = rng.normal(0, 0.3, x0.shape)
    const = lambda x, t: c  # noqa: E731
    x1 = rf_invert(const, x0, steps=steps)
    back = rf_sample(const, x1, steps=steps)
    const_err = float(np.max(np.abs(back - x0)))

    # 2) linear velocity: dx/dt = -x  ->  x(t) = x0 * exp(-t); 2nd order should win
    def linear(x, t):
        return -x

    e1 = rf_sample(linear, rf_invert(linear, x0, steps=steps, order=1), steps=steps, order=1)
    e2 = rf_sample(linear, rf_invert(linear, x0, steps=steps, order=2), steps=steps, order=2)
    e2_half = rf_sample(
        linear, rf_invert(linear, x0, steps=max(steps // 2, 2), order=2), steps=max(steps // 2, 2), order=2
    )
    rel = lambda r: float(np.linalg.norm(r - x0) / np.linalg.norm(x0))  # noqa: E731

    # 3a) DDIM algebra, checked against an oracle epsilon
    abar = alphas_cumprod(max(steps * 20, 200))
    span = len(abar) - 1

    def eps_oracle(x, t_norm):
        a = abar[int(round(t_norm * span))]
        return (x - np.sqrt(a) * x0) / np.sqrt(1 - a)

    oracle_back = ddim_sample(eps_oracle, ddim_invert(eps_oracle, x0, abar, steps=steps), abar, steps=steps)

    # 3b) DDIM with a predicted (linear) epsilon: honest, and expected to be worse
    k = 0.35
    eps_pred = lambda x, t: k * x  # noqa: E731
    approx_back = ddim_sample(eps_pred, ddim_invert(eps_pred, x0, abar, steps=steps), abar, steps=steps)

    return {
        "steps": steps,
        "const_field_max_abs_error": const_err,
        "const_field_psnr": roundtrip_psnr(x0, back),
        "linear_euler_rel_error": rel(e1),
        "linear_taylor_rel_error": rel(e2),
        "linear_taylor_convergence": rel(e2_half) / max(rel(e2), 1e-18),
        "linear_euler_psnr": roundtrip_psnr(x0, e1),
        "linear_taylor_psnr": roundtrip_psnr(x0, e2),
        "ddim_oracle_rel_error": rel(oracle_back),
        "ddim_approx_rel_error": rel(approx_back),
        "ddim_approx_psnr": roundtrip_psnr(x0, approx_back),
        "notes": "ddim_approx_* is the inherent DDIM gap, not a bug; L3 uses the oracle-exact path or RF",
    }


# ---------------------------------------------------------------- storage


def save_latents(path: str | Path, latent: Array, meta: Optional[Dict[str, Any]] = None) -> Path:
    """Store the inverted latent next to its provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        latent=np.asarray(latent, dtype=np.float32),
        meta=json.dumps(meta or {}, ensure_ascii=False),
    )
    return path


def load_latents(path: str | Path) -> Tuple[Array, Dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as blob:
        latent = blob["latent"]
        meta = json.loads(str(blob["meta"]))
    return latent, meta


# ---------------------------------------------------------------- diffusers bridge


def diffusers_available() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def invert_with_diffusers(
    model_id: str,
    image_path: str | Path,
    prompt: str,
    steps: int = 30,
    device: Optional[str] = None,
    guidance: float = 5.0,
    verify: bool = True,
    max_side: int = 512,
) -> Dict[str, Any]:
    """Run a real DDIM inversion through a ``diffusers`` pipeline.

    Kept in-tree so the recipe's L3 tier is executable, not aspirational. It needs
    a checkpoint (2-7 GB) and, realistically, a GPU: CPU works but is 50-100x slower.
    Returns the latent, the verification PSNR against the original, and metadata.
    """
    if not diffusers_available():
        raise RuntimeError("L3 inversion needs diffusers + torch: pip install diffusers torch transformers accelerate")
    import torch
    from diffusers import AutoPipelineForText2Image
    from PIL import Image

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    # Inversion needs a deterministic DDIM-style schedule. PNDM (the default for several
    # checkpoints) duplicates timesteps and carries its own multi-step state, so walking it
    # with a DDIM update reproduces nothing.
    from diffusers import DDIMScheduler

    if not isinstance(pipe.scheduler, DDIMScheduler):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    # match the pipeline's expected size while keeping the aspect ratio
    w, h = image.size
    scale = max_side / max(w, h)
    image = image.resize((max(32, int(w * scale) // 8 * 8), max(32, int(h * scale) // 8 * 8)), Image.LANCZOS)

    arr = np.asarray(image).astype(np.float32) / 127.5 - 1.0
    x0 = torch.from_numpy(arr).permute(2, 0, 1)[None].to(dev, dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(x0).latent_dist.sample() * pipe.vae.config.scaling_factor

    embeds, negative = pipe.encode_prompt(prompt, device=dev, num_images_per_prompt=1, do_classifier_free_guidance=guidance > 1)
    # classifier-free guidance needs BOTH branches in the batch, matching the latent concat
    context = torch.cat([negative, embeds]) if (guidance > 1 and negative is not None) else embeds
    sched = pipe.scheduler
    sched.set_timesteps(steps)
    ts = sched.timesteps  # descending: noise level -> clean
    if not hasattr(sched, "alphas_cumprod"):
        raise RuntimeError(f"scheduler {type(sched).__name__} exposes no alphas_cumprod; use a DDIM-style scheduler")

    def alpha(t):
        return torch.as_tensor(sched.alphas_cumprod[int(t.item())], device=dev, dtype=latent.dtype)

    def eps_at(x, t):
        model_in = torch.cat([x] * 2) if guidance > 1 else x
        out = pipe.unet(model_in, t, encoder_hidden_states=context).sample
        if guidance > 1:
            uncond, cond = out.chunk(2)
            out = uncond + guidance * (cond - uncond)
        return out

    def step(x, t_cur, t_next, invert):
        a_c, a_n = alpha(t_cur), alpha(t_next)
        eps = eps_at(x, t_cur)
        x0_hat = (x - torch.sqrt(1 - a_c) * eps) / torch.sqrt(a_c)
        return torch.sqrt(a_n) * x0_hat + torch.sqrt(1 - a_n) * eps

    # 1) inversion: clean -> noise, so walk t upward (the scheduler lists it downward)
    inv = latent.clone()
    with torch.no_grad():
        for i in range(len(ts) - 1):
            inv = step(inv, ts[-(i + 1)], ts[-(i + 2)], invert=True)

    # 2) sampling: noise -> clean, the same walk reversed
    cur = inv.clone()
    with torch.no_grad():
        for i in range(len(ts) - 1):
            cur = step(cur, ts[i], ts[i + 1], invert=False)

    result: Dict[str, Any] = {
        "method": "ddim",
        "model": model_id,
        "prompt": prompt,
        "steps": steps,
        "guidance": guidance,
        "device": dev,
        "shape": list(inv.shape),
        "latent": inv.detach().float().cpu().numpy(),
    }

    if verify:
        with torch.no_grad():
            # decode the *sampled-back* latent, never the noised one: at t=T-1 the latent is
            # almost pure noise and its PSNR would measure nothing but the noise floor
            decoded = pipe.vae.decode(cur / pipe.vae.config.scaling_factor).sample
            # baseline: VAE round trip with no inversion at all. If the inversion PSNR is
            # close to this, the solver is fine and the gap is the checkpoint, not the code.
            plain = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        rec = ((decoded[0].float().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255)
        base = ((plain[0].float().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255)
        original = np.asarray(image)
        result["psnr"] = roundtrip_psnr(original, rec)
        result["vae_psnr"] = roundtrip_psnr(original, base)
        result["psnr_gap_vs_vae"] = round(result["vae_psnr"] - result["psnr"], 2)
        result["passes_l3"] = bool(result["psnr"] >= 30.0)
        if result["passes_l3"]:
            result["quality"] = "L3 pixel-exact"
        elif result["psnr"] >= 15.0:
            result["quality"] = "L3-approximate (inverted, not pixel-exact)"
        else:
            result["quality"] = "below reconstruction floor"
        result["verdict"] = (
            "L3 pixel-exact"
            if result["passes_l3"]
            else (
                "inversion matches the VAE ceiling; the checkpoint is the limit"
                if result["psnr_gap_vs_vae"] < 3.0
                else "naive DDIM inversion diverges from the VAE ceiling - add null-text / RF refinement"
            )
        )
    return result
