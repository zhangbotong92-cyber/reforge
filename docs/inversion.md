# Inversion: what the numbers actually are

This page exists because "L3 = pixel exact (PSNR 30+ dB)" is easy to write in a README and
easy to fake. Here is what the code in `reforge/invert.py` really measures, on this machine,
with the checkpoints that are actually reachable from it.

## 1. The solver, checked without any checkpoint

`reforge invert` (no `--model`) runs `self_test()` against fields with closed-form solutions:

| claim | how it is checked | measured |
|---|---|---|
| integral is right | constant velocity field has a linear solution | max abs error **4.4e-16** |
| order is right | halve the steps; error should drop ~4x | **x3.9** per halving |
| 1st vs 2nd order | same field, Euler vs Taylor | 3.3e-02 vs **1.1e-03** relative |
| DDIM algebra is right | oracle epsilon derived from the ground-truth x0 | **2.2e-17** |
| DDIM is exact with a *predicted* eps | the same round trip with a learned eps | **2.7e-02**, and it grows with steps |

The last row is the whole story of L3. DDIM inversion evaluates the predicted epsilon at the
inverted point, not at the true one, so the residual compounds. Null-text inversion, RF-Solver
and friends exist purely to close that gap.

## 2. The real pipeline, on a real checkpoint

`nota-ai/bk-sdm-tiny` (0.5 B, CPU, torch 2.8, `DDIMScheduler`, 320 px, 20 steps),
target = a synthetic test card:

| guidance | inversion PSNR | VAE ceiling | gap |
|---|---|---|---|
| 1.0 (no CFG) | **16.40 dB** | 31.56 dB | 15.2 |
| 3.0 | 14.32 dB | 31.56 dB | 17.2 |
| 7.5 | 11.35 dB | 31.56 dB | 20.2 |

Two things to read from that table:

1. **The gap tracks the CFG scale monotonically.** That is the known failure mode, reproduced
   on demand — the implementation is behaving like a DDIM inversion should.
2. **A distilled tiny model cannot reach 30 dB.** Its epsilon is only good for 2–4 steps;
   used for a 20-step walk it drifts. The 16 dB ceiling here is the checkpoint, not the solver.

Reconstruction degradation measured in latent space on the same run: norm 58.45 -> 49.41 after
inversion -> 38.22 after sampling, cosine 0.58. Signal loss, not noise.

## 3. Three real bugs that this measurement found

Worth recording, because "it runs" hid all three:

| bug | symptom | fix |
|---|---|---|
| CFG batch not concatenated | `size of tensor a (8192) must match b (4096)` | build `context = cat([negative, embeds])` to match the latent concat |
| walk ran the wrong way | PSNR ~12 dB, indistinguishable from noise | inversion must ascend t (0 -> T), sampling descends |
| wrong scheduler | nothing reproduced | force `DDIMScheduler`; PNDM duplicates timesteps and keeps its own state |
| verified the wrong tensor | PSNR measured the decode of *pure noise* | decode the sampled-back latent, not the noised one |

## 4. What would actually reach 30 dB

- **Null-text refinement** (optimise the unconditional embedding per step; needs backprop through
  the UNet — GPU).
- **RF-Solver / RF-Edit** on a rectified-flow model (FLUX, SD3, Wan): the second-order term is
  already implemented and tested here; it needs a flow model's weights.
- **A full SD1.5/SD2.1 checkpoint at 50 steps**, which historically lands around 20–28 dB with
  plain DDIM inversion — better, still short of 30.

Until one of those is run on a GPU, REFORGE reports `passes_l3 = false` and
`quality = "L3-approximate"` for a checkpoint inversion. That is the honest state, and the
metric is what tells you so.
