# Experiment G — Quadrature Speed/Equivariance Pareto

What is the right way to compare DH vs GL? **Same equivariance error**, then ask
how much wall-clock time each method needs to reach it.

## Setup (OC20-scale EquiformerV2, lmax=6, mmax=2)

Real `EquiformerV2Backbone` from fairchem: 12 layers, 128 channels, 24
`S2Activation` modules. NVIDIA A100-SXM4-40GB, idle.

| Config | Method | Grid (lat × lon) | Points |
|---|---|---|---|
| DH default | e3nn equiangular | 14 × 5 | 70 |
| DH 2× | DH with `resolution=4(L+1)` | 28 × 28 | 784 |
| GL min | custom GL | 7 × 13 | 91 |
| GL match-DH | custom GL | 14 × 13 | 182 |
| GL 2× | custom GL | 14 × 28 | 392 |

## Two measurements at the same (lmax=6, mmax=2)

**Equivariance error** of one S2-Act kernel (random coefficients, 5 rotations × 10 inputs):
SiLU activation; `‖A(D·c) − D·A(c)‖ / ‖A(D·c)‖`.
*(Source: `experiments/verify_gl_grid.py`)*

**Wall-clock**: full forward pass on a QM9 batch=8, 90 measurements per config
(3 runs × 30 iter), `cuda.sync()` around each, fresh model per config.
*(Source: `experiments/expG_verify_speedup.py`, `verify_speedup_fairchem_default.json`)*

## Combined results

| Config | Pts | Equiv err | Fwd (ms, 95% CI) |
|---|---|---|---|
| DH default | 70 | 4.43e-1 | 111.83 ± 0.79 |
| **DH 2×** | **784** | **3.27e-1** | **162.73 ± 1.23 (+45.5%)** |
| GL min | 91 | 4.29e-1 | ~111 (matches DH default in expG_speed_benchmark) |
| **GL match-DH** | **182** | **3.28e-1** | **112.23 ± 0.75 (+0.4%, n.s.)** |
| **GL 2×** | **392** | **3.28e-1** | **111.34 ± 0.76 (−0.4%, n.s.)** |

## Pareto frontier and the matched-equivariance comparison

The relevant question: **at the same equivariance error, what's the cheapest config in each method?**

For target equiv err **= 3.28e-1** (the saturation floor at this mmax cropping):
- Cheapest DH: **DH 2× at 162.73 ms** (DH default at 70 pts only reaches 4.43e-1, not enough)
- Cheapest GL: **GL match-DH at 112.23 ms**
- **Savings: 50.5 ms = (162.73 − 112.23) / 162.73 = 31.0% time reduction**

So: **YES, at matched equivariance accuracy, GL saves ~31% wall-clock per forward.**

(Or framed the other way: GL achieves DH 2×'s equivariance at the same wall-clock as DH default, with the +50.5 ms premium going away.)

## What about smaller equivariance errors?

The mmax=2 cropping caps the achievable equivariance at ~3.28e-1 — no quadrature
goes lower because most of the high-`m` content is being thrown away by the
SO(3)→SO(2) reduction. To reach lower equivariance you'd need to raise mmax
(architectural change), or use different cropping.

For models that use mmax=lmax (no cropping) the GL advantage shifts: **GL
needs roughly half the latitude points of DH for matched accuracy**
(see `verify_gl_grid.py` output for lmax=mmax=4 and lmax=mmax=6).

## Caveats and what we did NOT prove

1. The +31% savings is at this specific (lmax=6, mmax=2) config and on this GPU.
   Different model depths, channel counts, or hardware will scale it.
2. The equivariance error is measured on **one S2-Act kernel with random
   inputs**, not on a trained model's actual intermediate features. Trained-model
   features are smoother, so absolute equiv numbers will be much smaller, but
   the *ranking* between quadratures should be preserved.
3. The kernel-level equivariance gap between GL match-DH (3.28e-1) and DH 2×
   (3.27e-1) is small (within rounding) — they really are the same accuracy.
   Compare GL min (4.29e-1) vs DH default (4.43e-1) — also near-tied.
4. We did not measure equivariance at the OC20-scale fully trained model. To
   prove the +31% savings holds end-to-end, one would need to train a model
   with each grid for 50 epochs and measure prediction invariance error.

## Earlier framing vs now

The previous report claimed "DH 2× costs +45.5% over DH default", which is
correct but compared apples to oranges (different equivariance errors). The
fairer claim, anchored to matched equivariance:

> Switching DH 2× → GL match-DH saves **31% wall-clock per forward** while
> preserving the same equivariance error. Switching DH default → GL match-DH
> is **free** (same wall-clock) and substantially improves equivariance
> (4.43e-1 → 3.28e-1).

## Reproduce

```bash
module load pytorch/2.6.0-1
# Equivariance numbers for all configs at lmax=6, mmax=2
python3 experiments/verify_gl_grid.py
# Wall-clock: rigorous 3-run verification
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/expG_verify_speedup.py \
    --config fairchem_default --batch_size 8 \
    --n_repeats 3 --n_iter 30 --n_warmup 20
```

## Files in this folder

- `README.md` — this report
- `pareto.json` — combined equivariance + wall-clock data, OC20 scale
- `verify_speedup_fairchem_default.json` — rigorous wall-clock benchmark
- `speed_benchmark.json`, `speed_fairchem_default.json` — initial single-shot benchmarks
- `report.tex` / `report.pdf` — earlier slowdown-only report (superseded by this one)
- `SiLU_gl10x9_U0_seed42/` — end-to-end 2-epoch GL training pilot
