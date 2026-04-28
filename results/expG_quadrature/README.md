# Experiment G — Quadrature Speed/Equivariance Pareto

What is the right way to compare DH vs GL? **Same equivariance error**,
then ask how much wall-clock time each method needs to reach it.

## Setup (fairchem default EquiformerV2, lmax=6, mmax=2)

Real `EquiformerV2Backbone` from fairchem: 12 layers, 128 channels,
24 `S2Activation` modules. NVIDIA A100-SXM4-40GB, idle.

These hyperparameters are the function-signature defaults of
`EquiformerV2` in `fairchem.core.models.equiformer_v2.equiformer_v2.py`
(`num_layers=12`, `sphere_channels=128`, `lmax_list=[6]`, `mmax_list=[2]`).
That makes them a reasonable mid-size config, but **not the published OC20
SOTA**: the EquiformerV2 paper (Liao et al., ICLR 2024) reports its best
OC20 S2EF model with 20 layers, lmax=8, mmax=8 (~150M params). The setting
here is smaller, so the savings reported below are likely a *lower bound*
for the production OC20 model.

The "24" S2 activation modules count is `2 × num_layers = 24` (each
`TransBlockV2` has one S2 activation in the graph-attention block and one
in the FFN).

| Config | Method | Grid (lat × lon) | Points |
|---|---|---|---|
| DH default | e3nn equiangular | 14 × 5 | 70 |
| DH 2× | DH with `resolution=4(L+1)` | 28 × 28 | 784 |
| GL min | custom GL | 7 × 13 | 91 |
| GL match-DH | custom GL | 14 × 13 | 182 |
| GL 2× | custom GL | 14 × 28 | 392 |

## What we measured (no pseudo-replication, no estimated points)

**Equivariance error** of one S2-Act kernel (SiLU activation, random
coefficients, 5 rotations × 10 inputs at lmax=6, mmax=2). This is a
weight-independent kernel-level test; it depends only on the grid.
*(Source: `experiments/verify_gl_grid.py`)*

**Wall-clock**: end-to-end forward pass on QM9 (batch=8 graphs).
Protocol:
- 5 independent runs per config (the unit of replication for CI).
- Each run uses 10 warmup + 30 measured forwards on **distinct QM9
  batches** drawn without replacement (no batch reuse within a run, fresh
  RNG seed per run, so within-run iterations are not pseudo-replicates).
- `torch.cuda.synchronize()` around every measurement.
- Fresh model construction per (run, config); `del model + empty_cache`
  before next config.
- 95% CI is `t_{n_runs-1, 0.975} × s.e.(run_means)` with the run mean as
  the unit (NOT pooled across iterations).
- Architecture-level timing: a fresh randomly-initialized
  `QM9Model(...)` is timed; **no checkpoint weights are loaded** for this
  benchmark. The cost depends on architecture and tensor shapes, not on
  weight values, so this faithfully measures the cost a deployed model
  would incur. The separate `dropin_pretrained.json` table validates
  that swapping the grid in an actual trained checkpoint preserves
  predictions.
*(Source: `experiments/expG_verify_speedup_v2.py`,
`verify_speedup_v2.json`)*

## Combined results

| Config | Pts | Equiv err | Fwd (ms, 95% CI on run means) |
|---|---|---|---|
| DH default   | 70  | 4.43 × 10⁻¹ | 113.50 ± 1.87 |
| **DH 2×**    | **784** | **3.27 × 10⁻¹** | **164.66 ± 1.62 (+45.1%, p < 0.0001)** |
| GL min       | 91  | 4.29 × 10⁻¹ | 112.68 ± 1.87 (−0.7%, p = 0.42, n.s.) |
| **GL match-DH** | **182** | **3.28 × 10⁻¹** | **113.17 ± 2.25 (−0.3%, p = 0.76, n.s.)** |
| **GL 2×**    | **392** | **3.28 × 10⁻¹** | **112.61 ± 2.35 (−0.8%, p = 0.44, n.s.)** |

(The "n.s." comparisons are with respect to DH default. Welch's t-test on
n = 5 run means; significance threshold 0.05.)

## Matched-equivariance comparison

For target equiv err **= 3.28 × 10⁻¹** (the saturation floor at this mmax
cropping):

- Cheapest DH: **DH 2× at 164.66 ms** (DH default at 70 pts only reaches
  4.43 × 10⁻¹, not enough)
- Cheapest GL: **GL match-DH at 113.17 ms**
- **Savings: 51.5 ms ± 2.36 (95% CI), p < 0.0001 = 31.3% time reduction**

Equivalently: GL achieves DH 2×'s equivariance at the same wall-clock as
DH default (the GL-vs-DH-default differences are statistically zero).
The +51.5 ms premium that DH pays to reach the equivariance floor goes
away under GL.

## Caveats and what we did NOT prove

1. The +31% savings is at this specific (lmax=6, mmax=2) config and on
   this GPU. Different model depths, channel counts, or hardware will
   scale it. At the smaller QM9-scale model (4 layers, lmax=4) the S2
   activation is only ~0.3% of forward and the gap collapses to noise.
2. The equivariance error is measured on **one S2-Act kernel with random
   inputs** at lmax=6, mmax=2, not on a trained model's actual
   intermediate features. Trained-model features are smoother (absolute
   equiv numbers will be much smaller); the *ranking* between
   quadratures is what is being claimed to transfer.
3. The kernel-level equivariance gap between GL match-DH and DH 2× is
   within rounding (3.28 × 10⁻¹ vs 3.27 × 10⁻¹) — they really are the
   same accuracy at this mmax cropping.
4. The mmax = 2 cropping caps achievable equivariance at ≈ 0.328
   regardless of grid; lower error needs mmax = lmax (architectural
   change). At lmax = mmax both methods can reach a ≈ 4 × 10⁻³ floor;
   there GL needs roughly half the latitude points of DH (see
   `verify_gl_grid.py` output).
5. The wall-clock benchmark above does NOT load any pretrained weights;
   it times architecture-level forward cost. The companion experiment
   `dropin_pretrained.json` validates that swapping the grid in actual
   trained checkpoints preserves predictions.

## Earlier version vs this one

- An earlier benchmark (`verify_speedup_fairchem_default.json`) reported
  90 pooled per-iteration timings as if IID, and silently filled
  GL-min's wall-clock as 111.0 ms because it was never measured. Both
  are corrected here. The headline number (≈ 31% savings) is unchanged
  but the uncertainty quantification is now sound: 95% CI ±2.36 ms on
  the difference, p < 0.0001 from Welch's t-test on n = 5 run means.

## Reproduce

```bash
module load pytorch/2.6.0-1
# Equivariance for all configs at lmax=6, mmax=2
python3 experiments/verify_gl_grid.py
# Wall-clock: rigorous IID protocol, 5 runs × 30 distinct batches
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/expG_verify_speedup_v2.py \
    --batch_size 8 --n_runs 5 --n_iter 30 --n_warmup 10
# Drop-in test on actual trained checkpoints (small model)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/expG_dropin_pretrained.py
```

## Files in this folder

- `README.md` — this report
- `pareto.json` — combined equivariance + wall-clock data
- `pareto.{png,pdf}` — Pareto frontier figure
- `verify_speedup_v2.json` — rigorous wall-clock benchmark (current)
- `verify_speedup_fairchem_default.json` — older V1 benchmark (kept for diff)
- `breakdown.json` — per-operation S2-act timing inside the model
- `dropin_pretrained.json` — drop-in swap on actual trained 4-layer checkpoints
- `report.tex`, `report.pdf` — short report in PDF
- `SiLU_gl10x9_U0_seed42/` — end-to-end 2-epoch GL training pilot
