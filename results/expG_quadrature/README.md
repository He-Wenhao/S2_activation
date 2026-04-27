# Experiment G — Quadrature Speed Verification

This document records exactly how the +45.5% slowdown of `DH 2×` over `DH default`
was measured, what was controlled for, and how to reproduce it.

## Question

Does swapping the `SO3_Grid` quadrature inside EquiformerV2 produce a real,
reproducible wall-clock difference, or is the gap a measurement artifact?

Specifically:
- Is the **+46% forward-time penalty for DH 2×** real?
- Are the GL configurations actually free, or just within noise?

## Setup

### Hardware
| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB (Perlmutter) |
| GPU utilization at start | 0% (no other processes on device) |
| PyTorch | 2.6.0 |
| Software stack | `module load pytorch/2.6.0-1` on Perlmutter |

We confirmed via `nvidia-smi --query-compute-apps` that no other process held
GPU memory before the run started.

### Model: OC20-scale EquiformerV2
The benchmark uses the *real* `EquiformerV2Backbone` from
`fairchem.core.models.equiformer_v2.equiformer_v2`, configured at the
fairchem default (OC20) scale, **not** the small QM9 model used in
Experiment F:

| Hyperparameter | Value |
|---|---|
| `num_layers` | 12 |
| `sphere_channels` | 128 |
| `attn_hidden_channels` | 128 |
| `ffn_hidden_channels` | 256 |
| `num_heads` | 8 |
| `lmax_list` | `[6]` |
| `mmax_list` | `[2]` |
| Total `S2Activation` modules | 24 (2 per layer × 12 layers) |

This config matches the activation density of the production OC20
EquiformerV2 model. The QM9 model in Experiment F uses 4 layers / 64 channels
/ lmax=4 (5.2M params) and S2 activation is only ~0.3% of forward there,
so we deliberately picked the larger config to test where the cost actually matters.

### Data
- Three QM9 batches (batch_size = 8 molecules), pre-fetched once and reused
  across **all** configs in **every** run.
- This guarantees identical GPU work per batch — only the grid changes.
- Adapted to EquiformerV2's expected interface via `qm9_adapt`
  (sets `atomic_numbers`, `natoms`).

### Quadrature configurations measured
| Label | Method | Grid | Total points |
|---|---|---|---|
| DH default | Driscoll–Healy (e3nn default) | `2(L+1)` lat × `2m+1` lon | 14 × 5 = 70 |
| DH 2× | DH with `resolution=4(L+1)` (overrides both dims) | 28 × 28 | 784 |
| GL match-DH | Custom GL via `CustomSO3Grid` | `2(L+1)` × `2L+1` | 14 × 13 = 182 |
| GL 2× | Custom GL with double longitude | `2(L+1)` × `4L+4` | 14 × 28 = 392 |

The custom GL grids are constructed in `src/equiformer_grid_patch.py` by
direct evaluation of e3nn real spherical harmonics on Gauss–Legendre latitude
nodes × uniform longitude nodes, then patched into
`model.backbone.SO3_grid` via `patch_so3_grid()`.

## Benchmark Protocol

For each (run, config) pair we executed:

1. **Build a fresh model**: instantiate `QM9Model(backbone_kwargs)`, patch
   `S2Activation` modules to use SiLU, optionally swap the `SO3_Grid` to GL.
2. **Warm up**: call `model(batch)` 20 times with `torch.no_grad()` to let
   CUDA kernel selection / cuDNN autotune settle.
3. **Measure**: 30 timed iterations; for each:
   ```python
   torch.cuda.synchronize()
   t0 = time.perf_counter()
   model(batch)            # forward only, no_grad
   torch.cuda.synchronize()
   times.append((time.perf_counter() - t0) * 1000)
   ```
4. **Tear down**: `del model`, `torch.cuda.empty_cache()` before the next
   config to prevent memory carryover from biasing later configs.

### Why these specific choices
- `cuda.synchronize()` around every measurement is essential — without it,
  CUDA kernels are launched async and you measure launch latency, not
  execution time.
- 20 warmup iterations is well above the empirical 5–10 needed to reach
  steady state on this model. The first few iterations are visibly slower
  due to CUDA kernel selection / memory pool growth.
- 30 measured iterations gives a sample standard error well below the
  effect size we want to detect.
- `del model + empty_cache` between configs is necessary because the
  CUDA memory allocator reuses freed blocks; if we don't release, the
  later configs see different fragmentation patterns than the earlier ones.

## Variance Sources Controlled For

A naive "run once, look at the means" benchmark can be misled by:

| Source | Mitigation |
|---|---|
| Cold-start CUDA kernel selection / autotune | 20-iteration warmup per config |
| Async kernel launches | `torch.cuda.synchronize()` around every measurement |
| Run-to-run drift (GPU thermal, system load) | Repeat the whole sweep 3 times |
| Configuration ordering (one config helping the next via cache reuse) | Run forward, reverse, AND seed-shuffled orderings |
| Different inputs across configs | Pre-fetch 3 QM9 batches once, reuse for ALL configs |
| Memory carryover between configs | `del model + cuda.empty_cache()` between configs |
| Other GPU users | Verified `nvidia-smi --query-compute-apps` returned no users |

The three orderings test the most insidious failure mode: if `DH 2×` always
ran first, you might worry it bears the cold-start cost and looks artificially
slow. By running it in position 1 (forward order), 3 (reverse), and the
shuffled position, we can see if its measured time depends on its position.

## Statistical Methodology

Each config has **3 independent runs** of 30 iterations = **90 total
measurements**. We report:

- **Mean**: arithmetic mean over all 90 iterations
- **Std**: pooled std of all 90 iterations (within-run + between-run)
- **Run-to-run std**: std of the 3 per-run means (a sanity check on
  reproducibility across full sweep restarts)
- **95% CI**: `1.96 × std / sqrt(90)` — the standard CI for a sample mean,
  assuming approximately normal iteration times (which holds well after
  warmup; we verified by inspecting the histogram is roughly Gaussian
  with no long tail)

For a comparison `Δ = config − DH_default`, we report:
- `Δ ms` = difference of means
- `Δ %` = `100 × Δ / DH_default_mean`
- **Combined CI** = `sqrt(CI_config² + CI_DH_default²)` — a conservative
  Welch-style bound on the difference of two means
- Significance: `**` if `|Δ| > combined_CI`, `(not sig.)` otherwise.

## Results

### Per-run measurements (mean ± std over 30 iters)

| Run | Order | DH default | **DH 2×** | GL match-DH | GL 2× |
|---|---|---|---|---|---|
| 1 | forward | 112.32 ± 4.17 | **162.60 ± 5.97** | 112.33 ± 3.41 | 111.08 ± 3.61 |
| 2 | reverse | 111.80 ± 3.54 | **162.35 ± 6.02** | 112.07 ± 3.83 | 111.13 ± 3.52 |
| 3 | shuffled (seed=42) | 111.37 ± 3.62 | **163.22 ± 5.85** | 112.30 ± 3.60 | 111.80 ± 3.86 |

`DH 2×` measures within 0.5% of itself across all three orderings — the
ordering does not affect the result.

### Aggregated (90 iterations per config)

| Config | Mean (ms) | Std (ms) | 95% CI (ms) | Run-to-run std |
|---|---|---|---|---|
| DH default | 111.83 | 3.81 | ±0.79 | 0.39 |
| **DH 2×** | **162.73** | 5.96 | ±1.23 | 0.36 |
| GL match-DH | 112.23 | 3.62 | ±0.75 | 0.12 |
| GL 2× | 111.34 | 3.68 | ±0.76 | 0.33 |

### Differences vs DH default

| Comparison | Δ (ms) | Δ (%) | Combined 95% CI | Significance |
|---|---|---|---|---|
| **DH 2× vs DH default** | **+50.90** | **+45.5%** | ±1.46 ms | **35× CI ✓ significant** |
| GL match-DH vs DH default | +0.40 | +0.4% | ±1.08 ms | not significant |
| GL 2× vs DH default | −0.49 | −0.4% | ±1.09 ms | not significant |

The 50.9 ms gap is 35× the combined 95% CI, so we can reject the null
hypothesis (`DH default = DH 2×`) at vastly stronger than p < 0.001.

### Sanity check: first-principles prediction

The S2 activation kernel timing alone (measured separately, batch=1024):
- DH default: 0.34 ms / call
- DH 2×: 2.36 ms / call

The model has 24 `S2Activation` modules. Per-forward kernel-time delta:
```
24 × (2.36 − 0.34) = 48.5 ms predicted
50.9 ms observed
```

The 2.4 ms gap is the kernel-launch overhead and slightly larger
intermediate tensors. The arithmetic checks out.

## How to Reproduce

```bash
# On a Perlmutter login or compute node with a free A100:
module load pytorch/2.6.0-1
cd /pscratch/sd/w/whe1/S2_activation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/expG_verify_speedup.py \
    --config fairchem_default \
    --batch_size 8 \
    --n_repeats 3 \
    --n_iter 30 \
    --n_warmup 20
```

Outputs:
- Console: per-run table, aggregate table, comparisons with significance
- File: `results/expG_quadrature/verify_speedup_fairchem_default.json`
  containing every individual iteration time

The full benchmark takes ~3 minutes on an A100-40GB.

## Files

```
results/expG_quadrature/
├── README.md                              ← this document
├── speed_benchmark.json                   ← initial small-config benchmark
├── speed_fairchem_default.json            ← initial OC20-config benchmark
├── verify_speedup_fairchem_default.json   ← rigorous 3-run verification (the one this README documents)
└── SiLU_gl10x9_U0_seed42/                 ← end-to-end GL training pilot (2-epoch)

experiments/
├── expG_quadrature_speed.py               ← initial single-run benchmark
└── expG_verify_speedup.py                 ← multi-run verification (used here)

src/
└── equiformer_grid_patch.py               ← CustomSO3Grid implementation +
                                            patch_so3_grid() helper
```

## Interpretation

The +45.5% number is solid:
1. Reproducible across 3 independent runs and 3 different config orderings.
2. Statistically significant by 35σ (combined CI).
3. Consistent with first-principles prediction (24 calls × per-call delta).
4. GL configs land within ±1 ms of DH default — no detectable cost.

**Practical implication for EquiformerV2 users:**
GL `2×` delivers DH `2×`'s equivariance (≈4 × 10⁻³ floor at lmax=6) at
DH default's wall-clock cost. Conversely, raising DH to `2×` costs +51 ms
per forward (and roughly proportional cost per training step) for the
same accuracy gain that GL provides for free. The speed advantage is a
property of the model scale: at QM9 scale (4 layers, lmax=4) the S2
activation is ~0.3% of forward and the gap collapses to noise; at
OC20 scale it grows to 7% / 32% (default / 2×) and matters in practice.
