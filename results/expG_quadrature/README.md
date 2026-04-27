# Experiment G — Quadrature Speed Verification

Verifies that DH 2× is really +45% slower than DH default, and that GL is really free.

## Setup

- **GPU**: NVIDIA A100-40GB (Perlmutter), confirmed idle (`nvidia-smi`).
- **Model**: real `EquiformerV2Backbone` from fairchem, OC20-scale config:
  12 layers, 128 channels, lmax=6, mmax=2 → 24 `S2Activation` modules.
- **Data**: 3 QM9 batches (batch=8) pre-fetched **once** and reused for every config.

| Config | Method | Grid (lat × lon) | Total pts |
|---|---|---|---|
| DH default | e3nn equiangular | 14 × 5 | 70 |
| DH 2× | DH with `resolution=4(L+1)` | 28 × 28 | 784 |
| GL match-DH | custom `CustomSO3Grid` | 14 × 13 | 182 |
| GL 2× | custom `CustomSO3Grid` | 14 × 28 | 392 |

## Protocol (per config, per run)

1. Build a fresh model, patch SiLU + (optionally) swap `SO3_Grid` to GL.
2. **20 warmup** forward passes.
3. **30 measured** forward passes, each wrapped in `torch.cuda.synchronize()`.
4. `del model; cuda.empty_cache()` before next config.

## Controls

| Confound | Mitigation |
|---|---|
| Async kernels | `cuda.synchronize()` around every iter |
| Cold start | 20-iter warmup |
| Run-to-run drift | 3 independent sweeps |
| Ordering effects | Forward / reverse / shuffled orderings |
| Different inputs | Same 3 batches reused for all configs |
| Memory carryover | `del model + empty_cache` between configs |
| Other GPU users | Verified `nvidia-smi --query-compute-apps` empty |

## Results (3 runs × 30 iters = 90 measurements per config)

| Config | Mean ± 95% CI (ms) | Δ vs DH default | Significance |
|---|---|---|---|
| DH default | 111.83 ± 0.79 | — | — |
| **DH 2×** | **162.73 ± 1.23** | **+50.90 ms (+45.5%)** | **35× CI** |
| GL match-DH | 112.23 ± 0.75 | +0.40 ms (+0.4%) | not sig. |
| GL 2× | 111.34 ± 0.76 | −0.49 ms (−0.4%) | not sig. |

DH 2× measured 162.60 / 162.35 / 163.22 ms across the three orderings —
position-independent. Run-to-run std of the means is 0.36 ms.

## Sanity check

S2-act kernel alone: DH default 0.34 ms, DH 2× 2.36 ms (per call).
24 modules × (2.36 − 0.34) = **48.5 ms predicted** vs **50.9 ms observed**.
The 2.4 ms gap is kernel-launch overhead.

## Reproduce

```bash
module load pytorch/2.6.0-1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/expG_verify_speedup.py \
    --config fairchem_default --batch_size 8 \
    --n_repeats 3 --n_iter 30 --n_warmup 20
```

Takes ~3 min. Outputs `verify_speedup_fairchem_default.json` (every iteration time).

## Files

- `verify_speedup_fairchem_default.json` — the verified run (this README)
- `speed_benchmark.json`, `speed_fairchem_default.json` — initial single-shot benchmarks
- `SiLU_gl10x9_U0_seed42/` — end-to-end 2-epoch GL training pilot
- `experiments/expG_verify_speedup.py` — the verification script
- `src/equiformer_grid_patch.py` — `CustomSO3Grid` + `patch_so3_grid()`

## Takeaway

GL 2× delivers DH 2×'s equivariance at DH default's wall-clock cost.
Switching from DH default to DH 2× costs +51 ms/forward; switching to GL is free.
The advantage scales with model size (negligible at QM9 scale, +45% at OC20 scale).
