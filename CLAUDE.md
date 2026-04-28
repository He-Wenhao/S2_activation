# CLAUDE.md — repo guide for AI assistants

This file is loaded automatically by Claude Code when it works in this repository.
Skim before doing anything; the gotchas are non-obvious.

## What's in this repo

`S2 Activation` is a study of pointwise nonlinearities applied on $S^2$ via
spherical-harmonic ↔ grid transforms inside SO(3)-equivariant networks
(SCN, eSCN, EquiformerV2 lineage). The current main story (in `paper/`) is
that switching the quadrature rule of EquiformerV2's `SO3_Grid` from
Driscoll–Healy (e3nn default) to Gauss–Legendre is a drop-in equivariance
upgrade with measurable inference-time savings at matched accuracy. An
older "collection of tests" draft lives in `paper_bk/`.

## Python environment

This is an HPC project on **NERSC Perlmutter**. Every Python script runs
inside the NERSC PyTorch module:

```bash
module load pytorch/2.6.0-1
```

After loading, `python3` resolves to
`/global/common/software/nersc9/pytorch/2.6.0-1/bin/python3`.

| Package | Version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.6.0 |
| CUDA | 12.4 |
| cuDNN | 9.05 |
| e3nn | 0.4.4 |
| fairchem-core | 1.10.0 |
| torch_geometric | 2.7.0 |
| scipy | 1.16.3 |
| numpy | 2.2.6 |
| matplotlib | 3.10.8 |

User packages (e3nn, fairchem) live in
`/global/homes/w/whe1/.local/perlmutter/pytorch2.6.0-1/lib/python3.12/site-packages/`.
The base PyTorch comes from the module itself.

## Two non-obvious gotchas

1. **`weights_only=True` breaks e3nn imports.** PyTorch 2.6 made
   `torch.load(weights_only=True)` the default. e3nn ships a `constants.pt`
   file containing a tuple with a `slice` object, which fails the new safe
   globals check. Every script in this repo includes the workaround:

   ```python
   import torch
   torch.serialization.add_safe_globals([slice])
   from e3nn import o3   # safe to import after the line above
   ```

   Skipping this line gives an opaque
   `_pickle.UnpicklingError: ... GLOBAL builtins.slice was not an allowed global`.

2. **OC20-scale benchmarks need `expandable_segments`.** The 12-layer /
   128-channel / lmax=6 model fragments the CUDA allocator badly when
   constructing several large grids in the same process. Set:

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 ...
   ```

   Without this you can OOM at batch=8 even on a 40 GB A100.

## Reproducing the speed results

```bash
module load pytorch/2.6.0-1
cd /pscratch/sd/w/whe1/S2_activation

# Numerical equivariance comparison at lmax=6, mmax=2 (~2 minutes)
python3 experiments/verify_gl_grid.py

# Wall-clock benchmark, 4 configs × 3 runs × 30 iter (~3 minutes)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/expG_verify_speedup.py \
    --config fairchem_default --batch_size 8 \
    --n_repeats 3 --n_iter 30 --n_warmup 20

# Per-operation breakdown (CUDA-event hooks into every S2Activation.forward, ~3 minutes)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/expG_breakdown.py --batch_size 8
```

All three need a single A100-class GPU. On a shared compute node, check
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv` first;
a contended GPU will distort timing measurements.

## Project layout

```
src/
  equiformer_grid_patch.py     CustomSO3Grid + patch_so3_grid()
  s2_activation.py             Standalone (toy) S2 activation for unit tests
  quadrature_methods.py        GL / Lebedev / uniform / Fibonacci

experiments/
  expA…expE_*.py               Earlier exploratory experiments (paper_bk/ topics)
  expF_equiformerv2_qm9.py     Train EquiformerV2 on QM9 for downstream eval
  expG_*.py                    Quadrature speed / equivariance / breakdown
  verify_gl_grid.py            Numerical sanity check of GL vs DH grids

results/
  expA…expE/                   Earlier results
  expF/runs/                   QM9 training runs (one folder per seed × act × grid)
  expG_quadrature/             Speed benchmarks, Pareto data, README/PDF report

paper/                         Current NeurIPS-style draft (GL-quadrature story)
paper_bk/                      Older collection-of-tests draft

scripts/
  run_expF.sh                  SLURM array launcher for expF
  run_expF_sequential.sh       Single-node interactive launcher
```

## Conventions worth following

- **No retraining for the GL story.** The wall-clock argument is purely
  architectural. Any wall-clock benchmark instantiates a fresh
  `QM9Model(...)` and never loads weights. State this clearly when writing
  about it; do not call it a "trained model" measurement.
- **Don't pseudo-replicate timings.** The proper unit of replication for
  CI / significance is a *run mean* (n = number of independent re-runs),
  not iteration count. Iterations within a run share batches, the
  allocator state, and the model instance, so they are correlated.
- **Don't fill in unmeasured points on Pareto plots.** If a
  configuration's wall-clock is missing, omit it from the figure and say
  so in the caption.
