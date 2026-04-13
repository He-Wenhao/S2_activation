# Experiments

This directory contains six experiments (A–F) that systematically study the effect
of pointwise activation functions on equivariance in the S2 Activation operator
used by equivariant neural networks (e.g., EquiformerV2).

---

## Experiment A — Spectral Leakage (`expA_spectral_leakage.py`)

**Question:** How much energy does each activation leak beyond the band limit l_max?

Applies each activation pointwise on the sphere and measures how much of the output
power spectrum falls above l_max. Activations are compared at l_max = 3, 6, 10
against a high-resolution (l_max_ref = 25) ground truth.

- **Metrics:** power spectrum P(l), leakage ratio R, effective bandwidth l_eff
- **Activations:** ReLU, |x|, LeakyReLU(0.1), SiLU, GELU, tanh, Softplus(β = 1, 3, 10, 30, 100), x², sin
- **Output:** `results/metrics/expA_spectral.json`, plots in `results/figures/`

---

## Experiment B — Coefficient Error (`expB_coefficient_error.py`)

**Question:** How does the full S2Activation pipeline's coefficient error decompose
into truncation error vs. aliasing error, and how does sampling strategy affect each?

Compares Gauss–Legendre (1×, 2×, 3× oversampling), Lebedev, and uniform grids.

- **Metrics:** relative coefficient error, truncation ratio, per-degree breakdown
- **Activations:** ReLU, |x|, SiLU, GELU, tanh, Softplus(β = 1, 10, 100)
- **l_max:** 3, 6, 10
- **Output:** `results/metrics/expB_coefficient_error.json`, plots in `results/figures/`

---

## Experiment C — Equivariance Error (`expC_equivariance_error.py`)

**Question:** How well does S2Activation commute with SO(3) rotations?

Measures ||S2Act(D·x) − D·S2Act(x)|| / ||S2Act(D·x)|| over random Wigner-D
rotations and inputs. Also plots equivariance error vs. spectral leakage to show
that leakage is a strong predictor of equivariance violation.

- **Metrics:** equivariance error (mean ± std), Pearson correlation with leakage
- **Activations:** ReLU, |x|, SiLU, GELU, tanh, Softplus(β = 1, 10, 100)
- **l_max:** 3, 6, 10;  20 rotations × 10 inputs per config
- **Sampling:** GL 1×/2×/3×, Lebedev, Uniform
- **Output:** `results/metrics/expC_equivariance.json`, plots in `results/figures/`

---

## Experiment D — Synthetic Task Performance (`expD_task_performance.py`)

**Question:** Does spectral leakage / equivariance error predict downstream task accuracy?

Trains a SphericalCNN classifier on a synthetic 5-class spherical dataset whose
classes are distinguished at high-l degrees, stress-testing S2Activation.

- **Metrics:** test accuracy, best validation accuracy, training time
- **Activations:** ReLU, SiLU, tanh, Softplus(β = 1, 10)
- **Setup:** l_max = 6, 15 epochs, 2 runs per config
- **Sampling:** GL 1×/2×, Lebedev
- **Output:** `results/metrics/expD_task.json`, plots in `results/figures/`

---

## Experiment E — Jacobian Isotropy / Expressibility (`expE_expressibility.py`)

**Question:** How many independent output directions can S2Activation explore locally?

Computes the Jacobian of S2Activation and measures its effective rank
(exp(entropy of normalised singular values)). High effective rank means the
activation preserves more of the input signal's diversity within the band limit.

- **Metrics:** effective rank, stable rank, spectral gap, Frobenius norm
- **Activations:** ReLU, |x|, SiLU, GELU, tanh, Softplus(β = 1, 10, 100)
- **l_max:** 3, 6, 10;  50 inputs per config
- **Output:** `results/metrics/expE_expressibility.json`, plots in `results/figures/`

---

## Experiment F — EquiformerV2 on QM9 (`expF_equiformerv2_qm9.py`)

**Question:** Do the theoretical findings (leakage, equivariance, isotropy) transfer
to a real molecular property-prediction task?

Swaps the S2 Activation nonlinearity inside EquiformerV2 and trains on QM9 (U0
target by default). Also measures per-layer equivariance error post-training.

- **Metrics:** test MAE, per-layer prediction invariance error, training history
- **Activations:** SiLU, Softplus(β = 1, 10), tanh, ReLU, |x|
- **Grid resolutions:** default, 2×, 3× (for SiLU and Softplus_1)
- **Setup:** l_max = 4, 50 epochs, 5 seeds per config, AdamW + cosine annealing
- **Output:** `results/expF/runs/{config}/results.json`, `results/expF/summary.json`

### Running Experiment F

```bash
# Single run
python experiments/expF_equiformerv2_qm9.py train \
    --act SiLU --grid default --target U0 --seed 42

# Full sweep (50 configs) via SLURM array
sbatch --array=0-49 scripts/run_expF.sh

# Aggregate results after all runs complete
python experiments/expF_equiformerv2_qm9.py aggregate
```
