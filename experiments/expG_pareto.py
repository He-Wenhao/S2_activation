"""
Combined Pareto benchmark: equivariance error vs wall-clock time, measured
at the SAME (lmax, mmax) configuration so the comparison is apples-to-apples.

For each grid configuration we report:
  (a) equivariance error of one S2-Activation kernel (random coefficients,
      averaged over 5 rotations × 10 inputs)
  (b) wall-clock per full EquiformerV2 forward pass on a real QM9 batch

We then identify, for each method (DH and GL), the cheapest config that
reaches each equivariance level, and compute the time savings of GL
relative to DH at matched accuracy.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "experiments"))

from src.equiformer_grid_patch import patch_so3_grid, CustomSO3Grid
from expF_equiformerv2_qm9 import (
    QM9Model, BACKBONE_DEFAULTS, patch_s2_activations, load_qm9, qm9_adapt,
)
from torch_geometric.loader import DataLoader
from fairchem.core.models.equiformer_v2.so3 import SO3_Grid


# ─── Equivariance error (S2 Activation kernel only) ────────────────────────

def equiv_error(grid, lmax, mmax, n_inputs=10, n_rots=5, seed=0):
    torch.manual_seed(seed)
    irreps = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
    l_harm, m_harm = [], []
    for lval in range(lmax + 1):
        for mval in range(-lval, lval + 1):
            l_harm.append(lval)
            m_harm.append(abs(mval))
    l_harm = torch.tensor(l_harm)
    m_harm = torch.tensor(m_harm)
    mask = torch.bitwise_and(l_harm.le(lmax), m_harm.le(mmax))
    n_keep = mask.sum().item()

    to_grid_mat = grid.get_to_grid_mat(device=None)
    from_grid_mat = grid.get_from_grid_mat(device=None)

    errors = []
    for ri in range(n_rots):
        torch.manual_seed(seed + ri)
        angles = torch.rand(3) * 2 * math.pi
        angles[1] = angles[1] / 2
        D_full = irreps.D_from_angles(angles[0:1], angles[1:2], angles[2:3]).squeeze(0)
        D = D_full[mask][:, mask]

        for ii in range(n_inputs):
            torch.manual_seed(seed + ri * 1000 + ii)
            c = torch.randn(n_keep, 1)
            grid_vals = torch.einsum("bai, ic -> bac", to_grid_mat, c)
            grid_vals = torch.nn.functional.silu(grid_vals)
            c_act = torch.einsum("bai, bac -> ic", from_grid_mat, grid_vals)
            target = D @ c_act

            c_rot = D @ c
            grid_vals_rot = torch.einsum("bai, ic -> bac", to_grid_mat, c_rot)
            grid_vals_rot = torch.nn.functional.silu(grid_vals_rot)
            c_act_rot = torch.einsum("bai, bac -> ic", from_grid_mat, grid_vals_rot)

            err = (target - c_act_rot).norm() / target.norm()
            errors.append(err.item())

    return float(np.mean(errors)), float(np.std(errors))


# ─── Full forward wall-clock ───────────────────────────────────────────────

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_forward(grid_config, batches, backbone_kwargs, device,
                  n_warmup=20, n_iter=30):
    backbone = dict(backbone_kwargs)
    if grid_config["method"] == "dh" and grid_config.get("resolution"):
        backbone["grid_resolution"] = grid_config["resolution"]
    model = QM9Model(backbone).to(device)
    patch_s2_activations(model, "SiLU")
    if grid_config["method"] == "gl":
        patch_so3_grid(model, method="gl",
                       n_beta=grid_config.get("n_beta"),
                       n_alpha=grid_config.get("n_alpha"))
    model.eval()
    iter_idx = [0]

    def fwd():
        with torch.no_grad():
            _ = model(batches[iter_idx[0] % len(batches)])
        iter_idx[0] += 1

    for _ in range(n_warmup):
        fwd()
    cuda_sync()
    times = []
    for _ in range(n_iter):
        cuda_sync()
        t0 = time.perf_counter()
        fwd()
        cuda_sync()
        times.append((time.perf_counter() - t0) * 1000)

    del model
    torch.cuda.empty_cache()
    times = np.array(times)
    return float(times.mean()), float(times.std()), 1.96 * float(times.std()) / math.sqrt(len(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_iter", type=int, default=30)
    p.add_argument("--n_warmup", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # OC20-scale config
    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    backbone_kwargs["num_layers"] = 12
    backbone_kwargs["sphere_channels"] = 128
    backbone_kwargs["attn_hidden_channels"] = 128
    backbone_kwargs["ffn_hidden_channels"] = 256
    backbone_kwargs["num_heads"] = 8
    backbone_kwargs["lmax_list"] = [6]
    backbone_kwargs["mmax_list"] = [2]
    LMAX, MMAX = 6, 2

    # Pre-fetch batches
    dataset, target_idx, train_idx, _, _ = load_qm9("U0", seed=42)
    loader = DataLoader(dataset[train_idx[: args.batch_size * 4]],
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    batches = []
    for b in loader:
        b = b.to(device)
        b, _ = qm9_adapt(b, target_idx)
        batches.append(b)
    batches = batches[:3]

    configs = [
        ("DH default",   {"method": "dh", "resolution": None}),
        ("DH 2x",        {"method": "dh", "resolution": 4 * (LMAX + 1)}),
        ("GL min",       {"method": "gl", "n_beta": LMAX + 1,    "n_alpha": 2 * LMAX + 1}),
        ("GL match-DH",  {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 2 * LMAX + 1}),
        ("GL 2x",        {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 4 * LMAX + 4}),
    ]

    results = []
    print(f"\n{'Config':<13s} {'Pts':>5s} {'Equiv err':>11s} {'Fwd (ms)':>11s}")
    print("-" * 50)
    for label, cfg in configs:
        # Build grid for equivariance test
        if cfg["method"] == "dh":
            grid = SO3_Grid(LMAX, MMAX, resolution=cfg.get("resolution"))
        else:
            grid = CustomSO3Grid(LMAX, MMAX, method="gl",
                                  n_beta=cfg.get("n_beta"),
                                  n_alpha=cfg.get("n_alpha"))
        npts = grid.get_to_grid_mat(device=None).shape[0] * \
               grid.get_to_grid_mat(device=None).shape[1]
        equiv_mean, equiv_std = equiv_error(grid, LMAX, MMAX)
        del grid

        # Wall-clock benchmark
        torch.cuda.empty_cache()
        fwd_mean, fwd_std, fwd_ci = time_forward(
            cfg, batches, backbone_kwargs, device,
            n_warmup=args.n_warmup, n_iter=args.n_iter,
        )

        results.append({
            "label": label, "config": cfg, "n_points": npts,
            "equiv_err_mean": equiv_mean, "equiv_err_std": equiv_std,
            "fwd_ms_mean": fwd_mean, "fwd_ms_std": fwd_std,
            "fwd_ms_ci95": fwd_ci,
        })
        print(f"{label:<13s} {npts:>5d} {equiv_mean:>11.3e} "
              f"{fwd_mean:>7.2f} ± {fwd_ci:.2f}")

    # Pareto analysis: for each method, sort by points, find time savings
    print(f"\n{'='*65}")
    print("PARETO ANALYSIS: cheapest config to reach each equivariance level")
    print("="*65)
    # All distinct equiv targets (rounded)
    equiv_targets = sorted({round(r["equiv_err_mean"], 3) for r in results})
    print(f"{'Target equiv':<14s} {'Cheapest DH':<25s} {'Cheapest GL':<25s} {'Savings':>10s}")
    print("-" * 75)
    for tgt in equiv_targets:
        dh_candidates = [r for r in results if r["config"]["method"] == "dh"
                          and r["equiv_err_mean"] <= tgt + 1e-6]
        gl_candidates = [r for r in results if r["config"]["method"] == "gl"
                          and r["equiv_err_mean"] <= tgt + 1e-6]
        dh_best = min(dh_candidates, key=lambda r: r["fwd_ms_mean"]) if dh_candidates else None
        gl_best = min(gl_candidates, key=lambda r: r["fwd_ms_mean"]) if gl_candidates else None
        if dh_best and gl_best:
            savings = (dh_best["fwd_ms_mean"] - gl_best["fwd_ms_mean"]) / dh_best["fwd_ms_mean"] * 100
            print(f"{tgt:<14.3f} {dh_best['label'] + f' ({dh_best['fwd_ms_mean']:.1f} ms)':<25s} "
                  f"{gl_best['label'] + f' ({gl_best['fwd_ms_mean']:.1f} ms)':<25s} "
                  f"{savings:>9.1f}%")

    out_path = Path("results/expG_quadrature/pareto.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": "OC20-scale: 12 layers, 128 channels, lmax=6, mmax=2",
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
