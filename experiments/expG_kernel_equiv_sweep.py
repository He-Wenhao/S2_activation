"""
Kernel-level equivariance error for the four grid configs at the two
(ℓ_max, m_max) settings used in the paper:
  - (L=4, M=2): QM9-small and OC20-31M architectures
  - (L=6, M=2): fairchem-default architecture

Kernel-level error is a property of the kernel only (single S2-act
forward on random SH coefficients, no model). So one measurement per
(L, M) covers all architectures with that (L, M).

Output: results/expG_quadrature/kernel_equiv_sweep.json
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
torch.serialization.add_safe_globals([slice])
from e3nn import o3

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from src.equiformer_grid_patch import CustomSO3Grid
from fairchem.core.models.equiformer_v2.so3 import SO3_Grid


def equiv_error(grid, lmax, mmax, n_inputs=10, n_rots=5, seed=0):
    """|Act(D@c) - D@Act(c)| / |Act(D@c)|, SiLU activation."""
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

    to_grid = grid.get_to_grid_mat(device=None)
    from_grid = grid.get_from_grid_mat(device=None)

    errors = []
    for ri in range(n_rots):
        g = torch.Generator().manual_seed(seed + ri)
        angles = torch.rand(3, generator=g) * 2 * math.pi
        angles[1] = angles[1] / 2
        D_full = irreps.D_from_angles(angles[0:1], angles[1:2], angles[2:3]).squeeze(0)
        D = D_full[mask][:, mask]

        for ii in range(n_inputs):
            torch.manual_seed(seed + ri * 1000 + ii)
            c = torch.randn(n_keep, 1)
            v = torch.einsum("bai,ic->bac", to_grid, c)
            v = torch.nn.functional.silu(v)
            c_act = torch.einsum("bai,bac->ic", from_grid, v)

            target = D @ c_act
            c_rot = D @ c
            v_rot = torch.einsum("bai,ic->bac", to_grid, c_rot)
            v_rot = torch.nn.functional.silu(v_rot)
            c_act_rot = torch.einsum("bai,bac->ic", from_grid, v_rot)

            errors.append(((target - c_act_rot).norm() / target.norm()).item())

    arr = np.array(errors)
    return float(arr.mean()), float(arr.std()), float(1.96 * arr.std() / math.sqrt(len(arr)))


def main():
    out = {}
    for L, M in [(4, 2), (6, 2)]:
        configs = [
            ("DH default", SO3_Grid(L, M, resolution=None)),
            ("DH 2x",      SO3_Grid(L, M, resolution=4 * (L + 1))),
            ("GL match-DH",CustomSO3Grid(L, M, method="gl",
                                         n_beta=2 * (L + 1), n_alpha=2 * L + 1)),
            ("GL 2x",      CustomSO3Grid(L, M, method="gl",
                                         n_beta=2 * (L + 1), n_alpha=4 * L + 4)),
        ]
        out[f"L{L}_M{M}"] = {}
        print(f"\n=== lmax={L}, mmax={M} ===")
        for label, grid in configs:
            mat = grid.get_to_grid_mat(device=None)
            npts = mat.shape[0] * mat.shape[1]
            mean, std, ci = equiv_error(grid, L, M)
            out[f"L{L}_M{M}"][label] = {
                "n_points": int(npts),
                "equiv_err_mean": mean,
                "equiv_err_std": std,
                "equiv_err_ci95": ci,
            }
            print(f"  {label:<13s}: {npts:>4d} pts, equiv = {mean:.4f} ± {ci:.4f}")

    out_path = _repo_root / "results/expG_quadrature/kernel_equiv_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
