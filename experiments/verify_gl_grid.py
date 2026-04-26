"""
Sanity-check the custom GL SO3_Grid against the default DH grid.

For each (lmax, activation), measure:
  1. Total grid points
  2. Roundtrip error: |from_grid(to_grid(c)) - c| / |c|  on band-limited input
  3. Equivariance error of S2Activation on random input + random rotation

We expect: GL with same point count as DH should have lower equivariance error
because it correctly integrates products of band-limited functions.
"""

import sys
import math
import numpy as np
import torch

torch.serialization.add_safe_globals([slice])

from e3nn import o3

sys.path.insert(0, "/pscratch/sd/w/whe1/S2_activation")
from src.equiformer_grid_patch import CustomSO3Grid

# Import EquiformerV2's SO3_Grid for comparison
from fairchem.core.models.equiformer_v2.so3 import SO3_Grid


def make_random_so3_rotation(seed=0):
    """Returns a Wigner-D matrix for a random SO(3) rotation."""
    g = torch.Generator().manual_seed(seed)
    angles = torch.rand(3, generator=g) * 2 * math.pi
    angles[1] = angles[1] / 2  # beta in [0, pi]
    return angles  # (alpha, beta, gamma)


def equivariance_error(grid, lmax, mmax, n_inputs=20, n_rots=5, seed=0):
    """Measure |Act(D@c) - D@Act(c)| / |Act(D@c)| for SiLU activation."""
    torch.manual_seed(seed)

    irreps = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
    # Restrict to |m| <= mmax: build mask
    l_harm, m_harm = [], []
    for lval in range(lmax + 1):
        for mval in range(-lval, lval + 1):
            l_harm.append(lval)
            m_harm.append(abs(mval))
    l_harm = torch.tensor(l_harm)
    m_harm = torch.tensor(m_harm)
    mask = torch.bitwise_and(l_harm.le(lmax), m_harm.le(mmax))
    n_keep = mask.sum().item()

    to_grid_mat = grid.get_to_grid_mat(device=None)  # [b, a, n_keep]
    from_grid_mat = grid.get_from_grid_mat(device=None)  # [b, a, n_keep]

    errors = []
    for ri in range(n_rots):
        angles = make_random_so3_rotation(seed=seed + ri)
        # Wigner-D for full irreps
        D_full = irreps.D_from_angles(angles[0:1], angles[1:2], angles[2:3]).squeeze(0)
        # Restrict to mask
        D = D_full[mask][:, mask]

        for ii in range(n_inputs):
            torch.manual_seed(seed + ri * 1000 + ii)
            c = torch.randn(n_keep, 1)  # [n_keep, 1 channel]

            # Act(c) using grid
            grid_vals = torch.einsum("bai, ic -> bac", to_grid_mat, c)
            grid_vals = torch.nn.functional.silu(grid_vals)
            c_act = torch.einsum("bai, bac -> ic", from_grid_mat, grid_vals)

            # D @ Act(c)
            target = D @ c_act

            # Act(D @ c)
            c_rot = D @ c
            grid_vals_rot = torch.einsum("bai, ic -> bac", to_grid_mat, c_rot)
            grid_vals_rot = torch.nn.functional.silu(grid_vals_rot)
            c_act_rot = torch.einsum("bai, bac -> ic", from_grid_mat, grid_vals_rot)

            err = (target - c_act_rot).norm() / target.norm()
            errors.append(err.item())

    return float(np.mean(errors)), float(np.std(errors))


def roundtrip_error(grid, lmax, mmax, n_inputs=10, seed=0):
    """Measure |from_grid(to_grid(c)) - c| / |c| for band-limited inputs."""
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

    torch.manual_seed(seed)
    errs = []
    for _ in range(n_inputs):
        c = torch.randn(n_keep, 1)
        g = torch.einsum("bai, ic -> bac", to_grid_mat, c)
        c2 = torch.einsum("bai, bac -> ic", from_grid_mat, g)
        errs.append(((c - c2).norm() / c.norm()).item())
    return float(np.mean(errs)), float(np.std(errs))


def run_table(LMAX, MMAX):
    print(f"\n=== Testing lmax={LMAX}, mmax={MMAX} ===\n")

    configs = [
        ("DH default (e3nn)",  SO3_Grid(LMAX, MMAX, resolution=None)),
        ("DH 2x (e3nn)",       SO3_Grid(LMAX, MMAX, resolution=4 * (LMAX + 1))),
        ("DH min (custom)",    CustomSO3Grid(LMAX, MMAX, method="dh",
                                              n_beta=2 * (LMAX + 1),
                                              n_alpha=2 * LMAX + 1)),
        ("GL min",             CustomSO3Grid(LMAX, MMAX, method="gl",
                                              n_beta=LMAX + 1,
                                              n_alpha=2 * LMAX + 1)),
        ("GL match-DH",        CustomSO3Grid(LMAX, MMAX, method="gl",
                                              n_beta=2 * (LMAX + 1),
                                              n_alpha=2 * LMAX + 1)),
        ("GL 2x",              CustomSO3Grid(LMAX, MMAX, method="gl",
                                              n_beta=2 * (LMAX + 1),
                                              n_alpha=4 * LMAX + 4)),
        ("GL high",            CustomSO3Grid(LMAX, MMAX, method="gl",
                                              n_beta=4 * (LMAX + 1),
                                              n_alpha=4 * LMAX + 4)),
    ]

    print(f"{'Config':<25s} {'Points':>8s} {'Roundtrip':>12s} {'Equiv err':>14s}")
    print("-" * 65)
    for label, grid in configs:
        m = grid.get_to_grid_mat(device=None)
        npts = m.shape[0] * m.shape[1]
        rt_mean, _ = roundtrip_error(grid, LMAX, MMAX)
        eq_mean, _ = equivariance_error(grid, LMAX, MMAX)
        print(f"{label:<25s} {npts:>8d} {rt_mean:>12.2e} {eq_mean:>14.2e}")


def main():
    # Real EquiformerV2 config: lossy mmax cropping
    run_table(4, 2)
    # Pure quadrature comparison: no mmax cropping
    run_table(4, 4)
    run_table(6, 6)


if __name__ == "__main__":
    main()
