"""
Four-way S2-activation kernel benchmark: DH-dense / DH-FFT / GL / Lebedev.

EquiformerV2's `S2Activation.forward` is

    x_grid = einsum("bai, zic -> zbac", to_grid_mat, inputs)
    x_grid = silu(x_grid)
    return einsum("bai, zbac -> zic", from_grid_mat, x_grid)

We compare four implementations:

  1. **DH-dense** — current EquiformerV2 default. Precomputed dense
     to/from-grid matrices via e3nn's equiangular Driscoll-Healy quadrature.
     Single einsum.
  2. **DH-FFT** — e3nn's ToS2Grid/FromS2Grid forward path with a grid that
     satisfies the FFT condition (n_alpha odd, n_alpha >= 2*lmax+1).
     Beta direction is still a dense matmul; alpha direction uses
     torch.fft.irfft / torch.fft.rfft.
  3. **GL** — Gauss-Legendre latitude × uniform longitude, dense matmul
     (no FFT structure for GL nodes).
  4. **Lebedev** — non-tensor-product Lebedev quadrature, dense matmul
     over a flat list of points.

For each method we report:
  - equivariance error (kernel-level, random coefficients)
  - per-call wall-clock (cuda.sync timing, n=5 runs of 30 forwards each
    on independent random inputs)
  - total grid points

The benchmark configures lmax=6, mmax=2 to match the headline.

Outputs: results/expG_quadrature/four_way.json
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

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

from e3nn import o3
from e3nn.o3 import ToS2Grid, FromS2Grid
from src.equiformer_grid_patch import (
    CustomSO3Grid, _real_sh_at_points, _angles_to_xyz,
    _lebedev_min_degree_for_lmax,
)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def t_critical(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(df, 1.96)


# ─── Method wrappers exposing a forward(inputs) interface ─────────────────

class _DenseGrid:
    """Wraps a CustomSO3Grid (or fairchem SO3_Grid) so .forward(x) does the
    standard dense einsum."""
    def __init__(self, to_mat, from_mat, name, n_points):
        self.to_mat = to_mat
        self.from_mat = from_mat
        self.name = name
        self.n_points = n_points

    def forward(self, x):
        # x: [batch, n_coeffs, channels]
        g = torch.einsum("bai, zic -> zbac", self.to_mat, x)
        g = torch.nn.functional.silu(g)
        return torch.einsum("bai, zbac -> zic", self.from_mat, g)


class _DHFastGrid:
    """e3nn-style ToS2Grid + FromS2Grid forward with FFT path enabled."""
    def __init__(self, lmax, mmax, n_beta, n_alpha, device, dtype):
        assert n_alpha >= 2 * lmax + 1, "alpha must be >= 2*lmax+1 for FFT"
        assert n_alpha % 2 == 1, "alpha must be odd for the e3nn FFT branch"
        self.to_grid = ToS2Grid(lmax, (n_beta, n_alpha),
                                  normalization="integral", dtype=dtype)
        self.from_grid = FromS2Grid((n_beta, n_alpha), lmax,
                                     normalization="integral", dtype=dtype)
        self.to_grid = self.to_grid.to(device)
        self.from_grid = self.from_grid.to(device)
        self.lmax = lmax
        self.mmax = mmax
        # Precompute mmax-cropping indices: full (lmax+1)^2 -> kept indices
        l_harm, m_harm = [], []
        for lv in range(lmax + 1):
            for mv in range(-lv, lv + 1):
                l_harm.append(lv)
                m_harm.append(abs(mv))
        l_harm = torch.tensor(l_harm)
        m_harm = torch.tensor(m_harm)
        mask = torch.bitwise_and(l_harm.le(lmax), m_harm.le(mmax))
        self.idx_keep = torch.arange(len(mask))[mask].to(device)
        self.n_full = (lmax + 1) ** 2
        self.name = "DH-FFT"
        self.n_points = n_beta * n_alpha

        # Apply the same mmax rescaling as fairchem SO3_Grid (lines 537-546).
        # Vectorize as a single per-coefficient gain vector.
        gain = torch.ones(self.n_full, device=device, dtype=dtype)
        for lv in range(lmax + 1):
            if lv <= mmax:
                continue
            length = 2 * lv + 1
            factor = math.sqrt(length / (2 * mmax + 1))
            start, end = lv ** 2, lv ** 2 + 2 * lv + 1
            gain[start:end] = factor
        # Apply gain only on kept indices for in-place efficiency
        self.gain_in = gain.unsqueeze(0).unsqueeze(-1)  # [1, n_full, 1]
        self.gain_out = gain.unsqueeze(0).unsqueeze(-1)  # [1, n_full, 1]

    def forward(self, x):
        # x: [batch, n_kept, channels] — fairchem convention
        B, _, C = x.shape
        # Inflate to full (lmax+1)^2 in one scatter
        x_full = x.new_zeros(B, self.n_full, C)
        x_full[:, self.idx_keep, :] = x
        x_full = x_full * self.gain_in  # rescale l>mmax blocks
        # ToS2Grid expects [..., (lmax+1)^2]; channel as a leading axis is fine
        x_lead = x_full.permute(0, 2, 1).contiguous()  # [B, C, M]
        g = self.to_grid(x_lead)                        # [B, C, beta, alpha]
        g = torch.nn.functional.silu(g)
        out_full = self.from_grid(g).permute(0, 2, 1).contiguous()  # [B, M, C]
        out_full = out_full * self.gain_out
        return out_full[:, self.idx_keep, :]


# ─── Equivariance test ────────────────────────────────────────────────────

def equivariance_error(grid_obj, lmax, mmax, n_inputs=10, n_rots=5,
                       device="cuda", seed=0):
    torch.manual_seed(seed)
    irreps = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
    l_harm, m_harm = [], []
    for lv in range(lmax + 1):
        for mv in range(-lv, lv + 1):
            l_harm.append(lv)
            m_harm.append(abs(mv))
    l_harm = torch.tensor(l_harm)
    m_harm = torch.tensor(m_harm)
    mask = torch.bitwise_and(l_harm.le(lmax), m_harm.le(mmax))
    n_keep = mask.sum().item()

    errors = []
    for ri in range(n_rots):
        torch.manual_seed(seed + ri)
        angles = torch.rand(3) * 2 * math.pi
        angles[1] = angles[1] / 2
        D_full = irreps.D_from_angles(
            angles[0:1], angles[1:2], angles[2:3]
        ).squeeze(0)
        D = D_full[mask][:, mask].to(device)

        for ii in range(n_inputs):
            torch.manual_seed(seed + ri * 1000 + ii)
            c = torch.randn(1, n_keep, 1, device=device)  # [1, n_kept, 1ch]
            c_act = grid_obj.forward(c)
            target = (D @ c_act.view(n_keep, -1)).view(1, n_keep, -1)

            c_rot_in = (D @ c.view(n_keep, -1)).view(1, n_keep, -1)
            c_act_rot = grid_obj.forward(c_rot_in)

            err = (target - c_act_rot).norm() / target.norm()
            errors.append(err.item())
    return float(np.mean(errors)), float(np.std(errors))


# ─── Kernel timing ────────────────────────────────────────────────────────

def time_kernel(grid_obj, n_kept, n_channels=128, batch=1024,
                n_warmup=20, n_iter=30, n_runs=5, device="cuda", seed=0):
    """Time grid_obj.forward(x) on independent random inputs.

    Each run uses fresh random inputs; CI is over n_runs run-means."""
    run_means = []
    for run_id in range(n_runs):
        torch.manual_seed(seed + 1000 * run_id + 1)
        # Pre-generate enough inputs for warmup + measurement (independent)
        n_total = n_warmup + n_iter
        xs = [torch.randn(batch, n_kept, n_channels, device=device)
              for _ in range(n_total)]

        # Warmup
        with torch.no_grad():
            for i in range(n_warmup):
                grid_obj.forward(xs[i])
        cuda_sync()

        # Measure
        times = []
        with torch.no_grad():
            for i in range(n_warmup, n_total):
                cuda_sync()
                t0 = time.perf_counter()
                grid_obj.forward(xs[i])
                cuda_sync()
                times.append((time.perf_counter() - t0) * 1000)
        run_means.append(float(np.mean(times)))

    arr = np.array(run_means)
    mean = arr.mean()
    se = arr.std(ddof=1) / math.sqrt(len(arr))
    ci95 = t_critical(len(arr) - 1) * se
    return {"mean_ms": mean, "se_ms": se, "ci95_ms": ci95,
            "run_means": run_means}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmax", type=int, default=6)
    p.add_argument("--mmax", type=int, default=2)
    p.add_argument("--batch", type=int, default=1024,
                    help="Token-batch size for the kernel benchmark "
                         "(roughly the number of edges/atoms passed through "
                         "S2Activation per forward pass)")
    p.add_argument("--n_channels", type=int, default=128)
    p.add_argument("--n_warmup", type=int, default=20)
    p.add_argument("--n_iter", type=int, default=30)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--out", type=str,
                    default="results/expG_quadrature/four_way.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = "cpu"
    print(f"GPU: {gpu}")
    print(f"Setup: lmax={args.lmax}, mmax={args.mmax}, batch={args.batch}, "
          f"channels={args.n_channels}\n")

    # Configurations to benchmark
    L = args.lmax
    M = args.mmax

    configs = []

    # 1) DH-dense default (e3nn equiangular at minimum density)
    g_dh_default = CustomSO3Grid(L, M, method="dh",
                                   n_beta=2 * (L + 1), n_alpha=2 * M + 1)
    g_dh_default.to(device)
    configs.append(("DH-dense default",
                    _DenseGrid(g_dh_default.get_to_grid_mat(),
                               g_dh_default.get_from_grid_mat(),
                               "DH-dense default",
                               g_dh_default.n_beta * g_dh_default.n_alpha)))

    # 2) DH-dense at the SAME grid as DH-FFT (apples-to-apples)
    n_alpha_fft = 2 * L + 1  # smallest odd value that triggers e3nn FFT
    if n_alpha_fft % 2 == 0:
        n_alpha_fft += 1
    g_dh_match = CustomSO3Grid(L, M, method="dh",
                                 n_beta=2 * (L + 1), n_alpha=n_alpha_fft)
    g_dh_match.to(device)
    configs.append(("DH-dense match",
                    _DenseGrid(g_dh_match.get_to_grid_mat(),
                               g_dh_match.get_from_grid_mat(),
                               "DH-dense match",
                               g_dh_match.n_beta * g_dh_match.n_alpha)))

    # 3) DH-FFT — e3nn forward path, n_alpha odd & >= 2*lmax+1 to trigger FFT
    g_dhfft = _DHFastGrid(L, M, n_beta=2 * (L + 1), n_alpha=n_alpha_fft,
                           device=device, dtype=dtype)
    g_dhfft.n_points = 2 * (L + 1) * n_alpha_fft
    configs.append(("DH-FFT", g_dhfft))

    # 4) GL match-DH (same n_beta, same n_alpha as DH-FFT for fair comparison)
    g_gl = CustomSO3Grid(L, M, method="gl",
                          n_beta=2 * (L + 1), n_alpha=n_alpha_fft)
    g_gl.to(device)
    configs.append(("GL match-DH",
                    _DenseGrid(g_gl.get_to_grid_mat(), g_gl.get_from_grid_mat(),
                               "GL match-DH", g_gl.n_beta * g_gl.n_alpha)))

    # 5) Lebedev (smallest rule that integrates products of degree-lmax exactly)
    deg_leb = _lebedev_min_degree_for_lmax(L)
    g_leb = CustomSO3Grid(L, M, method="lebedev", lebedev_degree=deg_leb)
    g_leb.to(device)
    configs.append(("Lebedev (low)",
                    _DenseGrid(g_leb.get_to_grid_mat(), g_leb.get_from_grid_mat(),
                               f"Lebedev d={deg_leb}",
                               g_leb.n_beta * g_leb.n_alpha)))

    # 6) Lebedev (high — to see if more nodes help equivariance)
    deg_leb_high = max(deg_leb + 8, 21)  # several rule levels above minimum
    g_leb_hi = CustomSO3Grid(L, M, method="lebedev",
                               lebedev_degree=deg_leb_high)
    g_leb_hi.to(device)
    configs.append((f"Lebedev (d={deg_leb_high})",
                    _DenseGrid(g_leb_hi.get_to_grid_mat(),
                               g_leb_hi.get_from_grid_mat(),
                               f"Lebedev d={deg_leb_high}",
                               g_leb_hi.n_beta * g_leb_hi.n_alpha)))

    # ─── Run benchmark ───
    print(f"{'Method':<20s} {'Pts':>5s}  {'Equiv err':>11s}  "
          f"{'Kernel (ms, 95% CI)':>24s}")
    print("-" * 70)
    rows = []
    for label, gobj in configs:
        eq_mean, eq_std = equivariance_error(gobj, L, M, device=device)
        timing = time_kernel(gobj, n_kept=g_dh_default.get_to_grid_mat().shape[-1],
                              n_channels=args.n_channels, batch=args.batch,
                              n_warmup=args.n_warmup, n_iter=args.n_iter,
                              n_runs=args.n_runs, device=device)
        n_pts = getattr(gobj, "n_points", "?")
        print(f"{label:<20s} {n_pts:>5d}  {eq_mean:>11.3e}  "
              f"{timing['mean_ms']:>9.3f} ± {timing['ci95_ms']:.3f}")
        rows.append({
            "method": label, "n_points": n_pts,
            "equiv_err_mean": eq_mean, "equiv_err_std": eq_std,
            "kernel_ms_mean": timing["mean_ms"],
            "kernel_ms_ci95": timing["ci95_ms"],
            "kernel_ms_run_means": timing["run_means"],
        })
        del gobj
        torch.cuda.empty_cache()

    out = {
        "args": vars(args),
        "device": gpu,
        "lmax": L, "mmax": M,
        "results": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
