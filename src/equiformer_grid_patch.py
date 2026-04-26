"""
Custom SO3_Grid implementations using alternative quadrature methods
(Gauss-Legendre, Lebedev) as drop-in replacements for EquiformerV2's
default Driscoll-Healy/equiangular grid.

We construct `to_grid_mat` and `from_grid_mat` by direct evaluation of
real spherical harmonics in e3nn's "integral" convention (unit L^2 norm)
on the chosen quadrature nodes, and pair them with the corresponding
quadrature weights.
"""

import math
import numpy as np
import torch
from e3nn import o3
from scipy.special import roots_legendre


def _angles_to_xyz(beta: torch.Tensor, alpha: torch.Tensor):
    """
    e3nn's spherical-harmonic convention:
      x = sin(beta) sin(alpha)
      y = cos(beta)
      z = sin(beta) cos(alpha)
    """
    sb = beta.sin()[:, None]    # [N_beta, 1]
    cb = beta.cos()[:, None]    # [N_beta, 1]
    sa = alpha.sin()[None, :]   # [1, N_alpha]
    ca = alpha.cos()[None, :]   # [1, N_alpha]

    x = sb * sa                 # [N_beta, N_alpha]
    y = cb.expand_as(x)
    z = sb * ca
    return torch.stack([x, y, z], dim=-1)  # [N_beta, N_alpha, 3]


def _real_sh_at_points(lmax: int, points_xyz: torch.Tensor):
    """
    Evaluate real spherical harmonics Y_l^m for l=0..lmax in e3nn's
    "integral" normalization (unit L^2 norm) at given Cartesian points.

    Returns: tensor of shape [..., (lmax+1)^2]
    """
    # e3nn's o3.spherical_harmonics returns shape [..., (lmax+1)^2] with
    # ordering (l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1), (l=2,m=-2), ...
    irreps = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
    # Use normalize=True to project onto unit sphere; normalization='integral'
    # gives unit L^2 norm.
    Y = o3.spherical_harmonics(
        irreps, points_xyz, normalize=True, normalization="integral"
    )
    return Y  # [..., (lmax+1)^2]


def _build_grid_matrices(
    betas: torch.Tensor,        # (N_beta,)
    alphas: torch.Tensor,       # (N_alpha,)
    qw_beta: torch.Tensor,      # (N_beta,)  weights for cos(beta) (sum=2)
    qw_alpha: torch.Tensor,     # (N_alpha,) weights for alpha (sum=2*pi)
    lmax: int,
    dtype=torch.float32,
    device="cpu",
):
    """
    Construct EquiformerV2-compatible to_grid_mat / from_grid_mat using
    direct SH evaluation. Both have shape [N_beta, N_alpha, (lmax+1)^2].

    Math:
      to_grid_mat[b, a, i]   = Y_i(beta_b, alpha_a)
      from_grid_mat[b, a, i] = qw_beta[b] * qw_alpha[a] * Y_i(beta_b, alpha_a)

    For band-limited f = sum_i c_i Y_i, we have
      f(b, a) = sum_i c_i to_grid_mat[b,a,i]
      sum_{b,a} from_grid_mat[b,a,i] f(b,a) = sum_i c_i  *  delta_{ii'}  = c_i
    when the quadrature is exact for products Y_i Y_{i'} of degree up to 2*lmax.
    """
    betas = betas.to(dtype=dtype, device=device)
    alphas = alphas.to(dtype=dtype, device=device)
    qw_beta = qw_beta.to(dtype=dtype, device=device)
    qw_alpha = qw_alpha.to(dtype=dtype, device=device)

    points = _angles_to_xyz(betas, alphas).to(dtype=dtype)  # [b, a, 3]
    Y = _real_sh_at_points(lmax, points)                    # [b, a, (lmax+1)^2]

    to_grid_mat = Y.contiguous()

    weight = qw_beta[:, None] * qw_alpha[None, :]            # [b, a]
    from_grid_mat = (Y * weight[:, :, None]).contiguous()

    return to_grid_mat, from_grid_mat


def gauss_legendre_grid_matrices(lmax: int, n_beta: int = None,
                                  n_alpha: int = None,
                                  dtype=torch.float32, device="cpu"):
    """Gauss-Legendre latitude × uniform longitude tensor-product grid."""
    if n_beta is None:
        n_beta = lmax + 1
    if n_alpha is None:
        n_alpha = 2 * lmax + 1

    cos_beta, w_cos = roots_legendre(n_beta)  # nodes in [-1,1], weights sum to 2
    betas = torch.from_numpy(np.arccos(cos_beta)).to(dtype=dtype)
    qw_beta = torch.from_numpy(w_cos).to(dtype=dtype)

    alphas = torch.arange(n_alpha, dtype=dtype) * (2 * math.pi / n_alpha)
    qw_alpha = torch.full((n_alpha,), 2 * math.pi / n_alpha, dtype=dtype)

    return _build_grid_matrices(betas, alphas, qw_beta, qw_alpha, lmax,
                                  dtype=dtype, device=device)


def driscoll_healy_grid_matrices(lmax: int, n_beta: int = None,
                                  n_alpha: int = None,
                                  dtype=torch.float32, device="cpu"):
    """
    Equiangular Driscoll-Healy grid (matches EquiformerV2 default mathematically).
    Useful for sanity checks against e3nn's implementation.
    """
    if n_beta is None:
        n_beta = 2 * (lmax + 1)
    if n_alpha is None:
        n_alpha = 2 * lmax + 1

    # DH equiangular nodes: beta_j = (j+0.5)/N * pi
    j = torch.arange(n_beta, dtype=dtype)
    betas = (j + 0.5) / n_beta * math.pi
    # DH weights for sin(beta) integration (sum to 2)
    # w_j = (2/N) * sin(beta_j) * Σ_{k=0}^{N/2-1} (1/(2k+1)) * sin((2j+1)(2k+1) pi / (2N))
    N = n_beta
    qw_beta = torch.zeros(N, dtype=dtype)
    for jj in range(N):
        s = 0.0
        for k in range(N // 2):
            s += (1.0 / (2 * k + 1)) * math.sin((2 * jj + 1) * (2 * k + 1) * math.pi / (2 * N))
        qw_beta[jj] = (2.0 / N) * math.sin((jj + 0.5) / N * math.pi) * s
    # Renormalize so weights sum exactly to 2 (∫_0^π sin β dβ = 2)
    qw_beta = qw_beta * (2.0 / qw_beta.sum())

    alphas = torch.arange(n_alpha, dtype=dtype) * (2 * math.pi / n_alpha)
    qw_alpha = torch.full((n_alpha,), 2 * math.pi / n_alpha, dtype=dtype)

    return _build_grid_matrices(betas, alphas, qw_beta, qw_alpha, lmax,
                                  dtype=dtype, device=device)


class CustomSO3Grid(torch.nn.Module):
    """
    Drop-in replacement for fairchem's SO3_Grid. Same `get_to_grid_mat` /
    `get_from_grid_mat` interface; configurable quadrature method.
    """

    def __init__(self, lmax: int, mmax: int, method: str = "gl",
                 n_beta: int = None, n_alpha: int = None):
        super().__init__()
        self.lmax = lmax
        self.mmax = mmax
        self.method = method

        if method == "gl":
            to_grid_mat, from_grid_mat = gauss_legendre_grid_matrices(
                lmax, n_beta=n_beta, n_alpha=n_alpha
            )
        elif method == "dh":
            to_grid_mat, from_grid_mat = driscoll_healy_grid_matrices(
                lmax, n_beta=n_beta, n_alpha=n_alpha
            )
        else:
            raise ValueError(f"Unknown quadrature method: {method}")

        # Apply mmax rescaling for l > mmax (matches SO3_Grid)
        if lmax != mmax:
            for lval in range(lmax + 1):
                if lval <= mmax:
                    continue
                start_idx = lval ** 2
                length = 2 * lval + 1
                rescale_factor = math.sqrt(length / (2 * mmax + 1))
                to_grid_mat[:, :, start_idx : start_idx + length] *= rescale_factor
                from_grid_mat[:, :, start_idx : start_idx + length] *= rescale_factor

        # Crop columns to keep only |m| <= mmax
        l_harm, m_harm = [], []
        for lval in range(lmax + 1):
            for mval in range(-lval, lval + 1):
                l_harm.append(lval)
                m_harm.append(abs(mval))
        l_harm = torch.tensor(l_harm)
        m_harm = torch.tensor(m_harm)
        mask = torch.bitwise_and(l_harm.le(lmax), m_harm.le(mmax))
        idx = torch.arange(len(mask))[mask]

        to_grid_mat = to_grid_mat[:, :, idx]
        from_grid_mat = from_grid_mat[:, :, idx]

        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)
        self.n_beta = to_grid_mat.shape[0]
        self.n_alpha = to_grid_mat.shape[1]

    def get_to_grid_mat(self, device=None):
        return self.to_grid_mat

    def get_from_grid_mat(self, device=None):
        return self.from_grid_mat

    def __repr__(self):
        return (
            f"CustomSO3Grid(method={self.method}, lmax={self.lmax}, mmax={self.mmax}, "
            f"n_beta={self.n_beta}, n_alpha={self.n_alpha}, "
            f"total_points={self.n_beta * self.n_alpha})"
        )


def patch_so3_grid(model: torch.nn.Module, method: str = "gl",
                   n_beta: int = None, n_alpha: int = None) -> int:
    """
    Replace all SO3_Grid instances in `model.backbone.SO3_grid` with
    CustomSO3Grid instances using the specified method.

    Returns: number of grids replaced.
    """
    backbone = model.backbone if hasattr(model, "backbone") else model
    so3_grid = backbone.SO3_grid

    n_replaced = 0
    device = next(model.parameters()).device

    new_outer = torch.nn.ModuleList()
    for inner_list in so3_grid:
        new_inner = torch.nn.ModuleList()
        for old_grid in inner_list:
            new_grid = CustomSO3Grid(
                old_grid.lmax, old_grid.mmax,
                method=method, n_beta=n_beta, n_alpha=n_alpha
            ).to(device)
            new_inner.append(new_grid)
            n_replaced += 1
        new_outer.append(new_inner)

    backbone.SO3_grid = new_outer
    return n_replaced
