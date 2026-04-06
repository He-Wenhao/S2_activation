"""
Spherical harmonics utilities.

Uses e3nn for l_max <= 11, falls back to scipy for higher degrees.
Provides forward (coefficients -> sphere values) and inverse (sphere values -> coefficients)
transforms via quadrature.
"""

import torch
import math
import numpy as np
from scipy.special import sph_harm

# Fix e3nn compatibility with PyTorch 2.6
torch.serialization.add_safe_globals([slice])
from e3nn import o3

E3NN_LMAX = 11  # e3nn 0.4.4 max supported l


def _real_sph_harm_scipy(l_max, points):
    """
    Compute real spherical harmonics using scipy for arbitrary l_max.
    Uses the same ordering as e3nn: for each l, m goes from -l to l.

    Normalization: 'integral' (orthonormal on the sphere, integral of Y^2 = 1).

    Args:
        l_max: maximum degree
        points: (N, 3) numpy array of unit vectors

    Returns:
        Y: (N, (l_max+1)^2) numpy array
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-30), -1, 1))  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle

    n_coeffs = (l_max + 1) ** 2
    N = len(points)
    Y = np.zeros((N, n_coeffs), dtype=np.float64)

    idx = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # scipy sph_harm uses (m, l, phi, theta) convention with |m|
            if m < 0:
                # Y_l^{-|m|} = sqrt(2) * (-1)^m * Im[Y_l^{|m|}]
                Y_complex = sph_harm(abs(m), l, phi, theta)
                Y[:, idx] = np.sqrt(2) * (-1)**m * Y_complex.imag
            elif m == 0:
                Y_complex = sph_harm(0, l, phi, theta)
                Y[:, idx] = Y_complex.real
            else:
                # Y_l^{+m} = sqrt(2) * (-1)^m * Re[Y_l^m]
                Y_complex = sph_harm(m, l, phi, theta)
                Y[:, idx] = np.sqrt(2) * (-1)**m * Y_complex.real
            idx += 1

    return Y


def spherical_harmonics_on_points(l_max, points):
    """
    Evaluate all real spherical harmonics Y_l^m up to degree l_max at given points.

    Args:
        l_max: maximum degree
        points: (N, 3) tensor of unit vectors

    Returns:
        Y: (N, (l_max+1)^2) tensor of SH values
    """
    if l_max <= E3NN_LMAX:
        irreps = o3.Irreps([(1, (l, 1)) for l in range(l_max + 1)])
        Y = o3.spherical_harmonics(irreps, points, normalize=True, normalization='integral')
        return Y
    else:
        pts_np = points.detach().cpu().numpy() if isinstance(points, torch.Tensor) else points
        Y_np = _real_sph_harm_scipy(l_max, pts_np)
        return torch.from_numpy(Y_np).to(dtype=points.dtype if isinstance(points, torch.Tensor) else torch.float64)


def expand_coefficients_to_sphere(coeffs, points, l_max):
    """
    Evaluate f(x) = sum_{l,m} c_{lm} Y_l^m(x) at given points.

    Args:
        coeffs: (..., (l_max+1)^2) SH coefficients
        points: (N, 3) unit vectors
        l_max: maximum degree

    Returns:
        values: (..., N) function values at the points
    """
    Y = spherical_harmonics_on_points(l_max, points)  # (N, n_coeffs)
    values = torch.einsum('...c,nc->...n', coeffs, Y.to(coeffs.dtype))
    return values


def project_to_coefficients(values, points, weights, l_max):
    """
    Compute SH coefficients via quadrature:
        c_{lm} = integral f(x) Y_l^m(x) dx ≈ sum_i w_i f(x_i) Y_l^m(x_i)

    Args:
        values: (..., N) function values at sampling points
        points: (N, 3) unit vectors
        weights: (N,) quadrature weights
        l_max: maximum degree

    Returns:
        coeffs: (..., (l_max+1)^2) reconstructed coefficients
    """
    Y = spherical_harmonics_on_points(l_max, points)  # (N, n_coeffs)
    wY = weights.unsqueeze(-1) * Y.to(weights.dtype)  # (N, n_coeffs)
    coeffs = torch.einsum('...n,nc->...c', values, wY)
    return coeffs


def generate_random_coefficients(l_max, distribution='random_normal', seed=None):
    """
    Generate random SH coefficients for testing.

    Args:
        l_max: maximum degree
        distribution: one of 'random_normal', 'random_uniform', 'sparse', 'polynomial'
        seed: random seed

    Returns:
        coeffs: ((l_max+1)^2,) tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    n = (l_max + 1) ** 2

    if distribution == 'random_normal':
        return torch.randn(n, dtype=torch.float64)
    elif distribution == 'random_uniform':
        return torch.rand(n, dtype=torch.float64)
    elif distribution == 'sparse':
        coeffs = torch.zeros(n, dtype=torch.float64)
        num_nonzero = max(1, n // 5)
        indices = torch.randperm(n)[:num_nonzero]
        coeffs[indices] = torch.randn(num_nonzero, dtype=torch.float64)
        return coeffs
    elif distribution == 'polynomial':
        coeffs = torch.zeros(n, dtype=torch.float64)
        idx = 0
        for l in range(l_max + 1):
            for m in range(2 * l + 1):
                coeffs[idx] = torch.randn(1, dtype=torch.float64).item() / (l + 1) ** 2
                idx += 1
        return coeffs
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
