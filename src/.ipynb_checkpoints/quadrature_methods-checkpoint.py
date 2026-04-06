"""
Sampling methods for the sphere S2.

Implements four quadrature/sampling strategies:
  1. Uniform equiangular grid
  2. Gauss-Legendre quadrature
  3. Lebedev quadrature
  4. Fibonacci sphere (baseline)
"""

import numpy as np
import torch
from scipy.integrate import lebedev_rule
from scipy.special import roots_legendre


class SamplingMethods:

    @staticmethod
    def uniform_grid(resolution_theta, resolution_phi):
        """
        Equiangular grid on S2.

        Returns:
            points: (N, 3) Cartesian coordinates on the unit sphere
            weights: (N,) quadrature weights (sum ≈ 4π)
        """
        theta = np.linspace(0, np.pi, resolution_theta, endpoint=False) + np.pi / (2 * resolution_theta)
        phi = np.linspace(0, 2 * np.pi, resolution_phi, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        theta_flat = theta_grid.ravel()
        phi_flat = phi_grid.ravel()

        x = np.sin(theta_flat) * np.cos(phi_flat)
        y = np.sin(theta_flat) * np.sin(phi_flat)
        z = np.cos(theta_flat)
        points = np.stack([x, y, z], axis=-1)

        # Weights: sin(theta) * dtheta * dphi
        dtheta = np.pi / resolution_theta
        dphi = 2 * np.pi / resolution_phi
        weights = np.sin(theta_flat) * dtheta * dphi

        return points, weights

    @staticmethod
    def gauss_legendre(l_max):
        """
        Gauss-Legendre quadrature on S2.
        Uses (l_max+1) GL nodes in cos(theta) and (2*l_max+2) uniform nodes in phi.
        Exact for polynomials up to degree 2*l_max+1 in cos(theta).

        Returns:
            points: (N, 3) Cartesian coordinates
            weights: (N,) quadrature weights (sum = 4π)
        """
        n_theta = l_max + 1
        n_phi = 2 * l_max + 2

        # GL nodes and weights in [-1, 1] for cos(theta)
        cos_theta, w_theta = roots_legendre(n_theta)
        theta = np.arccos(cos_theta)

        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        dphi = 2 * np.pi / n_phi

        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        w_theta_grid = np.meshgrid(w_theta, phi, indexing='ij')[0]

        theta_flat = theta_grid.ravel()
        phi_flat = phi_grid.ravel()

        x = np.sin(theta_flat) * np.cos(phi_flat)
        y = np.sin(theta_flat) * np.sin(phi_flat)
        z = np.cos(theta_flat)
        points = np.stack([x, y, z], axis=-1)

        weights = w_theta_grid.ravel() * dphi  # sum = 4π

        return points, weights

    @staticmethod
    def lebedev(degree):
        """
        Lebedev quadrature on S2.
        Uses scipy.integrate.lebedev_rule.

        Args:
            degree: algebraic degree of exactness (odd integer).
                    Available: 3,5,7,...,131

        Returns:
            points: (N, 3) Cartesian coordinates
            weights: (N,) quadrature weights (sum = 4π)
        """
        xyz, w = lebedev_rule(degree)
        points = xyz.T  # (N, 3)
        # scipy lebedev_rule weights sum to 4π
        weights = np.array(w)
        return points, weights

    @staticmethod
    def fibonacci_sphere(num_points):
        """
        Fibonacci spiral sampling on S2 (approximately uniform distribution).

        Returns:
            points: (N, 3) Cartesian coordinates
            weights: (N,) uniform weights (each = 4π/N)
        """
        golden_ratio = (1 + np.sqrt(5)) / 2
        i = np.arange(num_points)
        theta = np.arccos(1 - 2 * (i + 0.5) / num_points)
        phi = 2 * np.pi * i / golden_ratio

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.stack([x, y, z], axis=-1)

        weights = np.full(num_points, 4 * np.pi / num_points)
        return points, weights


def get_sampling(method, l_max=None, resolution=None, degree=None, num_points=None):
    """
    Convenience function to get sampling points and weights.

    Returns:
        points: torch.Tensor (N, 3)
        weights: torch.Tensor (N,)
    """
    if method == 'uniform':
        res_theta = resolution if resolution else 2 * (l_max + 1)
        res_phi = 2 * res_theta
        pts, wts = SamplingMethods.uniform_grid(res_theta, res_phi)
    elif method == 'gauss_legendre':
        pts, wts = SamplingMethods.gauss_legendre(l_max)
    elif method == 'lebedev':
        deg = degree if degree else 2 * l_max + 1
        pts, wts = SamplingMethods.lebedev(deg)
    elif method == 'fibonacci':
        n = num_points if num_points else (l_max + 1) ** 2
        pts, wts = SamplingMethods.fibonacci_sphere(n)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return torch.from_numpy(pts).double(), torch.from_numpy(wts).double()
