"""
S2Activation supporting multiple quadrature methods.

Drop-in replacement for e3nn.nn.S2Activation with configurable sampling.
"""

import torch
import torch.nn as nn
import math

torch.serialization.add_safe_globals([slice])
from e3nn import o3

from .quadrature_methods import get_sampling
from .spherical_harmonics_utils import spherical_harmonics_on_points


def normalize2mom(act):
    """Normalize activation so that the second moment is 1 on a standard normal input."""
    x = torch.linspace(-5, 5, 1000)
    y = act(x)
    mom2 = (y ** 2).mean().sqrt()
    if mom2.item() < 1e-10:
        return act
    return lambda x: act(x) / mom2


class S2Activation(nn.Module):
    """
    Pointwise activation on S2 via spherical harmonics.

    Process:
      1. coefficients -> sphere values (via SH evaluation at quadrature points)
      2. Apply pointwise nonlinearity
      3. sphere values -> coefficients (via quadrature integration)

    Args:
        irreps_in: e3nn Irreps of input
        act: pointwise activation function
        sampling_method: 'uniform', 'gauss_legendre', 'lebedev', 'fibonacci'
        resolution: for 'uniform' method, grid resolution
        degree: for 'lebedev', algebraic degree of exactness
        num_points: for 'fibonacci', number of points
        lmax_out: output l_max (default: same as input)
        normalization: 'integral' (default for our custom methods)
    """

    def __init__(self, irreps_in, act, sampling_method='gauss_legendre',
                 resolution=None, degree=None, num_points=None,
                 n_theta=None, n_phi=None,
                 lmax_out=None, normalization='integral'):
        super().__init__()

        irreps_in = o3.Irreps(irreps_in).simplify()
        _, (_, p_val) = irreps_in[0]
        _, (lmax, _) = irreps_in[-1]
        assert all(mul == 1 for mul, _ in irreps_in)
        assert irreps_in.ls == list(range(lmax + 1))

        if all(p == p_val for _, (l, p) in irreps_in):
            p_arg = 1
        elif all(p == p_val * (-1) ** l for _, (l, p) in irreps_in):
            p_arg = -1
        else:
            raise ValueError("Input parity not well defined")

        self.irreps_in = irreps_in
        self.lmax = lmax
        self.sampling_method = sampling_method

        if lmax_out is None:
            lmax_out = lmax
        self.lmax_out = lmax_out

        if p_val in (0, +1):
            self.irreps_out = o3.Irreps([(1, (l, p_val * p_arg ** l)) for l in range(lmax_out + 1)])
        elif p_val == -1:
            x = torch.linspace(0, 10, 256)
            a1, a2 = act(x), act(-x)
            if (a1 - a2).abs().max() < a1.abs().max() * 1e-10:
                self.irreps_out = o3.Irreps([(1, (l, p_arg ** l)) for l in range(lmax_out + 1)])
            elif (a1 + a2).abs().max() < a1.abs().max() * 1e-10:
                self.irreps_out = o3.Irreps([(1, (l, -p_arg ** l)) for l in range(lmax_out + 1)])
            else:
                raise ValueError("Parity violated by activation")

        self.act = normalize2mom(act)

        # Get quadrature points and weights
        points, weights = get_sampling(
            method=sampling_method,
            l_max=lmax,
            resolution=resolution,
            degree=degree,
            num_points=num_points,
            n_theta=n_theta,
            n_phi=n_phi,
        )

        # Pre-compute SH matrices for forward and inverse transforms
        # Y_fwd: (N_pts, n_in) for coeffs -> values
        Y_fwd = spherical_harmonics_on_points(lmax, points)
        # Y_inv: (N_pts, n_out) for values -> coeffs
        Y_inv = spherical_harmonics_on_points(lmax_out, points)

        # Weighted inverse: wY[i, c] = w_i * Y_c(x_i)
        wY_inv = weights.unsqueeze(-1) * Y_inv

        self.register_buffer('Y_fwd', Y_fwd.float())
        self.register_buffer('wY_inv', wY_inv.float())
        self.n_points = len(weights)

    def forward(self, features):
        """
        Args:
            features: (..., irreps_in.dim) tensor of SH coefficients

        Returns:
            (..., irreps_out.dim) tensor of SH coefficients after activation
        """
        # To sphere: (..., N_pts)
        f = torch.einsum('...c,nc->...n', features, self.Y_fwd.to(features.dtype))
        # Apply activation
        f = self.act(f)
        # From sphere: (..., n_out)
        coeffs = torch.einsum('...n,nc->...c', f, self.wY_inv.to(features.dtype))
        return coeffs

    def extra_repr(self):
        return (f"method={self.sampling_method}, n_points={self.n_points}, "
                f"lmax_in={self.lmax}, lmax_out={self.lmax_out}")
