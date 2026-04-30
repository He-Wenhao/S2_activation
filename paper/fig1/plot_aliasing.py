"""
Figure 1, bottom panels: sphere heatmaps and signed-coefficient bar
charts before and after the pointwise activation.

Pointwise sigma takes a band-limited f (with c_{ell m} = 0 for
ell > ell_max) and produces sigma(f) with c_{ell m} != 0 for *every*
ell. We make this concrete with two matched row pairs:
  - before sigma: field heatmap on the sphere + SH bar chart
  - after sigma: field heatmap on the sphere + SH bar chart

The pre-activation coefficient panel has bars only at ell <= LMAX (and
zeros at ell > LMAX). The post-activation coefficient panel has bars
at every ell up to LMAX_DISPLAY, with the truncation keeping only the
green-shaded ell <= LMAX region. The high-ell tail (coloured groups
outside the green region) is the content the finite-N grid quadrature
cannot integrate exactly, and aliases into the kept coefficients in a
rotation-dependent way.

Visual design:
  * x-axis enumerates all (ell, m) pairs, grouped by ell with a
    small gap between groups; within a group bars are placed at
    m = -ell, -ell+1, ..., +ell from left to right.
  * y-axis is the signed coefficient c_{ell m} (positive up, negative
    down, zero line drawn).
  * Each ell group has its own colour; a short horizontal line plus
    "ell=k" text label sits below the group.
  * The kept-by-truncation region (ell <= LMAX) is shaded green and
    bounded by a dashed line.
"""

import warnings
from math import factorial, pi, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _double_factorial(n):
    if n <= 0:
        return 1.0
    out = 1.0
    while n > 1:
        out *= n
        n -= 2
    return out


def _associated_legendre(lmax, x):
    """Return P[l, m, i] = P_l^m(x_i) with Condon-Shortley phase."""
    x = np.asarray(x, dtype=np.float64)
    P = np.zeros((lmax + 1, lmax + 1, x.size), dtype=np.float64)
    P[0, 0] = 1.0
    one_minus_x2 = np.clip(1.0 - x * x, 0.0, None)

    for m in range(1, lmax + 1):
        P[m, m] = ((-1.0) ** m) * _double_factorial(2 * m - 1) * \
            np.power(one_minus_x2, 0.5 * m)

    for m in range(lmax):
        P[m + 1, m] = (2 * m + 1) * x * P[m, m]

    for m in range(lmax + 1):
        for l in range(m + 2, lmax + 1):
            P[l, m] = (
                (2 * l - 1) * x * P[l - 1, m]
                - (l + m - 1) * P[l - 2, m]
            ) / (l - m)
    return P


def _real_sh_value(l, m, theta, phi, P):
    abs_m = abs(m)
    norm = sqrt(
        (2 * l + 1) / (4 * pi) *
        factorial(l - abs_m) / factorial(l + abs_m)
    )
    base = norm * P[l, abs_m]
    if m == 0:
        return base
    if m > 0:
        return sqrt(2.0) * base * np.cos(abs_m * phi)
    return sqrt(2.0) * base * np.sin(abs_m * phi)


def _real_sh_basis(LMAX_REF, theta, phi):
    """Real-SH basis matrix Y of shape [n_pts, (LMAX_REF+1)^2]."""
    n_pts = len(theta)
    n_basis = (LMAX_REF + 1) ** 2
    Y = np.zeros((n_pts, n_basis), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        cos_theta = np.cos(theta)
        P = _associated_legendre(LMAX_REF, cos_theta)
        idx = 0
        for l in range(LMAX_REF + 1):
            for m in range(-l, l + 1):
                Y[:, idx] = _real_sh_value(l, m, theta, phi, P)
                idx += 1
    return Y


def _gl_grid(N_theta, N_phi):
    cos_theta, w_lat = np.polynomial.legendre.leggauss(N_theta)
    theta = np.arccos(cos_theta)
    phi = np.linspace(0.0, 2 * np.pi, N_phi, endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    w = np.outer(w_lat, np.full(N_phi, 2 * np.pi / N_phi))
    return TT.ravel(), PP.ravel(), w.ravel()


def _regular_grid(n_theta=181, n_phi=361):
    theta = np.linspace(1e-4, np.pi - 1e-4, n_theta)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    return TT, PP


def _plot_sphere_heat(ax, values, title, panel_label, vlim):
    lon = values["phi"] - np.pi
    lat = (np.pi / 2.0) - values["theta"]
    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    mesh = ax.pcolormesh(lon, lat, values["field"], shading="auto",
                         cmap="coolwarm", norm=norm)
    ax.grid(True, color="#666666", alpha=0.25, linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.text(0.01, 1.06, panel_label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=10.5, pad=10)
    return mesh


def _plot_coeffs(ax, c_disp, LMAX_DISPLAY, LMAX, ymax, title,
                 ylabel=False, show_legend=False, panel_label=None):
    """Bar chart of signed c_{ell m} grouped by ell.

    Within each ell group, bars are placed at consecutive integer x
    positions in order m = -ell, ..., +ell (left to right). Groups
    are separated by a fixed gap and coloured by ell.
    """
    cmap = plt.get_cmap("tab10")
    bar_x, bar_y, bar_color = [], [], []
    group_xs, group_xe = [], []

    x = 0.0
    GAP = 1.5
    idx = 0
    for l in range(LMAX_DISPLAY + 1):
        gxs = x
        for _ in range(-l, l + 1):
            bar_x.append(x)
            bar_y.append(float(c_disp[idx]))
            bar_color.append(cmap(l % cmap.N))
            x += 1.0
            idx += 1
        group_xs.append(gxs)
        group_xe.append(x - 1.0)
        x += GAP
    x_end = x - GAP

    bar_x = np.array(bar_x)
    bar_y = np.array(bar_y)

    # Kept-by-truncation green shading and boundary
    if LMAX < LMAX_DISPLAY:
        keep_xe = group_xe[LMAX] + (GAP / 2.0)
        ax.axvspan(-1.0, keep_xe, color="#cdebd2", alpha=0.5, zorder=1,
                   label="kept by truncation "
                         "($\\ell \\leq \\ell_{\\max}$)")
        ax.axvline(keep_xe, color="black", linestyle="--", linewidth=0.9,
                   zorder=2)

    # Bars
    ax.bar(bar_x, bar_y, width=0.85, color=bar_color,
           edgecolor="black", linewidth=0.25, zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.7, zorder=2)

    ax.set_ylim(-ymax, ymax)

    # Group labels (short bracket-line + "ell=k" text below the bars)
    label_y = -ymax * 1.04
    text_y = -ymax * 1.20
    for l, (gxs, gxe) in enumerate(zip(group_xs, group_xe)):
        ax.plot([gxs - 0.45, gxe + 0.45], [label_y, label_y],
                color="black", linewidth=0.9, clip_on=False)
        ax.text((gxs + gxe) / 2.0, text_y, f"$\\ell={l}$",
                ha="center", va="top", fontsize=9.5, clip_on=False)

    ax.set_xlim(-1.0, x_end + 1.0)
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel("Coefficient $c_{\\ell m}$", fontsize=11)
    ax.set_title(title, fontsize=10.5)
    if panel_label is not None:
        ax.text(0.01, 1.03, panel_label, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13, fontweight="bold")
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    if show_legend:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)


def main():
    LMAX = 3            # band limit of input f
    LMAX_DISPLAY = 6    # ell groups drawn in the bar chart
    seed = 7

    # GL latitude exactness ~ 2 N_theta - 1; uniform-longitude
    # exactness ~ N_phi/2. Both are far above LMAX_DISPLAY, so the
    # c_{ell m} we project for sigma(f) at ell <= LMAX_DISPLAY are
    # genuine SH coefficients, not aliases of the projection itself.
    N_theta = 64
    N_phi = 127
    theta, phi, w = _gl_grid(N_theta, N_phi)
    Y = _real_sh_basis(LMAX_DISPLAY, theta, phi)

    # Band-limited input f from random SH coefficients up to LMAX
    rng = np.random.default_rng(seed)
    n_keep = (LMAX + 1) ** 2
    n_disp = (LMAX_DISPLAY + 1) ** 2
    c_in = np.zeros(n_disp)
    c_in[:n_keep] = rng.standard_normal(n_keep) / np.sqrt(n_keep)
    f = Y @ c_in

    sigma_f = _silu(f)
    c_sigma = Y.T @ (sigma_f * w)

    # Dense regular grid for visualizing the field on the sphere.
    TT_vis, PP_vis = _regular_grid()
    Y_vis = _real_sh_basis(LMAX, TT_vis.ravel(), PP_vis.ravel())
    f_vis = (Y_vis @ c_in[:n_keep]).reshape(TT_vis.shape)
    sigma_vis = _silu(f_vis)

    yabs = max(np.abs(c_in).max(), np.abs(c_sigma).max())
    ymax = yabs * 1.22
    vlim = max(np.abs(f_vis).max(), np.abs(sigma_vis).max())

    fig = plt.figure(figsize=(12.5, 6.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.15], hspace=0.5,
                          wspace=0.22)
    ax_heat_before = fig.add_subplot(gs[0, 0], projection="mollweide")
    ax_coeff_before = fig.add_subplot(gs[0, 1])
    ax_heat_after = fig.add_subplot(gs[1, 0], projection="mollweide")
    ax_coeff_after = fig.add_subplot(gs[1, 1])

    mesh_before = _plot_sphere_heat(
        ax_heat_before,
        {"theta": TT_vis, "phi": PP_vis, "field": f_vis},
        title="Before pointwise $\\sigma$:  $f(\\theta,\\phi)$",
        panel_label="(b)",
        vlim=vlim,
    )
    _plot_coeffs(
        ax_coeff_before, c_in, LMAX_DISPLAY, LMAX, ymax,
        title="Before pointwise $\\sigma$:  $f$ is band-limited",
        ylabel=True, show_legend=True, panel_label="(c)")

    mesh_after = _plot_sphere_heat(
        ax_heat_after,
        {"theta": TT_vis, "phi": PP_vis, "field": sigma_vis},
        title="After pointwise $\\sigma$:  $\\sigma(f(\\theta,\\phi))$",
        panel_label="(d)",
        vlim=vlim,
    )
    _plot_coeffs(
        ax_coeff_after, c_sigma, LMAX_DISPLAY, LMAX, ymax,
        title="After pointwise $\\sigma$:  "
              "$\\sigma(f)$ has content at every $\\ell$",
        ylabel=False, show_legend=True, panel_label="(e)")

    cbar_before = fig.colorbar(mesh_before, ax=ax_heat_before,
                               orientation="horizontal",
                               fraction=0.08, pad=0.10)
    cbar_before.ax.tick_params(labelsize=8)
    cbar_before.set_label("Field value", fontsize=9)
    cbar_after = fig.colorbar(mesh_after, ax=ax_heat_after,
                              orientation="horizontal",
                              fraction=0.08, pad=0.10)
    cbar_after.ax.tick_params(labelsize=8)
    cbar_after.set_label("Field value", fontsize=9)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08)
    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "fig_aliasing.pdf"
    out_png = out_dir / "fig_aliasing.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
