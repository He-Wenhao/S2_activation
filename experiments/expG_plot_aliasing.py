"""
Figure 1, panels (b)-(e): visualize f and sigma(f) in two
complementary ways -- as a function on the sphere (heat plot) and as
a bar chart of signed SH coefficients c_{ell m}.

Saves four panel PDFs that the LaTeX figure assembles via subcaption
into Figure 1 alongside the workflow cartoon (panel (a) from
expG_plot_workflow.py):

    paper/figures/fig_sphere_before.pdf   -- panel (b): f on the sphere
    paper/figures/fig_sphere_after.pdf    -- panel (c): sigma(f) on the sphere
    paper/figures/fig_coeffs_before.pdf   -- panel (d): c_{ell m} of f
    paper/figures/fig_coeffs_after.pdf    -- panel (e): c_{ell m} of sigma(f)

Visual story:
  * Pointwise sigma takes a band-limited f (with c_{ell m} = 0 for
    ell > ell_max) and produces sigma(f) with c_{ell m} != 0 for
    every ell. The two sphere panels (b, c) make this concrete as
    "the function on S^2 looks similar before and after sigma" while
    the two bar panels (d, e) make it concrete in the spectral
    domain "but the SH coefficients leak to all ell after sigma".
  * The high-ell tail in (e), outside the green-shaded
    ell <= ell_max region, is the content the finite-N grid
    quadrature cannot integrate exactly, and is the source of
    S^2-Activation equivariance error.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np
import torch

torch.serialization.add_safe_globals([slice])

# scipy.special.sph_harm was renamed to sph_harm_y in scipy 1.15.
try:
    from scipy.special import sph_harm_y

    def _Ylm_complex(m, l, theta, phi):
        # theta = polar (colatitude), phi = azimuth.
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm

    def _Ylm_complex(m, l, theta, phi):
        # Old API order: sph_harm(m, n, azimuth, polar).
        return sph_harm(m, l, phi, theta)
from scipy.special import roots_legendre


# ---------------------------------------------------------------------------
#  Spherical-harmonic helpers
# ---------------------------------------------------------------------------

def _real_sh_basis(LMAX_REF, theta, phi):
    """Real-SH basis matrix Y of shape [n_pts, (LMAX_REF+1)^2]."""
    n_pts = len(theta)
    n_basis = (LMAX_REF + 1) ** 2
    Y = np.zeros((n_pts, n_basis), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        idx = 0
        for l in range(LMAX_REF + 1):
            for m in range(-l, l + 1):
                if m == 0:
                    Y[:, idx] = _Ylm_complex(0, l, theta, phi).real
                elif m > 0:
                    Y[:, idx] = np.sqrt(2.0) * _Ylm_complex(
                        m, l, theta, phi).real
                else:
                    Y[:, idx] = np.sqrt(2.0) * _Ylm_complex(
                        -m, l, theta, phi).imag
                idx += 1
    return Y


def _gl_grid(N_theta, N_phi):
    cos_theta, w_lat = roots_legendre(N_theta)
    theta = np.arccos(cos_theta)
    phi = np.linspace(0.0, 2 * np.pi, N_phi, endpoint=False)
    TT, PP = np.meshgrid(theta, phi, indexing="ij")
    w = np.outer(w_lat, np.full(N_phi, 2 * np.pi / N_phi))
    return TT.ravel(), PP.ravel(), w.ravel()


# ---------------------------------------------------------------------------
#  Sphere heat plot
# ---------------------------------------------------------------------------

def _save_sphere_panel(TT, PP, values, *, title, vmin, vmax, out_path):
    """Save a 3D sphere shaded by `values`."""
    x = np.sin(TT) * np.cos(PP)
    y = np.sin(TT) * np.sin(PP)
    z = np.cos(TT)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("seismic")
    facecolors = cmap(norm(values))

    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, facecolors=facecolors,
                    rstride=1, cstride=1,
                    antialiased=False, linewidth=0, shade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=20, azim=35)
    ax.set_title(title, fontsize=11, pad=2)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.04,
                        format="%+.2f")
    cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
#  Signed-coefficient bar chart
# ---------------------------------------------------------------------------

def _draw_coeff_bars(ax, c_disp, LMAX_DISPLAY, LMAX, ymax, *,
                     title, ylabel, show_legend):
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

    if LMAX < LMAX_DISPLAY:
        keep_xe = group_xe[LMAX] + (GAP / 2.0)
        ax.axvspan(-1.0, keep_xe, color="#cdebd2", alpha=0.5, zorder=1,
                   label="kept by truncation "
                         "($\\ell \\leq \\ell_{\\max}$)")
        ax.axvline(keep_xe, color="black", linestyle="--", linewidth=0.9,
                   zorder=2)

    ax.bar(bar_x, bar_y, width=0.85, color=bar_color,
           edgecolor="black", linewidth=0.25, zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.7, zorder=2)
    ax.set_ylim(-ymax, ymax)

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
    ax.set_title(title, fontsize=11)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    if show_legend:
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)


def _save_coeffs_panel(c, LMAX_DISPLAY, LMAX, ymax, *, title, out_path):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    _draw_coeff_bars(ax, c, LMAX_DISPLAY, LMAX, ymax,
                     title=title, ylabel=True, show_legend=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    LMAX = 3            # band limit of input f
    LMAX_DISPLAY = 6    # ell groups drawn in the bar chart
    seed = 7

    # Projection grid: GL latitude exactness >> LMAX_DISPLAY so the
    # c_{ell m} we project from sigma(f) are genuine SH coefficients
    # rather than projection aliases.
    N_theta_p = 64
    N_phi_p = 127
    theta_p, phi_p, w_p = _gl_grid(N_theta_p, N_phi_p)
    Y_p = _real_sh_basis(LMAX_DISPLAY, theta_p, phi_p)

    # Random band-limited input f
    rng = np.random.default_rng(seed)
    n_keep = (LMAX + 1) ** 2
    n_disp = (LMAX_DISPLAY + 1) ** 2
    c_in = np.zeros(n_disp)
    c_in[:n_keep] = rng.standard_normal(n_keep) / np.sqrt(n_keep)

    f_p = Y_p @ c_in
    sigma_f_p = torch.nn.functional.silu(torch.tensor(f_p)).numpy()
    c_sigma = Y_p.T @ (sigma_f_p * w_p)

    # Visualization grid (uniform mesh in (theta, phi) for smooth shading)
    N_t_v, N_p_v = 80, 161
    theta_v = np.linspace(1e-3, np.pi - 1e-3, N_t_v)
    phi_v = np.linspace(0.0, 2 * np.pi, N_p_v)
    TT, PP = np.meshgrid(theta_v, phi_v, indexing="ij")
    Y_v = _real_sh_basis(LMAX, TT.ravel(), PP.ravel())
    f_viz = (Y_v @ c_in[:n_keep]).reshape(N_t_v, N_p_v)
    sigma_viz = torch.nn.functional.silu(torch.tensor(f_viz)).numpy()

    # Shared diverging color scale for the two sphere panels.
    # Slight clipping at 85% of the absolute max gives the colormap
    # room to saturate so the spatial structure is clearly visible.
    sphere_vmax = 0.85 * max(np.abs(f_viz).max(), np.abs(sigma_viz).max())

    # Shared y range for the two bar panels
    yabs = max(np.abs(c_in).max(), np.abs(c_sigma).max())
    coeffs_ymax = yabs * 1.22

    out_dir = Path("paper/fig1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Sphere panels (b), (c) ----
    _save_sphere_panel(
        TT, PP, f_viz,
        title="$f(\\theta, \\phi)$ before pointwise $\\sigma$",
        vmin=-sphere_vmax, vmax=sphere_vmax,
        out_path=out_dir / "fig_sphere_before.pdf")
    _save_sphere_panel(
        TT, PP, sigma_viz,
        title="$\\sigma(f(\\theta, \\phi))$ after pointwise $\\sigma$",
        vmin=-sphere_vmax, vmax=sphere_vmax,
        out_path=out_dir / "fig_sphere_after.pdf")

    # ---- Bar-chart panels (d), (e) ----
    _save_coeffs_panel(
        c_in, LMAX_DISPLAY, LMAX, coeffs_ymax,
        title="Before pointwise $\\sigma$:  $f$ is band-limited",
        out_path=out_dir / "fig_coeffs_before.pdf")
    _save_coeffs_panel(
        c_sigma, LMAX_DISPLAY, LMAX, coeffs_ymax,
        title="After pointwise $\\sigma$:  "
              "$\\sigma(f)$ has content at every $\\ell$",
        out_path=out_dir / "fig_coeffs_after.pdf")


if __name__ == "__main__":
    main()
