"""
Schematic figure for the Method section: 3-panel comparison of
Driscoll-Healy, Gauss-Legendre, and Lebedev quadrature node layouts
on the sphere, plus a 4th panel showing the S^2 activation pipeline.

Saves: paper/figures/fig_quadratures.{pdf,png}
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.integrate import lebedev_rule
from scipy.special import roots_legendre


def dh_nodes(lmax: int):
    n_lat = 2 * (lmax + 1)
    n_lon = 2 * lmax + 1
    j = np.arange(n_lat)
    betas = (j + 0.5) / n_lat * np.pi
    k = np.arange(n_lon)
    alphas = k * (2 * np.pi / n_lon)
    BB, AA = np.meshgrid(betas, alphas, indexing="ij")
    x = np.sin(BB) * np.cos(AA)
    y = np.sin(BB) * np.sin(AA)
    z = np.cos(BB)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1), n_lat * n_lon


def gl_nodes(lmax: int):
    n_lat = lmax + 1
    n_lon = 2 * lmax + 1
    cos_b, _ = roots_legendre(n_lat)
    betas = np.arccos(cos_b)
    k = np.arange(n_lon)
    alphas = k * (2 * np.pi / n_lon)
    BB, AA = np.meshgrid(betas, alphas, indexing="ij")
    x = np.sin(BB) * np.cos(AA)
    y = np.sin(BB) * np.sin(AA)
    z = np.cos(BB)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1), n_lat * n_lon


def lebedev_nodes(degree: int):
    xyz, _ = lebedev_rule(degree)
    return xyz.T, xyz.shape[1]


def plot_sphere_with_nodes(ax, points, title, color="#d62728"):
    # Solid translucent sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    su = np.outer(np.cos(u), np.sin(v))
    sv = np.outer(np.sin(u), np.sin(v))
    sw = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(su, sv, sw, color="white", alpha=0.85,
                     edgecolor="lightgray", linewidth=0.2, antialiased=True,
                     rstride=3, cstride=3)

    # Quadrature nodes
    ax.scatter(points[:, 0] * 1.02, points[:, 1] * 1.02, points[:, 2] * 1.02,
                s=35, c=color, edgecolor="black", linewidth=0.5,
                depthshade=True, alpha=0.95)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=30)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.pane.fill = False
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)


def main():
    LMAX = 4
    fig = plt.figure(figsize=(12, 4.2))

    pts_dh, n_dh = dh_nodes(LMAX)
    pts_gl, n_gl = gl_nodes(LMAX)
    pts_leb, n_leb = lebedev_nodes(11)

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    plot_sphere_with_nodes(ax1, pts_dh,
                           f"Driscoll–Healy\n$2(\\ell_{{\\max}}{{+}}1) \\times (2\\ell_{{\\max}}{{+}}1) = {2*(LMAX+1)} \\times {2*LMAX+1}$ = {n_dh} pts",
                           color="#d62728")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    plot_sphere_with_nodes(ax2, pts_gl,
                           f"Gauss–Legendre\n$(\\ell_{{\\max}}{{+}}1) \\times (2\\ell_{{\\max}}{{+}}1) = {LMAX+1} \\times {2*LMAX+1}$ = {n_gl} pts",
                           color="#1f77b4")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    plot_sphere_with_nodes(ax3, pts_leb,
                           f"Lebedev\ndegree 11, {n_leb} pts (non-tensor-product)",
                           color="#2ca02c")

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_pdf = Path("paper/figures/fig_quadratures.pdf")
    out_png = Path("paper/figures/fig_quadratures.png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
