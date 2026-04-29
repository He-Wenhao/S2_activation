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
    # Light, translucent wireframe sphere so back-facing nodes
    # remain visible (the user noted nodes were hidden previously).
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 24)
    su = np.outer(np.cos(u), np.sin(v))
    sv = np.outer(np.sin(u), np.sin(v))
    sw = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(su, sv, sw, color="lightgray", alpha=0.45,
                      linewidth=0.5, rstride=2, cstride=2)

    # Front-back depth cue via z-coordinate (relative to viewing direction)
    # but draw all nodes so none get culled.
    n = points.shape[0]
    # View direction at our (elev=18, azim=30) — approximate camera
    # vector pointing toward viewer in world coords:
    elev_rad, azim_rad = np.deg2rad(18), np.deg2rad(30)
    cam = np.array([np.cos(elev_rad) * np.cos(azim_rad),
                    np.cos(elev_rad) * np.sin(azim_rad),
                    np.sin(elev_rad)])
    dot = points @ cam  # +1 = facing camera, -1 = away
    front = dot >= 0
    back = ~front

    # Back-facing nodes drawn first, more transparent so the wireframe
    # shows through; front-facing on top, fully opaque.
    if back.any():
        ax.scatter(points[back, 0], points[back, 1], points[back, 2],
                   s=22, c=color, edgecolor="black", linewidth=0.3,
                   depthshade=False, alpha=0.45)
    if front.any():
        ax.scatter(points[front, 0] * 1.005, points[front, 1] * 1.005,
                   points[front, 2] * 1.005,
                   s=34, c=color, edgecolor="black", linewidth=0.5,
                   depthshade=False, alpha=1.0)

    ax.set_title(title, fontsize=11, pad=6)
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
