"""
Figure 1(a): cartoon of the S^2 activation workflow.

Shows:  SH coeffs c -- to_grid -->  values f(x_i) on grid  -- σ -->
        σ(f(x_i)) on grid  -- from_grid -->  new SH coeffs c'

We render this as a horizontal flow with four boxes / states and three
arrows labelled with the operations.

Saves: paper/figures/fig_workflow.{pdf,png}
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np


def add_box(ax, x, y, w, h, text, fc="#f5f7fa", ec="black"):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.02",
                        facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def add_arrow(ax, x0, x1, y, label):
    arr = FancyArrowPatch((x0, y), (x1, y),
                           arrowstyle="-|>", mutation_scale=14,
                           color="black", linewidth=1.2)
    ax.add_patch(arr)
    ax.text((x0 + x1) / 2, y + 0.07, label,
            ha="center", va="bottom", fontsize=9, color="#333333",
            style="italic")


def draw_sphere_with_dots(ax, cx, cy, r, dot_density="dh", color="#d62728"):
    """Tiny sphere icon with quadrature-like dots."""
    # Draw circle
    circ = Circle((cx, cy), r, facecolor="white", edgecolor="gray",
                   linewidth=0.7, zorder=3)
    ax.add_patch(circ)
    # Sample some "grid" dots
    if dot_density == "dh":
        thetas = np.linspace(0.15, np.pi - 0.15, 6)
        phis = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        thg, phg = np.meshgrid(thetas, phis, indexing="ij")
        u = (np.sin(thg) * np.cos(phg)).ravel() * 0.85
        v = np.cos(thg).ravel() * 0.85
    elif dot_density == "scattered":
        rng = np.random.default_rng(7)
        u = rng.uniform(-0.85, 0.85, 32)
        v = rng.uniform(-0.85, 0.85, 32)
        keep = u**2 + v**2 < 0.85**2
        u, v = u[keep], v[keep]
    else:
        u, v = np.zeros(0), np.zeros(0)
    ax.scatter(cx + u * r, cy + v * r, s=4, color=color, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    # Box centers (on y=1.3) — spaced wider so arrow labels fit
    y = 1.3
    cx = [1.2, 4.4, 7.6, 10.8]
    box_w = 1.7
    box_h = 1.2

    # Box 1: SH coefficients in
    add_box(ax, cx[0], y, box_w, box_h,
            "SH coefficients\n$c \\in \\mathbb{R}^{(\\ell_{\\max}+1)^2}$",
            fc="#fff7e6")

    # Box 2: grid values
    add_box(ax, cx[1], y, box_w, box_h,
            "Grid values\n$\\{ f(\\theta_i, \\phi_i) \\}_{i=1}^{N}$",
            fc="#f5f7fa")
    draw_sphere_with_dots(ax, cx[1], 0.35, 0.22, "dh", color="#d62728")

    # Box 3: activated grid values
    add_box(ax, cx[2], y, box_w, box_h,
            "Activated grid\n$\\{ \\sigma(f(\\theta_i, \\phi_i)) \\}_{i=1}^{N}$",
            fc="#e8f0ff")
    draw_sphere_with_dots(ax, cx[2], 0.35, 0.22, "dh", color="#1f77b4")

    # Box 4: SH coefficients out
    add_box(ax, cx[3], y, box_w, box_h,
            "New SH coeffs\n$c' = \\mathcal{A}_\\sigma(c)$",
            fc="#fff7e6")

    # Arrows (start/end leave a small gap from box edges)
    gap = box_w / 2 + 0.12
    add_arrow(ax, cx[0] + gap, cx[1] - gap, y, "to-grid")
    add_arrow(ax, cx[1] + gap, cx[2] - gap, y, "pointwise $\\sigma$")
    add_arrow(ax, cx[2] + gap, cx[3] - gap, y, "from-grid")

    # (No title — LaTeX caption (a) will provide one)

    out_pdf = Path("paper/figures/fig_workflow.pdf")
    out_png = Path("paper/figures/fig_workflow.png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
