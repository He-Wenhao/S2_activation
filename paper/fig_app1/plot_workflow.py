# -*- coding: utf-8 -*-
"""
Figure 1: detailed S^2-activation pipeline cartoon.

Visual story:
  - Stages 1-3 are exact mathematical operations on continuous
    objects (SH coeffs, sphere functions). They do NOT introduce
    equivariance error.
  - The final integration step is the only place approximation
    enters: the exact integral ∫ dΩ σ(f) Y_lm is replaced by a
    finite-quadrature sum Σ_i w_i σ(f(θ_i,φ_i)) Y_lm, and the gap
    between them is the source of S^2-Activation equivariance error.

We render this by:
  - Drawing stages 1, 2, 3 with a light-green background to mark
    them as exact / equivariant.
  - Showing two parallel "stage 4" boxes side-by-side: the IDEAL
    exact integral (top, green border, marked "exact, equivariant")
    and the ACTUAL quadrature approximation (bottom, red border,
    marked "computed in practice").
  - A red "≈" between the two with a label "source of equivariance
    error".

Saves: paper/fig1/fig_workflow.{pdf,png}
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


def add_box(ax, x, y, w, h, text, fc="#f5f7fa", ec="black",
             lw=1.1, fontsize=10):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.02",
                        facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def add_arrow(ax, x0, x1, y0, y1, label, color="black", lw=1.2,
              label_dy=0.10, label_color="#333333", style="italic"):
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                           arrowstyle="-|>", mutation_scale=14,
                           color=color, linewidth=lw)
    ax.add_patch(arr)
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + label_dy, label,
                ha="center", va="bottom", fontsize=9,
                color=label_color, style=style)


def main():
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold")

    # ── Stage 1, 2, 3 in a row, on a green-tinted "exact" band ──
    band_y, band_h = 2.5, 1.6
    band = Rectangle((0.4, band_y - band_h/2), 7.0, band_h,
                      facecolor="#e9f9ee", edgecolor="#7ac28d",
                      linewidth=0.8, linestyle="--", zorder=0)
    ax.add_patch(band)
    ax.text(0.5, band_y + band_h/2 - 0.12,
            "exact (continuous, equivariant)",
            ha="left", va="top", fontsize=8.5, color="#5a8b6c",
            style="italic")

    cx_in = [1.5, 4.0, 6.5]
    box_w_in = 2.0
    box_h_in = 1.2

    add_box(ax, cx_in[0], band_y, box_w_in, box_h_in,
            "SH coefficients\n"
            "$\\{c_{\\ell m}\\}_{\\ell,m}$\n"
            "$\\ell \\leq \\ell_{\\max}$",
            fc="white", ec="#3e8e5a", lw=1.2)
    add_box(ax, cx_in[1], band_y, box_w_in, box_h_in,
            "Spherical function\n"
            "$f(\\theta,\\phi) =$\n"
            "$\\sum_{\\ell,m} c_{\\ell m}\\, Y_{\\ell m}(\\theta,\\phi)$",
            fc="white", ec="#3e8e5a", lw=1.2)
    add_box(ax, cx_in[2], band_y, box_w_in, box_h_in,
            "Activated function\n"
            "$\\sigma( f(\\theta,\\phi) )$",
            fc="white", ec="#3e8e5a", lw=1.2)

    # Black arrows for the exact stages
    gap = box_w_in / 2 + 0.12
    add_arrow(ax, cx_in[0] + gap, cx_in[1] - gap, band_y, band_y,
              "expand in SH", color="black")
    add_arrow(ax, cx_in[1] + gap, cx_in[2] - gap, band_y, band_y,
              "pointwise $\\sigma$", color="black")

    # ── Stage 4 splits into two: ideal (top) and actual (bottom) ──
    cx_out = 10.5
    box_w_out = 4.0
    box_h_out = 1.05

    y_ideal = band_y + 0.85
    y_actual = band_y - 0.85

    # Ideal exact integral (top) — separate text label below the math
    p_top = FancyBboxPatch((cx_out - box_w_out/2, y_ideal - box_h_out/2),
                            box_w_out, box_h_out,
                            boxstyle="round,pad=0.02",
                            facecolor="#e9f9ee", edgecolor="#3e8e5a", linewidth=1.4)
    ax.add_patch(p_top)
    ax.text(cx_out, y_ideal + 0.18,
            "$c'_{\\ell m} = \\int_{S^2}\\sigma(f)\\,Y_{\\ell m}\\,d\\Omega$",
            ha="center", va="center", fontsize=11)
    ax.text(cx_out, y_ideal - 0.30,
            "exact, exactly equivariant",
            ha="center", va="center", fontsize=9, style="italic",
            color="#3e8e5a")

    # Actual quadrature (bottom)
    p_bot = FancyBboxPatch((cx_out - box_w_out/2, y_actual - box_h_out/2),
                            box_w_out, box_h_out,
                            boxstyle="round,pad=0.02",
                            facecolor="#fdecee", edgecolor="#c44", linewidth=1.4)
    ax.add_patch(p_bot)
    ax.text(cx_out, y_actual + 0.18,
            "$\\widehat{c}'_{\\ell m} = \\sum_i w_i\\,\\sigma(f(\\theta_i,\\phi_i))\\,Y_{\\ell m}(\\theta_i,\\phi_i)$",
            ha="center", va="center", fontsize=10)
    ax.text(cx_out, y_actual - 0.30,
            "the actual computation in EquiformerV2",
            ha="center", va="center", fontsize=9, style="italic",
            color="#c44")

    # Arrow from box 3 to ideal (top, black, exact) and to actual
    # (bottom, red, approximate)
    arr_x0 = cx_in[2] + gap
    arr_x1 = cx_out - box_w_out/2 - 0.12
    add_arrow(ax, arr_x0, arr_x1, band_y, y_ideal,
              "integrate vs. $Y_{\\ell m}$ (continuous)",
              color="black", lw=1.2, label_dy=0.06,
              label_color="#3e8e5a")
    add_arrow(ax, arr_x0, arr_x1, band_y, y_actual,
              "integrate vs. $Y_{\\ell m}$ (quadrature)",
              color="#c44", lw=1.6, label_dy=-0.40,
              label_color="#c44")

    # The "≈" between the two boxes, with a callout
    approx_x = cx_out - box_w_out/2 - 0.55
    ax.text(approx_x, band_y, "$\\approx$",
            ha="center", va="center", fontsize=24, color="#c44",
            weight="bold")
    ax.annotate(
        "Source of $S^2$-Activation\n"
        "equivariance error.\n"
        "Goes to zero only when\n"
        "the quadrature is exact\n"
        "(see appendix proof).",
        xy=(approx_x, band_y - 0.05),
        xytext=(approx_x - 1.7, band_y - 1.7),
        ha="center", va="top",
        fontsize=8.5, color="#9a1a26",
        arrowprops=dict(arrowstyle="->", color="#c44",
                         lw=1.0, connectionstyle="arc3,rad=0.15"),
    )

    out_dir = Path(__file__).resolve().parent
    out_pdf = out_dir / "fig_workflow.pdf"
    out_png = out_dir / "fig_workflow.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
