"""
3-panel companion to fig_sweep_models.pdf:
  one panel per architecture, equivariance error vs forward time.

Equivariance error: model-level pred-invariance from
  results/expG_quadrature/sweep_equiv.json
Forward time: from
  results/expG_quadrature/sweep_models.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    repo = Path(__file__).resolve().parents[2]
    sweep_eq = json.load(open(repo / "results/expG_quadrature/sweep_equiv.json"))
    sweep_t = json.load(open(repo / "results/expG_quadrature/sweep_models.json"))

    arch_order = list(sweep_t["configs"].keys())
    grid_order = ["DH default", "DH 2x", "GL match-DH", "GL 2x"]

    # n_points table per (lmax, grid) — derived once with verify_gl_grid.
    NPTS = {
        4: {"DH default": 50, "DH 2x": 400, "GL match-DH": 90,  "GL 2x": 200},
        6: {"DH default": 70, "DH 2x": 784, "GL match-DH": 182, "GL 2x": 392},
    }

    style = {
        "DH default":   dict(color="#d62728", marker="s"),
        "DH 2x":        dict(color="#a02020", marker="s"),
        "GL match-DH":  dict(color="#1f77b4", marker="o"),
        "GL 2x":        dict(color="#114488", marker="o"),
    }

    def label_for(arch_name, grid_label):
        # NB: grid_label "GL match-DH" contains the substring "DH", so we
        # cannot use `"DH" in grid_label`. Match by prefix instead.
        L = sweep_t["configs"][arch_name]["backbone"]["lmax_list"][0]
        method = "DH" if grid_label.startswith("DH") else "GL"
        return f"{method} ×{NPTS[L][grid_label]} pts"

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    short = {
        "QM9-small (4L/64ch/L4/M2, our expF model)":
            "QM9-small\n(4L / 64ch / lmax=4 / mmax=2)",
        "OC20-31M-public (8L/128ch/L4/M2, eq2_31M_ec4_allmd.pt)":
            "OC20 EqV2-31M\n(8L / 128ch / lmax=4 / mmax=2)",
        "fairchem-default (12L/128ch/L6/M2)":
            "fairchem default\n(12L / 128ch / lmax=6 / mmax=2)",
    }

    for ax, arch in zip(axes, arch_order):
        t = sweep_t["configs"][arch]["results"]
        e = sweep_eq["configs"][arch]["results"]

        xs, ys, xerrs, yerrs = [], [], [], []
        for label in grid_order:
            xs.append(t[label]["mean"])
            ys.append(e[label]["pred_invariance_error_mean"])
            xerrs.append(t[label]["ci95"])
            yerrs.append(e[label]["pred_invariance_error_ci95"])

        # Connector line in grid_order to show the trace
        ax.plot(xs, ys, color="gray", lw=0.8, alpha=0.5, zorder=1)

        for label, x, y, xe, ye in zip(grid_order, xs, ys, xerrs, yerrs):
            ax.errorbar(
                x, y, xerr=xe, yerr=ye,
                fmt=style[label]["marker"], ms=9,
                color=style[label]["color"], ecolor=style[label]["color"],
                mec="black", mew=0.7, elinewidth=1.0, capsize=3,
                label=label_for(arch, label), zorder=3,
            )

        # Matched-accuracy comparison (legitimate Pareto gap):
        # GL 2× vs DH 2× — both reach the ~10⁻⁶ pred-invariance floor.
        L = sweep_t["configs"][arch]["backbone"]["lmax_list"][0]
        dh2_x = t["DH 2x"]["mean"]
        gl2_x = t["GL 2x"]["mean"]
        dh2_y = e["DH 2x"]["pred_invariance_error_mean"]
        gl2_y = e["GL 2x"]["pred_invariance_error_mean"]
        if gl2_x < dh2_x:
            ax.annotate(
                "",
                xy=(gl2_x, gl2_y), xytext=(dh2_x, dh2_y),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.6, alpha=0.9),
            )
            saved_ms = dh2_x - gl2_x
            saved_pct = 100 * saved_ms / dh2_x
            ax.text(
                0.98, 0.97,
                (f"GL ×{NPTS[L]['GL 2x']} pts vs DH ×{NPTS[L]['DH 2x']} pts\n"
                 f"at matched invariance:\n"
                 f"−{saved_ms:.1f} ms ({saved_pct:.1f}%)"),
                transform=ax.transAxes,
                ha="right", va="top", color="green", fontsize=8.0, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="green", alpha=0.85, lw=0.6),
            )

        ax.set_yscale("log")
        ax.set_title(short.get(arch, arch), fontsize=10)
        ax.set_xlabel("Forward time (ms)", fontsize=10)
        ax.grid(True, which="both", alpha=0.3, zorder=0)
        ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    axes[0].set_ylabel("Pred-invariance error\n(random-init, lower better)", fontsize=10)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent
    fig.savefig(out_dir / "fig_sweep_pareto.png", dpi=200)
    fig.savefig(out_dir / "fig_sweep_pareto.pdf")
    print(f"Saved: {out_dir / 'fig_sweep_pareto.pdf'}")


if __name__ == "__main__":
    main()
