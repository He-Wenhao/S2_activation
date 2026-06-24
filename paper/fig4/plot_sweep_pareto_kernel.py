"""
Kernel-level companion to fig_sweep_pareto.pdf:
  one panel per architecture, kernel-level equivariance error vs forward time.

Kernel-level error: from results/expG_quadrature/kernel_equiv_sweep.json
  (L=4 numbers cover both QM9-small and OC20-31M, same kernel.)
Forward time:        from results/expG_quadrature/sweep_models.json
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    repo = Path(__file__).resolve().parents[2]
    sweep_t = json.load(open(repo / "results/expG_quadrature/sweep_models.json"))
    sweep_e = json.load(open(repo / "results/expG_quadrature/kernel_equiv_sweep.json"))

    arch_order = list(sweep_t["configs"].keys())
    grid_order = ["DH default", "DH 2x", "GL match-DH", "GL 2x"]

    style = {
        "DH default":   dict(color="#d62728", marker="s"),
        "DH 2x":        dict(color="#a02020", marker="s"),
        "GL match-DH":  dict(color="#1f77b4", marker="o"),
        "GL 2x":        dict(color="#114488", marker="o"),
    }

    short = {
        "QM9-small (4L/64ch/L4/M2, our expF model)":
            "QM9-small\n(4L / 64ch / lmax=4 / mmax=2)",
        "OC20-31M-public (8L/128ch/L4/M2, eq2_31M_ec4_allmd.pt)":
            "OC20 EqV2-31M\n(8L / 128ch / lmax=4 / mmax=2)",
        "fairchem-default (12L/128ch/L6/M2)":
            "fairchem default\n(12L / 128ch / lmax=6 / mmax=2)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    for ax, arch in zip(axes, arch_order):
        L = sweep_t["configs"][arch]["backbone"]["lmax_list"][0]
        eq_table = sweep_e[f"L{L}_M2"]
        t = sweep_t["configs"][arch]["results"]

        xs, ys, xerrs, yerrs = [], [], [], []
        for label in grid_order:
            xs.append(t[label]["mean"])
            ys.append(eq_table[label]["equiv_err_mean"])
            xerrs.append(t[label]["ci95"])
            yerrs.append(eq_table[label]["equiv_err_ci95"])

        ax.plot(xs, ys, color="gray", lw=0.8, alpha=0.5, zorder=1)

        for label, x, y, xe, ye in zip(grid_order, xs, ys, xerrs, yerrs):
            method = "DH" if label.startswith("DH") else "GL"
            npts = eq_table[label]["n_points"]
            ax.errorbar(
                x, y, xerr=xe, yerr=ye,
                fmt=style[label]["marker"], ms=9,
                color=style[label]["color"], ecolor=style[label]["color"],
                mec="black", mew=0.7, elinewidth=1.0, capsize=3,
                label=f"{method} ×{npts} pts", zorder=3,
            )

        # Kernel-level matched-equivariance comparison: GL ×N_match vs DH 2×
        # both at the saturation floor. GL ×N_match has the smallest pts at floor.
        gl_match_x = t["GL match-DH"]["mean"]
        dh2_x = t["DH 2x"]["mean"]
        gl_match_y = eq_table["GL match-DH"]["equiv_err_mean"]
        dh2_y = eq_table["DH 2x"]["equiv_err_mean"]
        if gl_match_x < dh2_x:
            ax.annotate(
                "",
                xy=(gl_match_x, gl_match_y), xytext=(dh2_x, dh2_y),
                arrowprops=dict(arrowstyle="->", color="green", lw=1.6, alpha=0.9),
            )
            saved_ms = dh2_x - gl_match_x
            saved_pct = 100 * saved_ms / dh2_x
            ax.text(
                0.98, 0.97,
                (f"GL ×{eq_table['GL match-DH']['n_points']} pts vs "
                 f"DH ×{eq_table['DH 2x']['n_points']} pts\n"
                 f"at matched kernel-level err.:\n"
                 f"−{saved_ms:.1f} ms ({saved_pct:.1f}%)"),
                transform=ax.transAxes,
                ha="right", va="top", color="green", fontsize=8.0, weight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="green", alpha=0.85, lw=0.6),
            )

        ax.set_title(short.get(arch, arch), fontsize=10)
        ax.set_xlabel("Forward time (ms)", fontsize=10)
        ax.grid(True, which="both", alpha=0.3, zorder=0)
        ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    axes[0].set_ylabel("Kernel-level equivariance error\n(SiLU, lower better)",
                       fontsize=10)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent
    fig.savefig(out_dir / "fig_sweep_pareto_kernel.png", dpi=200)
    fig.savefig(out_dir / "fig_sweep_pareto_kernel.pdf")
    print(f"Saved: {out_dir / 'fig_sweep_pareto_kernel.pdf'}")


if __name__ == "__main__":
    main()
