"""
Plot the model-size sweep results: forward time per config across
multiple architectures, with the matched-equivariance savings annotated.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("results/expG_quadrature/sweep_models.json")
    with open(src) as f:
        data = json.load(f)

    arch_order = list(data["configs"].keys())
    grid_order = ["DH default", "DH 2x", "GL match-DH", "GL 2x"]
    grid_styles = {
        "DH default":   ("#d62728", "",   "DH min"),
        "DH 2x":        ("#a02020", "xx", "DH 2× density"),
        "GL match-DH":  ("#1f77b4", "",   "GL min"),
        "GL 2x":        ("#114488", "xx", "GL 2× density"),
    }
    # n_points per (lmax, grid)
    NPTS = {
        4: {"DH default": 50, "DH 2x": 400, "GL match-DH": 90,  "GL 2x": 200},
        6: {"DH default": 70, "DH 2x": 784, "GL match-DH": 182, "GL 2x": 392},
    }

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bar_w = 0.20
    x = np.arange(len(arch_order))

    # Plot bars
    for i, label in enumerate(grid_order):
        means = [data["configs"][a]["results"][label]["mean"] for a in arch_order]
        cis = [data["configs"][a]["results"][label]["ci95"] for a in arch_order]
        offset = (i - 1.5) * bar_w
        color, hatch, leg_label = grid_styles[label]
        ax.bar(
            x + offset, means, bar_w,
            yerr=cis, capsize=3, label=leg_label,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.6,
            alpha=0.95, error_kw=dict(elinewidth=1.0, capthick=1.0),
        )
        # n_pts label above each bar
        for j, a in enumerate(arch_order):
            L = data["configs"][a]["backbone"]["lmax_list"][0]
            npts = NPTS[L][label]
            # NB: "GL match-DH" contains "DH" as substring; match by prefix.
            method_short = "DH" if label.startswith("DH") else "GL"
            ax.text(
                x[j] + offset, means[j] + cis[j] + 2.5,
                f"×{npts}", ha="center", va="bottom",
                fontsize=7.5, color="black", rotation=0,
            )

    # Annotate matched-invariance saving (GL 2× vs DH 2× — both reach ~10⁻⁶
    # pred-invariance, see fig:sweep bottom panels). Kernel-level "matched
    # equivariance" between GL ×N_match and DH 2× does NOT translate to
    # model-level, so we no longer label that as a saving.
    for i, a in enumerate(arch_order):
        dh2 = data["configs"][a]["results"]["DH 2x"]["mean"]
        gl2 = data["configs"][a]["results"]["GL 2x"]["mean"]
        saved = dh2 - gl2
        pct = 100 * saved / dh2 if dh2 > 0 else 0.0
        ymax = max(dh2, gl2) * 1.10
        ax.annotate(
            f"GL 2× vs DH 2× at matched\n"
            f"model-level invariance:\n"
            f"−{saved:.1f} ms ({pct:.1f}%)",
            xy=(x[i], ymax + 8), ha="center", fontsize=8.5, color="green",
            weight="bold",
        )

    ax.set_xticks(x)
    # Wrap long architecture labels
    short_labels = []
    for a in arch_order:
        if "QM9-small" in a:
            short_labels.append("QM9-small\n(4L / 64ch / lmax=4 / mmax=2)")
        elif "OC20-31M" in a:
            short_labels.append("OC20 EqV2-31M\n(8L / 128ch / lmax=4 / mmax=2)\n[public ckpt]")
        elif "fairchem-default" in a:
            short_labels.append("fairchem default\n(12L / 128ch / lmax=6 / mmax=2)")
        else:
            short_labels.append(a)
    ax.set_xticklabels(short_labels, fontsize=9)

    ax.set_ylabel("Forward wall-clock per batch=8 (ms, 95% CI)", fontsize=11)
    ax.set_title(
        "Quadrature-swap savings across EquiformerV2 architectures\n"
        "Architecture-level forward (random-init), "
        "$n=3$ runs $\\times$ 20 forwards on distinct QM9 batches each, "
        "NVIDIA A100-SXM4-40GB",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(
        max(data["configs"][a]["results"]["DH 2x"]["mean"]
            for a in arch_order) * 1.30,
        180,
    ))

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "fig_sweep_models.png"
    out_pdf = out_dir / "fig_sweep_models.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
