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
        "DH default":   ("#d62728", "//"),
        "DH 2x":        ("#a02020", "xx"),
        "GL match-DH":  ("#1f77b4", ""),
        "GL 2x":        ("#114488", ""),
    }

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bar_w = 0.20
    x = np.arange(len(arch_order))

    # Plot bars
    for i, label in enumerate(grid_order):
        means = [data["configs"][a]["results"][label]["mean"] for a in arch_order]
        cis = [data["configs"][a]["results"][label]["ci95"] for a in arch_order]
        offset = (i - 1.5) * bar_w
        color, hatch = grid_styles[label]
        bars = ax.bar(
            x + offset, means, bar_w,
            yerr=cis, capsize=3, label=label,
            color=color, hatch=hatch, edgecolor="black", linewidth=0.6,
            alpha=0.95, error_kw=dict(elinewidth=1.0, capthick=1.0),
        )

    # Annotate savings on top of each architecture group
    for i, a in enumerate(arch_order):
        sav = data["configs"][a]["matched_equiv_savings"]
        dh2_mean = data["configs"][a]["results"]["DH 2x"]["mean"]
        gl_mean = data["configs"][a]["results"]["GL match-DH"]["mean"]
        ymax = max(dh2_mean, gl_mean) * 1.08
        ax.annotate(
            f"GL→DH 2× saves\n{sav['savings_ms']:+.1f} ms ({sav['savings_pct']:+.1f}%)",
            xy=(x[i], ymax + 8), ha="center", fontsize=9.5, color="green",
            weight="bold",
        )

    ax.set_xticks(x)
    # Wrap long architecture labels
    short_labels = []
    for a in arch_order:
        if "QM9-small" in a:
            short_labels.append("QM9-small\n(4L / 64ch / lmax=4)")
        elif "OC20-31M" in a:
            short_labels.append("OC20 EqV2-31M\n(8L / 128ch / lmax=4)\n[public ckpt]")
        elif "fairchem-default" in a:
            short_labels.append("fairchem default\n(12L / 128ch / lmax=6)")
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
    for out in [Path("results/expG_quadrature/sweep_models.png"),
                 Path("results/expG_quadrature/sweep_models.pdf"),
                 Path("paper/figures/fig_sweep_models.png"),
                 Path("paper/figures/fig_sweep_models.pdf")]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
