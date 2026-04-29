"""
Plot the kernel-level Pareto frontier for all four (six) quadrature methods.

Reads results/expG_quadrature/four_way.json, writes
  results/expG_quadrature/four_way.png
  results/expG_quadrature/four_way.pdf
  paper/figures/fig_four_way.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("results/expG_quadrature/four_way.json")
    with open(src) as f:
        data = json.load(f)

    rows = data["results"]

    # Color-code by method family
    family_style = {
        "dh-dense": dict(color="#d62728", marker="s", label="Driscoll–Healy (dense, e3nn default)"),
        "dh-fft":   dict(color="#9467bd", marker="D", label="Driscoll–Healy (FFT, e3nn forward)"),
        "gl":       dict(color="#1f77b4", marker="o", label="Gauss–Legendre (dense, ours)"),
        "lebedev":  dict(color="#2ca02c", marker="^", label="Lebedev (dense, ours)"),
    }

    def family(label):
        if label.startswith("DH-FFT"):
            return "dh-fft"
        if label.startswith("DH"):
            return "dh-dense"
        if label.startswith("GL"):
            return "gl"
        if label.startswith("Lebedev"):
            return "lebedev"
        return "gl"

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Plot points and connect within each family
    by_family = {}
    for r in rows:
        fam = family(r["method"])
        by_family.setdefault(fam, []).append(r)

    # Per-family lines
    for fam, fam_rows in by_family.items():
        fam_rows.sort(key=lambda r: r["kernel_ms_mean"])
        if len(fam_rows) > 1:
            xs = [r["kernel_ms_mean"] for r in fam_rows]
            ys = [r["equiv_err_mean"] for r in fam_rows]
            ax.plot(xs, ys, "-", color=family_style[fam]["color"], alpha=0.4,
                     lw=1.5, zorder=1)

    # Scatter points
    for r in rows:
        fam = family(r["method"])
        st = family_style[fam]
        ax.errorbar(
            r["kernel_ms_mean"], r["equiv_err_mean"],
            xerr=r["kernel_ms_ci95"], fmt=st["marker"],
            color=st["color"], markersize=12,
            markeredgecolor="black", markeredgewidth=1.0, zorder=3,
        )

    # Legend (proxy artists)
    handles = [
        plt.Line2D([], [], color=family_style[k]["color"], marker=family_style[k]["marker"],
                    lw=1.5, ms=10, label=family_style[k]["label"])
        for k in ["dh-dense", "dh-fft", "gl", "lebedev"]
    ]
    ax.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=9)

    # Annotate each point
    annotations = {
        "DH-dense default":    (0.55, 0.470, "left",  True),
        "DH-dense match":      (1.40, 0.380, "left",  True),
        "DH-FFT":              (3.30, 0.300, "right", True),
        "GL match-DH":         (1.40, 0.305, "left",  True),
        "Lebedev (low)":       (0.30, 0.530, "left",  True),
    }
    # Find the high-Lebedev label dynamically
    leb_high_label = next((r["method"] for r in rows
                            if r["method"].startswith("Lebedev (d=")), None)
    if leb_high_label:
        annotations[leb_high_label] = (0.55, 0.345, "left", True)

    by_label = {r["method"]: r for r in rows}

    for label, (x_off_ms, y_data, ha, draw_leader) in annotations.items():
        if label not in by_label:
            continue
        r = by_label[label]
        text = f"{label}\n({r['n_points']} pts)"
        ax.text(
            x_off_ms, y_data, text,
            fontsize=8.5, ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.18",
                       facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4,
        )
        if draw_leader:
            ax.plot([x_off_ms, r["kernel_ms_mean"]],
                     [y_data, r["equiv_err_mean"]],
                     color="gray", lw=0.6, alpha=0.5, zorder=2)

    ax.set_xlabel("S2-Activation kernel time per forward call (ms, log scale)",
                   fontsize=11)
    ax.set_ylabel("Equivariance error of S2 Activation (lower is better)",
                   fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        f"Kernel-level Pareto: 4 quadrature methods at "
        f"$\\ell_{{\\max}}={data['lmax']}$, $m_{{\\max}}={data['mmax']}$"
        f" (S2-Activation alone, SiLU, batch={data['args']['batch']}, "
        f"{data['args']['n_channels']} ch, A100)",
        fontsize=10,
    )
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    for out in [Path("results/expG_quadrature/four_way.png"),
                 Path("results/expG_quadrature/four_way.pdf"),
                 Path("paper/figures/fig_four_way.png"),
                 Path("paper/figures/fig_four_way.pdf")]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
