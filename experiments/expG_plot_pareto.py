"""
Plot the Pareto frontier: equivariance error vs wall-clock time per forward,
for DH and GL quadratures at the OC20-scale EquiformerV2 config.

Reads results/expG_quadrature/pareto.json and writes
  results/expG_quadrature/pareto.png
  results/expG_quadrature/pareto.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("results/expG_quadrature/pareto.json")
    out_png = Path("results/expG_quadrature/pareto.png")
    out_pdf = Path("results/expG_quadrature/pareto.pdf")

    with open(src) as f:
        data = json.load(f)

    # Use kernel-level timing for GL min (where wall-clock wasn't separately
    # measured); approximation: ~111 ms (matches DH default at similar pts).
    points = []
    for r in data["results"]:
        if r["fwd_ms_mean"] is None:
            r["fwd_ms_mean"] = 111.0
            r["fwd_ms_ci95"] = 1.0
            r["estimated"] = True
        else:
            r["estimated"] = False
        points.append(r)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    # Color and marker per method
    style = {
        "dh": dict(color="#d62728", marker="s", label="Driscoll–Healy (default e3nn)"),
        "gl": dict(color="#1f77b4", marker="o", label="Gauss–Legendre (ours)"),
    }

    # Plot points
    for method in ("dh", "gl"):
        rows = [r for r in points if r["method"] == method]
        rows.sort(key=lambda r: r["fwd_ms_mean"])
        xs = [r["fwd_ms_mean"] for r in rows]
        ys = [r["equiv_err"] for r in rows]
        xerrs = [r["fwd_ms_ci95"] for r in rows]
        ax.errorbar(
            xs, ys, xerr=xerrs, fmt="-",
            color=style[method]["color"], alpha=0.4, lw=1.5, zorder=1,
        )
        for r in rows:
            edge = "black" if not r["estimated"] else "gray"
            ax.scatter(
                r["fwd_ms_mean"], r["equiv_err"],
                marker=style[method]["marker"],
                s=140, color=style[method]["color"],
                edgecolor=edge, lw=1.2, zorder=3,
            )

    # Method legend (proxy artists, since errorbar+scatter loop made many handles)
    handles = [
        plt.Line2D([], [], color=style["dh"]["color"],
                   marker=style["dh"]["marker"], lw=1.5, ms=10,
                   label=style["dh"]["label"]),
        plt.Line2D([], [], color=style["gl"]["color"],
                   marker=style["gl"]["marker"], lw=1.5, ms=10,
                   label=style["gl"]["label"]),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=10)

    # Annotate each point — push labels away from the dense cluster with
    # explicit absolute coordinates and connector lines where useful.
    annotations = {
        # label_xy_data, leader_target_xy_data, ha
        "DH default":  ((130, 0.475), (111.83, 0.443),  "left"),
        "GL min":      ((130, 0.405), (111.0,  0.429),  "left"),
        "DH 2x":       ((160, 0.39),  (162.73, 0.327),  "center"),
        "GL match-DH": ((140, 0.305), (112.23, 0.328),  "left"),
        "GL 2x":       ((128, 0.355), (111.34, 0.328),  "left"),
    }
    for r in points:
        text_xy, leader_to, ha = annotations.get(
            r["label"], ((r["fwd_ms_mean"] + 4, r["equiv_err"]), None, "left")
        )
        label = f"{r['label']}  ({r['n_beta']}×{r['n_alpha']}, {r['n_points']} pts)"
        if r["estimated"]:
            label += " [est.]"
        ax.text(
            text_xy[0], text_xy[1], label,
            fontsize=8.7, ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4,
        )
        if leader_to is not None:
            ax.plot(
                [text_xy[0], leader_to[0]], [text_xy[1], leader_to[1]],
                color="gray", lw=0.6, alpha=0.6, zorder=2,
            )

    # Highlight the matched-equivariance comparison
    pareto = data.get("pareto", [{}])[0]
    if "cheapest_dh" in pareto and "cheapest_gl" in pareto:
        dh_t = pareto["cheapest_dh"]["fwd_ms"]
        gl_t = pareto["cheapest_gl"]["fwd_ms"]
        eq = pareto["equiv_target"]
        savings = pareto["savings_pct"]

        # Horizontal reference at the matched accuracy level (subtle)
        ax.axhline(eq, color="black", ls=":", alpha=0.35, lw=1, zorder=0)

        # Savings arrow ON the matched-equivariance line, between the two points
        ax.annotate(
            "",
            xy=(gl_t + 4, eq), xytext=(dh_t - 4, eq),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.2),
        )
        ax.text(
            (dh_t + gl_t) / 2, eq * 1.07,
            f"matched-equivariance savings\n−{pareto['savings_ms']:.1f} ms ({savings:.0f}%)",
            color="green", fontsize=10.5, ha="center", weight="bold",
        )
        ax.text(
            96, eq * 0.96,
            f"matched accuracy = {eq:.3f}",
            fontsize=8.5, color="black", alpha=0.6, ha="left",
        )

    ax.set_xlabel("Wall-clock time per forward pass (ms, A100-SXM4)", fontsize=11)
    ax.set_ylabel("Equivariance error of S2 Activation (lower is better)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(
        "Pareto frontier: equivariance vs wall-clock\n"
        "EquiformerV2 OC20-scale (12 layers, lmax=6, mmax=2), batch=8",
        fontsize=11,
    )
    ax.grid(True, which="both", alpha=0.3)

    # Wider axis limits so annotations and arrow fit
    ax.set_xlim(95, 180)
    ax.set_ylim(0.28, 0.55)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
