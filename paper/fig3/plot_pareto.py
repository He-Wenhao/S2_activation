"""
Plot the Pareto frontier: equivariance error vs wall-clock time per forward,
for DH and GL quadratures at the OC20-scale EquiformerV2 config.

Reads results/expG_quadrature/pareto.json and writes
  paper/fig3/fig_pareto.png
  paper/fig3/fig_pareto.pdf
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    src = Path("results/expG_quadrature/pareto.json")
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / "fig_pareto.png"
    out_pdf = out_dir / "fig_pareto.pdf"

    with open(src) as f:
        data = json.load(f)

    # Only plot points with directly-measured wall-clock. If a config has
    # `fwd_ms_mean=None` in pareto.json, we OMIT it from the Pareto figure
    # rather than silently substituting an estimate (per adversarial review).
    points = []
    omitted = []
    for r in data["results"]:
        if r["fwd_ms_mean"] is None:
            omitted.append(r["label"])
            continue
        # Normalize new expG_pareto.py format (config dict + equiv_err_mean)
        # to the keys the plotter expects.
        if "method" not in r and "config" in r:
            r["method"] = r["config"].get("method", "")
            r["n_beta"] = r["config"].get("n_beta") or 0
            r["n_alpha"] = r["config"].get("n_alpha") or 0
        if "equiv_err" not in r and "equiv_err_mean" in r:
            r["equiv_err"] = r["equiv_err_mean"]
        r["estimated"] = False
        points.append(r)
    if omitted:
        print(f"Omitting (no measured wall-clock): {omitted}")

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
    # Leader-line targets are pulled from the actual measured (x, y) so we
    # don't hard-code stale coordinates.
    by_label = {r["label"]: r for r in points}

    def _leader(label_):
        r = by_label.get(label_)
        return (r["fwd_ms_mean"], r["equiv_err"]) if r is not None else None

    annotations = {
        # text_xy_data, draw_leader: bool, ha
        "DH default":  ((130, 0.475), True,  "left"),
        "GL min":      ((130, 0.405), True,  "left"),
        "DH 2x":       ((160, 0.39),  True,  "center"),
        "GL match-DH": ((140, 0.305), True,  "left"),
        "GL 2x":       ((128, 0.355), True,  "left"),
    }
    for r in points:
        text_xy, draw_leader, ha = annotations.get(
            r["label"], ((r["fwd_ms_mean"] + 4, r["equiv_err"]), False, "left")
        )
        method_short = "DH" if r["method"] == "dh" else (
            "GL" if r["method"] == "gl" else r["method"].upper())
        label = f"{method_short} ×{r['n_points']} pts"
        ax.text(
            text_xy[0], text_xy[1], label,
            fontsize=8.7, ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4,
        )
        leader_to = _leader(r["label"]) if draw_leader else None
        if leader_to is not None:
            ax.plot(
                [text_xy[0], leader_to[0]], [text_xy[1], leader_to[1]],
                color="gray", lw=0.6, alpha=0.6, zorder=2,
            )

    # Highlight the matched-equivariance comparison
    pareto_list = data.get("pareto") or []
    pareto = pareto_list[0] if pareto_list else {}
    if "cheapest_dh" not in pareto or "cheapest_gl" not in pareto:
        # New pareto.json format has no "pareto" block — synthesize.
        dh_pts = [r for r in points if r["method"] == "dh"]
        gl_pts = [r for r in points if r["method"] == "gl"]
        if dh_pts and gl_pts:
            target = max(min(r["equiv_err"] for r in dh_pts),
                         min(r["equiv_err"] for r in gl_pts))
            dh_ok = [r for r in dh_pts if r["equiv_err"] <= target + 1e-6]
            gl_ok = [r for r in gl_pts if r["equiv_err"] <= target + 1e-6]
            if dh_ok and gl_ok:
                dh_best = min(dh_ok, key=lambda r: r["fwd_ms_mean"])
                gl_best = min(gl_ok, key=lambda r: r["fwd_ms_mean"])
                pareto = {
                    "cheapest_dh": {"label": dh_best["label"], "fwd_ms": dh_best["fwd_ms_mean"]},
                    "cheapest_gl": {"label": gl_best["label"], "fwd_ms": gl_best["fwd_ms_mean"]},
                    "equiv_target": target,
                    "savings_ms": dh_best["fwd_ms_mean"] - gl_best["fwd_ms_mean"],
                    "savings_pct": 100 * (dh_best["fwd_ms_mean"] - gl_best["fwd_ms_mean"]) / dh_best["fwd_ms_mean"],
                }
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
        "EquiformerV2 fairchem default (12 layers, lmax=6, mmax=2), batch=8",
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
