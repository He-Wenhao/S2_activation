"""
Rigorous wall-clock benchmark, V2 — addresses adversarial review.

Fixes vs the V1 protocol (expG_verify_speedup.py):
  1. **Independent batches per iteration**: V1 cycled through 3 fixed
     batches across 30 iterations × 3 runs, so the 90 timings were not
     IID. V2 samples a fresh batch per iteration (no batch is reused
     within a run) and re-seeds across runs.
  2. **Run-mean CI, not pooled-iteration CI**: V1 reported
     `1.96 * std / sqrt(n_iter)` over 90 correlated samples, which
     understates uncertainty. V2 reports
     `t_{n_runs-1, 0.975} * s.e.(run means)` — the canonical IID CI
     where the unit of replication is a run.
  3. **GL min is measured directly**: V1 left it null and the plot
     script silently filled 111 ms. V2 measures it under the same
     protocol as everything else.
  4. **Architecture-level claim made explicit**: this benchmark
     instantiates fresh `QM9Model` objects (random init) and times
     forward passes. The numbers are properties of the architecture and
     batch shape, not of trained weights.

Outputs:
  results/expG_quadrature/verify_speedup_v2.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
torch.serialization.add_safe_globals([slice])

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "experiments"))

from src.equiformer_grid_patch import patch_so3_grid
from expF_equiformerv2_qm9 import (
    QM9Model, BACKBONE_DEFAULTS, patch_s2_activations, load_qm9, qm9_adapt,
)
from torch_geometric.loader import DataLoader

try:
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def t_critical(df: int, alpha: float = 0.025) -> float:
    """Two-sided 95% CI critical value."""
    if HAVE_SCIPY:
        return float(_scipy_stats.t.ppf(1 - alpha, df))
    # Fallback: hard-coded values for small df
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(df, 1.96)


def sample_independent_batches(dataset, idx_pool, batch_size, n_batches,
                                target_idx, device, rng_seed: int):
    """Draw n_batches non-overlapping mini-batches of size batch_size."""
    g = torch.Generator().manual_seed(rng_seed)
    n_needed = n_batches * batch_size
    if n_needed > len(idx_pool):
        # Sample with replacement of molecules but rotate through to
        # get n_batches distinct batches
        molec_idx = torch.randint(0, len(idx_pool), (n_needed,), generator=g).tolist()
    else:
        perm = torch.randperm(len(idx_pool), generator=g).tolist()
        molec_idx = [idx_pool[i] for i in perm[:n_needed]]

    batches = []
    for i in range(n_batches):
        chunk = molec_idx[i * batch_size:(i + 1) * batch_size]
        loader = DataLoader([dataset[j] for j in chunk],
                              batch_size=batch_size, shuffle=False, num_workers=0)
        b = next(iter(loader)).to(device)
        b, _ = qm9_adapt(b, target_idx)
        batches.append(b)
    return batches


def time_run(grid_config, dataset, idx_pool, target_idx, backbone_kwargs, device,
             rng_seed: int, n_warmup: int, n_iter: int, batch_size: int):
    """One independent run: fresh model, fresh batches, return per-iter times."""
    backbone = dict(backbone_kwargs)
    if grid_config["method"] == "dh" and grid_config.get("resolution"):
        backbone["grid_resolution"] = grid_config["resolution"]
    model = QM9Model(backbone).to(device).eval()
    patch_s2_activations(model, "SiLU")
    if grid_config["method"] == "gl":
        patch_so3_grid(model, method="gl",
                        n_beta=grid_config.get("n_beta"),
                        n_alpha=grid_config.get("n_alpha"))

    # Sample distinct batches: enough for warmup + measurement, no reuse.
    n_total = n_warmup + n_iter
    batches = sample_independent_batches(
        dataset, idx_pool, batch_size, n_total, target_idx, device,
        rng_seed=rng_seed,
    )

    # Warmup
    with torch.no_grad():
        for i in range(n_warmup):
            _ = model(batches[i])
    cuda_sync()

    # Measure
    times = []
    with torch.no_grad():
        for i in range(n_warmup, n_warmup + n_iter):
            cuda_sync()
            t0 = time.perf_counter()
            _ = model(batches[i])
            cuda_sync()
            times.append((time.perf_counter() - t0) * 1000)

    del model
    torch.cuda.empty_cache()
    return np.array(times)


def aggregate(per_run_times):
    """Compute run-mean CI and within-run std diagnostics."""
    run_means = np.array([r.mean() for r in per_run_times])
    pooled = np.concatenate(per_run_times)
    n_runs = len(run_means)

    summary = {
        "n_runs": n_runs,
        "iter_per_run": len(per_run_times[0]),
        "n_total_iter": len(pooled),
        "run_means": run_means.tolist(),
        "mean_of_run_means": float(run_means.mean()),
        "se_run_means": float(run_means.std(ddof=1) / math.sqrt(n_runs)) if n_runs > 1 else float("nan"),
        "ci95_run_means": float(t_critical(n_runs - 1) * run_means.std(ddof=1) / math.sqrt(n_runs)) if n_runs > 1 else float("nan"),
        "within_run_std_mean": float(np.mean([r.std() for r in per_run_times])),
        "pooled_mean": float(pooled.mean()),
        "pooled_std": float(pooled.std()),
    }
    return summary


def compare(s_a, s_b):
    """Welch's t comparison between two configs from their run means."""
    a = np.array(s_a["run_means"])
    b = np.array(s_b["run_means"])
    diff = a.mean() - b.mean()
    if HAVE_SCIPY and len(a) > 1 and len(b) > 1:
        t_stat, p = _scipy_stats.ttest_ind(a, b, equal_var=False)
        # 95% CI for the difference of means (Welch)
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        df_welch = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))**2 / (
            (a.var(ddof=1) / len(a))**2 / (len(a) - 1) +
            (b.var(ddof=1) / len(b))**2 / (len(b) - 1)
        )
        ci = t_critical(int(df_welch)) * se
    else:
        t_stat, p, ci = float("nan"), float("nan"), float("nan")
    return {"diff_ms": float(diff), "t_stat": float(t_stat), "p_value": float(p),
            "ci95_diff": float(ci)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_runs", type=int, default=5,
                    help="Independent re-runs per config (the unit of replication)")
    p.add_argument("--n_iter", type=int, default=30,
                    help="Distinct batches measured per run (no batch reused)")
    p.add_argument("--n_warmup", type=int, default=10,
                    help="Warmup forwards per run, also on distinct batches")
    p.add_argument("--out", type=str,
                    default="results/expG_quadrature/verify_speedup_v2.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"
    print(f"GPU: {gpu_name}")
    print(f"Protocol: {args.n_runs} independent runs × "
          f"{args.n_warmup} warmup + {args.n_iter} measured forwards")
    print("Each forward uses a *distinct* QM9 batch (no batch reuse within a run).")
    print("Architecture-level timing (random-init QM9Model, no checkpoint loaded).\n")

    # OC20-default backbone
    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    backbone_kwargs["num_layers"] = 12
    backbone_kwargs["sphere_channels"] = 128
    backbone_kwargs["attn_hidden_channels"] = 128
    backbone_kwargs["ffn_hidden_channels"] = 256
    backbone_kwargs["num_heads"] = 8
    backbone_kwargs["lmax_list"] = [6]
    backbone_kwargs["mmax_list"] = [2]
    LMAX = 6

    # Load QM9 once; share dataset across runs but resample batches.
    dataset, target_idx, train_idx, _, _ = load_qm9("U0", seed=42)
    idx_pool = list(train_idx)
    n_batches_needed_per_run = args.n_warmup + args.n_iter
    print(f"QM9 train pool: {len(idx_pool)} molecules, "
          f"need {n_batches_needed_per_run * args.batch_size} per run "
          f"(pool covers all configs).\n")

    configs = [
        ("DH default",   {"method": "dh", "resolution": None}),
        ("DH 2x",        {"method": "dh", "resolution": 4 * (LMAX + 1)}),
        ("GL min",       {"method": "gl", "n_beta": LMAX + 1, "n_alpha": 2 * LMAX + 1}),
        ("GL match-DH",  {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 2 * LMAX + 1}),
        ("GL 2x",        {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 4 * LMAX + 4}),
    ]

    summaries = {}
    raw = {}
    print(f"{'Config':<13s} {'Run-mean (ms)':>16s} {'95% CI (n=runs)':>18s} {'within-run std':>17s}")
    print("-" * 70)
    for label, cfg in configs:
        per_run = []
        for run_id in range(args.n_runs):
            seed = 1000 * (run_id + 1) + hash(label) % 1000
            t = time_run(cfg, dataset, idx_pool, target_idx, backbone_kwargs,
                         device, rng_seed=seed,
                         n_warmup=args.n_warmup, n_iter=args.n_iter,
                         batch_size=args.batch_size)
            per_run.append(t)
        s = aggregate(per_run)
        summaries[label] = s
        raw[label] = [r.tolist() for r in per_run]
        print(f"{label:<13s} {s['mean_of_run_means']:>13.2f}    "
              f"± {s['ci95_run_means']:>5.2f}        "
              f"{s['within_run_std_mean']:>5.2f}")

    # Comparisons against DH default
    print(f"\n{'Comparison':<35s} {'Δ (ms)':>10s} {'95% CI':>10s} {'p-value':>10s} {'sig.':>6s}")
    print("-" * 75)
    base = summaries["DH default"]
    comparisons = {}
    for label, s in summaries.items():
        if label == "DH default":
            continue
        cmp = compare(s, base)
        comparisons[label] = cmp
        sig = "**" if (not math.isnan(cmp["p_value"])) and cmp["p_value"] < 0.05 else "n.s."
        print(f"{label + ' vs DH default':<35s} {cmp['diff_ms']:>+10.2f} "
              f"±{cmp['ci95_diff']:>5.2f}    {cmp['p_value']:>9.4f}   {sig}")

    # Pareto comparison: GL match-DH vs DH 2x at matched equivariance floor
    if "DH 2x" in summaries and "GL match-DH" in summaries:
        cmp = compare(summaries["GL match-DH"], summaries["DH 2x"])
        savings_pct = 100 * cmp["diff_ms"] / summaries["DH 2x"]["mean_of_run_means"]
        print("\nMatched-equivariance savings:")
        print(f"  GL match-DH vs DH 2x: {cmp['diff_ms']:+.2f} ms "
              f"(±{cmp['ci95_diff']:.2f}, p={cmp['p_value']:.4f}) "
              f"= {savings_pct:+.1f}% of DH 2x time")

    out = {
        "args": vars(args),
        "device": gpu_name,
        "backbone": {k: str(v) for k, v in backbone_kwargs.items()},
        "summaries": summaries,
        "comparisons_vs_dh_default": comparisons,
        "raw_iter_times_ms": raw,
        "scipy_available": HAVE_SCIPY,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
