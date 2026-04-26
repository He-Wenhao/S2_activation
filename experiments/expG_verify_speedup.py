"""
Rigorous verification of the DH-default vs DH-2x wall-clock gap.

What could make the +46% number an artifact:
  (1) Order effects: caching / JIT warmup helping later configs
  (2) Insufficient warmup
  (3) Run-to-run variance dominating the gap
  (4) Differing batch contents (different molecules)

This script:
  - Re-runs each config N=3 times in *different orders* (shuffled, normal, reverse)
  - Uses 20-iteration warmup, 30-iteration measurement per run
  - Reports per-run mean and aggregated mean ± std across runs
  - Tests both the OC20-scale config and the small QM9 config
  - Uses identical batch content for all configs in a run
"""

import argparse
import json
import random
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


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, n_warmup=20, n_iter=30):
    for _ in range(n_warmup):
        fn()
    cuda_sync()
    times = []
    for _ in range(n_iter):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def make_model(grid_config, backbone_kwargs, device):
    backbone = dict(backbone_kwargs)
    if grid_config["method"] == "dh" and grid_config.get("resolution"):
        backbone["grid_resolution"] = grid_config["resolution"]
    model = QM9Model(backbone).to(device)
    patch_s2_activations(model, "SiLU")
    if grid_config["method"] == "gl":
        patch_so3_grid(model, method="gl",
                       n_beta=grid_config.get("n_beta"),
                       n_alpha=grid_config.get("n_alpha"))
    return model


def time_one_config(grid_config, batches, backbone_kwargs, device,
                    n_warmup=20, n_iter=30):
    model = make_model(grid_config, backbone_kwargs, device)
    model.eval()
    iter_idx = [0]

    def fwd():
        with torch.no_grad():
            _ = model(batches[iter_idx[0] % len(batches)])
        iter_idx[0] += 1

    times = time_fn(fwd, n_warmup=n_warmup, n_iter=n_iter)
    del model
    torch.cuda.empty_cache()
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["small", "fairchem_default"],
                    default="fairchem_default")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_repeats", type=int, default=3,
                    help="How many independent runs (each is shuffled order)")
    p.add_argument("--n_iter", type=int, default=30)
    p.add_argument("--n_warmup", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # Configure backbone
    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    if args.config == "fairchem_default":
        backbone_kwargs["num_layers"] = 12
        backbone_kwargs["sphere_channels"] = 128
        backbone_kwargs["attn_hidden_channels"] = 128
        backbone_kwargs["ffn_hidden_channels"] = 256
        backbone_kwargs["num_heads"] = 8
        backbone_kwargs["lmax_list"] = [6]
        backbone_kwargs["mmax_list"] = [2]
        LMAX, MMAX = 6, 2
    else:
        LMAX, MMAX = 4, 2
    print(f"Config: {args.config}, batch={args.batch_size}, "
          f"n_warmup={args.n_warmup}, n_iter={args.n_iter}, n_repeats={args.n_repeats}")
    print(f"  num_layers={backbone_kwargs['num_layers']}, "
          f"sphere_channels={backbone_kwargs['sphere_channels']}, "
          f"lmax={LMAX}, mmax={MMAX}\n")

    # Pre-fetch batches once and re-use across all configs (identical workload)
    dataset, target_idx, train_idx, _, _ = load_qm9("U0", seed=42)
    loader = DataLoader(dataset[train_idx[: args.batch_size * 4]],
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    batches = []
    for b in loader:
        b = b.to(device)
        b, _ = qm9_adapt(b, target_idx)
        batches.append(b)
    batches = batches[:3]
    print(f"Pre-fetched {len(batches)} identical batches for all configs\n")

    configs = {
        "DH default":   {"method": "dh", "resolution": None},
        "DH 2x":        {"method": "dh", "resolution": 4 * (LMAX + 1)},
        "GL match-DH":  {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 2 * LMAX + 1},
        "GL 2x":        {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 4 * LMAX + 4},
    }

    # Three orderings: forward, reverse, shuffled (with seeded permutation)
    order_seeds = [None, "reverse", 42]
    all_results = {label: [] for label in configs}

    for repeat in range(args.n_repeats):
        seed_or_mode = order_seeds[repeat % len(order_seeds)]
        labels = list(configs.keys())
        if seed_or_mode == "reverse":
            order = labels[::-1]
            order_label = "reverse"
        elif seed_or_mode is None:
            order = labels[:]
            order_label = "forward"
        else:
            rng = random.Random(seed_or_mode)
            order = labels[:]
            rng.shuffle(order)
            order_label = f"shuffled(seed={seed_or_mode})"

        print(f"--- Repeat {repeat+1}/{args.n_repeats}: {order_label} order ---")
        for label in order:
            cfg = configs[label]
            times = time_one_config(cfg, batches, backbone_kwargs, device,
                                     n_warmup=args.n_warmup, n_iter=args.n_iter)
            all_results[label].append(times)
            print(f"  {label:<13s}: {times.mean():>7.2f} ± {times.std():.2f} ms "
                  f"(min {times.min():.2f}, max {times.max():.2f})")
        print()

    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATE across {args.n_repeats} repeats × {args.n_iter} iterations each")
    print(f"{'='*70}")
    print(f"{'Config':<13s} {'Mean ms':>10s} {'Std (across runs)':>20s} {'CI 95%':>15s}")
    print("-" * 65)

    summary = {}
    for label in configs:
        per_run_means = np.array([r.mean() for r in all_results[label]])
        per_run_std_in = np.array([r.std() for r in all_results[label]])
        # Pooled iterations
        all_iter = np.concatenate(all_results[label])
        mean = all_iter.mean()
        std_total = all_iter.std()
        ci95 = 1.96 * std_total / np.sqrt(len(all_iter))
        summary[label] = {
            "mean_ms": float(mean),
            "std_ms": float(std_total),
            "ci95_ms": float(ci95),
            "per_run_means": per_run_means.tolist(),
            "n_iterations_total": len(all_iter),
        }
        print(f"{label:<13s} {mean:>10.2f} {std_total:>10.2f} ({per_run_means.std():.2f} run-to-run)"
              f" {ci95:>9.2f}")

    # Speedup analysis
    base = summary["DH default"]["mean_ms"]
    base_ci = summary["DH default"]["ci95_ms"]
    print(f"\n{'Comparison vs DH default':<35s} {'Δ ms':>10s} {'Δ %':>10s}")
    print("-" * 60)
    for label, s in summary.items():
        if label == "DH default":
            continue
        delta = s["mean_ms"] - base
        delta_pct = 100 * delta / base
        ci_combined = np.sqrt(s["ci95_ms"]**2 + base_ci**2)
        # Is the difference statistically significant?
        sig = "**" if abs(delta) > ci_combined else "(not sig.)"
        print(f"{label:<35s} {delta:>+10.2f} {delta_pct:>+9.1f}%  {sig} (combined CI ±{ci_combined:.2f} ms)")

    out = {
        "args": vars(args),
        "summary": summary,
        "raw_iterations": {k: [r.tolist() for r in v] for k, v in all_results.items()},
    }
    out_path = Path(f"results/expG_quadrature/verify_speedup_{args.config}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
