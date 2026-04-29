"""
Sweep wall-clock savings across published EquiformerV2 architectures.

Configs covered:
  1. QM9-small (our expF setting): 4 layers, 64 ch, lmax=4, mmax=2
  2. fairchem default (function signature): 12 layers, 128 ch, lmax=6, mmax=2
  3. EquiformerV2-31M (OC20, public ckpt): 8 layers, 128 ch, lmax=4, mmax=2,
     grid_resolution=18 (oversampled at training time!)
  4. EquiformerV2 OC20 ablation (table 9 of the paper): 12 layers, 128 ch,
     lmax=6, mmax=2 (= our fairchem default)

For each, time DH default vs GL match-DH vs DH 2x with the rigorous
n=3 runs × 20 distinct-batch protocol.

Outputs: results/expG_quadrature/sweep_models.json
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


def t_critical(df: int) -> float:
    if HAVE_SCIPY:
        return float(_scipy_stats.t.ppf(0.975, df))
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    return table.get(df, 1.96)


def time_run(grid_config, batches, backbone_kwargs, device,
             n_warmup=10, n_iter=20):
    backbone = dict(backbone_kwargs)
    if grid_config["method"] == "dh" and grid_config.get("resolution"):
        backbone["grid_resolution"] = grid_config["resolution"]
    model = QM9Model(backbone).to(device).eval()
    patch_s2_activations(model, "SiLU")
    if grid_config["method"] == "gl":
        patch_so3_grid(model, method="gl",
                        n_beta=grid_config.get("n_beta"),
                        n_alpha=grid_config.get("n_alpha"))
    elif grid_config["method"] == "lebedev":
        patch_so3_grid(model, method="lebedev",
                        lebedev_degree=grid_config.get("degree"))

    # Warmup
    with torch.no_grad():
        for i in range(n_warmup):
            _ = model(batches[i % len(batches)])
    cuda_sync()
    times = []
    with torch.no_grad():
        for i in range(n_iter):
            cuda_sync()
            t0 = time.perf_counter()
            _ = model(batches[(n_warmup + i) % len(batches)])
            cuda_sync()
            times.append((time.perf_counter() - t0) * 1000)
    n_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return np.array(times), n_params


def aggregate(per_run_ms):
    means = np.array([r.mean() for r in per_run_ms])
    n = len(means)
    return {
        "n_runs": n,
        "run_means": means.tolist(),
        "mean": float(means.mean()),
        "ci95": float(t_critical(n - 1) * means.std(ddof=1) / math.sqrt(n))
        if n > 1 else float("nan"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_runs", type=int, default=3)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--n_warmup", type=int, default=10)
    p.add_argument("--out", type=str,
                    default="results/expG_quadrature/sweep_models.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}\n")

    # Architecture configs (each one is a kwargs dict to override BACKBONE_DEFAULTS)
    arch_configs = {
        "QM9-small (4L/64ch/L4/M2, our expF model)": {
            "num_layers": 4, "sphere_channels": 64,
            "attn_hidden_channels": 64, "ffn_hidden_channels": 128,
            "num_heads": 4, "lmax_list": [4], "mmax_list": [2],
        },
        "OC20-31M-public (8L/128ch/L4/M2, eq2_31M_ec4_allmd.pt)": {
            "num_layers": 8, "sphere_channels": 128,
            "attn_hidden_channels": 64, "ffn_hidden_channels": 128,
            "num_heads": 8, "lmax_list": [4], "mmax_list": [2],
        },
        "fairchem-default (12L/128ch/L6/M2)": {
            "num_layers": 12, "sphere_channels": 128,
            "attn_hidden_channels": 128, "ffn_hidden_channels": 256,
            "num_heads": 8, "lmax_list": [6], "mmax_list": [2],
        },
    }

    # Load QM9 once for batches (any small molecular batch works for timing
    # — we are measuring architecture cost, not training).
    dataset, target_idx, train_idx, _, _ = load_qm9("U0", seed=42)
    idx_pool = list(train_idx)
    batch_size = 8
    n_batches_needed = args.n_warmup + args.n_iter

    out = {"device": gpu, "args": vars(args), "configs": {}}

    for arch_name, arch_kwargs in arch_configs.items():
        print(f"\n=== {arch_name} ===")
        backbone_kwargs = dict(BACKBONE_DEFAULTS)
        backbone_kwargs.update(arch_kwargs)
        L = arch_kwargs["lmax_list"][0]
        M = arch_kwargs["mmax_list"][0]

        # Per-arch grid configs
        grid_configs = [
            ("DH default", {"method": "dh", "resolution": None}),
            ("DH 2x",      {"method": "dh", "resolution": 4 * (L + 1)}),
            ("GL match-DH",{"method": "gl",
                             "n_beta": 2 * (L + 1), "n_alpha": 2 * L + 1}),
            ("GL 2x",      {"method": "gl",
                             "n_beta": 2 * (L + 1), "n_alpha": 4 * L + 4}),
        ]

        out["configs"][arch_name] = {"backbone": arch_kwargs, "results": {}}

        # Pre-fetch enough batches per run (re-sample per run for IID)
        n_pool_per_run = n_batches_needed
        rngs = [42 + 1000 * i for i in range(args.n_runs)]

        for label, gc in grid_configs:
            per_run = []
            n_params_check = None
            for run_id, seed in enumerate(rngs):
                # Sample a fresh batch sequence per run
                rng = torch.Generator().manual_seed(seed)
                perm = torch.randperm(len(idx_pool), generator=rng).tolist()
                idx_take = [idx_pool[i] for i in perm[:n_pool_per_run * batch_size]]
                batches = []
                for j in range(n_pool_per_run):
                    chunk = idx_take[j * batch_size:(j + 1) * batch_size]
                    loader = DataLoader([dataset[k] for k in chunk],
                                          batch_size=batch_size,
                                          shuffle=False, num_workers=0)
                    b = next(iter(loader)).to(device)
                    b, _ = qm9_adapt(b, target_idx)
                    batches.append(b)

                times, n_params = time_run(gc, batches, backbone_kwargs, device,
                                              n_warmup=args.n_warmup,
                                              n_iter=args.n_iter)
                per_run.append(times)
                if n_params_check is None:
                    n_params_check = n_params

            agg = aggregate(per_run)
            out["configs"][arch_name]["n_params"] = n_params_check
            out["configs"][arch_name]["results"][label] = agg
            print(f"  {label:<13s}: {agg['mean']:>7.2f} ± {agg['ci95']:.2f} ms"
                  f"   ({n_params_check:,} params total)")

        # Pareto computation: GL match-DH vs DH 2x
        gl = out["configs"][arch_name]["results"]["GL match-DH"]
        dh2 = out["configs"][arch_name]["results"]["DH 2x"]
        savings_ms = dh2["mean"] - gl["mean"]
        savings_pct = 100 * savings_ms / dh2["mean"]
        print(f"  → Matched-equiv savings (GL match-DH vs DH 2x): "
              f"{savings_ms:+.2f} ms = {savings_pct:+.1f}%")
        out["configs"][arch_name]["matched_equiv_savings"] = {
            "savings_ms": savings_ms, "savings_pct": savings_pct,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
