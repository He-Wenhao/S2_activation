"""
Sweep MODEL-LEVEL prediction-invariance error across the same three
EquiformerV2 architectures used in expG_sweep_models.py, at the four
grid configurations (DH default / DH 2x / GL match-DH / GL 2x).

For each (arch, grid) pair we build a random-init QM9Model, patch the
S2 activations to SiLU and the SO3_Grid to the requested method, and
call measure_equivariance() — for each of n_batches QM9 batches we run
the model on the original input and on n_rotations random SO(3)-rotated
copies, measuring relative prediction discrepancy.

Forward times are NOT measured here (those live in sweep_models.json);
we only measure model-level invariance error so that fig4 can plot
equiv error vs. forward time per architecture.

Output: results/expG_quadrature/sweep_equiv.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
torch.serialization.add_safe_globals([slice])

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "experiments"))

from src.equiformer_grid_patch import patch_so3_grid
from expF_equiformerv2_qm9 import (
    QM9Model, BACKBONE_DEFAULTS, patch_s2_activations,
    load_qm9, qm9_adapt, measure_equivariance,
)
from torch_geometric.loader import DataLoader


def make_loader(dataset, idx_pool, target_idx, batch_size, n_batches, seed, device):
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(idx_pool), generator=rng).tolist()
    take = [idx_pool[i] for i in perm[: n_batches * batch_size]]
    subset = [dataset[k] for k in take]
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_rotations", type=int, default=10)
    p.add_argument("--n_batches", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str,
                   default="results/expG_quadrature/sweep_equiv.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}\n")

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

    dataset, target_idx, train_idx, _, _ = load_qm9("U0", seed=42)
    idx_pool = list(train_idx)

    out = {"device": gpu, "args": vars(args), "configs": {}}

    for arch_name, arch_kwargs in arch_configs.items():
        print(f"\n=== {arch_name} ===")
        backbone_kwargs = dict(BACKBONE_DEFAULTS)
        backbone_kwargs.update(arch_kwargs)
        L = arch_kwargs["lmax_list"][0]

        grid_configs = [
            ("DH default", {"method": "dh", "resolution": None}),
            ("DH 2x",      {"method": "dh", "resolution": 4 * (L + 1)}),
            ("GL match-DH",{"method": "gl",
                              "n_beta": 2 * (L + 1), "n_alpha": 2 * L + 1}),
            ("GL 2x",      {"method": "gl",
                              "n_beta": 2 * (L + 1), "n_alpha": 4 * L + 4}),
        ]

        out["configs"][arch_name] = {"backbone": arch_kwargs, "results": {}}

        for label, gc in grid_configs:
            backbone = dict(backbone_kwargs)
            if gc["method"] == "dh" and gc.get("resolution"):
                backbone["grid_resolution"] = gc["resolution"]

            torch.manual_seed(args.seed)
            model = QM9Model(backbone).to(device).eval()
            patch_s2_activations(model, "SiLU")
            if gc["method"] == "gl":
                patch_so3_grid(model, method="gl",
                               n_beta=gc.get("n_beta"),
                               n_alpha=gc.get("n_alpha"))

            loader = make_loader(dataset, idx_pool, target_idx,
                                  args.batch_size, args.n_batches,
                                  args.seed, device)

            eq = measure_equivariance(
                model, loader, device,
                n_rotations=args.n_rotations,
                n_batches=args.n_batches,
            )

            err_mean = eq["prediction_invariance_error"]
            err_std = eq["prediction_invariance_std"]
            n_samples = args.n_rotations * args.n_batches
            ci95 = 1.96 * err_std / math.sqrt(max(n_samples, 1))

            out["configs"][arch_name]["results"][label] = {
                "pred_invariance_error_mean": float(err_mean),
                "pred_invariance_error_std": float(err_std),
                "pred_invariance_error_ci95": float(ci95),
                "n_samples": n_samples,
            }
            print(f"  {label:<13s}: pred-invariance = "
                  f"{err_mean:.3e} ± {ci95:.2e} (n={n_samples})")

            del model
            torch.cuda.empty_cache()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
