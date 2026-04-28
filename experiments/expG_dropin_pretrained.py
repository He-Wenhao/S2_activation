"""
Drop-in test: load an actual trained EquiformerV2 checkpoint, swap the
SO3_Grid quadrature without retraining, measure how prediction invariance
error and test MAE change.

This addresses the reviewer's concern that the V1 wall-clock script never
loaded weights but the paper claimed "inside a trained EquiformerV2".

We use the 50-epoch DH-default checkpoints from Experiment F (4 layers,
lmax=4, mmax=2; not OC20-scale, but actually trained from random init to
convergence on QM9 U0). For each (activation, seed):

  1. Build the architecture with DH default grid.
  2. Load best.pt weights.
  3. Measure: test MAE, prediction invariance error.   --> reference numbers
  4. Patch the SO3_Grid to GL match-DH with the *same* weights — no
     fine-tuning, no retraining.
  5. Re-measure on the SAME molecules.
  6. Report deltas.

Outputs:
  results/expG_quadrature/dropin_pretrained.json
  console table per (activation, seed) and aggregate
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
    QM9_TARGET_IDX, measure_equivariance, evaluate, grid_resolution_value,
)
from torch_geometric.loader import DataLoader


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_one(model, loader, target_idx, device):
    """Test MAE."""
    return evaluate(model, loader, target_idx, device)


def build_and_load(checkpoint_dir: Path, activation: str, grid_name: str,
                    device, lmax: int):
    """Build EquiformerV2 with DH default grid, patch S2Activation, load weights."""
    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    backbone_kwargs["grid_resolution"] = grid_resolution_value(grid_name, lmax)
    model = QM9Model(backbone_kwargs).to(device)
    patch_s2_activations(model, activation)
    state = torch.load(checkpoint_dir / "best.pt",
                        map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, backbone_kwargs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 123, 456, 789, 1024])
    p.add_argument("--activation", type=str, default="SiLU")
    p.add_argument("--orig_grid", type=str, default="default",
                    choices=["default", "2x", "3x"],
                    help="Grid that was used at TRAIN time")
    p.add_argument("--target", type=str, default="U0")
    p.add_argument("--n_rotations", type=int, default=10)
    p.add_argument("--n_equiv_batches", type=int, default=20)
    p.add_argument("--n_test_batches", type=int, default=50,
                    help="How many test batches to use for MAE evaluation")
    p.add_argument("--out", type=str,
                    default="results/expG_quadrature/dropin_pretrained.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {args.activation} / {args.orig_grid} checkpoints from "
          f"results/expF/runs/...\n")

    # Load QM9 once
    dataset, target_idx, train_idx, val_idx, test_idx = load_qm9(
        args.target, seed=42,
    )
    test_loader = DataLoader(
        dataset[test_idx[:args.n_test_batches * 32]],
        batch_size=32, shuffle=False, num_workers=2,
    )

    LMAX = BACKBONE_DEFAULTS["lmax_list"][0]   # = 4 for the expF small model

    # GL grids to test (drop-in replacements at inference)
    gl_configs = [
        ("GL min",     {"n_beta": LMAX + 1,    "n_alpha": 2 * LMAX + 1}),
        ("GL match-DH",{"n_beta": 2 * (LMAX + 1), "n_alpha": 2 * LMAX + 1}),
        ("GL 2x",      {"n_beta": 2 * (LMAX + 1), "n_alpha": 4 * LMAX + 4}),
    ]

    rows = []
    for seed in args.seeds:
        run_name = f"{args.activation}_{args.orig_grid}_{args.target}_seed{seed}"
        run_dir = _repo_root / "results" / "expF" / "runs" / run_name
        if not (run_dir / "best.pt").exists():
            print(f"[skip] {run_name}: no best.pt")
            continue
        if not (run_dir / "results.json").exists():
            print(f"[skip] {run_name}: no results.json")
            continue
        with open(run_dir / "results.json") as f:
            train_log = json.load(f)
        print(f"\n=== {run_name}  (training MAE {train_log['results']['test_mae']:.3f} eV) ===")

        # Reference: original DH grid (re-measured with our protocol)
        model, _ = build_and_load(run_dir, args.activation, args.orig_grid, device, LMAX)
        mae_ref = evaluate_one(model, test_loader, target_idx, device)
        eq_ref = measure_equivariance(model, test_loader, device,
                                        n_rotations=args.n_rotations,
                                        n_batches=args.n_equiv_batches)
        print(f"  [{args.orig_grid:11s}]  "
              f"test MAE = {mae_ref:.4f},  pred-invariance = "
              f"{eq_ref['prediction_invariance_error']:.3e}")

        result_row = {
            "checkpoint": run_name, "seed": seed,
            "trained_with": args.orig_grid,
            "metrics": {
                args.orig_grid: {
                    "test_mae": float(mae_ref),
                    "pred_invariance_error": float(eq_ref["prediction_invariance_error"]),
                    "per_layer_l0_error_mean": eq_ref["per_layer_l0_error_mean"],
                }
            },
        }

        for label, cfg in gl_configs:
            # Re-build to get fresh weights, then drop in GL grid
            model, _ = build_and_load(run_dir, args.activation, args.orig_grid, device, LMAX)
            n_replaced = patch_so3_grid(model, method="gl", n_beta=cfg["n_beta"],
                                          n_alpha=cfg["n_alpha"])
            mae = evaluate_one(model, test_loader, target_idx, device)
            eq = measure_equivariance(model, test_loader, device,
                                        n_rotations=args.n_rotations,
                                        n_batches=args.n_equiv_batches)
            print(f"  [{label:11s}]  "
                  f"test MAE = {mae:.4f}  (Δ {mae - mae_ref:+.4f}),  "
                  f"pred-invariance = {eq['prediction_invariance_error']:.3e}  "
                  f"(× {eq['prediction_invariance_error'] / max(eq_ref['prediction_invariance_error'], 1e-30):.2f})")
            result_row["metrics"][label] = {
                "test_mae": float(mae),
                "test_mae_delta": float(mae - mae_ref),
                "pred_invariance_error": float(eq["prediction_invariance_error"]),
                "pred_invariance_ratio_to_ref": float(eq["prediction_invariance_error"] /
                                                       max(eq_ref["prediction_invariance_error"], 1e-30)),
                "per_layer_l0_error_mean": eq["per_layer_l0_error_mean"],
                "n_grids_replaced": int(n_replaced),
                "n_beta": cfg["n_beta"], "n_alpha": cfg["n_alpha"],
            }
            del model
            torch.cuda.empty_cache()
        rows.append(result_row)

    # Aggregate
    print(f"\n{'='*70}")
    print(f"AGGREGATE across seeds (mean ± std)")
    print(f"{'='*70}")
    print(f"{'Grid':<13s}  {'Test MAE Δ (eV)':>20s}  {'Inv-err ratio':>16s}  N seeds")
    print("-" * 70)
    grids_to_show = [args.orig_grid, "GL min", "GL match-DH", "GL 2x"]
    aggregate_table = {}
    for g in grids_to_show:
        if g == args.orig_grid:
            mae_d = np.array([0.0 for r in rows])
            inv_r = np.array([1.0 for r in rows])
        else:
            mae_d = np.array([r["metrics"][g]["test_mae_delta"] for r in rows])
            inv_r = np.array([r["metrics"][g]["pred_invariance_ratio_to_ref"] for r in rows])
        aggregate_table[g] = {
            "test_mae_delta_mean": float(mae_d.mean()),
            "test_mae_delta_std": float(mae_d.std(ddof=1)) if len(mae_d) > 1 else 0.0,
            "inv_err_ratio_mean": float(inv_r.mean()),
            "inv_err_ratio_std": float(inv_r.std(ddof=1)) if len(inv_r) > 1 else 0.0,
            "n_seeds": int(len(mae_d)),
        }
        print(f"{g:<13s}  {mae_d.mean():>+10.4f} ± {mae_d.std(ddof=1) if len(mae_d) > 1 else 0:.4f}  "
              f"{inv_r.mean():>10.3e}      {len(mae_d)}")

    out = {
        "args": vars(args),
        "device": str(device),
        "per_seed": rows,
        "aggregate": aggregate_table,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
