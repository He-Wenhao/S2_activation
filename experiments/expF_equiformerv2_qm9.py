#!/usr/bin/env python
"""
Experiment F: EquiformerV2 on QM9 — Activation Swap & Oversampling Ablation

Tests the paper's S2 activation analysis on a real molecular property prediction
task using EquiformerV2 (a production SO(3)-equivariant GNN).

Tier 1 — Activation swap:
  Swap the S2 Activation nonlinearity in EquiformerV2 and measure downstream MAE
  on QM9 molecular properties with multiple seeds.

Tier 2 — Oversampling ablation:
  Vary the SO3_Grid resolution (controls quadrature oversampling) and measure
  how equivariance error at each layer correlates with downstream performance.

Usage:
  # Single run
  python expF_equiformerv2_qm9.py --act SiLU --grid default --target U0 --seed 42

  # Evaluate equivariance on a trained model
  python expF_equiformerv2_qm9.py --eval_equiv --checkpoint runs/SiLU_default_U0_42/best.pt

  # Run full sweep (use with SLURM array jobs)
  python expF_equiformerv2_qm9.py --sweep
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix e3nn / PyTorch >=2.6 compatibility
torch.serialization.add_safe_globals([slice])

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# EquiformerV2 imports
from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone
from fairchem.core.models.equiformer_v2.activation import (
    S2Activation as EqV2S2Activation,
    SeparableS2Activation as EqV2SepS2Activation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "expF"
DATA_DIR = ROOT / "data" / "qm9"

# QM9 target indices in data.y (PyG convention)
QM9_TARGET_IDX = {
    "mu": 0, "alpha": 1, "HOMO": 2, "LUMO": 3, "gap": 4,
    "R2": 5, "ZPVE": 6, "U0": 7, "U": 8, "H": 9, "G": 10,
    "Cv": 11,
}

# Units for reporting (eV for energies, meV for orbital energies)
QM9_UNITS = {
    "U0": "eV", "U": "eV", "H": "eV", "G": "eV",
    "HOMO": "eV", "LUMO": "eV", "gap": "eV",
    "ZPVE": "eV", "mu": "D", "alpha": "a0^3", "R2": "a0^2", "Cv": "cal/mol·K",
}

# Small EquiformerV2 config suitable for QM9 (trains in ~1-2h per run on 1 GPU)
BACKBONE_DEFAULTS = dict(
    use_pbc=False,
    use_pbc_single=False,
    regress_forces=False,
    otf_graph=True,
    max_neighbors=20,
    max_radius=5.0,
    max_num_elements=10,  # QM9: H(1), C(6), N(7), O(8), F(9)
    num_layers=4,
    sphere_channels=64,
    attn_hidden_channels=64,
    num_heads=4,
    attn_alpha_channels=16,
    attn_value_channels=16,
    ffn_hidden_channels=128,
    norm_type="rms_norm_sh",
    lmax_list=[4],
    mmax_list=[2],
    grid_resolution=None,  # default: 2*(lmax+1) for latitude
    num_sphere_samples=128,
    edge_channels=64,
    use_atom_edge_embedding=True,
    share_atom_edge_embedding=False,
    use_m_share_rad=False,
    distance_function="gaussian",
    num_distance_basis=128,
    attn_activation="scaled_silu",
    use_s2_act_attn=False,
    use_attn_renorm=True,
    ffn_activation="scaled_silu",
    use_gate_act=False,
    use_grid_mlp=False,
    use_sep_s2_act=True,
    alpha_drop=0.0,
    drop_path_rate=0.0,
    proj_drop=0.0,
    weight_init="normal",
    avg_num_nodes=18.0,  # QM9 average
    avg_degree=15.0,
)


# ─── Activation Definitions ────────────────────────────────────────────────

class TanhAct(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

class AbsAct(nn.Module):
    def forward(self, x):
        return torch.abs(x)

ACTIVATION_REGISTRY = {
    "SiLU": lambda: nn.SiLU(),
    "Softplus_1": lambda: nn.Softplus(beta=1),
    "Softplus_10": lambda: nn.Softplus(beta=10),
    "tanh": lambda: TanhAct(),
    "ReLU": lambda: nn.ReLU(),
    "abs": lambda: AbsAct(),
}


def patch_s2_activations(model, act_name):
    """Replace the S2 Activation nonlinearity in all EquiformerV2 layers.

    Only patches the activation applied on the sphere (to-grid → act → from-grid).
    Does NOT change the scalar activation in SeparableS2Activation, since that
    bypasses the S2 projection entirely.
    """
    act_fn = ACTIVATION_REGISTRY[act_name]()
    count = 0
    for module in model.modules():
        if isinstance(module, EqV2S2Activation):
            module.act = act_fn
            count += 1
    log.info(f"Patched {count} S2Activation modules → {act_name}")
    return count


# ─── Grid Resolution ────────────────────────────────────────────────────────

def grid_resolution_value(name, lmax):
    """Convert grid resolution name to the integer for SO3_Grid."""
    if name == "default":
        return None
    elif name == "2x":
        return 4 * (lmax + 1)
    elif name == "3x":
        return 6 * (lmax + 1)
    raise ValueError(f"Unknown grid resolution: {name}")


# ─── QM9 Data ──────────────────────────────────────────────────────────────

def load_qm9(target_name, seed=42):
    """Load QM9 and split into train/val/test."""
    dataset = QM9(root=str(DATA_DIR))
    target_idx = QM9_TARGET_IDX[target_name]

    # Standard random split: 110k / 10k / ~10k
    n = len(dataset)
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    train_idx = perm[:110000]
    val_idx = perm[110000:120000]
    test_idx = perm[120000:]

    return dataset, target_idx, train_idx, val_idx, test_idx


def qm9_adapt(batch, target_idx):
    """Adapt a PyG QM9 batch to the format EquiformerV2 expects.

    PyG DataLoader already produces a Batch object; we just add the fields
    that EquiformerV2Backbone.forward() needs.
    """
    batch.atomic_numbers = batch.z
    _, counts = torch.unique(batch.batch, return_counts=True)
    batch.natoms = counts
    targets = batch.y[:, target_idx]
    return batch, targets


# ─── Model ──────────────────────────────────────────────────────────────────

class QM9Model(nn.Module):
    """EquiformerV2 backbone + scalar head for QM9."""

    def __init__(self, backbone_kwargs, head_hidden=128):
        super().__init__()
        self.backbone = EquiformerV2Backbone(**backbone_kwargs)
        ch = backbone_kwargs["sphere_channels"]
        self.head = nn.Sequential(
            nn.Linear(ch, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, data):
        emb = self.backbone(data)
        node_emb = emb["node_embedding"]
        # Extract l=0 invariant features: embedding[:, 0, :]
        x = node_emb.embedding[:, 0, :]  # [N_nodes, sphere_channels]
        x = self.head(x).squeeze(-1)  # [N_nodes]
        # Sum over atoms per molecule
        out = torch.zeros(len(data.natoms), device=x.device, dtype=x.dtype)
        out.index_add_(0, data.batch, x)
        # Normalize by average number of atoms
        out = out / data.natoms.float().clamp(min=1)
        return out


# ─── Per-Layer Equivariance Measurement (Tier 2) ───────────────────────────

def random_rotation_matrix(device="cpu"):
    """Random SO(3) rotation via QR decomposition of a Gaussian matrix."""
    M = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diagonal(R))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@torch.no_grad()
def measure_equivariance(model, loader, device, n_rotations=10, n_batches=20):
    """Measure prediction invariance and per-layer l=0 invariance error.

    For a perfectly equivariant model predicting a scalar (invariant) property,
    rotating the input should not change the prediction. Equivariance violations
    in S2 Activation break this.

    We also hook into each transformer block to measure how much the l=0
    (invariant) component drifts under rotation at each layer — showing
    compounding equivariance error across depth.
    """
    model.eval()
    n_layers = len(model.backbone.blocks)

    layer_embs = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, out):
            # out is SO3_Embedding; l=0 is at position 0
            layer_embs[idx] = out.embedding[:, 0, :].detach().clone()
        return hook_fn

    for i, block in enumerate(model.backbone.blocks):
        hooks.append(block.register_forward_hook(make_hook(i)))

    pred_errors = []
    layer_errors = {i: [] for i in range(n_layers)}

    for batch_idx, batch_list in enumerate(loader):
        if batch_idx >= n_batches:
            break
        data, _ = qm9_adapt(batch_list, target_idx=0)  # target doesn't matter
        data = data.to(device)

        # Original forward
        layer_embs.clear()
        pred_orig = model(data)
        emb_orig = {k: v.clone() for k, v in layer_embs.items()}

        for _ in range(n_rotations):
            R = random_rotation_matrix(device)
            data_rot = data.clone()
            data_rot.pos = data.pos @ R.T

            layer_embs.clear()
            pred_rot = model(data_rot)

            # Prediction invariance error
            denom = pred_orig.abs().mean().clamp(min=1e-8)
            pred_errors.append((pred_orig - pred_rot).abs().mean().item() / denom.item())

            # Per-layer l=0 invariance error
            for i in range(n_layers):
                denom = emb_orig[i].norm().clamp(min=1e-8)
                diff = (emb_orig[i] - layer_embs[i]).norm().item() / denom.item()
                layer_errors[i].append(diff)

    for h in hooks:
        h.remove()

    return {
        "prediction_invariance_error": float(np.mean(pred_errors)),
        "prediction_invariance_std": float(np.std(pred_errors)),
        "per_layer_l0_error_mean": {i: float(np.mean(e)) for i, e in layer_errors.items()},
        "per_layer_l0_error_std": {i: float(np.std(e)) for i, e in layer_errors.items()},
    }


# ─── Training ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, target_idx, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch_list in loader:
        data, targets = qm9_adapt(batch_list, target_idx)
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = F.l1_loss(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n_samples += len(targets)

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, target_idx, device):
    model.eval()
    total_mae = 0.0
    n_samples = 0
    for batch_list in loader:
        data, targets = qm9_adapt(batch_list, target_idx)
        data, targets = data.to(device), targets.to(device)
        pred = model(data)
        total_mae += F.l1_loss(pred, targets, reduction="sum").item()
        n_samples += len(targets)
    return total_mae / n_samples


def run_training(args):
    """Run a single training experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Data ──
    dataset, target_idx, train_idx, val_idx, test_idx = load_qm9(
        args.target, seed=args.seed
    )
    # Use a list-based collate; we handle batching in qm9_adapt
    train_loader = DataLoader(
        dataset[train_idx], batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        dataset[val_idx], batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        dataset[test_idx], batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model ──
    lmax = BACKBONE_DEFAULTS["lmax_list"][0]
    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    backbone_kwargs["grid_resolution"] = grid_resolution_value(args.grid, lmax)

    model = QM9Model(backbone_kwargs).to(device)
    n_patched = patch_s2_activations(model, args.act)
    assert n_patched > 0, "No S2Activation modules found to patch"

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {n_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Output directory ──
    run_name = f"{args.act}_{args.grid}_{args.target}_seed{args.seed}"
    run_dir = RESULTS_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    best_val_mae = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_mae = train_one_epoch(model, train_loader, optimizer, target_idx, device)
        val_mae = evaluate(model, val_loader, target_idx, device)
        scheduler.step()

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            torch.save(model.state_dict(), run_dir / "best.pt")

        history.append({
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:3d} | train MAE {train_mae:.5f} | "
                f"val MAE {val_mae:.5f} {'*' if is_best else ''}"
            )

    train_time = time.time() - t0
    log.info(f"Training done in {train_time/60:.1f} min")

    # ── Test evaluation ──
    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=True))
    test_mae = evaluate(model, test_loader, target_idx, device)
    log.info(f"Test MAE: {test_mae:.5f} {QM9_UNITS.get(args.target, '')}")

    # ── Per-layer equivariance measurement ──
    log.info("Measuring per-layer equivariance error...")
    equiv_results = measure_equivariance(
        model, test_loader, device,
        n_rotations=args.n_rotations, n_batches=args.n_equiv_batches,
    )
    log.info(f"Prediction invariance error: {equiv_results['prediction_invariance_error']:.6f}")
    for layer_i, err in equiv_results["per_layer_l0_error_mean"].items():
        log.info(f"  Layer {layer_i}: l=0 invariance error = {err:.6f}")

    # ── Save results ──
    results = {
        "config": {
            "activation": args.act,
            "grid_resolution": args.grid,
            "target": args.target,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "n_params": n_params,
            "n_s2act_patched": n_patched,
            "backbone": {k: str(v) for k, v in backbone_kwargs.items()},
        },
        "results": {
            "best_val_mae": best_val_mae,
            "test_mae": test_mae,
            "train_time_sec": train_time,
            "units": QM9_UNITS.get(args.target, ""),
        },
        "equivariance": equiv_results,
        "history": history,
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {run_dir}")

    return results


# ─── Standalone Equivariance Eval ───────────────────────────────────────────

def run_eval_equiv(args):
    """Evaluate equivariance on a trained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    # Load config from results.json in the same directory
    results_path = ckpt_path.parent / "results.json"
    with open(results_path) as f:
        saved = json.load(f)

    cfg = saved["config"]
    lmax = BACKBONE_DEFAULTS["lmax_list"][0]
    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    backbone_kwargs["grid_resolution"] = grid_resolution_value(cfg["grid_resolution"], lmax)

    model = QM9Model(backbone_kwargs).to(device)
    patch_s2_activations(model, cfg["activation"])
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    dataset, target_idx, _, _, test_idx = load_qm9(cfg["target"], seed=cfg["seed"])
    test_loader = DataLoader(
        dataset[test_idx], batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    equiv = measure_equivariance(
        model, test_loader, device,
        n_rotations=args.n_rotations, n_batches=args.n_equiv_batches,
    )

    print(json.dumps(equiv, indent=2))
    return equiv


# ─── Aggregate Results ──────────────────────────────────────────────────────

def aggregate_results():
    """Collect all run results into a summary table."""
    runs_dir = RESULTS_DIR / "runs"
    if not runs_dir.exists():
        log.error(f"No runs found in {runs_dir}")
        return

    rows = []
    for run_dir in sorted(runs_dir.iterdir()):
        results_file = run_dir / "results.json"
        if not results_file.exists():
            continue
        with open(results_file) as f:
            r = json.load(f)
        cfg = r["config"]
        res = r["results"]
        eq = r.get("equivariance", {})
        rows.append({
            "activation": cfg["activation"],
            "grid": cfg["grid_resolution"],
            "target": cfg["target"],
            "seed": cfg["seed"],
            "test_mae": res["test_mae"],
            "best_val_mae": res["best_val_mae"],
            "pred_equiv_err": eq.get("prediction_invariance_error", None),
            "train_time": res["train_time_sec"],
        })

    if not rows:
        log.error("No results to aggregate")
        return

    # Group by (activation, grid, target) and compute mean/std
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r["activation"], r["grid"], r["target"])
        groups[key].append(r)

    summary = []
    for key, runs in sorted(groups.items()):
        act, grid, target = key
        maes = [r["test_mae"] for r in runs]
        equiv_errs = [r["pred_equiv_err"] for r in runs if r["pred_equiv_err"] is not None]
        summary.append({
            "activation": act,
            "grid": grid,
            "target": target,
            "n_seeds": len(runs),
            "test_mae_mean": float(np.mean(maes)),
            "test_mae_std": float(np.std(maes)),
            "equiv_err_mean": float(np.mean(equiv_errs)) if equiv_errs else None,
            "equiv_err_std": float(np.std(equiv_errs)) if equiv_errs else None,
        })

    out_file = RESULTS_DIR / "summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'Activation':<14} {'Grid':<8} {'Target':<6} {'Seeds':<5} "
          f"{'MAE (mean±std)':<20} {'Equiv Err':<12}")
    print("-" * 75)
    for s in summary:
        mae_str = f"{s['test_mae_mean']:.5f}±{s['test_mae_std']:.5f}"
        eq_str = f"{s['equiv_err_mean']:.6f}" if s["equiv_err_mean"] is not None else "N/A"
        print(f"{s['activation']:<14} {s['grid']:<8} {s['target']:<6} {s['n_seeds']:<5} "
              f"{mae_str:<20} {eq_str:<12}")

    log.info(f"Summary saved to {out_file}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Experiment F: EquiformerV2 on QM9")
    sub = p.add_subparsers(dest="command", help="Sub-command")

    # --- train ---
    tr = sub.add_parser("train", help="Train a single configuration")
    tr.add_argument("--act", type=str, default="SiLU",
                    choices=list(ACTIVATION_REGISTRY.keys()))
    tr.add_argument("--grid", type=str, default="default",
                    choices=["default", "2x", "3x"])
    tr.add_argument("--target", type=str, default="U0",
                    choices=list(QM9_TARGET_IDX.keys()))
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch_size", type=int, default=128)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--n_rotations", type=int, default=10,
                    help="Number of rotations for equivariance measurement")
    tr.add_argument("--n_equiv_batches", type=int, default=20,
                    help="Number of test batches for equivariance measurement")

    # --- eval_equiv ---
    eq = sub.add_parser("eval_equiv", help="Evaluate equivariance on a checkpoint")
    eq.add_argument("--checkpoint", type=str, required=True)
    eq.add_argument("--batch_size", type=int, default=64)
    eq.add_argument("--n_rotations", type=int, default=20)
    eq.add_argument("--n_equiv_batches", type=int, default=50)

    # --- aggregate ---
    sub.add_parser("aggregate", help="Aggregate results from all runs")

    # --- sweep ---
    sw = sub.add_parser("sweep", help="Print all (act, grid, seed) configs for job arrays")

    return p.parse_args()


def print_sweep():
    """Print all configurations for SLURM array job submission."""
    activations = ["SiLU", "Softplus_1", "Softplus_10", "tanh", "ReLU", "abs"]
    grids = ["default", "2x", "3x"]
    seeds = [42, 123, 456, 789, 1024]
    target = "U0"

    configs = []
    # Tier 1: all activations × default grid × 5 seeds
    for act in activations:
        for seed in seeds:
            configs.append(f"--act {act} --grid default --target {target} --seed {seed}")

    # Tier 2: SiLU + Softplus_1 × 2x/3x grids × 5 seeds
    for act in ["SiLU", "Softplus_1"]:
        for grid in ["2x", "3x"]:
            for seed in seeds:
                configs.append(f"--act {act} --grid {grid} --target {target} --seed {seed}")

    for i, cfg in enumerate(configs):
        print(f"{i}\t{cfg}")

    print(f"\n# Total configs: {len(configs)}", file=sys.stderr)


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "eval_equiv":
        run_eval_equiv(args)
    elif args.command == "aggregate":
        aggregate_results()
    elif args.command == "sweep":
        print_sweep()
    else:
        # Default: train with command-line args
        # Allow flat usage: python expF_... --act SiLU --seed 42
        p = argparse.ArgumentParser()
        p.add_argument("--act", default="SiLU", choices=list(ACTIVATION_REGISTRY.keys()))
        p.add_argument("--grid", default="default", choices=["default", "2x", "3x"])
        p.add_argument("--target", default="U0", choices=list(QM9_TARGET_IDX.keys()))
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--epochs", type=int, default=50)
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--n_rotations", type=int, default=10)
        p.add_argument("--n_equiv_batches", type=int, default=20)
        args = p.parse_args()
        run_training(args)
