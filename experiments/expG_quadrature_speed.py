"""
Experiment G: Does swapping the SO3_Grid quadrature actually speed up
EquiformerV2 inference / training?

The S2 Activation is one operation in many (attention, FFN, embedding, ...).
The grid forward/backward cost is linear in the total number of grid points.
We measure:

  1. S2 Activation alone (forward + backward) — best case
  2. Full EquiformerV2 forward — realistic inference
  3. Full forward + backward + optimizer step — realistic training

at multiple grid configurations, on real QM9 batches.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
torch.serialization.add_safe_globals([slice])

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "experiments"))

from src.equiformer_grid_patch import patch_so3_grid, CustomSO3Grid
from expF_equiformerv2_qm9 import (
    QM9Model, BACKBONE_DEFAULTS, patch_s2_activations, load_qm9, qm9_adapt,
)
from torch_geometric.loader import DataLoader
from fairchem.core.models.equiformer_v2.so3 import SO3_Grid


# ─── Benchmark utilities ────────────────────────────────────────────────────

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, n_warmup=3, n_iter=20):
    """Time a callable; returns (mean_ms, std_ms)."""
    for _ in range(n_warmup):
        fn()
    cuda_sync()
    times = []
    for _ in range(n_iter):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    import numpy as np
    return float(np.mean(times)), float(np.std(times))


# ─── Benchmark 1: S2 Activation alone ──────────────────────────────────────

def bench_s2act_alone(grid, batch_size=1024, n_channels=64, lmax=4, mmax=2,
                       device="cuda"):
    """Time the to-grid → SiLU → from-grid pipeline in isolation."""
    # Number of (l, m) coefficients with l <= lmax, |m| <= mmax
    n_coeffs = sum(min(2 * l + 1, 2 * mmax + 1) for l in range(lmax + 1))

    x = torch.randn(batch_size, n_coeffs, n_channels, device=device,
                     requires_grad=True)
    to_mat = grid.get_to_grid_mat(device=None).to(device)
    from_mat = grid.get_from_grid_mat(device=None).to(device)

    def fwd():
        g = torch.einsum("bai, zic -> zbac", to_mat, x)
        g = torch.nn.functional.silu(g)
        out = torch.einsum("bai, zbac -> zic", from_mat, g)
        return out

    def fwd_bwd():
        out = fwd()
        loss = out.pow(2).mean()
        loss.backward()
        x.grad = None
        return loss

    fwd_ms, _ = time_fn(fwd, n_iter=30)
    fwd_bwd_ms, _ = time_fn(fwd_bwd, n_iter=30)
    return fwd_ms, fwd_bwd_ms


# ─── Benchmark 2/3: Full EquiformerV2 forward / backward ───────────────────

def bench_full_model(grid_config, batch_size=64, n_iter=20, device="cuda"):
    """
    Time full EquiformerV2 forward (and forward+backward+step) on real QM9.

    grid_config: dict with method/n_beta/n_alpha or method='dh' for default
    """
    # Build model with default DH grid first
    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    if grid_config["method"] == "dh" and grid_config.get("resolution"):
        backbone_kwargs["grid_resolution"] = grid_config["resolution"]
    model = QM9Model(backbone_kwargs).to(device)
    patch_s2_activations(model, "SiLU")

    if grid_config["method"] == "gl":
        n_replaced = patch_so3_grid(
            model, method="gl",
            n_beta=grid_config.get("n_beta"),
            n_alpha=grid_config.get("n_alpha"),
        )

    # Load a small chunk of QM9 for batches
    dataset, target_idx, train_idx, val_idx, test_idx = load_qm9("U0", seed=42)
    loader = DataLoader(
        dataset[train_idx[:batch_size * 4]],
        batch_size=batch_size, shuffle=False, num_workers=0,
    )
    batches = []
    for b in loader:
        b = b.to(device)
        b, _ = qm9_adapt(b, target_idx)
        batches.append(b)
    batches = batches[:3]  # rotate through 3 batches to avoid trivial caching

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Forward only (eval mode)
    model.eval()
    iter_idx = [0]

    def fwd_only():
        with torch.no_grad():
            _ = model(batches[iter_idx[0] % len(batches)])
        iter_idx[0] += 1

    fwd_ms, fwd_std = time_fn(fwd_only, n_warmup=5, n_iter=n_iter)

    # Forward + backward + optimizer step (training mode)
    model.train()

    def fwd_bwd_step():
        b = batches[iter_idx[0] % len(batches)]
        optimizer.zero_grad()
        y = model(b)
        loss = (y.squeeze() - b.y[:, target_idx]).pow(2).mean()
        loss.backward()
        optimizer.step()
        iter_idx[0] += 1

    train_ms, train_std = time_fn(fwd_bwd_step, n_warmup=5, n_iter=n_iter)

    return {
        "forward_ms": fwd_ms,
        "forward_std": fwd_std,
        "train_step_ms": train_ms,
        "train_step_std": train_std,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--out", type=str,
                    default="results/expG_quadrature/speed_benchmark.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    LMAX, MMAX = 4, 2

    # Configurations to benchmark
    configs = [
        ("DH default",   {"method": "dh", "resolution": None}, 10, 5),
        ("DH 2x",        {"method": "dh", "resolution": 4 * (LMAX + 1)}, 20, 20),
        ("GL min",       {"method": "gl", "n_beta": LMAX + 1, "n_alpha": 2 * LMAX + 1}, LMAX + 1, 2 * LMAX + 1),
        ("GL match-N_b", {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 2 * LMAX + 1}, 10, 9),
        ("GL 2x",        {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 4 * LMAX + 4}, 10, 20),
    ]

    results = {"device": str(device), "batch_size": args.batch_size,
               "n_iter": args.n_iter, "configs": []}

    print("\n=== Benchmark 1: S2 Activation alone ===")
    print(f"{'Config':<14s} {'Pts':>5s} {'Fwd (ms)':>10s} {'Fwd+Bwd (ms)':>14s}")
    print("-" * 50)
    for label, cfg, n_b, n_a in configs:
        if cfg["method"] == "dh":
            grid = SO3_Grid(LMAX, MMAX, resolution=cfg.get("resolution"))
        else:
            grid = CustomSO3Grid(LMAX, MMAX, method="gl",
                                  n_beta=cfg.get("n_beta"),
                                  n_alpha=cfg.get("n_alpha"))
        npts = grid.get_to_grid_mat(device=None).shape[0] * \
               grid.get_to_grid_mat(device=None).shape[1]
        fwd, fb = bench_s2act_alone(grid, batch_size=1024, device=device)
        print(f"{label:<14s} {npts:>5d} {fwd:>10.3f} {fb:>14.3f}")
        results["configs"].append({
            "label": label, "config": cfg, "n_beta": n_b, "n_alpha": n_a,
            "n_points": npts, "s2act_fwd_ms": fwd, "s2act_fwd_bwd_ms": fb,
        })

    print("\n=== Benchmark 2: Full EquiformerV2 (QM9 batch) ===")
    print(f"{'Config':<14s} {'Pts':>5s} {'Fwd (ms)':>10s} {'Train step (ms)':>17s}")
    print("-" * 55)
    for i, (label, cfg, n_b, n_a) in enumerate(configs):
        r = bench_full_model(cfg, batch_size=args.batch_size, n_iter=args.n_iter,
                              device=device)
        results["configs"][i].update(r)
        print(f"{label:<14s} {results['configs'][i]['n_points']:>5d} "
              f"{r['forward_ms']:>10.3f} {r['train_step_ms']:>17.3f}")

    # Speedup analysis vs DH default
    base = results["configs"][0]
    print("\n=== Speedup vs DH default ===")
    print(f"{'Config':<14s} {'Pts ratio':>10s} {'S2act fwd':>11s} "
          f"{'Full fwd':>10s} {'Train step':>12s}")
    print("-" * 65)
    for c in results["configs"]:
        print(f"{c['label']:<14s} "
              f"{c['n_points'] / base['n_points']:>10.2f}x "
              f"{base['s2act_fwd_ms'] / c['s2act_fwd_ms']:>10.2f}x "
              f"{base['forward_ms'] / c['forward_ms']:>9.2f}x "
              f"{base['train_step_ms'] / c['train_step_ms']:>11.2f}x")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
