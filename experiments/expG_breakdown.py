"""
Per-operation breakdown of EquiformerV2 forward pass.

Hooks into every S2Activation module to time its actual forward (not a
synthetic standalone benchmark). Reports:

  - Per-call time of each S2 activation
  - Total S2-act time per forward
  - Total forward time
  - Fraction of forward that is S2 activation
  - "Everything else" time (attention + FFN + embeddings + norms)

For each grid configuration. This answers: does S2 activation dominate?
"""

import argparse
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
from fairchem.core.models.equiformer_v2.activation import S2Activation


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class S2ActTimer:
    """Wraps S2Activation.forward to record per-call times via CUDA events."""
    def __init__(self):
        self.starts = []
        self.ends = []
        self.handles = []

    def hook(self, model):
        s2act_modules = [m for m in model.modules() if isinstance(m, S2Activation)]
        for mod in s2act_modules:
            orig_forward = mod.forward

            def wrapped_forward(*args, _orig=orig_forward, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = _orig(*args, **kwargs)
                end.record()
                self.starts.append(start)
                self.ends.append(end)
                return out

            mod.forward = wrapped_forward
            self.handles.append((mod, orig_forward))
        return len(s2act_modules)

    def collect_ms(self):
        cuda_sync()
        per_call_ms = [s.elapsed_time(e) for s, e in zip(self.starts, self.ends)]
        return per_call_ms

    def reset(self):
        self.starts.clear()
        self.ends.clear()


def run_config(label, grid_config, batches, backbone_kwargs, device,
               n_warmup=20, n_iter=20):
    backbone = dict(backbone_kwargs)
    if grid_config["method"] == "dh" and grid_config.get("resolution"):
        backbone["grid_resolution"] = grid_config["resolution"]
    model = QM9Model(backbone).to(device).eval()
    patch_s2_activations(model, "SiLU")
    if grid_config["method"] == "gl":
        patch_so3_grid(model, method="gl",
                       n_beta=grid_config.get("n_beta"),
                       n_alpha=grid_config.get("n_alpha"))

    timer = S2ActTimer()
    n_modules = timer.hook(model)

    # Warmup
    for i in range(n_warmup):
        with torch.no_grad():
            _ = model(batches[i % len(batches)])
    timer.reset()
    cuda_sync()

    # Measure
    fwd_times = []
    s2_total_per_fwd = []
    for i in range(n_iter):
        timer.reset()
        cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(batches[i % len(batches)])
        cuda_sync()
        fwd_times.append((time.perf_counter() - t0) * 1000)
        per_call = timer.collect_ms()
        s2_total_per_fwd.append(sum(per_call))

    fwd = np.array(fwd_times)
    s2 = np.array(s2_total_per_fwd)

    print(f"\n=== {label} ===")
    print(f"  S2 modules:               {n_modules}")
    print(f"  Full forward:             {fwd.mean():.2f} ± {fwd.std():.2f} ms")
    print(f"  Total S2-act per fwd:     {s2.mean():.2f} ± {s2.std():.2f} ms"
          f"  ({100 * s2.mean() / fwd.mean():.1f}% of fwd)")
    print(f"  Per-call S2-act:          {s2.mean() / n_modules:.3f} ms")
    print(f"  Everything else (fwd-S2): {fwd.mean() - s2.mean():.2f} ms")

    del model
    torch.cuda.empty_cache()
    return {
        "label": label, "n_modules": n_modules,
        "fwd_ms": float(fwd.mean()), "fwd_std": float(fwd.std()),
        "s2_total_ms": float(s2.mean()), "s2_std": float(s2.std()),
        "s2_per_call_ms": float(s2.mean() / n_modules),
        "non_s2_ms": float(fwd.mean() - s2.mean()),
        "s2_fraction": float(s2.mean() / fwd.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_iter", type=int, default=20)
    p.add_argument("--n_warmup", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    backbone_kwargs = dict(BACKBONE_DEFAULTS)
    backbone_kwargs["num_layers"] = 12
    backbone_kwargs["sphere_channels"] = 128
    backbone_kwargs["attn_hidden_channels"] = 128
    backbone_kwargs["ffn_hidden_channels"] = 256
    backbone_kwargs["num_heads"] = 8
    backbone_kwargs["lmax_list"] = [6]
    backbone_kwargs["mmax_list"] = [2]
    LMAX = 6

    dataset, target_idx, train_idx, _, _ = load_qm9("U0", seed=42)
    loader = DataLoader(dataset[train_idx[: args.batch_size * 4]],
                        batch_size=args.batch_size, shuffle=False, num_workers=0)
    batches = []
    for b in loader:
        b = b.to(device)
        b, _ = qm9_adapt(b, target_idx)
        batches.append(b)
    batches = batches[:3]

    configs = [
        ("DH default",   {"method": "dh", "resolution": None}),
        ("DH 2x",        {"method": "dh", "resolution": 4 * (LMAX + 1)}),
        ("GL match-DH",  {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 2 * LMAX + 1}),
        ("GL 2x",        {"method": "gl", "n_beta": 2 * (LMAX + 1), "n_alpha": 4 * LMAX + 4}),
    ]

    results = []
    for label, cfg in configs:
        torch.cuda.empty_cache()
        results.append(run_config(label, cfg, batches, backbone_kwargs, device,
                                   n_warmup=args.n_warmup, n_iter=args.n_iter))

    # Summary table
    print(f"\n{'='*85}")
    print(f"{'Config':<14s} {'Fwd (ms)':>9s} {'S2 total (ms)':>15s} "
          f"{'Per-call (ms)':>15s} {'%S2':>6s} {'Non-S2 (ms)':>13s}")
    print("-" * 85)
    for r in results:
        print(f"{r['label']:<14s} {r['fwd_ms']:>9.2f} {r['s2_total_ms']:>15.2f} "
              f"{r['s2_per_call_ms']:>15.3f} {100*r['s2_fraction']:>5.1f}% "
              f"{r['non_s2_ms']:>13.2f}")

    # Sanity check: non-S2 time should be roughly constant across configs
    non_s2 = [r["non_s2_ms"] for r in results]
    print(f"\nNon-S2 time across configs: "
          f"mean={np.mean(non_s2):.2f}, std={np.std(non_s2):.2f} "
          f"({100 * np.std(non_s2) / np.mean(non_s2):.1f}% rel)")

    import json
    out_path = Path("results/expG_quadrature/breakdown.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
