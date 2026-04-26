"""
Verify that the GL grid works as a drop-in replacement inside a real
EquiformerV2 model: forward pass produces finite outputs that change
predictably under rotation.
"""

import sys
import math
import torch

torch.serialization.add_safe_globals([slice])

sys.path.insert(0, "/pscratch/sd/w/whe1/S2_activation")
sys.path.insert(0, "/pscratch/sd/w/whe1/S2_activation/experiments")

from src.equiformer_grid_patch import patch_so3_grid

# Borrow model construction from expF
from expF_equiformerv2_qm9 import (
    QM9Model, BACKBONE_DEFAULTS, patch_s2_activations, load_qm9, QM9_TARGET_IDX,
    qm9_adapt,
)
from torch_geometric.loader import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model with default DH grid first
    model = QM9Model(BACKBONE_DEFAULTS).to(device)
    n_patched = patch_s2_activations(model, "SiLU")
    print(f"Patched {n_patched} S2 activations to SiLU")

    # Load a tiny piece of QM9 for a forward pass
    dataset, target_idx, train_idx, val_idx, test_idx = load_qm9("U0", seed=42)
    loader = DataLoader(dataset[test_idx[:2]], batch_size=2, shuffle=False)
    batch = next(iter(loader)).to(device)
    batch, _ = qm9_adapt(batch, target_idx)

    # ── Forward pass with original DH grid ──
    model.eval()
    with torch.no_grad():
        y_dh = model(batch).squeeze()
    print(f"\nDH (default e3nn) output: {y_dh.cpu().numpy()}")
    print(f"  SO3_Grid shape: {model.backbone.SO3_grid[0][0].to_grid_mat.shape}")

    # ── Patch to GL grid (default sizes) ──
    n_replaced = patch_so3_grid(model, method="gl")
    print(f"\nReplaced {n_replaced} grids with GL")
    print(f"  SO3_Grid shape: {model.backbone.SO3_grid[0][0].to_grid_mat.shape}")
    with torch.no_grad():
        y_gl = model(batch).squeeze()
    print(f"GL output: {y_gl.cpu().numpy()}")
    print(f"Difference DH vs GL: {(y_dh - y_gl).abs().max().item():.6e}")

    # ── Patch to higher-res GL grid ──
    n_replaced = patch_so3_grid(model, method="gl", n_beta=8, n_alpha=16)
    print(f"\nReplaced grids with GL (n_beta=8, n_alpha=16)")
    print(f"  SO3_Grid shape: {model.backbone.SO3_grid[0][0].to_grid_mat.shape}")
    with torch.no_grad():
        y_gl_hi = model(batch).squeeze()
    print(f"GL hi-res output: {y_gl_hi.cpu().numpy()}")

    # Both DH and GL should produce ROUGHLY similar outputs (since both are
    # valid quadratures), but they will differ slightly because aliasing
    # patterns are different.

    # ── Equivariance test on the trained-from-scratch (random) model ──
    # Random rotation of input atom positions; energy should be invariant
    from scipy.spatial.transform import Rotation
    rotation = torch.tensor(
        Rotation.random(random_state=0).as_matrix(),
        dtype=batch.pos.dtype, device=device,
    )
    pos_rot = batch.pos @ rotation.T
    batch_rot = batch.clone()
    batch_rot.pos = pos_rot
    batch_rot, _ = qm9_adapt(batch_rot, target_idx)

    # Test each grid configuration
    for label, n_beta, n_alpha in [("GL hi", 8, 16), ("GL min", None, None)]:
        patch_so3_grid(model, method="gl", n_beta=n_beta, n_alpha=n_alpha)
        with torch.no_grad():
            y0 = model(batch).squeeze()
            y_r = model(batch_rot).squeeze()
        diff = (y0 - y_r).abs() / (y0.abs() + 1e-12)
        print(f"\n{label}: pred invariance error = {diff.mean().item():.2e}")


if __name__ == "__main__":
    main()
