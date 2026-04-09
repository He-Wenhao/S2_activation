"""
Experiment C: Equivariance Error Measurement

Directly measure how well S2Activation commutes with SO(3) rotations:
  Equivariance requires: S2Act(D·x) = D·S2Act(x)
  Error = ||S2Act(D·x) - D·S2Act(x)|| / ||S2Act(D·x)||

Chain: coefficient error → equivariance breaking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.serialization
torch.serialization.add_safe_globals([slice])

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from e3nn import o3

from src.quadrature_methods import get_sampling
from src.spherical_harmonics_utils import (
    generate_random_coefficients,
    expand_coefficients_to_sphere,
    project_to_coefficients,
)


def get_nonlinearities():
    return {
        'ReLU': torch.relu,
        'abs': torch.abs,
        'SiLU': nn.SiLU(),
        'GELU': nn.GELU(),
        'tanh': torch.tanh,
        'Softplus_1': nn.Softplus(beta=1),
        'Softplus_10': nn.Softplus(beta=10),
        'Softplus_100': nn.Softplus(beta=100),
    }


def build_wigner_D(l_max, R):
    """Build block-diagonal Wigner-D matrix for rotation R."""
    angles = o3.matrix_to_angles(R)
    blocks = [o3.wigner_D(l, *angles) for l in range(l_max + 1)]
    return torch.block_diag(*blocks).double()


def s2_activation(coeffs_in, l_max, act_fn, pts, wts):
    """Full S2Activation: coeffs -> sphere -> nonlinearity -> coeffs."""
    f_vals = expand_coefficients_to_sphere(coeffs_in, pts, l_max)
    with torch.no_grad():
        g_vals = act_fn(f_vals)
    coeffs_out = project_to_coefficients(g_vals, pts, wts, l_max)
    return coeffs_out


def run_experiment_C(output_dir='results'):
    print("=" * 60)
    print("Experiment C: Equivariance Error Measurement")
    print("=" * 60)

    l_max_values = [3, 6, 10]
    num_rotations = 20
    num_inputs = 10
    nonlinearities = get_nonlinearities()

    # Sampling configurations
    def get_sampling_configs(l_max):
        configs = []
        for mult in [1, 2, 3]:
            lm = mult * l_max
            try:
                pts, wts = get_sampling('gauss_legendre', l_max=lm)
                configs.append((f'GL_{mult}x', pts, wts))
            except Exception:
                pass

        valid_degs = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                     35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]
        for mult_label, deg_target in [('Leb_min', 2*l_max+1), ('Leb_2x', 4*l_max+1)]:
            candidates = [d for d in valid_degs if d >= deg_target]
            if candidates:
                best_deg = candidates[0]
            else:
                best_deg = valid_degs[-1]
            try:
                pts, wts = get_sampling('lebedev', degree=best_deg)
                configs.append((f'{mult_label}_d{best_deg}', pts, wts))
            except Exception:
                pass

        for res in [50]:
            try:
                pts, wts = get_sampling('uniform', resolution=res)
                configs.append((f'Uniform_{res}', pts, wts))
            except Exception:
                pass

        return configs

    # Pre-generate random rotations
    torch.manual_seed(42)
    rotations = [o3.rand_matrix().double() for _ in range(num_rotations)]

    all_results = {}

    for l_max in l_max_values:
        print(f"\n--- l_max = {l_max} ---")
        all_results[l_max] = {}
        configs = get_sampling_configs(l_max)

        # Pre-compute Wigner-D matrices
        D_matrices = [build_wigner_D(l_max, R) for R in rotations]

        for act_name, act_fn in nonlinearities.items():
            all_results[l_max][act_name] = {}

            for cfg_name, pts, wts in configs:
                equiv_errors = []

                for seed in range(num_inputs):
                    coeffs_in = generate_random_coefficients(l_max, 'random_normal', seed=seed)

                    for rot_idx, (R, D) in enumerate(zip(rotations, D_matrices)):
                        # Path A: rotate then activate
                        rotated_input = D @ coeffs_in
                        path_A = s2_activation(rotated_input, l_max, act_fn, pts, wts)

                        # Path B: activate then rotate
                        activated = s2_activation(coeffs_in, l_max, act_fn, pts, wts)
                        path_B = D @ activated

                        # Equivariance error
                        err = (path_A - path_B).norm().item()
                        norm = path_A.norm().item() + 1e-30
                        equiv_errors.append(err / norm)

                mean_err = float(np.mean(equiv_errors))
                std_err = float(np.std(equiv_errors))

                all_results[l_max][act_name][cfg_name] = {
                    'mean_equiv_error': mean_err,
                    'std_equiv_error': std_err,
                    'n_points': len(pts),
                }

                print(f"  {act_name:15s} | {cfg_name:20s} | N={len(pts):5d} | "
                      f"equiv_err={mean_err:.4e} ± {std_err:.4e}")

    # Save results
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/expC_equivariance.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Sanity checks
    sanity_check_C(all_results)

    # Plots
    plot_equiv_vs_activation(all_results, output_dir)
    plot_equiv_vs_oversampling(all_results, output_dir)
    plot_equiv_vs_leakage(all_results, output_dir)

    return all_results


def sanity_check_C(results):
    print("\n--- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    for l_max in results:
        data = results[l_max]

        # Check 1: Equivariance error should decrease with oversampling (GL_1x > GL_3x)
        checks_total += 1
        improved = 0
        total = 0
        for act_name in data:
            if 'GL_1x' in data[act_name] and 'GL_3x' in data[act_name]:
                e1 = data[act_name]['GL_1x']['mean_equiv_error']
                e3 = data[act_name]['GL_3x']['mean_equiv_error']
                total += 1
                if e3 <= e1 * 1.05:  # allow 5% tolerance
                    improved += 1
        ok = improved > total * 0.5
        print(f"  [{'PASS' if ok else 'FAIL'}] GL_3x ≤ GL_1x equivariance in {improved}/{total} "
              f"activations at l_max={l_max}")
        if ok:
            checks_passed += 1

        # Check 2: Smooth activations should have better equivariance
        checks_total += 1
        if 'Softplus_1' in data and 'ReLU' in data:
            cfg = 'GL_1x'
            if cfg in data['Softplus_1'] and cfg in data['ReLU']:
                e_smooth = data['Softplus_1'][cfg]['mean_equiv_error']
                e_sharp = data['ReLU'][cfg]['mean_equiv_error']
                ok = e_smooth < e_sharp * 1.1  # allow some tolerance
                print(f"  [{'PASS' if ok else 'FAIL'}] Softplus_1 equiv ({e_smooth:.4e}) ≤ "
                      f"ReLU equiv ({e_sharp:.4e}) at l_max={l_max}")
                if ok:
                    checks_passed += 1

        # Check 3: Equivariance error should be > 0 (nonlinearity breaks it)
        checks_total += 1
        all_nonzero = all(
            data[act]['GL_1x']['mean_equiv_error'] > 1e-10
            for act in data if 'GL_1x' in data[act]
        )
        print(f"  [{'PASS' if all_nonzero else 'FAIL'}] All equivariance errors > 0 at l_max={l_max}")
        if all_nonzero:
            checks_passed += 1

    print(f"\n  Sanity checks: {checks_passed}/{checks_total} passed")


def plot_equiv_vs_activation(results, output_dir):
    """Plot C1: equivariance error for each activation, grouped by sampling."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax_idx, (l_max, data) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[ax_idx]
        act_names = sorted(data.keys())
        cfg_names = ['GL_1x', 'GL_2x', 'GL_3x']

        x = np.arange(len(act_names))
        width = 0.25

        for i, cfg in enumerate(cfg_names):
            vals = []
            errs = []
            for act in act_names:
                if cfg in data[act]:
                    vals.append(data[act][cfg]['mean_equiv_error'])
                    errs.append(data[act][cfg]['std_equiv_error'])
                else:
                    vals.append(0)
                    errs.append(0)
            ax.bar(x + i * width, vals, width, yerr=errs, label=cfg, capsize=2, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(act_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Equivariance error')
        ax.set_title(f'l_max = {l_max}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Experiment C: Equivariance Error by Activation', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expC_equiv_vs_activation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expC_equiv_vs_activation.png")


def plot_equiv_vs_oversampling(results, output_dir):
    """Plot C2: equivariance error vs oversampling ratio."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    act_colors = {
        'ReLU': '#d62728', 'abs': '#ff7f0e', 'SiLU': '#1f77b4',
        'GELU': '#17becf', 'tanh': '#9467bd', 'Softplus_1': '#2ca02c',
        'Softplus_10': '#98df8a', 'Softplus_100': '#e74c3c',
    }

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax_idx, (l_max, data) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[ax_idx]
        for act_name, configs in data.items():
            mults = []
            errs = []
            for cfg_name, cfg_data in configs.items():
                if cfg_name.startswith('GL_') and cfg_name.endswith('x'):
                    mult = int(cfg_name.split('_')[1].replace('x', ''))
                    mults.append(mult)
                    errs.append(cfg_data['mean_equiv_error'])
            if mults:
                order = np.argsort(mults)
                ax.plot([mults[i] for i in order], [errs[i] for i in order],
                        marker='o', label=act_name,
                        color=act_colors.get(act_name, 'gray'), linewidth=1.5)

        ax.set_xlabel('GL oversampling ratio')
        ax.set_ylabel('Equivariance error')
        ax.set_yscale('log')
        ax.set_title(f'l_max = {l_max}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3])

    plt.suptitle('Experiment C: Equivariance Error vs Oversampling', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expC_equiv_vs_oversampling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expC_equiv_vs_oversampling.png")


def plot_equiv_vs_leakage(results, output_dir):
    """Plot C3: equivariance error vs leakage ratio (from Experiment A)."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    # Try to load Experiment A results
    leakage_file = f'{output_dir}/metrics/expA_spectral.json'
    if not os.path.exists(leakage_file):
        print("Skipping C3: Experiment A results not found")
        return

    with open(leakage_file) as f:
        leakage_data = json.load(f)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax_idx, (l_max, data) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[ax_idx]
        l_key = str(l_max)

        if l_key not in leakage_data:
            continue

        # Use GL_1x for equivariance (to see raw effect)
        cfg = 'GL_1x'
        xs, ys, labels = [], [], []
        for act_name in data:
            if cfg not in data[act_name]:
                continue
            if act_name not in leakage_data[l_key]:
                continue
            R = leakage_data[l_key][act_name]['leakage_ratio']
            E = data[act_name][cfg]['mean_equiv_error']
            xs.append(R)
            ys.append(E)
            labels.append(act_name)

        ax.scatter(xs, ys, s=50, zorder=3)
        for i, label in enumerate(labels):
            ax.annotate(label, (xs[i], ys[i]), fontsize=7, xytext=(5, 5),
                       textcoords='offset points')

        ax.set_xlabel('Leakage ratio R (from Exp A)')
        ax.set_ylabel('Equivariance error (GL_1x)')
        ax.set_title(f'l_max = {l_max}')
        ax.grid(True, alpha=0.3)

        # Fit correlation line if enough points
        if len(xs) >= 3:
            from scipy.stats import pearsonr
            r, p = pearsonr(xs, ys)
            ax.set_title(f'l_max = {l_max} (r={r:.2f}, p={p:.3f})')

    plt.suptitle('Experiment C: Equivariance Error vs Spectral Leakage', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expC_equiv_vs_leakage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expC_equiv_vs_leakage.png")


if __name__ == '__main__':
    run_experiment_C()
