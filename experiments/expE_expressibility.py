"""
Experiment E: Expressibility of S2 Activation

Measure the effective rank of the Jacobian of S2Activation for each
activation function. The effective rank counts how many independent
directions in output SH coefficient space the operator can locally explore,
providing a measure of expressibility that is orthogonal to spectral leakage.

Metric: EffRank(J) = exp(H(p)) where p_i = s_i^2 / sum s_j^2 and
s_i are the singular values of the Jacobian J = dS2Act(c)/dc.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.serialization
torch.serialization.add_safe_globals([slice])
from e3nn import o3

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.s2_activation import S2Activation
from src.spherical_harmonics_utils import generate_random_coefficients


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


def effective_rank(singular_values):
    """
    Compute the effective rank from singular values.
    EffRank = exp(H(p)) where p_i = s_i^2 / sum(s_j^2).
    """
    s2 = singular_values ** 2
    total = s2.sum()
    if total < 1e-30:
        return 1.0
    p = s2 / total
    # Filter out zeros to avoid log(0)
    p = p[p > 1e-30]
    H = -(p * torch.log(p)).sum()
    return torch.exp(H).item()


def stable_rank(singular_values):
    """Stable rank = ||J||_F^2 / ||J||_2^2 = sum(s_i^2) / s_1^2."""
    s2 = singular_values ** 2
    if s2[0] < 1e-30:
        return 1.0
    return (s2.sum() / s2[0]).item()


def compute_jacobian_metrics(s2_act, coeffs_in):
    """
    Compute the Jacobian of S2Activation and return expressibility metrics.

    Uses autograd for correctness (handles normalize2mom wrapper).
    """
    c = coeffs_in.detach().float().requires_grad_(True)

    # Compute Jacobian column by column via backward passes
    d_in = c.shape[0]
    c_out = s2_act(c)
    d_out = c_out.shape[0]

    J = torch.zeros(d_out, d_in, dtype=torch.float32)
    for j in range(d_out):
        if c.grad is not None:
            c.grad.zero_()
        c_out[j].backward(retain_graph=True)
        J[j, :] = c.grad.clone()

    # SVD
    sv = torch.linalg.svdvals(J)

    return {
        'effective_rank': effective_rank(sv),
        'stable_rank': stable_rank(sv),
        'singular_values': sv.detach().cpu().numpy().tolist(),
        'spectral_gap': (sv[0] / sv[1]).item() if sv[1] > 1e-30 else float('inf'),
        'jacobian_frobenius': torch.norm(J, 'fro').item(),
    }


def run_experiment_E(output_dir='results'):
    print("=" * 60)
    print("Experiment E: Expressibility (Effective Rank of Jacobian)")
    print("=" * 60)

    l_max_values = [3, 6, 10]
    num_inputs = 50
    nonlinearities = get_nonlinearities()

    all_results = {}

    for l_max in l_max_values:
        print(f"\n--- l_max = {l_max} ---")
        all_results[l_max] = {}
        d = (l_max + 1) ** 2

        irreps = o3.Irreps([(1, (l, 1)) for l in range(l_max + 1)])

        for act_name, act_fn in nonlinearities.items():
            print(f"  {act_name}...", end='', flush=True)

            # Create S2Activation with GL_1x (standard)
            try:
                s2_act = S2Activation(irreps, act_fn, sampling_method='gauss_legendre')
            except Exception as e:
                print(f" FAILED: {e}")
                continue

            eff_ranks = []
            stab_ranks = []
            spectral_gaps = []

            for seed in range(num_inputs):
                coeffs = generate_random_coefficients(l_max, 'random_normal', seed=seed)
                metrics = compute_jacobian_metrics(s2_act, coeffs)
                eff_ranks.append(metrics['effective_rank'])
                stab_ranks.append(metrics['stable_rank'])
                spectral_gaps.append(metrics['spectral_gap'])

            result = {
                'mean_effective_rank': float(np.mean(eff_ranks)),
                'std_effective_rank': float(np.std(eff_ranks)),
                'mean_stable_rank': float(np.mean(stab_ranks)),
                'std_stable_rank': float(np.std(stab_ranks)),
                'mean_spectral_gap': float(np.mean(spectral_gaps)),
                'max_rank': d,
                'n_inputs': num_inputs,
            }
            all_results[l_max][act_name] = result

            print(f" EffRank = {result['mean_effective_rank']:.1f} / {d}"
                  f" (StableRank = {result['mean_stable_rank']:.1f})")

    # Save metrics
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/expE_expressibility.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir}/metrics/expE_expressibility.json")

    # Sanity checks
    sanity_check_E(all_results)

    # Plots
    plot_expressibility_bar(all_results, output_dir)
    plot_expressibility_vs_leakage(all_results, output_dir)

    print("\nExperiment E complete.")


def sanity_check_E(results):
    print("\n--- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    for l_max, data in sorted(results.items()):
        d = (l_max + 1) ** 2

        # Check 1: All effective ranks should be in [1, d]
        checks_total += 1
        all_in_range = all(
            1.0 <= data[a]['mean_effective_rank'] <= d
            for a in data
        )
        ok = all_in_range
        print(f"  [{'PASS' if ok else 'FAIL'}] All EffRank in [1, {d}] at l_max={l_max}")
        if ok:
            checks_passed += 1

        # Check 2: tanh should have lower EffRank than Softplus(1)
        # (saturation kills directions: tanh'(x)->0 for large |x|)
        checks_total += 1
        if 'Softplus_1' in data and 'tanh' in data:
            sp1 = data['Softplus_1']['mean_effective_rank']
            th = data['tanh']['mean_effective_rank']
            ok = th < sp1
            print(f"  [{'PASS' if ok else 'FAIL'}] tanh EffRank ({th:.1f}) "
                  f"< Softplus_1 EffRank ({sp1:.1f}) at l_max={l_max}"
                  f" (saturation reduces rank)")
            if ok:
                checks_passed += 1

        # Check 3: Softplus EffRank should decrease with beta
        # (higher beta -> more ReLU-like -> zeroes out negative half -> lower rank)
        checks_total += 1
        sp_keys = sorted(
            [k for k in data if k.startswith('Softplus_')],
            key=lambda x: float(x.split('_')[1])
        )
        if len(sp_keys) >= 2:
            sp_ranks = [data[k]['mean_effective_rank'] for k in sp_keys]
            ok = sp_ranks[0] > sp_ranks[-1]
            print(f"  [{'PASS' if ok else 'FAIL'}] Softplus EffRank decreases with beta: "
                  f"{[f'{r:.1f}' for r in sp_ranks]} at l_max={l_max}")
            if ok:
                checks_passed += 1

    print(f"\n  Sanity checks: {checks_passed}/{checks_total} passed")


def plot_expressibility_bar(results, output_dir):
    """Bar chart of effective rank for each activation, grouped by l_max."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax_idx, (l_max, data) in enumerate(sorted(results.items())):
        ax = axes[ax_idx]
        d = (l_max + 1) ** 2

        # Sort by effective rank
        items = sorted(data.items(), key=lambda x: x[1]['mean_effective_rank'])
        names = [k for k, v in items]
        ranks = [v['mean_effective_rank'] for k, v in items]
        stds = [v['std_effective_rank'] for k, v in items]

        colors = []
        for n in names:
            if n in ('ReLU', 'abs'):
                colors.append('#d62728')
            elif n.startswith('Softplus'):
                colors.append('#2ca02c')
            else:
                colors.append('#1f77b4')

        ax.barh(range(len(names)), ranks, xerr=stds, color=colors,
                alpha=0.8, capsize=3)
        ax.axvline(x=d, color='black', linestyle='--', alpha=0.5,
                   label=f'max = {d}')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Effective Rank')
        ax.set_title(f'l_max = {l_max} (d = {d})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Experiment E: Expressibility (Effective Rank of S2Activation Jacobian)',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expE_expressibility_bar.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expE_expressibility_bar.png")


def plot_expressibility_vs_leakage(results, output_dir):
    """Scatter plot: expressibility (EffRank) vs leakage ratio R."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    # Load leakage data from Experiment A
    leakage_path = f'{output_dir}/metrics/expA_spectral.json'
    if not os.path.exists(leakage_path):
        print(f"Warning: {leakage_path} not found, skipping expressibility vs leakage plot")
        return

    with open(leakage_path) as f:
        expA = json.load(f)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    color_map = {
        'ReLU': '#d62728', 'abs': '#ff7f0e',
        'SiLU': '#1f77b4', 'GELU': '#17becf', 'tanh': '#9467bd',
        'Softplus_1': '#2ca02c', 'Softplus_10': '#98df8a',
        'Softplus_100': '#e74c3c',
    }

    for ax_idx, (l_max, data) in enumerate(sorted(results.items())):
        ax = axes[ax_idx]
        d = (l_max + 1) ** 2
        lmax_str = str(l_max)

        for act_name, vals in data.items():
            if lmax_str not in expA or act_name not in expA[lmax_str]:
                continue
            R = expA[lmax_str][act_name]['leakage_ratio']
            eff_rank = vals['mean_effective_rank']
            color = color_map.get(act_name, 'gray')

            ax.scatter(R, eff_rank, color=color, s=80, zorder=5, edgecolors='black',
                       linewidths=0.5)
            ax.annotate(act_name, (R, eff_rank), fontsize=7,
                        xytext=(4, 4), textcoords='offset points')

        ax.set_xlabel('Spectral Leakage Ratio R')
        ax.set_ylabel('Effective Rank (Expressibility)')
        ax.set_title(f'l_max = {l_max} (d = {d})')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Expressibility vs. Spectral Leakage', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expE_express_vs_leakage.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expE_express_vs_leakage.png")


if __name__ == '__main__':
    run_experiment_E()
