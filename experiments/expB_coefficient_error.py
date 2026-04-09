"""
Experiment B: S2Activation Output Error

Measure how the full S2Activation pipeline's coefficient error depends on
(activation function × sampling method × sampling resolution).

Separates:
  - Truncation error: energy lost above l_max (independent of sampling)
  - Aliasing error: corruption of l <= l_max coefficients due to insufficient sampling

Chain: spectral leakage × sampling → coefficient error
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

from src.quadrature_methods import get_sampling
from src.spherical_harmonics_utils import (
    generate_random_coefficients,
    expand_coefficients_to_sphere,
    project_to_coefficients,
    spherical_harmonics_on_points,
)


def get_nonlinearities():
    """Same set as Experiment A for consistency."""
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


def s2_activation(coeffs_in, l_max, act_fn, pts, wts, l_max_out=None):
    """Full S2Activation: coeffs -> sphere -> nonlinearity -> coeffs."""
    if l_max_out is None:
        l_max_out = l_max
    f_vals = expand_coefficients_to_sphere(coeffs_in, pts, l_max)
    with torch.no_grad():
        g_vals = act_fn(f_vals)
    coeffs_out = project_to_coefficients(g_vals, pts, wts, l_max_out)
    return coeffs_out


def run_experiment_B(output_dir='results'):
    print("=" * 60)
    print("Experiment B: S2Activation Coefficient Error")
    print("=" * 60)

    l_max_values = [3, 6, 10]
    l_max_ref = 25  # high-res ground truth
    num_inputs = 10
    nonlinearities = get_nonlinearities()

    # Ground truth sampling (GL at l_max_ref)
    pts_gt, wts_gt = get_sampling('gauss_legendre', l_max=l_max_ref)
    print(f"Ground truth grid: GL l_max_ref={l_max_ref}, {len(pts_gt)} points")

    # Sampling configurations to test
    def get_sampling_configs(l_max):
        configs = []
        # GL at various oversampling ratios
        for mult in [1, 2, 3]:
            lm = mult * l_max
            try:
                pts, wts = get_sampling('gauss_legendre', l_max=lm)
                configs.append((f'GL_{mult}x', pts, wts))
            except Exception:
                pass

        # Lebedev at various degrees
        for mult_label, deg_target in [('Leb_min', 2*l_max+1), ('Leb_2x', 4*l_max+1)]:
            # Find closest valid Lebedev degree
            valid_degs = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                         35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]
            best_deg = min(valid_degs, key=lambda d: abs(d - deg_target) if d >= deg_target else 1000)
            if best_deg < deg_target:
                # No valid degree >= target, take the largest <= 131
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

        # Uniform grids
        for res in [20, 50, 100]:
            try:
                pts, wts = get_sampling('uniform', resolution=res)
                configs.append((f'Uniform_{res}', pts, wts))
            except Exception:
                pass

        return configs

    all_results = {}

    for l_max in l_max_values:
        print(f"\n--- l_max = {l_max} ---")
        all_results[l_max] = {}
        configs = get_sampling_configs(l_max)

        for act_name, act_fn in nonlinearities.items():
            all_results[l_max][act_name] = {}

            # Compute ground truth: S2Act with high-res GL, then truncate to l_max
            gt_errors_total = []  # for sanity
            for seed in range(num_inputs):
                coeffs_in = generate_random_coefficients(l_max, 'random_normal', seed=seed)

                # Ground truth output (project at l_max_ref, then take first l_max coefficients)
                coeffs_gt_full = s2_activation(coeffs_in, l_max, act_fn, pts_gt, wts_gt, l_max_out=l_max_ref)
                n_coeffs_lmax = (l_max + 1) ** 2
                coeffs_gt = coeffs_gt_full[:n_coeffs_lmax]  # truncated to l_max

                # Truncation error: energy in coefficients above l_max
                trunc_energy = coeffs_gt_full[n_coeffs_lmax:].norm().item()
                total_energy = coeffs_gt_full.norm().item()
                trunc_ratio = trunc_energy / (total_energy + 1e-30)

                for cfg_name, pts, wts in configs:
                    # Actual S2Activation at this sampling
                    coeffs_out = s2_activation(coeffs_in, l_max, act_fn, pts, wts, l_max_out=l_max)

                    # Total error
                    total_err = (coeffs_out - coeffs_gt).norm().item()
                    rel_err = total_err / (coeffs_gt.norm().item() + 1e-30)

                    # Per-degree error
                    errors_by_l = []
                    idx = 0
                    for l in range(l_max + 1):
                        n = 2 * l + 1
                        err_l = (coeffs_out[idx:idx+n] - coeffs_gt[idx:idx+n]).norm().item()
                        errors_by_l.append(err_l)
                        idx += n

                    key = cfg_name
                    if key not in all_results[l_max][act_name]:
                        all_results[l_max][act_name][key] = {
                            'rel_errors': [],
                            'trunc_ratios': [],
                            'n_points': len(pts),
                            'errors_by_l': [],
                        }
                    all_results[l_max][act_name][key]['rel_errors'].append(float(rel_err))
                    all_results[l_max][act_name][key]['trunc_ratios'].append(float(trunc_ratio))
                    all_results[l_max][act_name][key]['errors_by_l'].append(errors_by_l)

            # Aggregate
            for cfg_name in list(all_results[l_max][act_name].keys()):
                d = all_results[l_max][act_name][cfg_name]
                d['mean_rel_error'] = float(np.mean(d['rel_errors']))
                d['std_rel_error'] = float(np.std(d['rel_errors']))
                d['mean_trunc_ratio'] = float(np.mean(d['trunc_ratios']))
                d['mean_errors_by_l'] = [float(np.mean([e[l] for e in d['errors_by_l']]))
                                          for l in range(l_max + 1)]
                # Clean up raw data for JSON
                del d['rel_errors']
                del d['trunc_ratios']
                del d['errors_by_l']

            # Print summary
            for cfg_name, cfg_data in all_results[l_max][act_name].items():
                print(f"  {act_name:15s} | {cfg_name:20s} | N={cfg_data['n_points']:5d} | "
                      f"rel_err={cfg_data['mean_rel_error']:.2e} | trunc={cfg_data['mean_trunc_ratio']:.4f}")

    # Save results
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/expB_coefficient_error.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Sanity checks
    sanity_check_B(all_results)

    # Plots
    plot_error_vs_npoints(all_results, output_dir)
    plot_error_heatmap(all_results, output_dir)
    plot_oversampling_decay(all_results, output_dir)

    return all_results


def sanity_check_B(results):
    """Verify results make physical sense."""
    print("\n--- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    for l_max in results:
        data = results[l_max]

        # Check 1: GL_3x should have lower error than GL_1x for all activations
        checks_total += 1
        gl1x_better = 0
        gl3x_better = 0
        for act_name in data:
            if 'GL_1x' in data[act_name] and 'GL_3x' in data[act_name]:
                e1 = data[act_name]['GL_1x']['mean_rel_error']
                e3 = data[act_name]['GL_3x']['mean_rel_error']
                if e3 < e1:
                    gl3x_better += 1
                else:
                    gl1x_better += 1
        ok = gl3x_better > gl1x_better
        print(f"  [{'PASS' if ok else 'FAIL'}] GL_3x better than GL_1x in {gl3x_better}/{gl3x_better+gl1x_better} "
              f"activations at l_max={l_max}")
        if ok:
            checks_passed += 1

        # Check 2: Smooth activations should have lower error than sharp ones at GL_1x
        checks_total += 1
        if 'Softplus_1' in data and 'ReLU' in data:
            if 'GL_1x' in data['Softplus_1'] and 'GL_1x' in data['ReLU']:
                e_smooth = data['Softplus_1']['GL_1x']['mean_rel_error']
                e_sharp = data['ReLU']['GL_1x']['mean_rel_error']
                ok = e_smooth < e_sharp
                print(f"  [{'PASS' if ok else 'FAIL'}] Softplus_1 GL_1x error ({e_smooth:.2e}) < "
                      f"ReLU GL_1x error ({e_sharp:.2e}) at l_max={l_max}")
                if ok:
                    checks_passed += 1

        # Check 3: Truncation ratio should be consistent across sampling methods
        # (truncation is a property of the activation, not the sampling)
        checks_total += 1
        if 'ReLU' in data:
            trunc_vals = [v['mean_trunc_ratio'] for v in data['ReLU'].values()]
            if len(trunc_vals) >= 2:
                spread = max(trunc_vals) - min(trunc_vals)
                ok = spread < 0.05  # should be nearly the same
                print(f"  [{'PASS' if ok else 'FAIL'}] ReLU truncation ratio spread = {spread:.4f} "
                      f"across samplings at l_max={l_max} (should be ~0)")
                if ok:
                    checks_passed += 1

        # Check 4: Oversampling should reduce aliasing: GL_3x ≤ GL_1x for all activations
        # (errors are dominated by truncation, so improvement is small but consistent)
        checks_total += 1
        all_improve = True
        for act_name in data:
            if 'GL_1x' in data[act_name] and 'GL_3x' in data[act_name]:
                e1 = data[act_name]['GL_1x']['mean_rel_error']
                e3 = data[act_name]['GL_3x']['mean_rel_error']
                if e3 > e1 * 1.01:  # allow 1% tolerance
                    all_improve = False
        ok = all_improve
        print(f"  [{'PASS' if ok else 'FAIL'}] GL_3x ≤ GL_1x for all activations at l_max={l_max}")
        if ok:
            checks_passed += 1

    print(f"\n  Sanity checks: {checks_passed}/{checks_total} passed")


def plot_error_vs_npoints(results, output_dir):
    """Plot B1: error vs number of sampling points, grouped by activation."""
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
            ns = []
            errs = []
            for cfg_name, cfg_data in configs.items():
                ns.append(cfg_data['n_points'])
                errs.append(cfg_data['mean_rel_error'])
            # Sort by n_points
            order = np.argsort(ns)
            ns = [ns[i] for i in order]
            errs = [errs[i] for i in order]
            ax.plot(ns, errs, marker='o', label=act_name,
                    color=act_colors.get(act_name, 'gray'), linewidth=1.5, markersize=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of sampling points')
        ax.set_ylabel('Relative coefficient error')
        ax.set_title(f'l_max = {l_max}')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment B: S2Activation Error vs Sampling Points', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expB_error_vs_npoints.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expB_error_vs_npoints.png")


def plot_error_heatmap(results, output_dir):
    """Plot B2: heatmap of error (activation × sampling method)."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    for l_max, data in sorted(results.items(), key=lambda x: int(x[0])):
        act_names = sorted(data.keys())
        cfg_names = sorted(set(cfg for act in data.values() for cfg in act.keys()))

        matrix = np.full((len(act_names), len(cfg_names)), np.nan)
        for i, act in enumerate(act_names):
            for j, cfg in enumerate(cfg_names):
                if cfg in data[act]:
                    matrix[i, j] = np.log10(data[act][cfg]['mean_rel_error'] + 1e-30)

        fig, ax = plt.subplots(figsize=(max(8, len(cfg_names)*1.2), max(5, len(act_names)*0.5)))
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(cfg_names)))
        ax.set_xticklabels(cfg_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(act_names)))
        ax.set_yticklabels(act_names, fontsize=9)
        plt.colorbar(im, ax=ax, label='log10(relative error)')
        ax.set_title(f'Coefficient Error Heatmap (l_max={l_max})')

        # Add text annotations
        for i in range(len(act_names)):
            for j in range(len(cfg_names)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/figures/expB_heatmap_lmax{l_max}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir}/figures/expB_heatmap_lmax{l_max}.png")


def plot_oversampling_decay(results, output_dir):
    """Plot B3: error decay with GL oversampling ratio."""
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
                    errs.append(cfg_data['mean_rel_error'])
            if mults:
                order = np.argsort(mults)
                mults = [mults[i] for i in order]
                errs = [errs[i] for i in order]
                ax.plot(mults, errs, marker='o', label=act_name,
                        color=act_colors.get(act_name, 'gray'), linewidth=1.5)

        ax.set_xlabel('GL oversampling ratio')
        ax.set_ylabel('Relative coefficient error')
        ax.set_yscale('log')
        ax.set_title(f'l_max = {l_max}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3])

    plt.suptitle('Experiment B: Error Decay with Oversampling', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expB_oversampling_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expB_oversampling_decay.png")


if __name__ == '__main__':
    run_experiment_B()
