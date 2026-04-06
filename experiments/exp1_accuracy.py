"""
Experiment 1: Accuracy Comparison

Compare SH coefficient reconstruction error across sampling methods
for various l_max values and coefficient distributions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.quadrature_methods import get_sampling, SamplingMethods
from src.spherical_harmonics_utils import (
    generate_random_coefficients,
    expand_coefficients_to_sphere,
    project_to_coefficients,
)

# Available Lebedev degrees and their point counts
LEBEDEV_DEGREES = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                    35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]


def get_lebedev_npoints(degree):
    pts, _ = get_sampling('lebedev', degree=degree)
    return len(pts)


def reconstruction_error(coeffs, points, weights, l_max):
    """Compute reconstruction error metrics."""
    values = expand_coefficients_to_sphere(coeffs, points, l_max)
    recon = project_to_coefficients(values, points, weights, l_max)

    diff = coeffs - recon
    l2_error = diff.norm().item()
    rel_error = l2_error / (coeffs.norm().item() + 1e-30)
    max_error = diff.abs().max().item()

    # Per-degree error
    errors_by_degree = []
    idx = 0
    for l in range(l_max + 1):
        n = 2 * l + 1
        deg_err = diff[idx:idx + n].norm().item()
        errors_by_degree.append(deg_err)
        idx += n

    return {
        'l2_error': l2_error,
        'relative_error': rel_error,
        'max_error': max_error,
        'errors_by_degree': errors_by_degree,
    }


def run_experiment_1(output_dir='results'):
    print("=" * 60)
    print("Experiment 1: Accuracy Comparison")
    print("=" * 60)

    l_max_values = [3, 5, 7, 10, 15, 20]
    distributions = ['random_normal', 'polynomial']
    num_seeds = 3

    # Define sampling configurations (method, kwargs, label)
    def get_configs(l_max):
        configs = []

        # Uniform grids (fewer large grids for big l_max)
        uniform_resolutions = [20, 50, 100] if l_max <= 10 else [20, 50, 100]
        for res in uniform_resolutions:
            n = res * (2 * res)
            configs.append(('uniform', {'resolution': res}, f'Uniform {res}x{2*res} (N={n})'))

        # Gauss-Legendre at various resolutions
        # Sample ~6 values from 2 to l_max+2
        nt_exact = l_max + 1
        nt_values = sorted(set([2, max(2, nt_exact // 4), max(2, nt_exact // 2),
                                max(2, 3 * nt_exact // 4), nt_exact, nt_exact + 1]))
        for nt in nt_values:
            pts, _ = get_sampling('gauss_legendre', l_max=l_max, n_theta=nt)
            n = len(pts)
            exact_tag = " [exact]" if nt >= nt_exact else ""
            configs.append(('gauss_legendre', {'n_theta': nt}, f'GL n_t={nt} (N={n}){exact_tag}'))

        # Lebedev: only test degrees around the threshold (2*l_max+1) and a few below/above
        target_deg = 2 * l_max + 1
        relevant_degs = [d for d in LEBEDEV_DEGREES
                         if d <= target_deg + 10]
        # Keep at most 8 below threshold and 3 above
        below = [d for d in relevant_degs if d < target_deg][-6:]
        above = [d for d in LEBEDEV_DEGREES if d >= target_deg][:3]
        for deg in below + above:
            try:
                pts, _ = get_sampling('lebedev', degree=deg)
                n = len(pts)
                configs.append(('lebedev', {'degree': deg}, f'Lebedev deg={deg} (N={n})'))
            except Exception:
                continue

        # Fibonacci
        for n_pts in [50, 200, 600, 1000]:
            configs.append(('fibonacci', {'num_points': n_pts}, f'Fibonacci (N={n_pts})'))

        return configs

    all_results = {}

    for l_max in l_max_values:
        print(f"\n--- l_max = {l_max} ---")
        configs = get_configs(l_max)
        all_results[l_max] = {}

        for dist in distributions:
            all_results[l_max][dist] = {}

            for method, extra_kw, label in configs:
                errors_across_seeds = []
                for seed in range(num_seeds):
                    coeffs = generate_random_coefficients(l_max, dist, seed=seed)
                    kw = {'l_max': l_max}
                    kw.update(extra_kw)

                    try:
                        pts, wts = get_sampling(method, **kw)
                    except Exception as e:
                        break

                    err = reconstruction_error(coeffs, pts, wts, l_max)
                    errors_across_seeds.append(err)

                if not errors_across_seeds:
                    continue

                avg_rel = np.mean([e['relative_error'] for e in errors_across_seeds])
                std_rel = np.std([e['relative_error'] for e in errors_across_seeds])
                avg_l2 = np.mean([e['l2_error'] for e in errors_across_seeds])
                avg_max = np.mean([e['max_error'] for e in errors_across_seeds])
                n_pts = len(pts)

                all_results[l_max][dist][label] = {
                    'method': method,
                    'n_points': n_pts,
                    'avg_relative_error': float(avg_rel),
                    'std_relative_error': float(std_rel),
                    'avg_l2_error': float(avg_l2),
                    'avg_max_error': float(avg_max),
                    'avg_errors_by_degree': [
                        float(np.mean([e['errors_by_degree'][l] for e in errors_across_seeds]))
                        for l in range(l_max + 1)
                    ],
                }

                print(f"  {label:40s} | rel_err = {avg_rel:.2e} ± {std_rel:.2e}")

    # Save results
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/exp1_accuracy.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate figures
    plot_accuracy_curves(all_results, output_dir)
    plot_error_by_degree(all_results, output_dir)
    print_summary_tables(all_results, output_dir)

    return all_results


def plot_accuracy_curves(results, output_dir):
    """Plot relative error vs number of sampling points for each l_max."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    l_max_values = sorted([int(k) for k in results.keys()])

    method_colors = {
        'uniform': 'tab:blue',
        'gauss_legendre': 'tab:orange',
        'lebedev': 'tab:red',
        'fibonacci': 'tab:green',
    }
    method_markers = {
        'uniform': 's',
        'gauss_legendre': '^',
        'lebedev': 'o',
        'fibonacci': 'D',
    }

    for ax_idx, l_max in enumerate(l_max_values):
        ax = axes[ax_idx]
        dist = 'random_normal'
        data = results[l_max][dist] if l_max in results or str(l_max) in results else results[str(l_max)][dist]

        # Group by method
        method_data = {}
        for label, vals in data.items():
            m = vals['method']
            if m not in method_data:
                method_data[m] = {'n': [], 'err': []}
            method_data[m]['n'].append(vals['n_points'])
            method_data[m]['err'].append(vals['avg_relative_error'])

        for method_name, md in method_data.items():
            sorted_pairs = sorted(zip(md['n'], md['err']))
            ns, errs = zip(*sorted_pairs)
            ax.plot(ns, errs,
                    color=method_colors.get(method_name, 'gray'),
                    marker=method_markers.get(method_name, 'x'),
                    label=method_name, linewidth=1.5, markersize=5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of sampling points')
        ax.set_ylabel('Relative reconstruction error')
        ax.set_title(f'l_max = {l_max}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 1: Reconstruction Error vs Sampling Points', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp1_accuracy_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp1_accuracy_curves.png")


def plot_error_by_degree(results, output_dir):
    """Plot per-degree error for selected l_max values."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    selected_lmax = [5, 10, 20]
    dist = 'random_normal'

    for ax_idx, l_max in enumerate(selected_lmax):
        ax = axes[ax_idx]
        key = l_max if l_max in results else str(l_max)
        if key not in results:
            continue
        data = results[key][dist]

        # Pick one representative config per method
        best_per_method = {}
        for label, vals in data.items():
            m = vals['method']
            if m not in best_per_method or vals['avg_relative_error'] < best_per_method[m]['avg_relative_error']:
                best_per_method[m] = vals
                best_per_method[m]['label'] = label

        degrees = list(range(l_max + 1))
        width = 0.2
        offsets = {'uniform': -1.5, 'gauss_legendre': -0.5, 'lebedev': 0.5, 'fibonacci': 1.5}
        colors = {'uniform': 'tab:blue', 'gauss_legendre': 'tab:orange', 'lebedev': 'tab:red', 'fibonacci': 'tab:green'}

        for method_name, vals in best_per_method.items():
            errs = vals['avg_errors_by_degree']
            offset = offsets.get(method_name, 0)
            ax.bar([d + offset * width for d in degrees], errs, width,
                   label=f"{method_name} (N={vals['n_points']})",
                   color=colors.get(method_name, 'gray'), alpha=0.8)

        ax.set_xlabel('Degree l')
        ax.set_ylabel('L2 error per degree')
        ax.set_title(f'l_max = {l_max}')
        ax.set_yscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Error Distribution by Spherical Harmonic Degree', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp1_error_by_degree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp1_error_by_degree.png")


def print_summary_tables(results, output_dir):
    """Print and save summary tables."""
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    lines = []

    for l_max_key in sorted(results.keys(), key=lambda x: int(x)):
        l_max = int(l_max_key)
        lines.append(f"\n{'='*80}")
        lines.append(f"l_max = {l_max}")
        lines.append(f"{'='*80}")
        lines.append(f"{'Method':<40s} | {'N pts':>7s} | {'Rel Err':>10s} | {'Max Err':>10s}")
        lines.append("-" * 80)

        dist = 'random_normal'
        data = results[l_max_key][dist]
        sorted_items = sorted(data.items(), key=lambda x: x[1]['avg_relative_error'])
        for label, vals in sorted_items:
            lines.append(f"{label:<40s} | {vals['n_points']:>7d} | {vals['avg_relative_error']:>10.2e} | {vals['avg_max_error']:>10.2e}")

    table_str = '\n'.join(lines)
    print(table_str)
    with open(f'{output_dir}/tables/exp1_summary.txt', 'w') as f:
        f.write(table_str)


if __name__ == '__main__':
    run_experiment_1()
