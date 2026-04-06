"""
Experiment 4: Resolution Scaling Analysis

Analyze how the minimum number of sampling points needed for a target accuracy
scales with l_max, and fit asymptotic complexity.
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
from scipy.optimize import curve_fit

from src.quadrature_methods import get_sampling
from src.spherical_harmonics_utils import (
    generate_random_coefficients,
    expand_coefficients_to_sphere,
    project_to_coefficients,
)


def measure_accuracy(l_max, method, method_kw, num_seeds=5):
    """Measure average relative reconstruction error for a given config."""
    errors = []
    for seed in range(num_seeds):
        coeffs = generate_random_coefficients(l_max, 'random_normal', seed=seed)
        kw = {'l_max': l_max}
        kw.update(method_kw)
        try:
            pts, wts = get_sampling(method, **kw)
        except Exception:
            return None, 0
        vals = expand_coefficients_to_sphere(coeffs, pts, l_max)
        recon = project_to_coefficients(vals, pts, wts, l_max)
        rel_err = (coeffs - recon).norm().item() / (coeffs.norm().item() + 1e-30)
        errors.append(rel_err)
    return np.mean(errors), len(pts)


def measure_time(l_max, method, method_kw, batch_size=32, num_trials=50):
    """Measure forward pass time."""
    kw = {'l_max': l_max}
    kw.update(method_kw)
    try:
        pts, wts = get_sampling(method, **kw)
    except Exception:
        return None, 0

    from src.spherical_harmonics_utils import spherical_harmonics_on_points
    Y = spherical_harmonics_on_points(l_max, pts)
    wY = wts.unsqueeze(-1) * Y
    n_coeffs = (l_max + 1) ** 2

    x = torch.randn(batch_size, n_coeffs, dtype=torch.float64)

    # Warmup
    for _ in range(5):
        f = x @ Y.T
        _ = f @ wY

    times = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        f = x @ Y.T
        _ = f @ wY
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), len(pts)


def run_experiment_4(output_dir='results'):
    print("=" * 60)
    print("Experiment 4: Resolution Scaling Analysis")
    print("=" * 60)

    l_max_range = [3, 5, 7, 10]
    error_targets = [1e-2, 1e-3, 1e-4, 1e-5]

    # Part 1: Efficiency frontier
    print("\n--- Part 1: Efficiency Frontier ---")

    # For each method and l_max, find minimum N to achieve target error
    methods_configs = {
        'uniform': lambda lmax: [{'resolution': r} for r in [10, 20, 30, 50, 75, 100, 150, 200, 300]],
        'gauss_legendre': lambda lmax: [{}],  # GL is exact for its l_max
        'lebedev': lambda lmax: [{'degree': d} for d in [3, 5, 7, 9, 11, 15, 21, 27, 31, 35, 41]
                                  if d >= 3],
        'fibonacci': lambda lmax: [{'num_points': n} for n in [20, 50, 100, 200, 400, 600, 800, 1000, 2000, 5000]],
    }

    frontier_results = {}

    for l_max in l_max_range:
        print(f"\n  l_max = {l_max}")
        frontier_results[l_max] = {}

        for method, config_fn in methods_configs.items():
            configs = config_fn(l_max)
            # Collect (n_points, error) pairs
            pairs = []
            for kw in configs:
                err, n_pts = measure_accuracy(l_max, method, kw)
                if err is not None and n_pts > 0:
                    pairs.append((n_pts, err))

            if not pairs:
                continue

            pairs.sort(key=lambda x: x[0])
            frontier_results[l_max][method] = {
                'points_errors': [(int(n), float(e)) for n, e in pairs],
            }

            # Find min N for each target
            for target in error_targets:
                min_n = None
                for n, e in pairs:
                    if e <= target:
                        min_n = n
                        break
                key = f'min_n_for_{target}'
                frontier_results[l_max][method][key] = min_n

            print(f"    {method}: {len(pairs)} configs tested, "
                  f"best error = {min(e for _, e in pairs):.2e} at N={pairs[-1][0]}")

    # Part 2: Timing analysis
    print("\n--- Part 2: Timing Analysis ---")
    timing_results = {}

    for l_max in l_max_range:
        timing_results[l_max] = {}
        for method in ['uniform', 'gauss_legendre', 'lebedev']:
            if method == 'uniform':
                kw = {'resolution': max(20, 2 * (l_max + 1))}
            elif method == 'gauss_legendre':
                kw = {}
            elif method == 'lebedev':
                deg = min(31, 2 * l_max + 1)
                if deg < 3:
                    deg = 3
                kw = {'degree': deg}

            t, n = measure_time(l_max, method, kw)
            if t is not None:
                timing_results[l_max][method] = {'time_ms': float(t), 'n_points': int(n)}

    # Part 3: Asymptotic complexity fitting
    print("\n--- Part 3: Asymptotic Complexity ---")
    complexity_fits = {}

    for method in ['uniform', 'gauss_legendre', 'lebedev']:
        lmaxs = []
        npts = []
        for l_max in l_max_range:
            if l_max in frontier_results and method in frontier_results[l_max]:
                pairs = frontier_results[l_max][method]['points_errors']
                if pairs:
                    # Use the smallest N that achieves < 1e-10 (or best available)
                    best_n = pairs[-1][0]  # Largest N tested
                    for n, e in pairs:
                        if e < 1e-10:
                            best_n = n
                            break
                    lmaxs.append(l_max)
                    npts.append(best_n)

        if len(lmaxs) >= 3:
            # Fit N = a * l_max^b
            try:
                log_l = np.log(np.array(lmaxs, dtype=float))
                log_n = np.log(np.array(npts, dtype=float))
                coeffs = np.polyfit(log_l, log_n, 1)
                exponent = coeffs[0]
                prefactor = np.exp(coeffs[1])
                complexity_fits[method] = {
                    'exponent': float(exponent),
                    'prefactor': float(prefactor),
                    'lmaxs': [int(x) for x in lmaxs],
                    'npts': [int(x) for x in npts],
                    'formula': f'N ≈ {prefactor:.1f} * l_max^{exponent:.2f}',
                }
                print(f"  {method}: {complexity_fits[method]['formula']}")
            except Exception as e:
                print(f"  {method}: fitting failed ({e})")

    # Combine all results
    all_results = {
        'frontier': frontier_results,
        'timing': timing_results,
        'complexity': complexity_fits,
    }

    # Save
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)

    # Convert int keys to strings for JSON
    def stringify_keys(d):
        if isinstance(d, dict):
            return {str(k): stringify_keys(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [stringify_keys(x) for x in d]
        return d

    with open(f'{output_dir}/metrics/exp4_scaling.json', 'w') as f:
        json.dump(stringify_keys(all_results), f, indent=2)

    plot_efficiency_frontier(all_results, output_dir)
    plot_asymptotic_complexity(all_results, output_dir)

    return all_results


def plot_efficiency_frontier(results, output_dir):
    """Plot min N required for target accuracy vs l_max."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    frontier = results['frontier']
    targets = [1e-2, 1e-4]
    colors = {'uniform': 'tab:blue', 'gauss_legendre': 'tab:orange',
              'lebedev': 'tab:red', 'fibonacci': 'tab:green'}
    markers = {'uniform': 's', 'gauss_legendre': '^', 'lebedev': 'o', 'fibonacci': 'D'}

    fig, axes = plt.subplots(1, len(targets), figsize=(7 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]

    for ax_idx, target in enumerate(targets):
        ax = axes[ax_idx]
        for method in ['uniform', 'gauss_legendre', 'lebedev', 'fibonacci']:
            lmaxs = []
            min_ns = []
            for l_max_key in sorted(frontier.keys(), key=lambda x: int(x)):
                l_max = int(l_max_key)
                if method in frontier[l_max_key]:
                    key = f'min_n_for_{target}'
                    min_n = frontier[l_max_key][method].get(key)
                    if min_n is not None:
                        lmaxs.append(l_max)
                        min_ns.append(min_n)

            if lmaxs:
                ax.plot(lmaxs, min_ns, color=colors.get(method, 'gray'),
                        marker=markers.get(method, 'x'), label=method, linewidth=1.5)

        ax.set_xlabel('l_max')
        ax.set_ylabel('Min sampling points N')
        ax.set_title(f'Target error = {target}')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 4: Efficiency Frontier', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp4_efficiency_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp4_efficiency_frontier.png")


def plot_asymptotic_complexity(results, output_dir):
    """Plot N vs l_max on log-log scale with fitted lines."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    complexity = results['complexity']
    colors = {'uniform': 'tab:blue', 'gauss_legendre': 'tab:orange', 'lebedev': 'tab:red'}

    fig, ax = plt.subplots(figsize=(8, 6))

    for method, fit in complexity.items():
        lmaxs = np.array(fit['lmaxs'])
        npts = np.array(fit['npts'])

        ax.scatter(lmaxs, npts, color=colors.get(method, 'gray'),
                   s=60, zorder=3)

        # Plot fit line
        l_range = np.linspace(min(lmaxs), max(lmaxs), 100)
        n_fit = fit['prefactor'] * l_range ** fit['exponent']
        ax.plot(l_range, n_fit, '--', color=colors.get(method, 'gray'),
                label=f"{method}: {fit['formula']}", linewidth=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('l_max')
    ax.set_ylabel('Number of sampling points N')
    ax.set_title('Asymptotic Complexity: N vs l_max')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp4_asymptotic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp4_asymptotic.png")


if __name__ == '__main__':
    run_experiment_4()
