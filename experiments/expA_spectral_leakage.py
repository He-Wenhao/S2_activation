"""
Experiment A: Spectral Leakage Analysis

Measure how different nonlinear activation functions spread energy
into high-frequency spherical harmonic components beyond l_max.

Chain: activation smoothness → spectral leakage
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
)


def get_nonlinearities():
    """Return dict of activation functions to test."""
    return {
        # Sharp (C^0 at origin)
        'ReLU': torch.relu,
        'abs': torch.abs,
        'LeakyReLU_0.1': nn.LeakyReLU(0.1),
        # Smooth (C^inf)
        'SiLU': nn.SiLU(),
        'GELU': nn.GELU(),
        'tanh': torch.tanh,
        # Parametric smoothness: Softplus(β) → ReLU as β→∞
        'Softplus_1': nn.Softplus(beta=1),
        'Softplus_3': nn.Softplus(beta=3),
        'Softplus_10': nn.Softplus(beta=10),
        'Softplus_30': nn.Softplus(beta=30),
        'Softplus_100': nn.Softplus(beta=100),
        # Other
        'x^2': lambda x: x ** 2,
        'sin': torch.sin,
    }


def compute_power_spectrum(coeffs, l_max):
    """Compute power per degree: P(l) = sum_m |c_lm|^2."""
    P = []
    idx = 0
    for l in range(l_max + 1):
        n = 2 * l + 1
        P.append((coeffs[idx:idx + n] ** 2).sum().item())
        idx += n
    return np.array(P)


def run_experiment_A(output_dir='results'):
    print("=" * 60)
    print("Experiment A: Spectral Leakage Analysis")
    print("=" * 60)

    l_max_values = [3, 6, 10]
    l_max_ref = 25  # reference resolution for ground truth spectrum (reduced from 40 for login node)
    num_inputs = 10
    nonlinearities = get_nonlinearities()

    # Use high-resolution GL quadrature for accurate spectral analysis
    print(f"Using GL quadrature with l_max_ref={l_max_ref} for spectral analysis")
    pts_ref, wts_ref = get_sampling('gauss_legendre', l_max=l_max_ref)
    print(f"Reference grid: {len(pts_ref)} points")

    all_results = {}

    for l_max in l_max_values:
        print(f"\n--- l_max = {l_max} ---")
        all_results[l_max] = {}

        for act_name, act_fn in nonlinearities.items():
            spectra = []  # collect P(l) across random inputs
            leakage_ratios = []

            for seed in range(num_inputs):
                # Generate random band-limited input
                coeffs_in = generate_random_coefficients(l_max, 'random_normal', seed=seed)

                # Evaluate f(x) on reference grid
                f_vals = expand_coefficients_to_sphere(coeffs_in, pts_ref, l_max)

                # Apply nonlinearity
                with torch.no_grad():
                    g_vals = act_fn(f_vals)

                # Project back to get full spectrum up to l_max_ref
                coeffs_out = project_to_coefficients(g_vals, pts_ref, wts_ref, l_max_ref)

                # Power spectrum
                P = compute_power_spectrum(coeffs_out, l_max_ref)
                spectra.append(P)

                # Leakage ratio: energy above l_max / total energy
                total_energy = P.sum()
                leaked_energy = P[l_max + 1:].sum()
                R = leaked_energy / (total_energy + 1e-30)
                leakage_ratios.append(R)

            mean_spectrum = np.mean(spectra, axis=0)
            std_spectrum = np.std(spectra, axis=0)
            mean_R = np.mean(leakage_ratios)
            std_R = np.std(leakage_ratios)

            # Effective bandwidth: first l where P(l) < 1e-6 * P(0)
            threshold = 1e-6 * mean_spectrum[0] if mean_spectrum[0] > 0 else 1e-30
            l_eff = l_max_ref
            for l in range(l_max_ref + 1):
                if mean_spectrum[l] < threshold:
                    l_eff = l
                    break

            all_results[l_max][act_name] = {
                'mean_spectrum': mean_spectrum.tolist(),
                'std_spectrum': std_spectrum.tolist(),
                'leakage_ratio': float(mean_R),
                'leakage_ratio_std': float(std_R),
                'l_effective': int(l_eff),
            }

            print(f"  {act_name:20s} | R = {mean_R:.4f} ± {std_R:.4f} | l_eff = {l_eff}")

    # Save results
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/expA_spectral.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Sanity checks
    sanity_check_A(all_results)

    # Plots
    plot_power_spectra(all_results, output_dir)
    plot_leakage_ratios(all_results, output_dir)
    plot_softplus_transition(all_results, output_dir)

    return all_results


def sanity_check_A(results):
    """Verify results make physical sense."""
    print("\n--- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    for l_max in results:
        data = results[l_max]

        # Check 1: x^2 should have exactly l_max*2 bandwidth (product of two l_max signals)
        # but since input is random with all l up to l_max, x^2 should have support up to 2*l_max
        checks_total += 1
        if 'x^2' in data:
            P = np.array(data['x^2']['mean_spectrum'])
            # P should be near zero for l > 2*l_max (if l_max_ref is large enough)
            if 2 * l_max < len(P):
                high_energy = P[2 * l_max + 1:].sum()
                low_energy = P[:2 * l_max + 1].sum()
                ratio = high_energy / (low_energy + 1e-30)
                ok = ratio < 0.01
                print(f"  [{'PASS' if ok else 'FAIL'}] x^2 with l_max={l_max}: "
                      f"energy above 2*l_max = {ratio:.2e} (should be ~0)")
                if ok:
                    checks_passed += 1

        # Check 2: ReLU leakage > Softplus_1 leakage (Softplus_1 is genuinely smoother)
        checks_total += 1
        if 'ReLU' in data and 'Softplus_1' in data:
            r_relu = data['ReLU']['leakage_ratio']
            r_sp1 = data['Softplus_1']['leakage_ratio']
            ok = r_relu > r_sp1
            print(f"  [{'PASS' if ok else 'FAIL'}] ReLU leakage ({r_relu:.4f}) > "
                  f"Softplus_1 leakage ({r_sp1:.4f}) at l_max={l_max}")
            if ok:
                checks_passed += 1

        # Check 3: Softplus monotonicity — higher β → more leakage
        checks_total += 1
        sp_keys = sorted([k for k in data.keys() if k.startswith('Softplus_')],
                         key=lambda x: float(x.split('_')[1]))
        if len(sp_keys) >= 2:
            sp_ratios = [data[k]['leakage_ratio'] for k in sp_keys]
            monotone = all(sp_ratios[i] <= sp_ratios[i + 1] + 0.01
                           for i in range(len(sp_ratios) - 1))
            print(f"  [{'PASS' if monotone else 'FAIL'}] Softplus leakage monotone with β "
                  f"at l_max={l_max}: {[f'{r:.4f}' for r in sp_ratios]}")
            if monotone:
                checks_passed += 1

        # Check 4: tanh leakage should be very small (C^inf + bounded)
        checks_total += 1
        if 'tanh' in data:
            r_tanh = data['tanh']['leakage_ratio']
            ok = r_tanh < 0.15  # tanh still generates some high-l content
            print(f"  [{'PASS' if ok else 'FAIL'}] tanh leakage = {r_tanh:.4f} "
                  f"(should be small) at l_max={l_max}")
            if ok:
                checks_passed += 1

    print(f"\n  Sanity checks: {checks_passed}/{checks_total} passed")


def plot_power_spectra(results, output_dir):
    """Plot P(l) vs l for each l_max."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    # Color scheme: sharp=red tones, smooth=blue tones, softplus=green gradient
    color_map = {
        'ReLU': '#d62728', 'abs': '#ff7f0e', 'LeakyReLU_0.1': '#e377c2',
        'SiLU': '#1f77b4', 'GELU': '#17becf', 'tanh': '#9467bd',
        'x^2': '#8c564b', 'sin': '#bcbd22',
    }
    # Softplus colors: gradient from blue (smooth) to red (sharp)
    sp_colors = {'Softplus_1': '#2ca02c', 'Softplus_3': '#55a868',
                 'Softplus_10': '#98df8a', 'Softplus_30': '#f0b27a',
                 'Softplus_100': '#e74c3c'}

    for ax_idx, (l_max, data) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[ax_idx]

        # Plot non-softplus first
        for act_name in ['ReLU', 'abs', 'SiLU', 'GELU', 'tanh', 'x^2']:
            if act_name not in data:
                continue
            P = np.array(data[act_name]['mean_spectrum'])
            P_pos = np.maximum(P, 1e-30)
            color = color_map.get(act_name, 'gray')
            ax.semilogy(range(len(P)), P_pos, label=act_name, color=color, linewidth=1.5)

        # Vertical line at l_max
        ax.axvline(x=int(l_max), color='black', linestyle='--', alpha=0.5, label=f'l_max={l_max}')

        ax.set_xlabel('Degree l')
        ax.set_ylabel('Power P(l)')
        ax.set_title(f'l_max = {l_max}')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(4 * int(l_max), 25))

    plt.suptitle('Experiment A: Power Spectrum After Nonlinearity', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expA_power_spectra.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expA_power_spectra.png")

    # Separate plot for Softplus family
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax_idx, (l_max, data) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[ax_idx]

        # Plot ReLU as reference
        if 'ReLU' in data:
            P = np.array(data['ReLU']['mean_spectrum'])
            ax.semilogy(range(len(P)), np.maximum(P, 1e-30),
                        label='ReLU', color='red', linewidth=2, linestyle='--')

        # Plot Softplus family
        for act_name in sorted([k for k in data if k.startswith('Softplus_')],
                                key=lambda x: float(x.split('_')[1])):
            P = np.array(data[act_name]['mean_spectrum'])
            color = sp_colors.get(act_name, 'gray')
            ax.semilogy(range(len(P)), np.maximum(P, 1e-30),
                        label=act_name, color=color, linewidth=1.5)

        ax.axvline(x=int(l_max), color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Degree l')
        ax.set_ylabel('Power P(l)')
        ax.set_title(f'Softplus Family (l_max={l_max})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(4 * int(l_max), 25))

    plt.suptitle('Experiment A: Softplus(β) Transition from Smooth to Sharp', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expA_softplus_spectra.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expA_softplus_spectra.png")


def plot_leakage_ratios(results, output_dir):
    """Bar chart of leakage ratio R for each activation."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax_idx, (l_max, data) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[ax_idx]

        # Sort by leakage ratio
        items = sorted(data.items(), key=lambda x: x[1]['leakage_ratio'])
        names = [k for k, v in items]
        ratios = [v['leakage_ratio'] for k, v in items]
        stds = [v['leakage_ratio_std'] for k, v in items]

        colors = []
        for n in names:
            if n in ('ReLU', 'abs', 'LeakyReLU_0.1'):
                colors.append('#d62728')
            elif n.startswith('Softplus'):
                colors.append('#2ca02c')
            else:
                colors.append('#1f77b4')

        bars = ax.barh(range(len(names)), ratios, xerr=stds, color=colors, alpha=0.8, capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Leakage Ratio R')
        ax.set_title(f'l_max = {l_max}')
        ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Experiment A: Spectral Leakage Ratio (energy above l_max / total)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expA_leakage_ratios.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expA_leakage_ratios.png")


def plot_softplus_transition(results, output_dir):
    """Plot leakage ratio R vs β for Softplus family."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for l_max, data in sorted(results.items(), key=lambda x: int(x[0])):
        betas = []
        ratios = []
        stds = []
        for act_name, vals in data.items():
            if act_name.startswith('Softplus_'):
                beta = float(act_name.split('_')[1])
                betas.append(beta)
                ratios.append(vals['leakage_ratio'])
                stds.append(vals['leakage_ratio_std'])

        if betas:
            order = np.argsort(betas)
            betas = np.array(betas)[order]
            ratios = np.array(ratios)[order]
            stds = np.array(stds)[order]

            ax.errorbar(betas, ratios, yerr=stds, marker='o', linewidth=1.5, capsize=3,
                        label=f'l_max={l_max}')

        # Add ReLU reference as horizontal line
        if 'ReLU' in data:
            ax.axhline(y=data['ReLU']['leakage_ratio'], linestyle='--', alpha=0.4,
                        label=f'ReLU (l_max={l_max})')

    ax.set_xscale('log')
    ax.set_xlabel('Softplus β (→ ReLU as β → ∞)')
    ax.set_ylabel('Leakage Ratio R')
    ax.set_title('Softplus(β): Continuous Transition from Smooth to Sharp')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expA_softplus_transition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expA_softplus_transition.png")


if __name__ == '__main__':
    run_experiment_A()
