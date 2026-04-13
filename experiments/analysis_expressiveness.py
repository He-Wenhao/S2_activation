"""
Analysis: Expressiveness metrics for S2 activations and their correlation
with downstream QM9 performance.

Computes multiple expressiveness metrics from the S2 activation Jacobian:
  - Effective rank (from expE)
  - Stable rank (from expE)
  - log|det(J)| (information-theoretic volume)
  - Condition number (numerical stability)
  - Frobenius norm (output magnitude)
  - Spectral gap (dominance of leading direction)
  - Normalized effective rank (EffRank / d)
  - Output variance (dynamic range of output coefficients)
  - Entropy of singular value distribution
  - Nuclear norm (sum of singular values, trace norm)

Correlates each metric with:
  - Spectral leakage ratio R (smoothness proxy from expA)
  - Downstream QM9 test MAE (from expF)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
import json
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from src.s2_activation import S2Activation
from src.spherical_harmonics_utils import generate_random_coefficients
from e3nn import o3


# ─── Configuration ──────────────────────────────────────────────────────

L_MAX = 6
NUM_INPUTS = 50
ACTIVATIONS = ['SiLU', 'Softplus_1', 'Softplus_10', 'tanh', 'ReLU', 'abs']

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


# ─── Activation constructors ────────────────────────────────────────────

def get_act_fn(name):
    """Return (act_fn, display_name) for a given activation name."""
    mapping = {
        'ReLU': (torch.relu, 'ReLU'),
        'abs': (torch.abs, 'abs'),
        'SiLU': (nn.SiLU(), 'SiLU'),
        'tanh': (torch.tanh, 'tanh'),
        'Softplus_1': (nn.Softplus(beta=1), 'Softplus(1)'),
        'Softplus_10': (nn.Softplus(beta=10), 'Softplus(10)'),
    }
    return mapping[name]


# ─── Jacobian computation ───────────────────────────────────────────────

def compute_full_jacobian(s2_act, coeffs_in):
    """Compute the full Jacobian matrix via autograd."""
    c = coeffs_in.detach().float().requires_grad_(True)
    c_out = s2_act(c)
    d_in = c.shape[0]
    d_out = c_out.shape[0]

    J = torch.zeros(d_out, d_in, dtype=torch.float32)
    for j in range(d_out):
        if c.grad is not None:
            c.grad.zero_()
        c_out[j].backward(retain_graph=True)
        J[j, :] = c.grad.clone()

    return J


def compute_all_metrics(s2_act, coeffs_in):
    """Compute a comprehensive set of expressiveness metrics from the Jacobian."""
    J = compute_full_jacobian(s2_act, coeffs_in)
    sv = torch.linalg.svdvals(J)
    d = J.shape[0]

    # --- Effective rank ---
    s2 = sv ** 2
    total = s2.sum()
    if total < 1e-30:
        eff_rank = 1.0
    else:
        p = s2 / total
        p = p[p > 1e-30]
        H = -(p * torch.log(p)).sum()
        eff_rank = torch.exp(H).item()

    # --- Stable rank ---
    if s2[0] < 1e-30:
        stab_rank = 1.0
    else:
        stab_rank = (s2.sum() / s2[0]).item()

    # --- log|det(J)| (information volume) ---
    # Use sum of log(sv) to avoid overflow; only over nonzero SVs
    sv_pos = sv[sv > 1e-30]
    log_det = torch.log(sv_pos).sum().item() if len(sv_pos) > 0 else float('-inf')

    # --- Condition number ---
    if sv[-1] > 1e-30:
        cond = (sv[0] / sv[-1]).item()
    else:
        # Use ratio of largest to smallest nonzero
        cond = (sv[0] / sv_pos[-1]).item() if len(sv_pos) > 1 else float('inf')

    # --- Frobenius norm ---
    frob = torch.norm(J, 'fro').item()

    # --- Spectral gap ---
    if len(sv) > 1 and sv[1] > 1e-30:
        spec_gap = (sv[0] / sv[1]).item()
    else:
        spec_gap = float('inf')

    # --- Nuclear norm (sum of singular values) ---
    nuclear = sv.sum().item()

    # --- Entropy of normalized SV distribution ---
    # This gives a different view than effective rank
    sv_norm = sv / sv.sum() if sv.sum() > 1e-30 else sv
    sv_norm_pos = sv_norm[sv_norm > 1e-30]
    sv_entropy = -(sv_norm_pos * torch.log(sv_norm_pos)).sum().item()

    # --- Output variance (dynamic range) ---
    c_out = s2_act(coeffs_in.detach().float())
    out_var = c_out.var().item()
    out_range = (c_out.max() - c_out.min()).item()

    return {
        'effective_rank': eff_rank,
        'stable_rank': stab_rank,
        'log_det_J': log_det,
        'condition_number': cond,
        'frobenius_norm': frob,
        'spectral_gap': spec_gap,
        'nuclear_norm': nuclear,
        'sv_entropy': sv_entropy,
        'output_variance': out_var,
        'output_range': out_range,
        'normalized_eff_rank': eff_rank / d,
    }


# ─── Data loading ───────────────────────────────────────────────────────

def load_expA():
    """Load spectral leakage data."""
    path = os.path.join(RESULTS_DIR, 'metrics', 'expA_spectral.json')
    with open(path) as f:
        data = json.load(f)
    return {act: data[str(L_MAX)][act]['leakage_ratio'] for act in ACTIVATIONS}


def load_expE():
    """Load existing expressibility metrics."""
    path = os.path.join(RESULTS_DIR, 'metrics', 'expE_expressibility.json')
    with open(path) as f:
        data = json.load(f)
    return {act: data[str(L_MAX)][act] for act in ACTIVATIONS}


def load_expF():
    """Load downstream QM9 test MAE (default grid, excluding SiLU seed42 with epochs=2)."""
    runs_dir = os.path.join(RESULTS_DIR, 'expF', 'runs')
    mae_by_act = {act: [] for act in ACTIVATIONS}

    for entry in os.listdir(runs_dir):
        if '_default_' not in entry:
            continue
        rpath = os.path.join(runs_dir, entry, 'results.json')
        if not os.path.exists(rpath):
            continue
        with open(rpath) as f:
            r = json.load(f)
        cfg = r['config']
        act = cfg['activation']
        if act not in ACTIVATIONS:
            continue
        # Exclude the failed run
        if act == 'SiLU' and cfg['seed'] == 42 and cfg['epochs'] == 2:
            continue
        mae_by_act[act].append(r['results']['test_mae'])

    return {act: (np.mean(vals), np.std(vals), len(vals))
            for act, vals in mae_by_act.items() if vals}


# ─── Fresh Jacobian metrics computation ─────────────────────────────────

def compute_jacobian_metrics_all():
    """Compute comprehensive Jacobian metrics for all activations."""
    d = (L_MAX + 1) ** 2
    irreps = o3.Irreps([(1, (l, 1)) for l in range(L_MAX + 1)])

    results = {}
    for act_name in ACTIVATIONS:
        act_fn, display = get_act_fn(act_name)
        print(f"  Computing Jacobian metrics for {act_name}...", end='', flush=True)

        s2_act = S2Activation(irreps, act_fn, sampling_method='gauss_legendre')

        all_metrics = []
        for seed in range(NUM_INPUTS):
            coeffs = generate_random_coefficients(L_MAX, 'random_normal', seed=seed)
            m = compute_all_metrics(s2_act, coeffs)
            all_metrics.append(m)

        # Aggregate
        agg = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            # Filter out infinities for mean computation
            finite_vals = [v for v in vals if np.isfinite(v)]
            if finite_vals:
                agg[f'mean_{key}'] = float(np.mean(finite_vals))
                agg[f'std_{key}'] = float(np.std(finite_vals))
                agg[f'median_{key}'] = float(np.median(finite_vals))
            else:
                agg[f'mean_{key}'] = float('inf')
                agg[f'std_{key}'] = 0.0
                agg[f'median_{key}'] = float('inf')

        results[act_name] = agg
        print(f" EffRank={agg['mean_effective_rank']:.1f}/{d}, "
              f"log|det|={agg['mean_log_det_J']:.2f}, "
              f"cond={agg['mean_condition_number']:.1f}")

    return results


# ─── Correlation analysis ───────────────────────────────────────────────

def compute_correlations(leakage, mae_data, jac_metrics):
    """Compute Spearman and Pearson correlations between all metric pairs."""
    acts = sorted(set(leakage.keys()) & set(mae_data.keys()) & set(jac_metrics.keys()))

    # Build vectors
    R = np.array([leakage[a] for a in acts])
    MAE_mean = np.array([mae_data[a][0] for a in acts])
    MAE_std = np.array([mae_data[a][1] for a in acts])

    # Collect all metric names
    metric_names = [k.replace('mean_', '') for k in jac_metrics[acts[0]] if k.startswith('mean_')]

    metric_vectors = {}
    for mname in metric_names:
        key = f'mean_{mname}'
        vals = [jac_metrics[a][key] for a in acts]
        if all(np.isfinite(v) for v in vals):
            metric_vectors[mname] = np.array(vals)

    # Print data table
    print("\n" + "=" * 90)
    print("DATA TABLE (l_max=6)")
    print("=" * 90)
    header = f"{'Activation':<14} {'Leakage R':>10} {'MAE (meV)':>10} {'MAE std':>10}"
    for mn in sorted(metric_vectors.keys()):
        header += f" {mn[:15]:>16}"
    print(header)
    print("-" * len(header))

    for a in acts:
        row = f"{a:<14} {leakage[a]:>10.6f} {mae_data[a][0]:>10.4f} {mae_data[a][1]:>10.4f}"
        for mn in sorted(metric_vectors.keys()):
            row += f" {jac_metrics[a][f'mean_{mn}']:>16.4f}"
        print(row)

    # Compute correlations
    print("\n" + "=" * 90)
    print("CORRELATION TABLE")
    print("=" * 90)

    corr_results = {}

    print(f"\n{'Metric':<25} {'r(vs MAE)':>10} {'p(vs MAE)':>10} "
          f"{'rho(vs MAE)':>12} {'r(vs R)':>10} {'rho(vs R)':>12}")
    print("-" * 80)

    # Leakage vs MAE first
    r_pear, p_pear = stats.pearsonr(R, MAE_mean)
    rho_spear, p_spear = stats.spearmanr(R, MAE_mean)
    print(f"{'Leakage R':<25} {r_pear:>10.4f} {p_pear:>10.4f} {rho_spear:>12.4f} "
          f"{'---':>10} {'---':>12}")
    corr_results['Leakage R'] = {
        'pearson_vs_mae': r_pear, 'spearman_vs_mae': rho_spear,
        'p_pearson_vs_mae': p_pear
    }

    for mn in sorted(metric_vectors.keys()):
        mv = metric_vectors[mn]
        r_pear_mae, p_pear_mae = stats.pearsonr(mv, MAE_mean)
        rho_spear_mae, _ = stats.spearmanr(mv, MAE_mean)
        r_pear_R, p_pear_R = stats.pearsonr(mv, R)
        rho_spear_R, _ = stats.spearmanr(mv, R)

        print(f"{mn:<25} {r_pear_mae:>10.4f} {p_pear_mae:>10.4f} {rho_spear_mae:>12.4f} "
              f"{r_pear_R:>10.4f} {rho_spear_R:>12.4f}")

        corr_results[mn] = {
            'pearson_vs_mae': r_pear_mae,
            'spearman_vs_mae': rho_spear_mae,
            'pearson_vs_leakage': r_pear_R,
            'spearman_vs_leakage': rho_spear_R,
            'p_pearson_vs_mae': p_pear_mae,
        }

    return corr_results, acts, R, MAE_mean, MAE_std, metric_vectors


# ─── Plotting ───────────────────────────────────────────────────────────

def plot_scatter_grid(corr_results, acts, R, MAE_mean, MAE_std, metric_vectors, output_dir):
    """Scatter plots: leakage (x) vs each expressiveness metric (y), colored by MAE."""
    os.makedirs(output_dir, exist_ok=True)

    # Select metrics to plot (most interesting ones)
    plot_metrics = [
        'effective_rank', 'stable_rank', 'log_det_J',
        'condition_number', 'nuclear_norm', 'sv_entropy',
        'normalized_eff_rank', 'output_variance',
    ]
    plot_metrics = [m for m in plot_metrics if m in metric_vectors]

    n_metrics = len(plot_metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    # Color map by MAE
    norm = Normalize(vmin=MAE_mean.min() - 0.1, vmax=MAE_mean.max() + 0.1)
    cmap = cm.RdYlGn_r  # Red = high MAE (bad), green = low MAE (good)

    color_map = {
        'SiLU': '#1f77b4', 'Softplus_1': '#2ca02c', 'Softplus_10': '#98df8a',
        'tanh': '#9467bd', 'ReLU': '#d62728', 'abs': '#ff7f0e',
    }
    markers = {
        'SiLU': 'o', 'Softplus_1': 's', 'Softplus_10': 'D',
        'tanh': '^', 'ReLU': 'v', 'abs': 'P',
    }

    for idx, mn in enumerate(plot_metrics):
        ax = axes[idx]
        mv = metric_vectors[mn]

        # Correlation info
        r_val, p_val = stats.pearsonr(mv, MAE_mean)
        rho_val, _ = stats.spearmanr(mv, MAE_mean)

        sc = ax.scatter(R, mv, c=MAE_mean, cmap=cmap, norm=norm,
                        s=120, edgecolors='black', linewidths=0.8, zorder=5)

        for i, a in enumerate(acts):
            ax.annotate(a, (R[i], mv[i]), fontsize=7,
                        xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('Spectral Leakage R (smoothness)')
        ax.set_ylabel(mn.replace('_', ' ').title())
        ax.set_title(f'{mn}\nr(MAE)={r_val:.3f}, rho(MAE)={rho_val:.3f}',
                      fontsize=10)
        ax.grid(True, alpha=0.3)

    # Remove unused axes
    for idx in range(len(plot_metrics), len(axes)):
        axes[idx].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label('QM9 Test MAE (meV)', fontsize=10)

    fig.suptitle('Expressiveness Metrics vs Spectral Leakage, Colored by Downstream MAE\n'
                 '(l_max=6, QM9 U0 target)', fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.91, 1.0])
    path = os.path.join(output_dir, 'analysis_expressiveness_scatter.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_mae_vs_metrics(corr_results, acts, R, MAE_mean, MAE_std, metric_vectors, output_dir):
    """Direct scatter of each metric vs MAE with error bars."""
    os.makedirs(output_dir, exist_ok=True)

    plot_metrics = [
        'effective_rank', 'stable_rank', 'log_det_J',
        'condition_number', 'nuclear_norm', 'normalized_eff_rank',
    ]
    plot_metrics = [m for m in plot_metrics if m in metric_vectors]

    color_map = {
        'SiLU': '#1f77b4', 'Softplus_1': '#2ca02c', 'Softplus_10': '#98df8a',
        'tanh': '#9467bd', 'ReLU': '#d62728', 'abs': '#ff7f0e',
    }

    n_metrics = len(plot_metrics)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, mn in enumerate(plot_metrics):
        ax = axes[idx]
        mv = metric_vectors[mn]

        r_val, p_val = stats.pearsonr(mv, MAE_mean)
        rho_val, _ = stats.spearmanr(mv, MAE_mean)

        for i, a in enumerate(acts):
            ax.errorbar(mv[i], MAE_mean[i], yerr=MAE_std[i],
                        fmt='o', color=color_map.get(a, 'gray'),
                        markersize=10, capsize=4, label=a,
                        markeredgecolor='black', markeredgewidth=0.5)

        # Trend line
        z = np.polyfit(mv, MAE_mean, 1)
        x_line = np.linspace(mv.min(), mv.max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', alpha=0.5)

        ax.set_xlabel(mn.replace('_', ' ').title())
        ax.set_ylabel('QM9 Test MAE (meV)')
        ax.set_title(f'r={r_val:.3f} (p={p_val:.3f}), rho={rho_val:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    for idx in range(len(plot_metrics), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('QM9 Test MAE vs Expressiveness Metrics\n(l_max=6, U0 target, 50 epochs)',
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'analysis_mae_vs_metrics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_leakage_vs_mae(acts, R, MAE_mean, MAE_std, output_dir):
    """Simple leakage vs MAE scatter with error bars."""
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        'SiLU': '#1f77b4', 'Softplus_1': '#2ca02c', 'Softplus_10': '#98df8a',
        'tanh': '#9467bd', 'ReLU': '#d62728', 'abs': '#ff7f0e',
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, a in enumerate(acts):
        ax.errorbar(R[i], MAE_mean[i], yerr=MAE_std[i],
                    fmt='o', color=color_map.get(a, 'gray'),
                    markersize=12, capsize=5, label=a,
                    markeredgecolor='black', markeredgewidth=0.8)

    r_pear, p_pear = stats.pearsonr(R, MAE_mean)
    rho, _ = stats.spearmanr(R, MAE_mean)

    # Trend line
    z = np.polyfit(R, MAE_mean, 1)
    x_line = np.linspace(R.min() - 0.01, R.max() + 0.01, 50)
    ax.plot(x_line, np.polyval(z, x_line), '--', color='gray', alpha=0.5)

    ax.set_xlabel('Spectral Leakage Ratio R (smoothness proxy)', fontsize=12)
    ax.set_ylabel('QM9 Test MAE (meV)', fontsize=12)
    ax.set_title(f'Smoothness vs Downstream Performance\n'
                 f'r={r_pear:.3f} (p={p_pear:.3f}), Spearman rho={rho:.3f}',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'analysis_leakage_vs_mae.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_tradeoff_summary(acts, R, MAE_mean, metric_vectors, output_dir):
    """2D scatter: leakage (x) vs best expressiveness metric (y), sized by MAE."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    color_map = {
        'SiLU': '#1f77b4', 'Softplus_1': '#2ca02c', 'Softplus_10': '#98df8a',
        'tanh': '#9467bd', 'ReLU': '#d62728', 'abs': '#ff7f0e',
    }

    # Left: Leakage vs log|det(J)|, sized by MAE
    ax = axes[0]
    if 'log_det_J' in metric_vectors:
        mv = metric_vectors['log_det_J']
        sizes = 100 + 300 * (MAE_mean - MAE_mean.min()) / (MAE_mean.max() - MAE_mean.min())
        for i, a in enumerate(acts):
            ax.scatter(R[i], mv[i], s=sizes[i],
                       color=color_map.get(a, 'gray'),
                       edgecolors='black', linewidths=0.8, zorder=5, label=a)
            ax.annotate(a, (R[i], mv[i]), fontsize=8,
                        xytext=(6, 6), textcoords='offset points')
        ax.set_xlabel('Spectral Leakage R (smoothness)')
        ax.set_ylabel('log|det(J)| (information volume)')
        ax.set_title('Smoothness-Expressiveness Tradeoff\n(marker size ~ MAE, larger = worse)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Right: Leakage vs stable rank, sized by MAE
    ax = axes[1]
    if 'stable_rank' in metric_vectors:
        mv = metric_vectors['stable_rank']
        sizes = 100 + 300 * (MAE_mean - MAE_mean.min()) / (MAE_mean.max() - MAE_mean.min())
        for i, a in enumerate(acts):
            ax.scatter(R[i], mv[i], s=sizes[i],
                       color=color_map.get(a, 'gray'),
                       edgecolors='black', linewidths=0.8, zorder=5, label=a)
            ax.annotate(a, (R[i], mv[i]), fontsize=8,
                        xytext=(6, 6), textcoords='offset points')
        ax.set_xlabel('Spectral Leakage R (smoothness)')
        ax.set_ylabel('Stable Rank')
        ax.set_title('Smoothness vs Stable Rank\n(marker size ~ MAE, larger = worse)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'analysis_tradeoff_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Expressiveness Analysis: S2 Activation Smoothness-Expressiveness Tradeoff")
    print("=" * 70)

    # Load data
    print("\n[1] Loading experiment data...")
    leakage = load_expA()
    print(f"  ExpA leakage ratios (l_max={L_MAX}): {leakage}")

    expE = load_expE()
    print(f"  ExpE loaded for {list(expE.keys())}")

    mae_data = load_expF()
    print(f"  ExpF MAE data:")
    for act, (mean, std, n) in sorted(mae_data.items()):
        print(f"    {act}: {mean:.4f} +/- {std:.4f} meV (n={n} seeds)")

    # Compute fresh Jacobian metrics
    print(f"\n[2] Computing Jacobian metrics (l_max={L_MAX}, {NUM_INPUTS} inputs)...")
    jac_metrics = compute_jacobian_metrics_all()

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, 'metrics', 'analysis_expressiveness_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    # Convert for JSON serialization
    jac_save = {}
    for act, m in jac_metrics.items():
        jac_save[act] = {k: v if np.isfinite(v) else str(v) for k, v in m.items()}
    jac_save['_meta'] = {
        'l_max': L_MAX,
        'num_inputs': NUM_INPUTS,
        'activations': ACTIVATIONS,
    }
    with open(metrics_path, 'w') as f:
        json.dump(jac_save, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # Compute correlations
    print(f"\n[3] Computing correlations...")
    corr_results, acts, R, MAE_mean, MAE_std, metric_vectors = \
        compute_correlations(leakage, mae_data, jac_metrics)

    # Print summary of best predictors
    print("\n" + "=" * 70)
    print("BEST PREDICTORS OF DOWNSTREAM MAE (by |Pearson r|)")
    print("=" * 70)
    ranked = sorted(corr_results.items(),
                    key=lambda x: abs(x[1].get('pearson_vs_mae', 0)),
                    reverse=True)
    for rank, (mn, vals) in enumerate(ranked, 1):
        r = vals.get('pearson_vs_mae', 0)
        p = vals.get('p_pearson_vs_mae', 1)
        rho = vals.get('spearman_vs_mae', 0)
        sig = '*' if p < 0.05 else ' '
        print(f"  {rank}. {mn:<25} r={r:>7.4f} (p={p:.4f}){sig}  rho={rho:>7.4f}")

    # Plots
    print(f"\n[4] Generating plots...")
    fig_dir = os.path.join(RESULTS_DIR, 'figures')
    plot_scatter_grid(corr_results, acts, R, MAE_mean, MAE_std, metric_vectors, fig_dir)
    plot_mae_vs_metrics(corr_results, acts, R, MAE_mean, MAE_std, metric_vectors, fig_dir)
    plot_leakage_vs_mae(acts, R, MAE_mean, MAE_std, fig_dir)
    plot_tradeoff_summary(acts, R, MAE_mean, metric_vectors, fig_dir)

    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
