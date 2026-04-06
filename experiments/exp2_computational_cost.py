"""
Experiment 2: Computational Cost

Benchmark forward/backward pass time and memory for S2Activation
across different sampling methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.serialization
torch.serialization.add_safe_globals([slice])
from e3nn import o3
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.s2_activation import S2Activation


def benchmark_forward(act_module, x, num_trials=100, warmup=5):
    """Benchmark forward pass time in ms."""
    device = x.device
    # Warmup
    for _ in range(warmup):
        _ = act_module(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = act_module(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)


def benchmark_backward(act_module, x, num_trials=100, warmup=5):
    """Benchmark backward pass time in ms."""
    device = x.device
    # Warmup
    for _ in range(warmup):
        x_w = x.clone().requires_grad_(True)
        out = act_module(x_w)
        out.sum().backward()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(num_trials):
        x_w = x.clone().requires_grad_(True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = act_module(x_w)
        out.sum().backward()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)


def measure_memory(act_module, x):
    """Measure peak GPU memory for forward+backward (GPU only)."""
    if not torch.cuda.is_available():
        return {'peak_memory_mb': 0, 'note': 'CPU only'}

    device = torch.device('cuda')
    act_module = act_module.to(device)
    x = x.to(device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1e6

    x_w = x.clone().requires_grad_(True)
    out = act_module(x_w)
    mem_after_fwd = torch.cuda.memory_allocated() / 1e6

    out.sum().backward()
    torch.cuda.synchronize()
    mem_peak = torch.cuda.max_memory_allocated() / 1e6

    return {
        'model_memory_mb': float(mem_before),
        'activations_memory_mb': float(mem_after_fwd - mem_before),
        'peak_memory_mb': float(mem_peak),
    }


def run_experiment_2(output_dir='results'):
    print("=" * 60)
    print("Experiment 2: Computational Cost")
    print("=" * 60)

    device = torch.device('cpu')  # CPU benchmarks for fair comparison
    batch_sizes = [1, 8, 32]
    l_max_values = [5, 10]
    num_trials = 30

    sampling_configs = {
        'uniform_50': ('uniform', {'resolution': 50}),
        'uniform_100': ('uniform', {'resolution': 100}),
        'uniform_150': ('uniform', {'resolution': 150}),
        'uniform_200': ('uniform', {'resolution': 200}),
        'gauss_legendre': ('gauss_legendre', {}),
        'lebedev_11': ('lebedev', {'degree': 11}),
        'lebedev_15': ('lebedev', {'degree': 15}),
        'lebedev_19': ('lebedev', {'degree': 19}),
        'lebedev_21': ('lebedev', {'degree': 21}),
        'lebedev_25': ('lebedev', {'degree': 25}),
    }

    all_results = {}

    for l_max in l_max_values:
        print(f"\n--- l_max = {l_max} ---")
        irreps = o3.Irreps([(1, (l, 1)) for l in range(l_max + 1)])
        n_features = irreps.dim
        all_results[l_max] = {}

        for config_name, (method, kw) in sampling_configs.items():
            try:
                act = S2Activation(irreps, torch.relu, sampling_method=method, **kw)
            except Exception as e:
                print(f"  {config_name}: SKIP ({e})")
                continue

            act.eval()
            n_pts = act.n_points
            all_results[l_max][config_name] = {'n_points': n_pts, 'method': method, 'batch_results': {}}

            for bs in batch_sizes:
                x = torch.randn(bs, n_features)
                fwd_mean, fwd_std = benchmark_forward(act, x, num_trials)
                bwd_mean, bwd_std = benchmark_backward(act, x, num_trials)

                all_results[l_max][config_name]['batch_results'][bs] = {
                    'forward_ms': float(fwd_mean),
                    'forward_std': float(fwd_std),
                    'backward_ms': float(bwd_mean),
                    'backward_std': float(bwd_std),
                    'total_ms': float(fwd_mean + bwd_mean),
                    'time_per_sample_ms': float((fwd_mean + bwd_mean) / bs),
                }

            bs32 = all_results[l_max][config_name]['batch_results'].get(32, {})
            print(f"  {config_name:20s} (N={n_pts:5d}) | fwd={bs32.get('forward_ms',0):.2f}ms bwd={bs32.get('backward_ms',0):.2f}ms total={bs32.get('total_ms',0):.2f}ms")

    # Save
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/exp2_cost.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    plot_time_scaling(all_results, output_dir)
    plot_memory_comparison(all_results, output_dir)
    print_cost_tables(all_results, output_dir)

    return all_results


def plot_time_scaling(results, output_dir):
    """Plot forward+backward time vs number of sampling points."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    colors = {'uniform': 'tab:blue', 'gauss_legendre': 'tab:orange', 'lebedev': 'tab:red'}
    markers = {'uniform': 's', 'gauss_legendre': '^', 'lebedev': 'o'}

    for ax_idx, (l_max, data) in enumerate(sorted(results.items(), key=lambda x: int(x[0]))):
        ax = axes[ax_idx]
        method_pts = {}
        bs = 32

        for config_name, vals in data.items():
            m = vals['method']
            if m not in method_pts:
                method_pts[m] = {'n': [], 't': []}
            if bs in vals['batch_results']:
                method_pts[m]['n'].append(vals['n_points'])
                method_pts[m]['t'].append(vals['batch_results'][bs]['total_ms'])

        for method_name, md in method_pts.items():
            if not md['n']:
                continue
            sorted_pairs = sorted(zip(md['n'], md['t']))
            ns, ts = zip(*sorted_pairs)
            ax.plot(ns, ts, color=colors.get(method_name, 'gray'),
                    marker=markers.get(method_name, 'x'), label=method_name, linewidth=1.5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of sampling points')
        ax.set_ylabel('Total time (ms), batch=32')
        ax.set_title(f'l_max = {l_max}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 2: Computational Time vs Sampling Points', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp2_time_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp2_time_scaling.png")


def plot_memory_comparison(results, output_dir):
    """Plot total time for different batch sizes (bar chart)."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    # Pick l_max=10 if available
    l_max_key = 10 if 10 in results else list(results.keys())[0]
    data = results[l_max_key]

    fig, ax = plt.subplots(figsize=(12, 6))
    batch_sizes = [1, 8, 32, 64]
    width = 0.15
    x_pos = np.arange(len(batch_sizes))

    config_names = sorted(data.keys())
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(config_names)))

    for i, config_name in enumerate(config_names):
        vals = data[config_name]
        times = [vals['batch_results'].get(bs, {}).get('total_ms', 0) for bs in batch_sizes]
        ax.bar(x_pos + i * width, times, width, label=f"{config_name} (N={vals['n_points']})",
               color=colors_list[i])

    ax.set_xlabel('Batch size')
    ax.set_ylabel('Total time (ms)')
    ax.set_title(f'Forward+Backward Time by Batch Size (l_max={l_max_key})')
    ax.set_xticks(x_pos + width * len(config_names) / 2)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp2_batch_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp2_batch_comparison.png")


def print_cost_tables(results, output_dir):
    """Print summary table."""
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    lines = []

    for l_max_key in sorted(results.keys(), key=lambda x: int(x)):
        data = results[l_max_key]
        lines.append(f"\n{'='*90}")
        lines.append(f"l_max = {l_max_key}, batch_size = 32")
        lines.append(f"{'='*90}")
        lines.append(f"{'Config':<25s} | {'N pts':>7s} | {'Fwd (ms)':>10s} | {'Bwd (ms)':>10s} | {'Total (ms)':>10s} | {'ms/sample':>10s}")
        lines.append("-" * 90)

        sorted_items = sorted(data.items(), key=lambda x: x[1]['batch_results'].get(32, {}).get('total_ms', 999))
        for config_name, vals in sorted_items:
            bs32 = vals['batch_results'].get(32, {})
            lines.append(f"{config_name:<25s} | {vals['n_points']:>7d} | {bs32.get('forward_ms',0):>10.2f} | {bs32.get('backward_ms',0):>10.2f} | {bs32.get('total_ms',0):>10.2f} | {bs32.get('time_per_sample_ms',0):>10.3f}")

    table_str = '\n'.join(lines)
    print(table_str)
    with open(f'{output_dir}/tables/exp2_summary.txt', 'w') as f:
        f.write(table_str)


if __name__ == '__main__':
    run_experiment_2()
