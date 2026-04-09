"""
Experiment D: Task Performance

Verify that spectral leakage / equivariance analysis predicts downstream performance.
Train a SphericalCNN classifier with different (activation × sampling) configs.

Chain: equivariance error → model performance
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
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.s2_activation import S2Activation
from src.spherical_harmonics_utils import spherical_harmonics_on_points


def create_spherical_dataset(lmax=6, num_train=4000, num_val=1000, num_test=1000, seed=42):
    """
    Synthetic spherical classification task designed to stress-test S2Activation.
    Classes differ primarily in high-l coefficients, making the task sensitive
    to how well S2Activation preserves spectral information.
    """
    torch.manual_seed(seed)
    n_coeffs = (lmax + 1) ** 2
    num_classes = 5

    # Class prototypes: differences mainly in higher l degrees
    prototypes = torch.zeros(num_classes, n_coeffs, dtype=torch.float32)

    # Shared low-l baseline (same for all classes)
    shared_low = torch.randn(n_coeffs) * 0.3
    for c in range(num_classes):
        prototypes[c] = shared_low.clone()
        # Class-discriminative signal only at high l
        for l in range(max(1, lmax // 2), lmax + 1):
            start = l ** 2
            end = (l + 1) ** 2
            phase = 2 * np.pi * c / num_classes
            for m_idx in range(end - start):
                prototypes[c, start + m_idx] += 0.5 * np.sin(phase + m_idx * 0.7)

    total = num_train + num_val + num_test
    labels = torch.randint(0, num_classes, (total,))
    # High noise to make it challenging
    data = prototypes[labels] + torch.randn(total, n_coeffs) * 0.8

    splits = [num_train, num_train + num_val]
    return (
        (data[:splits[0]], labels[:splits[0]]),
        (data[splits[0]:splits[1]], labels[splits[0]:splits[1]]),
        (data[splits[1]:], labels[splits[1]:]),
    )


class SphericalCNN(nn.Module):
    def __init__(self, lmax, act_fn, sampling_method='gauss_legendre',
                 sampling_kw=None, num_classes=5):
        super().__init__()
        if sampling_kw is None:
            sampling_kw = {}

        n_features = (lmax + 1) ** 2
        irreps_in = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

        self.fc1 = nn.Linear(n_features, n_features)
        self.act1 = S2Activation(irreps_in, act_fn,
                                 sampling_method=sampling_method, **sampling_kw)
        n_out1 = self.act1.irreps_out.dim

        self.fc2 = nn.Linear(n_out1, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return self.fc3(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total


def run_experiment_D(output_dir='results'):
    print("=" * 60)
    print("Experiment D: Task Performance")
    print("=" * 60)

    device = torch.device('cpu')
    lmax = 6
    epochs = 15
    batch_size = 64
    lr = 1e-3
    num_runs = 2

    print("Creating dataset...")
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = \
        create_spherical_dataset(lmax=lmax)
    train_loader = DataLoader(TensorDataset(train_data, train_labels),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size)

    # Activation functions to test
    activations = {
        'ReLU': torch.relu,
        'SiLU': nn.SiLU(),
        'tanh': torch.tanh,
        'Softplus_1': nn.Softplus(beta=1),
        'Softplus_10': nn.Softplus(beta=10),
    }

    # Sampling configs
    sampling_configs = {
        'GL_1x': ('gauss_legendre', {}),
        'GL_2x': ('gauss_legendre', {'n_theta': 2 * (lmax + 1), 'n_phi': 4 * (lmax + 1)}),
        'Leb_min': ('lebedev', {'degree': 2 * lmax + 1}),
    }

    all_results = {}

    for act_name, act_fn in activations.items():
        for samp_name in ['GL_1x', 'GL_2x', 'Leb_min']:
            config_key = f"{act_name}_{samp_name}"
            print(f"\n--- {config_key} ---")

            # Build sampling kwargs
            if samp_name == 'GL_1x':
                method = 'gauss_legendre'
                kw = {}
            elif samp_name == 'GL_2x':
                method = 'gauss_legendre'
                kw = {'n_theta': 2 * (lmax + 1), 'n_phi': 4 * (lmax + 1)}
            elif samp_name == 'Leb_min':
                method = 'lebedev'
                valid_degs = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
                deg_target = 2 * lmax + 1
                candidates = [d for d in valid_degs if d >= deg_target]
                kw = {'degree': candidates[0] if candidates else valid_degs[-1]}

            run_results = []

            for run_idx in range(num_runs):
                torch.manual_seed(run_idx * 100 + 1)
                try:
                    model = SphericalCNN(lmax, act_fn, sampling_method=method,
                                        sampling_kw=kw, num_classes=5).to(device)
                except Exception as e:
                    print(f"  Failed to create model: {e}")
                    break

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                train_losses = []
                val_accs = []
                t_start = time.time()

                for epoch in range(epochs):
                    loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                    val_acc = evaluate(model, val_loader, device)
                    train_losses.append(loss)
                    val_accs.append(val_acc)

                train_time = time.time() - t_start
                test_acc = evaluate(model, test_loader, device)

                run_results.append({
                    'test_acc': test_acc,
                    'best_val_acc': max(val_accs),
                    'final_val_acc': val_accs[-1],
                    'train_time': train_time,
                    'train_losses': train_losses,
                    'val_accs': val_accs,
                })

                print(f"  Run {run_idx}: test_acc={test_acc:.4f}, "
                      f"val_acc={val_accs[-1]:.4f}, time={train_time:.1f}s")

            if run_results:
                all_results[config_key] = {
                    'activation': act_name,
                    'sampling': samp_name,
                    'mean_test_acc': float(np.mean([r['test_acc'] for r in run_results])),
                    'std_test_acc': float(np.std([r['test_acc'] for r in run_results])),
                    'mean_best_val_acc': float(np.mean([r['best_val_acc'] for r in run_results])),
                    'mean_train_time': float(np.mean([r['train_time'] for r in run_results])),
                    'runs': run_results,
                }

    # Save
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/expD_task.json', 'w') as f:
        # Make JSON serializable
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean(v) for v in obj]
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            return obj
        json.dump(clean(all_results), f, indent=2)

    # Sanity checks
    sanity_check_D(all_results)

    # Plots
    plot_accuracy_comparison(all_results, output_dir)
    plot_accuracy_vs_equivariance(all_results, output_dir)

    return all_results


def sanity_check_D(results):
    print("\n--- Sanity Checks ---")
    checks_passed = 0
    checks_total = 0

    # Check 1: All models should beat random chance (1/5 = 20%)
    checks_total += 1
    all_above_chance = all(v['mean_test_acc'] > 0.25 for v in results.values())
    print(f"  [{'PASS' if all_above_chance else 'FAIL'}] All models above random chance (20%)")
    accs = {k: f"{v['mean_test_acc']:.3f}" for k, v in results.items()}
    print(f"    Accuracies: {accs}")
    if all_above_chance:
        checks_passed += 1

    # Check 2: GL_2x should generally match or beat GL_1x
    checks_total += 1
    improvements = 0
    comparisons = 0
    for act in ['ReLU', 'SiLU', 'tanh', 'Softplus_1', 'Softplus_10']:
        k1 = f"{act}_GL_1x"
        k2 = f"{act}_GL_2x"
        if k1 in results and k2 in results:
            comparisons += 1
            if results[k2]['mean_test_acc'] >= results[k1]['mean_test_acc'] - 0.03:
                improvements += 1
    ok = improvements >= comparisons * 0.5 if comparisons > 0 else False
    print(f"  [{'PASS' if ok else 'FAIL'}] GL_2x ≥ GL_1x in {improvements}/{comparisons} activations")
    if ok:
        checks_passed += 1

    # Check 3: Training converges (loss decreases)
    checks_total += 1
    all_converge = True
    for k, v in results.items():
        losses = v['runs'][0]['train_losses']
        if len(losses) >= 3 and losses[-1] > losses[0]:
            all_converge = False
    print(f"  [{'PASS' if all_converge else 'FAIL'}] All models converge (final loss < initial loss)")
    if all_converge:
        checks_passed += 1

    print(f"\n  Sanity checks: {checks_passed}/{checks_total} passed")


def plot_accuracy_comparison(results, output_dir):
    """Bar chart of test accuracy for each (activation × sampling) config."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    # Group by activation
    activations = sorted(set(v['activation'] for v in results.values()))
    samplings = sorted(set(v['sampling'] for v in results.values()))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(activations))
    width = 0.25

    for i, samp in enumerate(samplings):
        accs = []
        stds = []
        for act in activations:
            key = f"{act}_{samp}"
            if key in results:
                accs.append(results[key]['mean_test_acc'])
                stds.append(results[key]['std_test_acc'])
            else:
                accs.append(0)
                stds.append(0)
        ax.bar(x + i * width, accs, width, yerr=stds, label=samp, capsize=3, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(activations)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Experiment D: Task Performance by (Activation × Sampling)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expD_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expD_accuracy_comparison.png")


def plot_accuracy_vs_equivariance(results, output_dir):
    """Scatter: test accuracy vs equivariance error (from Exp C)."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    equiv_file = f'{output_dir}/metrics/expC_equivariance.json'
    if not os.path.exists(equiv_file):
        print("Skipping accuracy vs equivariance plot: Exp C results not found")
        return

    with open(equiv_file) as f:
        equiv_data = json.load(f)

    # Use l_max=6 equivariance data (same as task)
    equiv_lmax = equiv_data.get('6', {})

    fig, ax = plt.subplots(figsize=(8, 6))

    for config_key, config_data in results.items():
        act_name = config_data['activation']
        samp_name = config_data['sampling']

        # Find matching equivariance error
        equiv_key = samp_name.replace('_', '_')  # e.g., GL_1x
        if act_name in equiv_lmax and equiv_key in equiv_lmax[act_name]:
            e_err = equiv_lmax[act_name][equiv_key]['mean_equiv_error']
            acc = config_data['mean_test_acc']
            ax.scatter(e_err, acc, s=60, zorder=3)
            ax.annotate(f"{act_name}\n{samp_name}", (e_err, acc),
                       fontsize=7, xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Equivariance Error (from Exp C)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Task Performance vs Equivariance Error')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/expD_acc_vs_equiv.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/expD_acc_vs_equiv.png")


if __name__ == '__main__':
    run_experiment_D()
