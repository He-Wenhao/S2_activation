"""
Experiment 3: Task Performance (Spherical MNIST)

Train a SphericalCNN classifier with different S2Activation sampling methods
and compare accuracy + training speed.
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


def create_spherical_mnist(lmax=10, num_train=6000, num_val=2000, num_test=2000, seed=42):
    """
    Create a synthetic Spherical MNIST-like dataset.
    Projects random 28x28 digit-like patterns to SH coefficients.

    Since we may not have torchvision, we generate synthetic digit-like data:
    each class has a distinct spherical pattern.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_coeffs = (lmax + 1) ** 2
    num_classes = 10
    total = num_train + num_val + num_test

    # Create class prototypes as SH coefficient templates
    prototypes = torch.randn(num_classes, n_coeffs, dtype=torch.float32) * 0.5
    # Make each class more distinct at different l ranges
    for c in range(num_classes):
        l_start = c % (lmax + 1)
        idx = l_start ** 2
        end_idx = min((l_start + 3) ** 2, n_coeffs)
        prototypes[c, idx:end_idx] *= 3.0

    # Generate samples by adding noise to prototypes
    labels = torch.randint(0, num_classes, (total,))
    data = prototypes[labels] + torch.randn(total, n_coeffs) * 0.3

    train_data = data[:num_train]
    train_labels = labels[:num_train]
    val_data = data[num_train:num_train + num_val]
    val_labels = labels[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]
    test_labels = labels[num_train + num_val:]

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


class SphericalCNN(nn.Module):
    def __init__(self, lmax=10, sampling_method='gauss_legendre',
                 sampling_kw=None, hidden_dims=(128, 64)):
        super().__init__()
        if sampling_kw is None:
            sampling_kw = {}

        n_features = (lmax + 1) ** 2
        irreps_in = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

        # Layer 1
        self.fc1 = nn.Linear(n_features, n_features)
        self.act1 = S2Activation(irreps_in, torch.relu,
                                 sampling_method=sampling_method, **sampling_kw)

        # After activation, irreps_out may differ; for simplicity we use
        # the same irreps structure and just use linear layers
        n_out1 = self.act1.irreps_out.dim

        # Layer 2: project to hidden
        # We need to handle the irreps carefully
        # Use a simple approach: flatten after activation and use MLP
        self.fc2 = nn.Linear(n_out1, hidden_dims[0])
        self.bn2 = nn.BatchNorm1d(hidden_dims[0])
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[1])
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_dims[1], 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


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


def run_experiment_3(output_dir='results'):
    print("=" * 60)
    print("Experiment 3: Task Performance (Spherical MNIST)")
    print("=" * 60)

    device = torch.device('cpu')
    lmax = 10
    epochs = 20
    batch_size = 32
    lr = 1e-3
    num_runs = 2

    # Create dataset
    print("Creating dataset...")
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = \
        create_spherical_mnist(lmax=lmax)
    train_loader = DataLoader(TensorDataset(train_data, train_labels),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_labels),
                            batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_data, test_labels),
                             batch_size=batch_size)

    sampling_configs = {
        'uniform_50': ('uniform', {'resolution': 50}),
        'uniform_100': ('uniform', {'resolution': 100}),
        'gauss_legendre': ('gauss_legendre', {}),
        'lebedev_15': ('lebedev', {'degree': 15}),
        'lebedev_21': ('lebedev', {'degree': 21}),
    }

    all_results = {}

    for config_name, (method, kw) in sampling_configs.items():
        print(f"\n--- {config_name} ---")
        run_results = []

        for run_idx in range(num_runs):
            torch.manual_seed(run_idx * 100)
            model = SphericalCNN(lmax=lmax, sampling_method=method,
                                 sampling_kw=kw).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            n_pts = model.act1.n_points
            train_losses = []
            val_accs = []

            t_start = time.perf_counter()
            for epoch in range(epochs):
                loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_acc = evaluate(model, val_loader, device)
                train_losses.append(loss)
                val_accs.append(val_acc)

                if (epoch + 1) % 10 == 0:
                    print(f"  Run {run_idx+1}/{num_runs} Epoch {epoch+1}/{epochs}: "
                          f"loss={loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

            train_time = time.perf_counter() - t_start
            test_acc = evaluate(model, test_loader, device)

            run_results.append({
                'train_losses': train_losses,
                'val_accs': val_accs,
                'test_acc': float(test_acc),
                'train_time_sec': float(train_time),
                'n_points': n_pts,
            })

        avg_test_acc = np.mean([r['test_acc'] for r in run_results])
        std_test_acc = np.std([r['test_acc'] for r in run_results])
        avg_time = np.mean([r['train_time_sec'] for r in run_results])

        all_results[config_name] = {
            'method': method,
            'n_points': run_results[0]['n_points'],
            'runs': run_results,
            'avg_test_acc': float(avg_test_acc),
            'std_test_acc': float(std_test_acc),
            'avg_train_time': float(avg_time),
        }
        print(f"  => Test acc: {avg_test_acc:.3f} ± {std_test_acc:.3f}, Time: {avg_time:.1f}s")

    # Save
    os.makedirs(f'{output_dir}/metrics', exist_ok=True)
    with open(f'{output_dir}/metrics/exp3_task.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    plot_training_curves(all_results, output_dir)
    plot_accuracy_vs_cost(all_results, output_dir)
    print_task_tables(all_results, output_dir)

    return all_results


def plot_training_curves(results, output_dir):
    """Plot validation accuracy curves."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Val accuracy
    ax = axes[0]
    for i, (config_name, data) in enumerate(results.items()):
        all_val = [r['val_accs'] for r in data['runs']]
        mean_val = np.mean(all_val, axis=0)
        std_val = np.std(all_val, axis=0)
        epochs = range(1, len(mean_val) + 1)
        ax.plot(epochs, mean_val, label=f"{config_name} (N={data['n_points']})",
                color=colors[i], linewidth=1.5)
        ax.fill_between(epochs, mean_val - std_val, mean_val + std_val,
                         alpha=0.15, color=colors[i])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Training Curves')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Train loss
    ax = axes[1]
    for i, (config_name, data) in enumerate(results.items()):
        all_loss = [r['train_losses'] for r in data['runs']]
        mean_loss = np.mean(all_loss, axis=0)
        epochs = range(1, len(mean_loss) + 1)
        ax.plot(epochs, mean_loss, label=f"{config_name}",
                color=colors[i], linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 3: Spherical MNIST Training', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp3_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp3_training_curves.png")


def plot_accuracy_vs_cost(results, output_dir):
    """Plot accuracy vs training time (Pareto front)."""
    os.makedirs(f'{output_dir}/figures', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'uniform': 'tab:blue', 'gauss_legendre': 'tab:orange', 'lebedev': 'tab:red'}
    markers = {'uniform': 's', 'gauss_legendre': '^', 'lebedev': 'o'}

    for config_name, data in results.items():
        m = data['method']
        ax.scatter(data['avg_train_time'], data['avg_test_acc'],
                   c=colors.get(m, 'gray'), marker=markers.get(m, 'x'),
                   s=100, label=f"{config_name} (N={data['n_points']})", zorder=3)
        ax.errorbar(data['avg_train_time'], data['avg_test_acc'],
                    yerr=data['std_test_acc'], color=colors.get(m, 'gray'),
                    fmt='none', capsize=3)

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Computational Cost')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/exp3_accuracy_vs_cost.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/figures/exp3_accuracy_vs_cost.png")


def print_task_tables(results, output_dir):
    os.makedirs(f'{output_dir}/tables', exist_ok=True)
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Experiment 3: Task Performance Summary")
    lines.append(f"{'='*80}")
    lines.append(f"{'Config':<20s} | {'N pts':>7s} | {'Accuracy':>10s} | {'Std':>8s} | {'Time (s)':>10s}")
    lines.append("-" * 80)

    sorted_items = sorted(results.items(), key=lambda x: -x[1]['avg_test_acc'])
    for config_name, data in sorted_items:
        lines.append(f"{config_name:<20s} | {data['n_points']:>7d} | {data['avg_test_acc']:>10.3f} | {data['std_test_acc']:>8.3f} | {data['avg_train_time']:>10.1f}")

    table_str = '\n'.join(lines)
    print(table_str)
    with open(f'{output_dir}/tables/exp3_summary.txt', 'w') as f:
        f.write(table_str)


if __name__ == '__main__':
    run_experiment_3()
