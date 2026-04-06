"""
Run all experiments and generate results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
torch.serialization.add_safe_globals([slice])

import time


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Spherical Activation Sampling Comparison")
    print("=" * 60)

    t_total = time.perf_counter()

    # Experiment 1
    print("\n[1/4] Accuracy Comparison...")
    t0 = time.perf_counter()
    from experiments.exp1_accuracy import run_experiment_1
    run_experiment_1(output_dir)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Experiment 2
    print("\n[2/4] Computational Cost...")
    t0 = time.perf_counter()
    from experiments.exp2_computational_cost import run_experiment_2
    run_experiment_2(output_dir)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Experiment 3
    print("\n[3/4] Task Performance...")
    t0 = time.perf_counter()
    from experiments.exp3_task_performance import run_experiment_3
    run_experiment_3(output_dir)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Experiment 4
    print("\n[4/4] Scaling Analysis...")
    t0 = time.perf_counter()
    from experiments.exp4_resolution_scaling import run_experiment_4
    run_experiment_4(output_dir)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    total_time = time.perf_counter() - t_total
    print(f"\nAll experiments completed in {total_time:.1f}s")
    print(f"Results saved to {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
