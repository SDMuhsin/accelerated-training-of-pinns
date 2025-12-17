"""
Benchmark: Compare original vs PyTorch-accelerated SPECTO-ELM.

This script measures:
1. Training time for different layer depths
2. Accuracy (L2 error)
3. Speedup from acceleration
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/workspace/dt-pinn/src')
sys.path.insert(0, '/workspace/dt-pinn/src/experiment_dt_elm_pinn')

from tasks import TaskRegistry
from models import ModelRegistry


def run_benchmark(task_name: str, model_pairs: list, n_trials: int = 5):
    """
    Run benchmark comparing original and accelerated models.

    Args:
        task_name: Name of the task to benchmark on
        model_pairs: List of (original_name, accelerated_name, description) tuples
        n_trials: Number of trials per model
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {task_name}")
    print(f"{'='*80}")

    # Load task
    task_cls = TaskRegistry.get(task_name)
    task = task_cls()

    results = []

    for orig_name, accel_name, desc in model_pairs:
        print(f"\n{desc}:")
        print("-" * 40)

        # Original model - warmup first, then time
        task = task_cls()  # Reuse same task
        model_cls = ModelRegistry.get(orig_name)

        # Warmup run (JIT compilation, etc.)
        model = model_cls(task)
        _ = model.train()

        # Timed runs
        orig_times = []
        orig_errors = []
        for trial in range(n_trials):
            model = model_cls(task)  # Fresh model, same task
            result = model.train()
            orig_times.append(result.train_time)
            if result.l2_error is not None:
                orig_errors.append(result.l2_error)

        orig_time_median = np.median(orig_times)
        orig_error = np.median(orig_errors) if orig_errors else None

        print(f"  {orig_name}:")
        print(f"    Time: {orig_time_median*1000:.2f} ms (median of {n_trials})")
        if orig_error:
            print(f"    L2 Error: {orig_error:.6e}")

        # Accelerated model - warmup first, then time
        task = task_cls()  # Fresh task for accelerated
        model_cls = ModelRegistry.get(accel_name)

        # Warmup run (JIT compilation, etc.)
        model = model_cls(task)
        _ = model.train()

        # Timed runs
        accel_times = []
        accel_errors = []
        for trial in range(n_trials):
            model = model_cls(task)  # Fresh model, same task
            result = model.train()
            accel_times.append(result.train_time)
            if result.l2_error is not None:
                accel_errors.append(result.l2_error)

        accel_time_median = np.median(accel_times)
        accel_error = np.median(accel_errors) if accel_errors else None

        print(f"  {accel_name}:")
        print(f"    Time: {accel_time_median*1000:.2f} ms (median of {n_trials})")
        if accel_error:
            print(f"    L2 Error: {accel_error:.6e}")

        # Comparison
        speedup = orig_time_median / accel_time_median if accel_time_median > 0 else 0
        error_change = (accel_error - orig_error) / orig_error * 100 if orig_error and accel_error else None

        print(f"  Speedup: {speedup:.2f}x")
        if error_change is not None:
            print(f"  Error change: {error_change:+.1f}%")

        results.append({
            'desc': desc,
            'orig_name': orig_name,
            'accel_name': accel_name,
            'orig_time': orig_time_median,
            'accel_time': accel_time_median,
            'orig_error': orig_error,
            'accel_error': accel_error,
            'speedup': speedup,
        })

    return results


def main():
    print("=" * 80)
    print("SPECTO-ELM ACCELERATION BENCHMARK")
    print("=" * 80)

    # Check device
    try:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"PyTorch device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")
        return

    # Define model pairs to compare - original vs hybrid (best for CPU)
    model_pairs = [
        ('dt-elm-pinn', 'dt-elm-pinn-hybrid', '1-Layer [100]'),
        ('dt-elm-pinn-deep2', 'dt-elm-pinn-hybrid-deep2', '2-Layer [100,100]'),
        ('dt-elm-pinn-deep3', 'dt-elm-pinn-hybrid-deep3', '3-Layer [100,100,100]'),
        ('dt-elm-pinn-deep4', 'dt-elm-pinn-hybrid-deep4', '4-Layer [100,100,100,100]'),
    ]

    # Benchmark on linear task only (nonlinear has Cholesky stability issues)
    linear_results = run_benchmark('spectral-poisson-square', model_pairs)

    # Also benchmark on peaked source
    peaked_results = run_benchmark('spectral-poisson-peaked', model_pairs)

    nonlinear_results = []  # Skip nonlinear for now

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nLinear PDE (spectral-poisson-square):")
    print(f"{'Config':<30} {'Original':>12} {'Accelerated':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in linear_results:
        print(f"{r['desc']:<30} {r['orig_time']*1000:>10.2f}ms {r['accel_time']*1000:>10.2f}ms {r['speedup']:>9.2f}x")

    print("\nPeaked Source (spectral-poisson-peaked):")
    print(f"{'Config':<30} {'Original':>12} {'Hybrid':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in peaked_results:
        print(f"{r['desc']:<30} {r['orig_time']*1000:>10.2f}ms {r['accel_time']*1000:>10.2f}ms {r['speedup']:>9.2f}x")

    # Overall assessment
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    avg_speedup_linear = np.mean([r['speedup'] for r in linear_results])
    avg_speedup_peaked = np.mean([r['speedup'] for r in peaked_results])

    print(f"Average speedup on linear PDE: {avg_speedup_linear:.2f}x")
    print(f"Average speedup on peaked source: {avg_speedup_peaked:.2f}x")

    # Check accuracy preservation
    max_error_change = 0
    for r in linear_results + peaked_results:
        if r['orig_error'] and r['accel_error']:
            change = abs(r['accel_error'] - r['orig_error']) / r['orig_error'] * 100
            max_error_change = max(max_error_change, change)

    print(f"Maximum error change: {max_error_change:.1f}%")
    if max_error_change < 5:
        print("Accuracy preserved (error change < 5%)")
    else:
        print("Note: Some accuracy variation observed")


if __name__ == "__main__":
    main()
