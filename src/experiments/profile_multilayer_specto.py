"""
Profile SPECTO-ELM to identify exact bottleneck in multilayer architectures.

This script measures time spent in each component:
1. Hidden layer construction (forward pass)
2. Operator products (L @ H, B @ H)
3. Least-squares solve (the suspected bottleneck)
4. Newton iterations (for nonlinear PDEs)
"""

import numpy as np
import scipy.linalg
import time
import sys
sys.path.insert(0, '/workspace/dt-pinn/src')
sys.path.insert(0, '/workspace/dt-pinn/src/experiment_dt_elm_pinn')

from tasks import TaskRegistry


def profile_lstsq_methods(A, b, n_trials=5):
    """Compare different least-squares solvers."""
    results = {}

    # Cholesky method (with regularization for numerical stability)
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        AtA = A.T @ A
        Atb = A.T @ b
        AtA += 1e-10 * np.eye(AtA.shape[0])  # Regularization
        c, low = scipy.linalg.cho_factor(AtA)
        x = scipy.linalg.cho_solve((c, low), Atb)
        times.append(time.perf_counter() - start)
    results['cholesky'] = np.median(times)

    # SVD method
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        times.append(time.perf_counter() - start)
    results['svd'] = np.median(times)

    # QR method
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        Q, R = np.linalg.qr(A)
        x = scipy.linalg.solve_triangular(R, Q.T @ b)
        times.append(time.perf_counter() - start)
    results['qr'] = np.median(times)

    return results


def profile_specto_elm(task_name, hidden_sizes, n_trials=3):
    """Profile SPECTO-ELM for a specific configuration."""
    print(f"\n{'='*60}")
    print(f"Task: {task_name}, Hidden sizes: {hidden_sizes}")
    print(f"{'='*60}")

    # Load task
    task_cls = TaskRegistry.get(task_name)
    task = task_cls()
    data = task.data
    X = data.X_full
    L = data.L  # Sparse Laplacian operator
    B = data.B  # Sparse boundary operator
    N_ib = data.N_ib
    f = data.f
    g = data.g

    print(f"Points: N_total={X.shape[0]}, N_interior={N_ib}, N_bc={X.shape[0]-N_ib}")
    print(f"Input dim: {X.shape[1]}")
    print(f"L shape: {L.shape}, nnz: {L.nnz}")

    np.random.seed(42)
    precision = X.dtype

    # 1. Profile hidden layer construction
    times_hidden = []
    for _ in range(n_trials):
        start = time.perf_counter()
        H_layers = []
        h = X
        input_dim = X.shape[1]
        for n_hidden in hidden_sizes:
            W = np.random.randn(input_dim, n_hidden).astype(precision) * np.sqrt(2.0 / input_dim)
            b_vec = np.random.randn(n_hidden).astype(precision) * 0.1
            h = np.tanh(h @ W + b_vec)
            H_layers.append(h)
            input_dim = n_hidden
        H = np.hstack(H_layers)
        times_hidden.append(time.perf_counter() - start)
        np.random.seed(42)  # Reset for consistency

    total_features = H.shape[1]
    print(f"Total features M: {total_features}")
    print(f"H shape: {H.shape}")
    print(f"Hidden layer construction: {np.median(times_hidden)*1000:.2f} ms")

    # 2. Profile operator products
    times_LH = []
    for _ in range(n_trials):
        start = time.perf_counter()
        LH_full = L @ H
        LH = LH_full[:N_ib, :]
        times_LH.append(time.perf_counter() - start)
    print(f"L @ H computation: {np.median(times_LH)*1000:.2f} ms")

    times_BH = []
    for _ in range(n_trials):
        start = time.perf_counter()
        BH = B @ H
        times_BH.append(time.perf_counter() - start)
    print(f"B @ H computation: {np.median(times_BH)*1000:.2f} ms")

    # 3. Profile system assembly
    LH = (L @ H)[:N_ib, :]
    BH = B @ H

    times_vstack = []
    for _ in range(n_trials):
        start = time.perf_counter()
        A = np.vstack([LH, BH])
        b_vec = np.concatenate([f, g])
        times_vstack.append(time.perf_counter() - start)
    print(f"System assembly: {np.median(times_vstack)*1000:.2f} ms")

    A = np.vstack([LH, BH])
    b_vec = np.concatenate([f, g])
    print(f"A shape: {A.shape}")

    # 4. Profile least-squares solve (THE BOTTLENECK)
    print(f"\nLeast-squares solve breakdown:")

    # Profile AtA
    times_AtA = []
    for _ in range(n_trials):
        start = time.perf_counter()
        AtA = A.T @ A
        times_AtA.append(time.perf_counter() - start)
    print(f"  A^T @ A: {np.median(times_AtA)*1000:.2f} ms")

    # Profile Atb
    times_Atb = []
    for _ in range(n_trials):
        start = time.perf_counter()
        Atb = A.T @ b_vec
        times_Atb.append(time.perf_counter() - start)
    print(f"  A^T @ b: {np.median(times_Atb)*1000:.2f} ms")

    AtA = A.T @ A
    Atb = A.T @ b_vec

    # Profile Cholesky factorization
    times_cho_factor = []
    for _ in range(n_trials):
        start = time.perf_counter()
        c, low = scipy.linalg.cho_factor(AtA + 1e-10 * np.eye(AtA.shape[0]))
        times_cho_factor.append(time.perf_counter() - start)
    print(f"  Cholesky factor: {np.median(times_cho_factor)*1000:.2f} ms")

    c, low = scipy.linalg.cho_factor(AtA + 1e-10 * np.eye(AtA.shape[0]))

    # Profile Cholesky solve
    times_cho_solve = []
    for _ in range(n_trials):
        start = time.perf_counter()
        x = scipy.linalg.cho_solve((c, low), Atb)
        times_cho_solve.append(time.perf_counter() - start)
    print(f"  Cholesky solve: {np.median(times_cho_solve)*1000:.2f} ms")

    # Compare all solvers
    print(f"\nFull solve comparison:")
    solver_times = profile_lstsq_methods(A, b_vec, n_trials)
    for solver, t in solver_times.items():
        print(f"  {solver}: {t*1000:.2f} ms")

    # Total time summary
    total_time = (
        np.median(times_hidden) +
        np.median(times_LH) +
        np.median(times_BH) +
        np.median(times_vstack) +
        solver_times['cholesky']
    )

    print(f"\nTotal time breakdown (Cholesky):")
    print(f"  Hidden layers: {np.median(times_hidden)/total_time*100:.1f}%")
    print(f"  L @ H: {np.median(times_LH)/total_time*100:.1f}%")
    print(f"  B @ H: {np.median(times_BH)/total_time*100:.1f}%")
    print(f"  Assembly: {np.median(times_vstack)/total_time*100:.1f}%")
    print(f"  lstsq solve: {solver_times['cholesky']/total_time*100:.1f}%")
    print(f"  TOTAL: {total_time*1000:.2f} ms")

    return {
        'task': task_name,
        'hidden_sizes': hidden_sizes,
        'total_features': total_features,
        'time_hidden': np.median(times_hidden),
        'time_LH': np.median(times_LH),
        'time_BH': np.median(times_BH),
        'time_assembly': np.median(times_vstack),
        'time_AtA': np.median(times_AtA),
        'time_cho_factor': np.median(times_cho_factor),
        'time_cho_solve': np.median(times_cho_solve),
        'time_lstsq_cholesky': solver_times['cholesky'],
        'time_lstsq_svd': solver_times['svd'],
        'time_lstsq_qr': solver_times['qr'],
        'total_time': total_time,
    }


def main():
    print("=" * 70)
    print("SPECTO-ELM PROFILING: Identifying Multilayer Bottleneck")
    print("=" * 70)

    # Test with different layer configurations
    task_name = "spectral-poisson-square"  # Linear task for clean profiling

    configs = [
        [100],          # 1-layer: M=100
        [100, 100],     # 2-layer: M=200
        [100, 100, 100],  # 3-layer: M=300
        [100, 100, 100, 100],  # 4-layer: M=400
        [200],          # 1-layer wider: M=200
        [400],          # 1-layer widest: M=400
    ]

    results = []
    for hidden_sizes in configs:
        result = profile_specto_elm(task_name, hidden_sizes)
        results.append(result)

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Scaling Analysis")
    print("=" * 70)
    print(f"{'Config':<25} {'M':>6} {'Total':>10} {'lstsq%':>8} {'AtA%':>8} {'factor%':>8}")
    print("-" * 70)
    for r in results:
        config_str = str(r['hidden_sizes'])
        lstsq_pct = r['time_lstsq_cholesky'] / r['total_time'] * 100
        AtA_pct = r['time_AtA'] / r['total_time'] * 100
        factor_pct = r['time_cho_factor'] / r['total_time'] * 100
        print(f"{config_str:<25} {r['total_features']:>6} {r['total_time']*1000:>9.2f}ms {lstsq_pct:>7.1f}% {AtA_pct:>7.1f}% {factor_pct:>7.1f}%")

    # Scaling analysis
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)
    baseline = results[0]['total_time']
    baseline_m = results[0]['total_features']
    for r in results:
        ratio = r['total_features'] / baseline_m
        expected_cubic = ratio ** 3
        actual = r['total_time'] / baseline
        print(f"M={r['total_features']:>3}: Actual {actual:.1f}x, Expected O(M³)={expected_cubic:.1f}x")


if __name__ == "__main__":
    main()
