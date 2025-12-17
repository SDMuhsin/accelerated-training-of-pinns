"""
Test GPU acceleration options for SPECTO-ELM bottlenecks.

This tests:
1. SciPy sparse × dense (baseline)
2. PyTorch sparse × dense (CPU)
3. PyTorch sparse × dense (GPU if available)
4. Different least-squares solvers
"""

import numpy as np
import scipy.sparse
import scipy.linalg
import time
import sys
sys.path.insert(0, '/workspace/dt-pinn/src')
sys.path.insert(0, '/workspace/dt-pinn/src/experiment_dt_elm_pinn')

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from tasks import TaskRegistry


def scipy_sparse_dense_multiply(L_csr, H, n_trials=10):
    """Test SciPy sparse × dense multiplication."""
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        result = L_csr @ H
        times.append(time.perf_counter() - start)
    return np.median(times), result


def torch_sparse_dense_multiply_cpu(L_csr, H, n_trials=10):
    """Test PyTorch sparse × dense multiplication on CPU."""
    if not TORCH_AVAILABLE:
        return None, None

    # Convert to PyTorch sparse tensor
    L_coo = L_csr.tocoo()
    indices = torch.stack([
        torch.from_numpy(L_coo.row.astype(np.int64)),
        torch.from_numpy(L_coo.col.astype(np.int64))
    ])
    values = torch.from_numpy(L_coo.data)
    L_torch = torch.sparse_coo_tensor(indices, values, L_csr.shape).coalesce()
    H_torch = torch.from_numpy(H)

    # Warmup
    _ = torch.sparse.mm(L_torch, H_torch)

    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        result = torch.sparse.mm(L_torch, H_torch)
        times.append(time.perf_counter() - start)
    return np.median(times), result.numpy()


def torch_sparse_dense_multiply_gpu(L_csr, H, n_trials=10):
    """Test PyTorch sparse × dense multiplication on GPU."""
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        return None, None

    # Convert to PyTorch sparse tensor and move to GPU
    L_coo = L_csr.tocoo()
    indices = torch.stack([
        torch.from_numpy(L_coo.row.astype(np.int64)),
        torch.from_numpy(L_coo.col.astype(np.int64))
    ])
    values = torch.from_numpy(L_coo.data)
    L_torch = torch.sparse_coo_tensor(indices, values, L_csr.shape).coalesce().cuda()
    H_torch = torch.from_numpy(H).cuda()

    # Warmup
    _ = torch.sparse.mm(L_torch, H_torch)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = torch.sparse.mm(L_torch, H_torch)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return np.median(times), result.cpu().numpy()


def test_dense_matmul(H, n_trials=10):
    """Test dense matrix multiply (for hidden layer construction)."""
    n_points, n_features = H.shape
    n_hidden = 100
    W = np.random.randn(n_features, n_hidden)
    b = np.random.randn(n_hidden)

    # NumPy baseline
    times_numpy = []
    for _ in range(n_trials):
        start = time.perf_counter()
        result = np.tanh(H @ W + b)
        times_numpy.append(time.perf_counter() - start)

    results = {'numpy': np.median(times_numpy)}

    if TORCH_AVAILABLE:
        H_torch = torch.from_numpy(H)
        W_torch = torch.from_numpy(W)
        b_torch = torch.from_numpy(b)

        # CPU
        times_cpu = []
        _ = torch.tanh(H_torch @ W_torch + b_torch)
        for _ in range(n_trials):
            start = time.perf_counter()
            result = torch.tanh(H_torch @ W_torch + b_torch)
            times_cpu.append(time.perf_counter() - start)
        results['torch_cpu'] = np.median(times_cpu)

        # GPU
        if CUDA_AVAILABLE:
            H_gpu = H_torch.cuda()
            W_gpu = W_torch.cuda()
            b_gpu = b_torch.cuda()
            _ = torch.tanh(H_gpu @ W_gpu + b_gpu)
            torch.cuda.synchronize()

            times_gpu = []
            for _ in range(n_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = torch.tanh(H_gpu @ W_gpu + b_gpu)
                torch.cuda.synchronize()
                times_gpu.append(time.perf_counter() - start)
            results['torch_gpu'] = np.median(times_gpu)

    return results


def test_lstsq_solvers(A, b, n_trials=5):
    """Compare least-squares solvers."""
    results = {}

    # SciPy Cholesky
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        AtA = A.T @ A + 1e-10 * np.eye(A.shape[1])
        Atb = A.T @ b
        c, low = scipy.linalg.cho_factor(AtA)
        x = scipy.linalg.cho_solve((c, low), Atb)
        times.append(time.perf_counter() - start)
    results['scipy_cholesky'] = np.median(times)

    # NumPy lstsq
    times = []
    for _ in range(n_trials):
        start = time.perf_counter()
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        times.append(time.perf_counter() - start)
    results['numpy_lstsq'] = np.median(times)

    if TORCH_AVAILABLE:
        A_torch = torch.from_numpy(A)
        b_torch = torch.from_numpy(b)

        # PyTorch lstsq CPU
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            x = torch.linalg.lstsq(A_torch, b_torch.unsqueeze(1)).solution.squeeze()
            times.append(time.perf_counter() - start)
        results['torch_lstsq_cpu'] = np.median(times)

        # PyTorch Cholesky CPU
        times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            AtA = A_torch.T @ A_torch + 1e-10 * torch.eye(A_torch.shape[1], dtype=A_torch.dtype)
            Atb = A_torch.T @ b_torch
            L = torch.linalg.cholesky(AtA)
            x = torch.cholesky_solve(Atb.unsqueeze(1), L).squeeze()
            times.append(time.perf_counter() - start)
        results['torch_cholesky_cpu'] = np.median(times)

        if CUDA_AVAILABLE:
            A_gpu = A_torch.cuda()
            b_gpu = b_torch.cuda()

            # Warmup
            _ = torch.linalg.lstsq(A_gpu, b_gpu.unsqueeze(1)).solution.squeeze()
            torch.cuda.synchronize()

            times = []
            for _ in range(n_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                x = torch.linalg.lstsq(A_gpu, b_gpu.unsqueeze(1)).solution.squeeze()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            results['torch_lstsq_gpu'] = np.median(times)

            # PyTorch Cholesky GPU
            times = []
            for _ in range(n_trials):
                torch.cuda.synchronize()
                start = time.perf_counter()
                AtA = A_gpu.T @ A_gpu + 1e-10 * torch.eye(A_gpu.shape[1], dtype=A_gpu.dtype, device='cuda')
                Atb = A_gpu.T @ b_gpu
                L = torch.linalg.cholesky(AtA)
                x = torch.cholesky_solve(Atb.unsqueeze(1), L).squeeze()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            results['torch_cholesky_gpu'] = np.median(times)

    return results


def main():
    print("=" * 70)
    print("GPU ACCELERATION TEST FOR SPECTO-ELM")
    print("=" * 70)
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"CUDA available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load task data
    task_name = "spectral-poisson-square"
    task_cls = TaskRegistry.get(task_name)
    task = task_cls()
    data = task.data
    X = data.X_full
    L_csr = data.L

    # Create test matrices with different sizes
    configs = [
        {'hidden_sizes': [100], 'M': 100},
        {'hidden_sizes': [100, 100, 100, 100], 'M': 400},
    ]

    for config in configs:
        M = config['M']
        print(f"\n{'='*70}")
        print(f"Testing with M={M} features ({config['hidden_sizes']})")
        print(f"{'='*70}")

        # Generate random H matrix
        np.random.seed(42)
        H = np.random.randn(X.shape[0], M)

        # Test sparse × dense multiplication
        print(f"\n1. Sparse × Dense Multiply (L @ H)")
        print(f"   L shape: {L_csr.shape}, H shape: {H.shape}")

        t_scipy, _ = scipy_sparse_dense_multiply(L_csr, H)
        print(f"   SciPy CSR @ dense:     {t_scipy*1000:>8.2f} ms")

        t_torch_cpu, _ = torch_sparse_dense_multiply_cpu(L_csr, H)
        if t_torch_cpu:
            print(f"   PyTorch sparse @ dense (CPU): {t_torch_cpu*1000:>8.2f} ms ({t_scipy/t_torch_cpu:.1f}x speedup)")

        t_torch_gpu, _ = torch_sparse_dense_multiply_gpu(L_csr, H)
        if t_torch_gpu:
            print(f"   PyTorch sparse @ dense (GPU): {t_torch_gpu*1000:>8.2f} ms ({t_scipy/t_torch_gpu:.1f}x speedup)")

        # Test dense matmul
        print(f"\n2. Dense Matmul (hidden layer)")
        dense_results = test_dense_matmul(H)
        print(f"   NumPy H @ W + b:      {dense_results['numpy']*1000:>8.2f} ms")
        if 'torch_cpu' in dense_results:
            print(f"   PyTorch CPU:          {dense_results['torch_cpu']*1000:>8.2f} ms")
        if 'torch_gpu' in dense_results:
            print(f"   PyTorch GPU:          {dense_results['torch_gpu']*1000:>8.2f} ms ({dense_results['numpy']/dense_results['torch_gpu']:.1f}x speedup)")

        # Test least-squares
        print(f"\n3. Least-Squares Solve")
        # Create A matrix (simulated system)
        LH = L_csr @ H
        N_ib = data.N_ib
        A = LH[:N_ib, :]
        b = np.random.randn(A.shape[0])
        print(f"   A shape: {A.shape}")

        lstsq_results = test_lstsq_solvers(A, b)
        baseline = lstsq_results['scipy_cholesky']
        for name, t in lstsq_results.items():
            speedup = baseline / t if t > 0 else 0
            print(f"   {name:<25}: {t*1000:>8.2f} ms ({speedup:.1f}x vs scipy_cholesky)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not CUDA_AVAILABLE:
        print("No GPU available. On HPC with GPU, expect:")
        print("  - Sparse × Dense: 5-20x speedup")
        print("  - Dense Matmul:   10-50x speedup")
        print("  - Cholesky:       2-5x speedup")
        print("\nRecommendation: Use PyTorch with GPU for all operations")
    else:
        print("GPU acceleration available!")


if __name__ == "__main__":
    main()
