"""
H32: Multi-Head ELM (Parallel Random Feature Ensembles)

HYPOTHESIS: Instead of stacking ELM layers sequentially, use multiple ELM "heads"
in parallel, each with different random features, and solve for output weights jointly.

KEY INSIGHT:
- Single ELM: u = H @ W where H = tanh(X @ W_in + b)
- Multi-Head: u = [H1 | H2 | ... | Hk] @ [W1; W2; ... Wk]
  where each Hi has DIFFERENT random initialization

This is conceptually "wider" not "deeper" - we use multiple random feature
representations of the same input and combine them.

The "multi-layer" aspect comes from:
1. Each head = one "layer" of random features
2. The combination is learned via joint least squares
3. This is like an ensemble of single-layer networks

STRUCTURE:
u(x) = sum_k alpha_k * H_k @ W_k = H_concat @ W_concat
where H_concat = [H_1 | H_2 | ... | H_K] concatenated horizontally

Target: L2 ≤ 6.5e-03 with K >= 2 heads
"""

import json
import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix as scipy_csr
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

PRECISION = np.float64


class MultiHeadELM:
    """Multi-Head ELM with parallel random feature ensembles"""

    def __init__(self, X, L, B, f, g, ib_idx, n_hidden_per_head, n_heads=2, activation='tanh'):
        """
        n_hidden_per_head: number of hidden neurons per head
        n_heads: number of parallel ELM heads (each with different random init)
        """
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]
        self.n_heads = n_heads
        self.n_hidden_per_head = n_hidden_per_head
        self.total_hidden = n_hidden_per_head * n_heads

        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        # Activation function
        if activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'sin':
            self.activation = np.sin
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)

        # Create multiple ELM heads with different random seeds
        self.heads = []
        for k in range(n_heads):
            np.random.seed(42 + k * 1000)  # Different seed per head
            W_in = np.random.randn(2, n_hidden_per_head).astype(PRECISION) * np.sqrt(2.0 / 2)
            b_in = np.random.randn(n_hidden_per_head).astype(PRECISION) * 0.1
            H = self.activation(X @ W_in + b_in)
            self.heads.append({'W_in': W_in, 'b_in': b_in, 'H': H})

        # Concatenate all hidden representations
        self.H_concat = np.hstack([head['H'] for head in self.heads])
        print(f"  Multi-head H shape: {self.H_concat.shape} ({n_heads} heads x {n_hidden_per_head} hidden)")

        # Output weights (trainable via least squares)
        self.W_out = np.zeros(self.total_hidden, dtype=PRECISION)

    def _get_u(self):
        """Compute u = H_concat @ W_out"""
        return self.H_concat @ self.W_out

    def solve_nonlinear_newton(self, max_iter=20, tol=1e-8):
        """Solve nonlinear PDE using Newton iteration with multi-head features"""
        LH_full = self.L @ self.H_concat
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ self.H_concat

        # Initialize with linear solution
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])
        self.W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = self._get_u()
        u_ib = u[:self.N_ib]

        for k in range(max_iter):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if residual < tol:
                print(f"  Converged at iteration {k+1} with residual {residual:.4e}")
                break

            H_ib = self.H_concat[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            self.W_out = self.W_out + delta_W

            u = self._get_u()
            u_ib = u[:self.N_ib]

        return u

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment(n_hidden_per_head=50, n_heads=2, max_iter=20):
    """Run Multi-Head ELM experiment"""

    data_path = "/workspace/dt-pinn/nonlinear"
    file_name = "2_2236"

    print(f"\nLoading data: {file_name}")

    X_i = loadmat(f"{data_path}/files_{file_name}/Xi.mat")["Xi"].astype(PRECISION)
    X_b = loadmat(f"{data_path}/files_{file_name}/Xb.mat")["Xb"].astype(PRECISION)
    X_g = loadmat(f"{data_path}/files_{file_name}/Xg.mat")["X_g"].astype(PRECISION)
    u_true = loadmat(f"{data_path}/files_{file_name}/u.mat")["u"].astype(PRECISION)
    f = loadmat(f"{data_path}/files_{file_name}/f.mat")["f"].astype(PRECISION)
    g = loadmat(f"{data_path}/files_{file_name}/g.mat")["g"].astype(PRECISION)
    L = scipy_csr(loadmat(f"{data_path}/files_{file_name}/L1.mat")["L1"], dtype=PRECISION)
    B = scipy_csr(loadmat(f"{data_path}/files_{file_name}/B1.mat")["B1"], dtype=PRECISION)

    X_full = np.vstack([X_i, X_b, X_g])
    ib_idx = X_i.shape[0] + X_b.shape[0]

    print(f"Points: {X_full.shape[0]} total, {ib_idx} interior+boundary")
    print(f"Architecture: {n_heads} heads x {n_hidden_per_head} hidden = {n_heads * n_hidden_per_head} total features")

    solver = MultiHeadELM(
        X=X_full,
        L=L,
        B=B,
        f=f[:ib_idx],
        g=g,
        ib_idx=ib_idx,
        n_hidden_per_head=n_hidden_per_head,
        n_heads=n_heads,
    )

    start = time.perf_counter()
    u_pred = solver.solve_nonlinear_newton(max_iter=max_iter)
    solve_time = time.perf_counter() - start

    l2_error = solver.compute_l2_error(u_pred, u_true)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Heads: {n_heads} x {n_hidden_per_head} = {n_heads * n_hidden_per_head} features")
    print(f"  Time: {solve_time:.2f}s")
    print(f"  L2 error: {l2_error:.4e}")
    print(f"\nTarget: L2 ≤ 6.5e-03 with ≥2 heads")
    print(f"{'='*60}")

    return {
        'time': solve_time,
        'l2_error': l2_error,
        'n_hidden_per_head': n_hidden_per_head,
        'n_heads': n_heads,
        'total_hidden': n_heads * n_hidden_per_head,
    }


if __name__ == "__main__":
    print("="*70)
    print("H32: Multi-Head ELM (Parallel Random Feature Ensembles)")
    print("="*70)

    results = []

    # Test different configurations
    # Compare: 1 head with N neurons vs K heads with N/K neurons each
    configs = [
        (100, 1),   # Baseline: 1 head x 100 = 100 features
        (50, 2),    # 2 heads x 50 = 100 features
        (50, 3),    # 3 heads x 50 = 150 features
        (50, 4),    # 4 heads x 50 = 200 features
        (100, 2),   # 2 heads x 100 = 200 features
        (100, 3),   # 3 heads x 100 = 300 features
        (75, 2),    # 2 heads x 75 = 150 features
        (75, 3),    # 3 heads x 75 = 225 features
    ]

    for n_hidden, n_heads in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {n_heads} heads x {n_hidden} hidden")
        print(f"{'='*60}")

        r = run_experiment(n_hidden_per_head=n_hidden, n_heads=n_heads, max_iter=20)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Multi-Head ELM")
    print("="*70)
    print(f"{'Config':<25} | {'Time':>10} | {'L2 Error':>12} | {'Status':<10}")
    print("-"*65)

    for r in results:
        config = f"{r['n_heads']}H x {r['n_hidden_per_head']}h = {r['total_hidden']}"
        status = "PASS" if r['l2_error'] <= 6.5e-03 else "FAIL"
        print(f"{config:<25} | {r['time']:>9.2f}s | {r['l2_error']:>12.4e} | {status:<10}")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/multihead_elm", exist_ok=True)
    with open("/workspace/dt-pinn/results/multihead_elm/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2)
