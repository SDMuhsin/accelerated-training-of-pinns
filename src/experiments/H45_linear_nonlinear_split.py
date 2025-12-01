"""
H45: Linear/Nonlinear Problem Decomposition

HYPOTHESIS: Decompose the nonlinear Poisson into:
  ∇²u = f + exp(u)

Split as:
  u = u_L + u_corr

Where:
  ∇²u_L = f + c  (linear, solve exactly via lstsq)
  u_corr corrects for the nonlinearity exp(u) - c

KEY INSIGHT: If we solve the linear part exactly, the ELM only needs
to learn a smaller correction term, potentially improving accuracy.

Target: L2 < 6.7e-03 (break the floor)
"""

import json
import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix as scipy_csr
from scipy.sparse.linalg import spsolve
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

PRECISION = np.float64


class LinearNonlinearSplitELM:
    """Solve by decomposing into linear + nonlinear correction"""

    def __init__(self, X, L, B, f, g, ib_idx, n_hidden=100):
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]

        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        self.n_hidden = n_hidden
        self._build_elm_features()

    def _build_elm_features(self):
        """Build ELM hidden layer"""
        np.random.seed(42)
        W_in = np.random.randn(self.X.shape[1], self.n_hidden).astype(PRECISION) * np.sqrt(2.0 / self.X.shape[1])
        b_in = np.random.randn(self.n_hidden).astype(PRECISION) * 0.1
        self.H = np.tanh(self.X @ W_in + b_in)

    def solve_linear_poisson(self, c=1.0):
        """
        Solve the linear Poisson: ∇²u_L = f + c with BCs u_L = g
        Uses ELM direct solve (no Newton needed for linear problem)
        """
        LH = (self.L @ self.H)[:self.N_ib, :]
        BH = self.B @ self.H

        A = np.vstack([LH, BH])
        b = np.concatenate([self.f + c, self.g])

        W_out, *_ = np.linalg.lstsq(A, b, rcond=None)
        u_L = self.H @ W_out

        return u_L, W_out

    def solve_with_correction(self, max_iter=20, tol=1e-10):
        """
        Iterative correction approach:
        1. Solve linear Poisson with constant approximation to exp(u)
        2. Compute correction needed for actual exp(u)
        3. Iterate until convergence
        """
        # Initial linear solve with exp(0) = 1
        u, W_out = self.solve_linear_poisson(c=1.0)
        u_ib = u[:self.N_ib]

        LH = (self.L @ self.H)[:self.N_ib, :]
        BH = self.B @ self.H

        for k in range(max_iter):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if k % 5 == 0:
                print(f"    Iter {k}: residual = {residual:.4e}")

            if residual < tol:
                print(f"    Converged at iteration {k+1}")
                break

            # Newton step on the full nonlinear problem
            H_ib = self.H[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            W_out = W_out + delta_W

            u = self.H @ W_out
            u_ib = u[:self.N_ib]

        return u

    def solve_defect_correction(self, max_outer=5, max_inner=10):
        """
        Defect correction approach:
        1. Solve linear problem exactly: ∇²u_L = f + exp(u_old)
        2. Use u_L as new estimate
        3. Repeat

        This separates the linear solve from the nonlinearity.
        """
        # Start with u = 0
        u = np.zeros(self.N_total, dtype=PRECISION)
        u_ib = u[:self.N_ib]

        LH = (self.L @ self.H)[:self.N_ib, :]
        BH = self.B @ self.H

        for outer in range(max_outer):
            # Current nonlinear term
            exp_u = np.exp(u_ib)

            # Solve LINEAR problem: ∇²u_new = f + exp(u_old)
            A = np.vstack([LH, BH])
            b = np.concatenate([self.f + exp_u, self.g])

            W_out, *_ = np.linalg.lstsq(A, b, rcond=None)
            u_new = self.H @ W_out
            u_new_ib = u_new[:self.N_ib]

            # Check convergence
            Lu = (self.L @ u_new)[:self.N_ib]
            exp_u_new = np.exp(u_new_ib)
            F_pde = Lu - self.f - exp_u_new
            F_bc = self.B @ u_new - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            print(f"    Outer iter {outer}: residual = {residual:.4e}")

            # Update
            u = u_new
            u_ib = u_new_ib

        # Final Newton polish
        print("    Final Newton polish...")
        for k in range(max_inner):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))

            if residual < 1e-10:
                break

            H_ib = self.H[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            W_out = W_out + delta_W

            u = self.H @ W_out
            u_ib = u[:self.N_ib]

        return u

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


class TwoNetworkApproach:
    """
    Use TWO separate ELM networks:
    1. Network 1: Solves linear Poisson (larger, more accurate)
    2. Network 2: Learns the nonlinear correction (smaller, flexible)

    u = u_linear + u_correction
    """

    def __init__(self, X, L, B, f, g, ib_idx, n_hidden_linear=200, n_hidden_corr=50):
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]

        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        # Build two separate networks
        self._build_networks(n_hidden_linear, n_hidden_corr)

    def _build_networks(self, n_linear, n_corr):
        """Build two ELM networks"""
        # Network 1: Linear solver (larger capacity)
        np.random.seed(42)
        W1 = np.random.randn(self.X.shape[1], n_linear).astype(PRECISION) * np.sqrt(2.0 / self.X.shape[1])
        b1 = np.random.randn(n_linear).astype(PRECISION) * 0.1
        self.H_linear = np.tanh(self.X @ W1 + b1)

        # Network 2: Correction (smaller, different features)
        np.random.seed(123)
        W2 = np.random.randn(self.X.shape[1], n_corr).astype(PRECISION) * np.sqrt(2.0 / self.X.shape[1])
        b2 = np.random.randn(n_corr).astype(PRECISION) * 0.1
        self.H_corr = np.tanh(self.X @ W2 + b2)

        self.n_linear = n_linear
        self.n_corr = n_corr

    def solve(self, max_iter=20, tol=1e-10):
        """
        Joint optimization of both networks.
        Combined features H = [H_linear, H_corr]
        """
        H = np.hstack([self.H_linear, self.H_corr])

        LH_full = self.L @ H
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ H

        # Initial solve
        A = np.vstack([LH, BH])
        b = np.concatenate([self.f + 1.0, self.g])

        W_out, *_ = np.linalg.lstsq(A, b, rcond=None)
        u = H @ W_out
        u_ib = u[:self.N_ib]

        # Newton iterations
        for k in range(max_iter):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))

            if residual < tol:
                print(f"    Converged at iteration {k+1}")
                break

            H_ib = H[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            W_out = W_out + delta_W

            u = H @ W_out
            u_ib = u[:self.N_ib]

        return u

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment():
    """Test linear/nonlinear splitting approaches"""

    data_path = "/workspace/dt-pinn/nonlinear"
    file_name = "2_2236"

    print(f"Loading data: {file_name}")

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

    results = []

    # Baseline: Standard Newton ELM
    print("\n" + "="*60)
    print("Baseline: Standard Newton ELM (100 hidden)")
    print("="*60)

    start = time.perf_counter()
    solver = LinearNonlinearSplitELM(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx, n_hidden=100
    )
    u_pred = solver.solve_with_correction(max_iter=20)
    baseline_time = time.perf_counter() - start
    baseline_error = solver.compute_l2_error(u_pred, u_true)
    print(f"  Time: {baseline_time:.2f}s, L2 Error: {baseline_error:.4e}")
    results.append({'method': 'baseline', 'l2': baseline_error, 'time': baseline_time})

    # Method 1: Defect correction
    print("\n" + "="*60)
    print("Method 1: Defect Correction (100 hidden)")
    print("="*60)

    start = time.perf_counter()
    solver = LinearNonlinearSplitELM(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx, n_hidden=100
    )
    u_pred = solver.solve_defect_correction(max_outer=5, max_inner=10)
    defect_time = time.perf_counter() - start
    defect_error = solver.compute_l2_error(u_pred, u_true)
    print(f"  Time: {defect_time:.2f}s, L2 Error: {defect_error:.4e}")
    results.append({'method': 'defect_correction', 'l2': defect_error, 'time': defect_time})

    # Method 2: Two-network approach (various configurations)
    print("\n" + "="*60)
    print("Method 2: Two-Network Approach")
    print("="*60)

    configs = [
        (100, 50),   # 100 linear + 50 correction
        (150, 50),   # 150 linear + 50 correction
        (100, 100),  # 100 linear + 100 correction
        (200, 50),   # 200 linear + 50 correction
    ]

    for n_lin, n_corr in configs:
        print(f"\n  Config: {n_lin} linear + {n_corr} correction")
        start = time.perf_counter()
        solver = TwoNetworkApproach(
            X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx,
            n_hidden_linear=n_lin, n_hidden_corr=n_corr
        )
        u_pred = solver.solve(max_iter=20)
        solve_time = time.perf_counter() - start
        error = solver.compute_l2_error(u_pred, u_true)
        print(f"    Time: {solve_time:.2f}s, L2 Error: {error:.4e}")
        results.append({
            'method': f'two_network_{n_lin}_{n_corr}',
            'l2': error,
            'time': solve_time
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Linear/Nonlinear Split Methods")
    print("="*70)
    print(f"{'Method':<30} | {'L2 Error':>12} | {'Time':>8} | {'vs Baseline':>12}")
    print("-"*70)

    for r in results:
        vs_baseline = "BETTER" if r['l2'] < baseline_error else "SAME/WORSE"
        print(f"{r['method']:<30} | {r['l2']:>12.4e} | {r['time']:>7.2f}s | {vs_baseline:>12}")

    print(f"\nTarget: L2 < 6.7e-03 (break the floor)")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/linear_nonlinear_split", exist_ok=True)
    with open("/workspace/dt-pinn/results/linear_nonlinear_split/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)

    return results


if __name__ == "__main__":
    print("="*70)
    print("H45: Linear/Nonlinear Problem Decomposition")
    print("="*70)
    run_experiment()
