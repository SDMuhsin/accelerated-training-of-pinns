"""
H27: DT-ELM-PINN (Discrete-Trained Extreme Learning Machine PINN)

HYPOTHESIS: Combine DT-PINN's precomputed RBF-FD operators with ELM's
single-shot linear solve training paradigm.

KEY INNOVATION:
- Traditional PIELM uses autodiff to form the linear system
- DT-ELM-PINN uses PRECOMPUTED sparse matrices (L, B) for derivatives
- This eliminates gradient computation entirely

STRUCTURE:
Network: u(x) = W_out @ tanh(W_in @ x + b_in)
- W_in, b_in: Random, FIXED
- W_out: Trainable via linear solve (not gradient descent)

For nonlinear PDE (∇²u = f + exp(u)):
Use Newton iteration:
1. Linearize: ∇²u^{k+1} = f + exp(u^k) + exp(u^k)*(u^{k+1} - u^k)
2. Rearrange: (L - diag(exp(u^k))) @ u^{k+1} = f + exp(u^k) - exp(u^k)*u^k
3. Solve for u^{k+1} (linear in u)

NOVELTY CHECK:
- PIELM exists (2019) but uses autodiff for derivatives
- DT-PINN exists but uses iterative training
- Combining precomputed operators with ELM training is NOVEL

Target: Time < 100s, L2 ≤ 3.0e-02
"""

import json
import os
import sys
import torch
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix as scipy_csr
from scipy.sparse.linalg import spsolve, lsqr
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(42)
np.random.seed(42)

PRECISION = np.float64


class DTELMPINN:
    """Discrete-Trained ELM PINN solver"""

    def __init__(self, X, L, B, f, g, ib_idx, n_hidden=100, activation='tanh'):
        """
        X: (N_total, 2) collocation points
        L: (N_ib, N_total) sparse Laplacian operator
        B: (N_bc, N_total) sparse boundary operator
        f: (N_ib, 1) source term
        g: (N_bc, 1) boundary values
        ib_idx: number of interior+boundary points
        """
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]
        self.n_hidden = n_hidden

        # Store operators (scipy sparse)
        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        # Initialize random hidden layer (FIXED, not trainable)
        self.W_in = np.random.randn(2, n_hidden).astype(PRECISION) * np.sqrt(2.0 / 2)
        self.b_in = np.random.randn(n_hidden).astype(PRECISION) * 0.1

        # Activation function
        if activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)

        # Compute hidden layer output (fixed)
        # H = activation(X @ W_in + b_in), shape: (N_total, n_hidden)
        self.H = self.activation(X @ self.W_in + self.b_in)

        # Output weights (trainable)
        self.W_out = np.zeros(n_hidden, dtype=PRECISION)

    def _get_u(self):
        """Compute u = H @ W_out"""
        return self.H @ self.W_out

    def _get_u_ib(self):
        """Get u at interior+boundary points"""
        return (self.H @ self.W_out)[:self.N_ib]

    def solve_linear_pde(self):
        """
        Solve linear PDE: L @ u = f with B @ u = g

        System:
        [L_ib @ H] @ W_out = f  (PDE at interior+boundary)
        [B @ H] @ W_out = g     (boundary conditions)

        Stack into: A @ W_out = b
        """
        # L @ H (only need first N_ib rows of result)
        LH = self.L @ self.H  # (N_ib, n_hidden)

        # B @ H
        BH = self.B @ self.H  # (N_bc, n_hidden)

        # Stack
        A = np.vstack([LH, BH])  # (N_ib + N_bc, n_hidden)
        b = np.concatenate([self.f, self.g])  # (N_ib + N_bc,)

        # Solve least squares: min ||A @ W_out - b||^2
        W_out, *_ = np.linalg.lstsq(A, b, rcond=None)
        self.W_out = W_out

        return self._get_u()

    def solve_nonlinear_newton(self, max_iter=20, tol=1e-8, damping=1.0):
        """
        Solve nonlinear PDE: L @ u - f - exp(u) = 0

        Newton iteration:
        F(u) = L @ u - f - exp(u) = 0
        J(u) = L - diag(exp(u))
        u^{k+1} = u^k - J^{-1} @ F(u^k)

        In ELM form, u = H @ W_out, so:
        J @ H @ delta_W = -F
        delta_W = -(J @ H)^{-1} @ F
        """
        # L @ H gives (2395, n_hidden) but we only need first N_ib rows for PDE
        LH_full = self.L @ self.H  # (2395, n_hidden)
        LH = LH_full[:self.N_ib, :]  # (N_ib, n_hidden) - only interior+boundary rows
        BH = self.B @ self.H  # (N_bc, n_hidden)

        # Initialize with linear solution (ignoring exp(u))
        # Solve L @ u = f + 1 (rough estimate with exp(0)=1)
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])
        self.W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = self._get_u()
        u_ib = u[:self.N_ib]

        residual_history = []

        for k in range(max_iter):
            # Compute residual F = L @ u - f - exp(u) at interior+boundary
            Lu_full = self.L @ u  # (2395,)
            Lu = Lu_full[:self.N_ib]  # Only first N_ib rows
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u  # (N_ib,)

            # Boundary residual
            Bu = self.B @ u
            F_bc = Bu - self.g  # (N_bc,)

            # Full residual
            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            residual_history.append(residual)

            if residual < tol:
                print(f"  Converged at iteration {k+1} with residual {residual:.4e}")
                break

            # Jacobian J = L - diag(exp(u)) (only modifies rows corresponding to ib)
            # For the system, we need: (J @ H) @ delta_W = -F
            # where delta_W = W_out^{k+1} - W_out^k

            # Construct modified operator: L - diag(exp(u)) acting on H
            # J @ H = L @ H - diag(exp(u)) @ H_ib
            # Note: LH is already sliced to (N_ib, n_hidden)
            H_ib = self.H[:self.N_ib, :]  # Hidden layer at interior+boundary points
            JH = LH - exp_u[:, np.newaxis] * H_ib  # (N_ib, n_hidden)

            # Stack PDE and BC
            A = np.vstack([JH, BH])  # (N_ib + N_bc, n_hidden)
            F = np.concatenate([-F_pde, -F_bc])  # Negative because we solve J @ delta = -F

            # Solve for delta_W
            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)

            # Update
            self.W_out = self.W_out + damping * delta_W

            # Recompute u
            u = self._get_u()
            u_ib = u[:self.N_ib]

            if (k + 1) % 5 == 0:
                print(f"  Iteration {k+1}: residual = {residual:.4e}")

        return u, residual_history

    def compute_l2_error(self, u_true):
        """Compute relative L2 error"""
        u_pred = self._get_u_ib()
        u_true_flat = u_true.flatten()[:self.N_ib]
        diff = u_pred - u_true_flat
        return np.linalg.norm(diff) / np.linalg.norm(u_true_flat)


def run_experiment(n_hidden=100, max_iter=20, damping=1.0):
    """Run DT-ELM-PINN experiment"""

    data_path = "/workspace/dt-pinn/nonlinear"
    file_name = "2_2236"

    print(f"\nLoading data: {file_name}")

    # Load data
    X_i = loadmat(f"{data_path}/files_{file_name}/Xi.mat")["Xi"].astype(PRECISION)
    X_b = loadmat(f"{data_path}/files_{file_name}/Xb.mat")["Xb"].astype(PRECISION)
    X_g = loadmat(f"{data_path}/files_{file_name}/Xg.mat")["X_g"].astype(PRECISION)
    u_true = loadmat(f"{data_path}/files_{file_name}/u.mat")["u"].astype(PRECISION)
    f = loadmat(f"{data_path}/files_{file_name}/f.mat")["f"].astype(PRECISION)
    g = loadmat(f"{data_path}/files_{file_name}/g.mat")["g"].astype(PRECISION)
    L = scipy_csr(loadmat(f"{data_path}/files_{file_name}/L1.mat")["L1"], dtype=PRECISION)
    B = scipy_csr(loadmat(f"{data_path}/files_{file_name}/B1.mat")["B1"], dtype=PRECISION)

    # Combine points
    X_full = np.vstack([X_i, X_b, X_g])
    ib_idx = X_i.shape[0] + X_b.shape[0]

    print(f"Points: {X_full.shape[0]} total, {ib_idx} interior+boundary")
    print(f"L shape: {L.shape}, B shape: {B.shape}")
    print(f"Hidden neurons: {n_hidden}")

    # Create solver
    solver = DTELMPINN(
        X=X_full,
        L=L,
        B=B,
        f=f[:ib_idx],  # f is only defined at interior+boundary
        g=g,
        ib_idx=ib_idx,
        n_hidden=n_hidden,
        activation='tanh'
    )

    # Time the solve
    start = time.perf_counter()

    print("\nSolving nonlinear PDE with Newton iteration...")
    u_pred, residuals = solver.solve_nonlinear_newton(max_iter=max_iter, damping=damping)

    solve_time = time.perf_counter() - start

    # Compute L2 error
    l2_error = solver.compute_l2_error(u_true)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Time: {solve_time:.2f}s")
    print(f"  L2 error: {l2_error:.4e}")
    print(f"  Newton iterations: {len(residuals)}")
    print(f"  Final residual: {residuals[-1]:.4e}")
    print(f"\nTarget: Time < 100s, L2 ≤ 3.0e-02")
    print(f"{'='*60}")

    return {
        'time': solve_time,
        'l2_error': l2_error,
        'residuals': residuals,
        'n_hidden': n_hidden,
        'max_iter': max_iter,
    }


if __name__ == "__main__":
    print("="*70)
    print("H27: DT-ELM-PINN (Discrete-Trained Extreme Learning Machine PINN)")
    print("="*70)

    # Test with different hidden layer sizes
    for n_hidden in [50, 100, 200, 500]:
        print(f"\n{'='*60}")
        print(f"Testing with {n_hidden} hidden neurons")
        print(f"{'='*60}")
        results = run_experiment(n_hidden=n_hidden, max_iter=30, damping=1.0)

        # Save results
        os.makedirs("/workspace/dt-pinn/results/dt_elm_pinn", exist_ok=True)
        with open(f"/workspace/dt-pinn/results/dt_elm_pinn/results_h{n_hidden}.json", "w") as f_out:
            json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist()
                      for k, v in results.items()}, f_out, indent=2)
