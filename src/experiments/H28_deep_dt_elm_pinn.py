"""
H28: Deep-DT-ELM-PINN (Multi-Layer Random Features)

HYPOTHESIS: Extend DT-ELM-PINN to multiple layers by fixing ALL hidden layers
(random initialization) and only solving for the final output weights.

STRUCTURE:
u(x) = W_out @ h_L(x)
where h_L = tanh(W_{L-1} @ h_{L-1} + b_{L-1})
      h_1 = tanh(W_0 @ x + b_0)
All W_0, ..., W_{L-1}, b_0, ..., b_{L-1} are FIXED (random).
Only W_out is solved via least squares + Newton iteration.

RESEARCH QUESTIONS:
1. Does deep random feature extraction maintain accuracy?
2. What is the optimal depth/width tradeoff?
3. Does deeper = better or worse for this problem?

Target: Time < 30s, L2 ≤ 6.5e-03
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


class DeepDTELMPINN:
    """Deep Discrete-Trained ELM PINN solver with multiple random hidden layers"""

    def __init__(self, X, L, B, f, g, ib_idx, layer_sizes, activation='tanh'):
        """
        X: (N_total, 2) collocation points
        L: (N_ib, N_total) sparse Laplacian operator
        B: (N_bc, N_total) sparse boundary operator
        f: (N_ib, 1) source term
        g: (N_bc, 1) boundary values
        ib_idx: number of interior+boundary points
        layer_sizes: list of hidden layer widths, e.g., [100] for 1-layer, [50, 50] for 2-layer
        """
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Store operators (scipy sparse)
        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        # Activation function
        if activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + np.exp(-x))
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)

        # Initialize random hidden layers (ALL FIXED, not trainable)
        self.weights = []
        self.biases = []

        input_dim = 2  # x, y coordinates
        for i, hidden_dim in enumerate(layer_sizes):
            # Xavier initialization
            W = np.random.randn(input_dim, hidden_dim).astype(PRECISION) * np.sqrt(2.0 / input_dim)
            b = np.random.randn(hidden_dim).astype(PRECISION) * 0.1
            self.weights.append(W)
            self.biases.append(b)
            input_dim = hidden_dim

        # Compute hidden layer output (fixed) by forward pass through all layers
        self.H = self._compute_hidden_output(X)

        # Output weights (trainable via least squares)
        self.W_out = np.zeros(layer_sizes[-1], dtype=PRECISION)

    def _compute_hidden_output(self, X):
        """Forward pass through all fixed hidden layers"""
        h = X
        for W, b in zip(self.weights, self.biases):
            h = self.activation(h @ W + b)
        return h  # Shape: (N_total, layer_sizes[-1])

    def _get_u(self):
        """Compute u = H @ W_out"""
        return self.H @ self.W_out

    def _get_u_ib(self):
        """Get u at interior+boundary points"""
        return (self.H @ self.W_out)[:self.N_ib]

    def solve_nonlinear_newton(self, max_iter=20, tol=1e-8, damping=1.0):
        """
        Solve nonlinear PDE: L @ u - f - exp(u) = 0 using Newton iteration
        """
        # L @ H gives (2395, n_hidden) but we only need first N_ib rows for PDE
        LH_full = self.L @ self.H
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ self.H

        # Initialize with linear solution (ignoring exp(u))
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])
        self.W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = self._get_u()
        u_ib = u[:self.N_ib]

        residual_history = []

        for k in range(max_iter):
            # Compute residual F = L @ u - f - exp(u) at interior+boundary
            Lu_full = self.L @ u
            Lu = Lu_full[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u

            # Boundary residual
            Bu = self.B @ u
            F_bc = Bu - self.g

            # Full residual
            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            residual_history.append(residual)

            if residual < tol:
                print(f"  Converged at iteration {k+1} with residual {residual:.4e}")
                break

            # Jacobian: J @ H = L @ H - diag(exp(u)) @ H_ib
            H_ib = self.H[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            # Stack PDE and BC
            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

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


def run_experiment(layer_sizes, max_iter=20, damping=1.0):
    """Run Deep-DT-ELM-PINN experiment"""

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
    print(f"Layer sizes: {layer_sizes} ({len(layer_sizes)} hidden layers)")

    # Create solver
    solver = DeepDTELMPINN(
        X=X_full,
        L=L,
        B=B,
        f=f[:ib_idx],
        g=g,
        ib_idx=ib_idx,
        layer_sizes=layer_sizes,
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
    print(f"  Layers: {layer_sizes}")
    print(f"  Time: {solve_time:.2f}s")
    print(f"  L2 error: {l2_error:.4e}")
    print(f"  Newton iterations: {len(residuals)}")
    print(f"  Final residual: {residuals[-1]:.4e}")
    print(f"\nTarget: Time < 30s, L2 ≤ 6.5e-03")
    print(f"{'='*60}")

    return {
        'time': solve_time,
        'l2_error': l2_error,
        'residuals': residuals,
        'layer_sizes': layer_sizes,
        'n_layers': len(layer_sizes),
        'max_iter': max_iter,
    }


if __name__ == "__main__":
    print("="*70)
    print("H28: Deep-DT-ELM-PINN (Multi-Layer Random Features)")
    print("="*70)

    # Compare different architectures
    # Baseline: 1-layer with 100 neurons (from H27)
    # Test: 2-layer, 3-layer, 4-layer with same total neurons

    architectures = [
        [100],           # 1-layer, 100 neurons (baseline)
        [50, 50],        # 2-layer, 50+50 neurons
        [100, 100],      # 2-layer, 100+100 neurons
        [50, 50, 50],    # 3-layer, 50+50+50 neurons
        [100, 100, 100], # 3-layer, 100+100+100 neurons
        [50, 50, 50, 50],  # 4-layer
    ]

    all_results = []

    for layer_sizes in architectures:
        print(f"\n{'='*60}")
        print(f"Testing architecture: {layer_sizes}")
        print(f"{'='*60}")

        results = run_experiment(layer_sizes=layer_sizes, max_iter=30, damping=1.0)
        all_results.append(results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Deep-DT-ELM-PINN Architecture Comparison")
    print("="*70)
    print(f"{'Architecture':<25} | {'Time':>8} | {'L2 Error':>12} | {'Status':<10}")
    print("-"*70)

    for r in all_results:
        layers_str = str(r['layer_sizes'])
        status = "PASS" if r['l2_error'] <= 6.5e-03 else "FAIL"
        print(f"{layers_str:<25} | {r['time']:>7.2f}s | {r['l2_error']:>12.4e} | {status:<10}")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/deep_dt_elm_pinn", exist_ok=True)
    with open("/workspace/dt-pinn/results/deep_dt_elm_pinn/results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
