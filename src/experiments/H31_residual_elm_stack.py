"""
H31: Residual ELM Stacking (Multi-Layer via Residual Learning)

HYPOTHESIS: Stack multiple ELM layers where each layer learns to correct
the residual error from the previous layer.

KEY INSIGHT:
- Single ELM: u = u_1 = H_1 @ W_1
- Two-layer residual: u = u_1 + u_2 where u_2 = H_2 @ W_2 learns to correct u_1's error
- Each layer is trained via ELM (direct solve), no gradient descent needed

STRUCTURE:
Layer 1: Solve for u_1 that minimizes ||L @ u_1 - f - exp(u_1)||^2 + BC
Layer 2: Given u_1, solve for δu that minimizes ||L @ (u_1 + δu) - f - exp(u_1 + δu)||^2 + BC
         Linearize: exp(u_1 + δu) ≈ exp(u_1) + exp(u_1) * δu
         So: (L - diag(exp(u_1))) @ δu = f + exp(u_1) - L @ u_1
Layer 3+: Repeat for additional refinement

This is essentially Newton iteration, but with DIFFERENT random features at each step.
The "multi-layer" comes from using different feature spaces for each correction.

Target: L2 ≤ 6.5e-03 with ≥2 ELM layers
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


class ELMLayer:
    """Single ELM layer with random hidden features"""

    def __init__(self, input_dim, n_hidden, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_hidden = n_hidden
        self.W_in = np.random.randn(input_dim, n_hidden).astype(PRECISION) * np.sqrt(2.0 / input_dim)
        self.b_in = np.random.randn(n_hidden).astype(PRECISION) * 0.1
        self.W_out = None

    def compute_hidden(self, X):
        """Compute hidden layer output H = tanh(X @ W_in + b_in)"""
        return np.tanh(X @ self.W_in + self.b_in)


class ResidualELMStack:
    """Stack of ELM layers using residual learning"""

    def __init__(self, X, L, B, f, g, ib_idx, n_hidden_per_layer, n_layers=2):
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]
        self.n_layers = n_layers

        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        # Create ELM layers with different random seeds
        self.elm_layers = []
        for i in range(n_layers):
            layer = ELMLayer(input_dim=2, n_hidden=n_hidden_per_layer, seed=42 + i * 100)
            layer.H = layer.compute_hidden(X)  # Precompute hidden features
            self.elm_layers.append(layer)

    def solve_layer1(self, max_iter=20, tol=1e-8):
        """Solve first ELM layer using Newton iteration"""
        layer = self.elm_layers[0]
        H = layer.H

        LH_full = self.L @ H
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ H

        # Initialize
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])
        layer.W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = H @ layer.W_out
        u_ib = u[:self.N_ib]

        for k in range(max_iter):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if residual < tol:
                break

            H_ib = H[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            layer.W_out = layer.W_out + delta_W

            u = H @ layer.W_out
            u_ib = u[:self.N_ib]

        return u

    def solve_residual_layer(self, layer_idx, u_prev, max_iter=10, tol=1e-8):
        """Solve residual layer: find δu such that u_new = u_prev + δu improves the solution"""
        layer = self.elm_layers[layer_idx]
        H = layer.H

        LH_full = self.L @ H
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ H

        u_prev_ib = u_prev[:self.N_ib]

        # Compute current residual
        Lu_prev = (self.L @ u_prev)[:self.N_ib]
        exp_u_prev = np.exp(u_prev_ib)

        # Initialize δu = 0
        layer.W_out = np.zeros(layer.n_hidden, dtype=PRECISION)

        for k in range(max_iter):
            # Current correction
            delta_u = H @ layer.W_out
            delta_u_ib = delta_u[:self.N_ib]

            # Total solution
            u_total = u_prev + delta_u
            u_total_ib = u_total[:self.N_ib]

            # Residual with current correction
            Lu_total = (self.L @ u_total)[:self.N_ib]
            exp_u_total = np.exp(u_total_ib)
            F_pde = Lu_total - self.f - exp_u_total
            F_bc = self.B @ u_total - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if residual < tol:
                break

            # Jacobian for δu
            H_ib = H[:self.N_ib, :]
            JH = LH - exp_u_total[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            layer.W_out = layer.W_out + delta_W

        return u_prev + H @ layer.W_out

    def solve(self, max_iter_per_layer=20):
        """Solve using all stacked layers"""
        print(f"  Layer 1: Solving base ELM...")
        u = self.solve_layer1(max_iter=max_iter_per_layer)

        # Compute L2 after layer 1
        u_ib = u[:self.N_ib]

        for i in range(1, self.n_layers):
            print(f"  Layer {i+1}: Solving residual correction...")
            u = self.solve_residual_layer(i, u, max_iter=max_iter_per_layer)

        return u

    def compute_l2_error(self, u, u_true):
        u_pred = u[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment(n_hidden=100, n_layers=2, max_iter=20):
    """Run Residual ELM Stack experiment"""

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
    print(f"Architecture: {n_layers} ELM layers x {n_hidden} hidden neurons each")

    # Create solver
    solver = ResidualELMStack(
        X=X_full,
        L=L,
        B=B,
        f=f[:ib_idx],
        g=g,
        ib_idx=ib_idx,
        n_hidden_per_layer=n_hidden,
        n_layers=n_layers
    )

    # Time the solve
    start = time.perf_counter()
    u_pred = solver.solve(max_iter_per_layer=max_iter)
    solve_time = time.perf_counter() - start

    # Compute L2 error
    l2_error = solver.compute_l2_error(u_pred, u_true)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Layers: {n_layers} x {n_hidden} hidden")
    print(f"  Time: {solve_time:.2f}s")
    print(f"  L2 error: {l2_error:.4e}")
    print(f"\nTarget: L2 ≤ 6.5e-03 with ≥2 layers")
    print(f"{'='*60}")

    return {
        'time': solve_time,
        'l2_error': l2_error,
        'n_hidden': n_hidden,
        'n_layers': n_layers,
    }


if __name__ == "__main__":
    print("="*70)
    print("H31: Residual ELM Stacking (Multi-Layer via Residual Learning)")
    print("="*70)

    results = []

    # Test different configurations
    configs = [
        (100, 1),   # Baseline: 1 layer
        (100, 2),   # 2 layers
        (100, 3),   # 3 layers
        (100, 4),   # 4 layers
        (50, 2),    # Smaller layers
        (50, 4),    # More smaller layers
    ]

    for n_hidden, n_layers in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {n_layers} layers x {n_hidden} hidden")
        print(f"{'='*60}")

        r = run_experiment(n_hidden=n_hidden, n_layers=n_layers, max_iter=20)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Residual ELM Stacking")
    print("="*70)
    print(f"{'Config':<20} | {'Time':>10} | {'L2 Error':>12} | {'Status':<10}")
    print("-"*60)

    for r in results:
        config = f"{r['n_layers']}L x {r['n_hidden']}h"
        status = "PASS" if r['l2_error'] <= 6.5e-03 else "FAIL"
        print(f"{config:<20} | {r['time']:>9.2f}s | {r['l2_error']:>12.4e} | {status:<10}")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/residual_elm_stack", exist_ok=True)
    with open("/workspace/dt-pinn/results/residual_elm_stack/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2)
