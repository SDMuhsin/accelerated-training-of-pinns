"""
H35: Layer-wise Greedy ELM Training

HYPOTHESIS: Train each layer sequentially using a greedy approach:
1. Train layer 1 using ELM
2. Freeze layer 1, train layer 2 using ELM on the residual
3. Continue until all layers are trained

This is different from H31 (residual stacking) because here we're training
CONSECUTIVE layers in a deep network, not parallel ELM heads.

KEY INSIGHT:
- Problem with H28-H32: Random features in deep layers compound the limitation
- Solution: Train each layer optimally before adding the next

STRUCTURE:
Layer 1: x -> h1 = tanh(W1 @ x + b1), ELM-solve W1 to minimize ||L @ (W_out @ h1) - f - exp(W_out @ h1)||
Layer 2: h1 -> h2 = tanh(W2 @ h1 + b2), fix h1, ELM-solve W2 on residual
...and so on

Target: L2 ≤ 6.5e-03 with ≥2 hidden layers
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


class LayerwiseELM:
    """Layer-wise greedy ELM training"""

    def __init__(self, X, L, B, f, g, ib_idx, hidden_sizes):
        """
        hidden_sizes: list of hidden layer sizes, e.g., [100, 100] for 2 hidden layers
        """
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]

        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        self.hidden_sizes = hidden_sizes
        self.n_layers = len(hidden_sizes)

        # Store layer weights and biases
        self.weights = []  # Input weights for each layer
        self.biases = []   # Biases for each layer
        self.H_layers = [] # Hidden activations for each layer

        # Output weights (solved last)
        self.W_out = None

    def train_first_layer(self, n_hidden, max_iter=20, tol=1e-8):
        """Train the first hidden layer using ELM + Newton iteration"""
        print(f"  Training layer 1: ({self.X.shape[1]} -> {n_hidden})")

        # Random input weights
        np.random.seed(42)
        W1 = np.random.randn(self.X.shape[1], n_hidden).astype(PRECISION) * np.sqrt(2.0 / self.X.shape[1])
        b1 = np.random.randn(n_hidden).astype(PRECISION) * 0.1

        # Compute hidden activations
        H = np.tanh(self.X @ W1 + b1)

        # ELM solve with Newton iteration
        LH_full = self.L @ H
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ H

        # Initialize output weights
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])
        W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = H @ W_out
        u_ib = u[:self.N_ib]

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

        # Store layer
        self.weights.append(W1)
        self.biases.append(b1)
        self.H_layers.append(H)
        self.W_out = W_out

        return u, residual

    def add_layer_and_refine(self, n_hidden, max_iter=20, tol=1e-8):
        """Add a new hidden layer and refine the solution"""
        layer_idx = len(self.weights)
        prev_H = self.H_layers[-1]

        print(f"  Training layer {layer_idx + 1}: ({prev_H.shape[1]} -> {n_hidden})")

        # Random weights for new layer
        np.random.seed(42 + layer_idx * 1000)
        W_new = np.random.randn(prev_H.shape[1], n_hidden).astype(PRECISION) * np.sqrt(2.0 / prev_H.shape[1])
        b_new = np.random.randn(n_hidden).astype(PRECISION) * 0.1

        # Compute new hidden activations
        H_new = np.tanh(prev_H @ W_new + b_new)

        # ELM solve with Newton iteration
        LH_full = self.L @ H_new
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ H_new

        # Initialize output weights
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])
        W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = H_new @ W_out
        u_ib = u[:self.N_ib]

        for k in range(max_iter):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if residual < tol:
                print(f"    Converged at iteration {k+1}")
                break

            H_ib = H_new[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            W_out = W_out + delta_W

            u = H_new @ W_out
            u_ib = u[:self.N_ib]

        # Store layer
        self.weights.append(W_new)
        self.biases.append(b_new)
        self.H_layers.append(H_new)
        self.W_out = W_out

        return u, residual

    def solve(self, max_iter=20):
        """Train all layers sequentially"""
        # Train first layer
        u, res = self.train_first_layer(self.hidden_sizes[0], max_iter=max_iter)
        print(f"    Layer 1 residual: {res:.4e}")

        # Add remaining layers
        for i, n_hidden in enumerate(self.hidden_sizes[1:]):
            u, res = self.add_layer_and_refine(n_hidden, max_iter=max_iter)
            print(f"    Layer {i+2} residual: {res:.4e}")

        return u

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment(hidden_sizes, max_iter=20):
    """Run Layer-wise ELM experiment"""

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
    print(f"Architecture: {hidden_sizes} ({len(hidden_sizes)} hidden layers)")

    start = time.perf_counter()

    solver = LayerwiseELM(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx,
        hidden_sizes=hidden_sizes
    )

    u_pred = solver.solve(max_iter=max_iter)

    total_time = time.perf_counter() - start

    l2_error = solver.compute_l2_error(u_pred, u_true)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Architecture: {hidden_sizes}")
    print(f"  Hidden layers: {len(hidden_sizes)}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  L2 error: {l2_error:.4e}")
    print(f"\nTarget: L2 ≤ 6.5e-03 with ≥2 layers")
    print(f"{'='*60}")

    return {
        'time': total_time,
        'l2_error': l2_error,
        'hidden_sizes': hidden_sizes,
        'n_layers': len(hidden_sizes),
    }


if __name__ == "__main__":
    print("="*70)
    print("H35: Layer-wise Greedy ELM Training")
    print("="*70)

    results = []

    # Test different configurations
    configs = [
        [100],           # 1 layer baseline
        [100, 100],      # 2 layers
        [100, 100, 100], # 3 layers
        [50, 50],        # 2 smaller layers
        [150, 150],      # 2 larger layers
        [100, 50],       # Pyramid
        [50, 100],       # Reverse pyramid
    ]

    for hidden_sizes in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {hidden_sizes}")
        print(f"{'='*60}")

        np.random.seed(42)

        r = run_experiment(hidden_sizes=hidden_sizes, max_iter=20)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Layer-wise Greedy ELM")
    print("="*70)
    print(f"{'Config':<25} | {'Time':>10} | {'L2 Error':>12} | {'Status':<10}")
    print("-"*65)

    for r in results:
        config = str(r['hidden_sizes'])
        status = "PASS" if r['l2_error'] <= 6.5e-03 else "FAIL"
        print(f"{config:<25} | {r['time']:>9.2f}s | {r['l2_error']:>12.4e} | {status:<10}")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/layerwise_elm", exist_ok=True)
    with open("/workspace/dt-pinn/results/layerwise_elm/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
