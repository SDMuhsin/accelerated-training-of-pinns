"""
H34: Alternating ELM Optimization

HYPOTHESIS: Alternate between:
1. ELM solve: Fix hidden layers, solve for output weights via Newton iteration
2. Gradient step: Fix output weights, update hidden layers via gradient descent

This is a principled way to train a multi-layer network while preserving ELM's
direct solve property for the output layer.

KEY INSIGHT:
- Pure ELM: Random features may not be optimal
- Pure gradient: Slow convergence for deep networks
- Alternating: Get benefits of both approaches

STRUCTURE:
For each alternating iteration:
  Step A: Given current hidden weights, ELM-solve for output weights (Newton)
  Step B: Given current output weights, gradient descent on hidden weights

This maintains:
1. DT-PINN: Uses precomputed sparse operators L, B
2. ELM: Direct solve for output layer each iteration
3. Multi-layer: Hidden layers are trained (not random)

Target: L2 ≤ 6.5e-03 with ≥2 hidden layers
"""

import json
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from scipy.io import loadmat
from scipy.sparse import csr_matrix as scipy_csr
import cupy
from cupy.sparse import csr_matrix as cupy_csr
from torch.utils.dlpack import to_dlpack, from_dlpack
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(42)
np.random.seed(42)

PRECISION = np.float64
TORCH_PRECISION = torch.float64
device = 'cuda'


class MultiLayerNetwork(torch.nn.Module):
    """Multi-layer network with separate hidden and output layers"""

    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                torch.nn.Linear(layer_sizes[i], layer_sizes[i+1], dtype=TORCH_PRECISION)
            )
            torch.nn.init.xavier_normal_(self.layers[-1].weight)
            torch.nn.init.zeros_(self.layers[-1].bias)

    def forward_hidden(self, x):
        """Compute hidden layer outputs (all but last layer)"""
        h = x
        for layer in self.layers[:-1]:
            h = torch.tanh(layer(h))
        return h

    def forward(self, x):
        """Full forward pass"""
        h = self.forward_hidden(x)
        return self.layers[-1](h)


class AlternatingELMSolver:
    """Alternating optimization between ELM and gradient descent"""

    def __init__(self, X, L, B, f, g, ib_idx, layer_sizes):
        """
        layer_sizes: e.g., [2, 50, 50, 1] for 2 hidden layers with 50 neurons each
        """
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]

        self.L_scipy = L
        self.B_scipy = B
        self.f_np = f.flatten()
        self.g_np = g.flatten()
        self.X_np = X

        # Convert to torch tensors
        self.X_t = torch.tensor(X, dtype=TORCH_PRECISION, device=device)
        self.f_t = torch.tensor(f, dtype=TORCH_PRECISION, device=device)
        self.g_t = torch.tensor(g, dtype=TORCH_PRECISION, device=device)

        # Convert scipy sparse to cupy sparse
        self.L_cupy = cupy_csr(L.astype(np.float64))
        self.B_cupy = cupy_csr(B.astype(np.float64))
        self.L_t_cupy = cupy_csr(L.T.astype(np.float64))
        self.B_t_cupy = cupy_csr(B.T.astype(np.float64))

        # Create network
        self.layer_sizes = layer_sizes
        self.network = MultiLayerNetwork(layer_sizes).to(device)

        # Number of hidden layers (excluding input and output)
        self.n_hidden_layers = len(layer_sizes) - 2

        print(f"  Network architecture: {layer_sizes}")
        print(f"  Hidden layers: {self.n_hidden_layers}")

    def _sparse_mul_L(self, u):
        """L @ u using cupy sparse"""
        u_cupy = cupy.from_dlpack(to_dlpack(u))
        result = self.L_cupy.dot(u_cupy)
        return from_dlpack(result.toDlpack())

    def _sparse_mul_B(self, u):
        """B @ u using cupy sparse"""
        u_cupy = cupy.from_dlpack(to_dlpack(u))
        result = self.B_cupy.dot(u_cupy)
        return from_dlpack(result.toDlpack())

    def elm_solve_output(self, max_iter=10, tol=1e-8):
        """
        ELM step: Fix hidden layers, solve for output weights via Newton iteration
        """
        with torch.no_grad():
            # Get hidden layer output
            H = self.network.forward_hidden(self.X_t)
            H_np = H.cpu().numpy()

        # Compute L @ H and B @ H using scipy sparse
        LH_full = self.L_scipy @ H_np
        LH = LH_full[:self.N_ib, :]
        BH = self.B_scipy @ H_np

        # Get current output weights
        W_out = self.network.layers[-1].weight.detach().cpu().numpy().flatten()

        # Newton iteration for output weights
        for k in range(max_iter):
            u = H_np @ W_out
            u_ib = u[:self.N_ib]

            Lu = (self.L_scipy @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f_np - exp_u
            F_bc = self.B_scipy @ u - self.g_np

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if residual < tol:
                break

            H_ib = H_np[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            W_out = W_out + delta_W

        # Update network output weights
        with torch.no_grad():
            self.network.layers[-1].weight.copy_(
                torch.tensor(W_out.reshape(1, -1), dtype=TORCH_PRECISION, device=device)
            )

        return residual

    def gradient_step_hidden(self, lr=0.01, n_steps=10):
        """
        Gradient step: Fix output weights, update hidden layers
        """
        # Only optimize hidden layer parameters
        hidden_params = []
        for layer in self.network.layers[:-1]:
            hidden_params.extend(layer.parameters())

        optimizer = optim.Adam(hidden_params, lr=lr)

        # Custom sparse matmul for autograd
        L_cupy, B_cupy = self.L_cupy, self.B_cupy
        L_t_cupy, B_t_cupy = self.L_t_cupy, self.B_t_cupy

        class SparseMul_L(torch.autograd.Function):
            @staticmethod
            def forward(ctx, u):
                return from_dlpack(L_cupy.dot(cupy.from_dlpack(to_dlpack(u))).toDlpack())
            @staticmethod
            def backward(ctx, grad):
                return from_dlpack(L_t_cupy.dot(cupy.from_dlpack(to_dlpack(grad))).toDlpack())

        class SparseMul_B(torch.autograd.Function):
            @staticmethod
            def forward(ctx, u):
                return from_dlpack(B_cupy.dot(cupy.from_dlpack(to_dlpack(u))).toDlpack())
            @staticmethod
            def backward(ctx, grad):
                return from_dlpack(B_t_cupy.dot(cupy.from_dlpack(to_dlpack(grad))).toDlpack())

        L_mul = SparseMul_L.apply
        B_mul = SparseMul_B.apply

        for step in range(n_steps):
            optimizer.zero_grad()

            u = self.network(self.X_t)
            Lu = L_mul(u)[:self.N_ib]
            exp_u = torch.exp(u[:self.N_ib])
            pde_residual = Lu - self.f_t[:self.N_ib] - exp_u
            pde_loss = torch.mean(pde_residual**2)

            Bu = B_mul(u)
            bc_residual = Bu - self.g_t
            bc_loss = torch.mean(bc_residual**2)

            loss = pde_loss + bc_loss
            loss.backward()
            optimizer.step()

        return loss.item()

    def solve(self, n_alternations=10, elm_iters=10, grad_steps=20, grad_lr=0.01):
        """
        Main solve loop: alternate between ELM and gradient steps
        """
        history = []

        for alt in range(n_alternations):
            # Step A: ELM solve for output weights
            elm_residual = self.elm_solve_output(max_iter=elm_iters)

            # Step B: Gradient descent on hidden layers
            grad_loss = self.gradient_step_hidden(lr=grad_lr, n_steps=grad_steps)

            # Compute L2 error
            with torch.no_grad():
                u_pred = self.network(self.X_t).cpu().numpy()

            history.append({
                'alternation': alt + 1,
                'elm_residual': elm_residual,
                'grad_loss': grad_loss,
            })

            if (alt + 1) % 2 == 0:
                print(f"  Alt {alt+1}: ELM res={elm_residual:.4e}, grad loss={grad_loss:.4e}")

        # Final ELM solve
        self.elm_solve_output(max_iter=elm_iters)

        with torch.no_grad():
            u_pred = self.network(self.X_t).cpu().numpy()

        return u_pred, history

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib].flatten()
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment(layer_sizes, n_alternations=10, elm_iters=10, grad_steps=20, grad_lr=0.01):
    """Run Alternating ELM experiment"""

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

    start = time.perf_counter()

    solver = AlternatingELMSolver(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx,
        layer_sizes=layer_sizes
    )

    print(f"\n  Running {n_alternations} alternations...")
    u_pred, history = solver.solve(
        n_alternations=n_alternations,
        elm_iters=elm_iters,
        grad_steps=grad_steps,
        grad_lr=grad_lr
    )

    total_time = time.perf_counter() - start

    l2_error = solver.compute_l2_error(u_pred, u_true)
    n_hidden = len(layer_sizes) - 2

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Architecture: {layer_sizes}")
    print(f"  Hidden layers: {n_hidden}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  L2 error: {l2_error:.4e}")
    print(f"\nTarget: L2 ≤ 6.5e-03 with ≥2 hidden layers")
    print(f"{'='*60}")

    return {
        'time': total_time,
        'l2_error': l2_error,
        'layer_sizes': layer_sizes,
        'n_hidden_layers': n_hidden,
        'n_alternations': n_alternations,
    }


if __name__ == "__main__":
    print("="*70)
    print("H34: Alternating ELM Optimization")
    print("="*70)

    results = []

    # Test different configurations
    configs = [
        # (layer_sizes, n_alternations, elm_iters, grad_steps, grad_lr)
        ([2, 50, 50, 1], 10, 10, 20, 0.01),    # 2 hidden layers, 50 neurons each
        ([2, 100, 100, 1], 10, 10, 20, 0.01),  # 2 hidden layers, 100 neurons each
        ([2, 50, 50, 1], 20, 10, 20, 0.01),    # More alternations
        ([2, 50, 50, 50, 1], 10, 10, 20, 0.01), # 3 hidden layers
        ([2, 100, 1], 10, 10, 20, 0.01),       # 1 hidden layer (comparison)
    ]

    for layer_sizes, n_alt, elm_it, grad_st, lr in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {layer_sizes}, {n_alt} alternations")
        print(f"{'='*60}")

        torch.manual_seed(42)
        np.random.seed(42)

        r = run_experiment(
            layer_sizes=layer_sizes,
            n_alternations=n_alt,
            elm_iters=elm_it,
            grad_steps=grad_st,
            grad_lr=lr
        )
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Alternating ELM Optimization")
    print("="*70)
    print(f"{'Config':<30} | {'Time':>10} | {'L2 Error':>12} | {'Status':<10}")
    print("-"*70)

    for r in results:
        config = f"{r['layer_sizes']} ({r['n_alternations']}alt)"
        status = "PASS" if r['l2_error'] <= 6.5e-03 else "FAIL"
        print(f"{config:<30} | {r['time']:>9.2f}s | {r['l2_error']:>12.4e} | {status:<10}")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/alternating_elm", exist_ok=True)
    with open("/workspace/dt-pinn/results/alternating_elm/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
