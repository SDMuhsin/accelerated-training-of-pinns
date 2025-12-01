"""
H33: Hybrid Trainable ELM (Gradient-Optimized Hidden Layer + ELM Output)

HYPOTHESIS: The limitation of pure ELM is that random features may not be optimal
for the specific PDE. Use gradient descent to optimize the FIRST hidden layer,
then use ELM (direct solve) for all subsequent layers.

KEY INSIGHT:
- Problem with H28-H32: Random features are not adapted to the PDE
- Problem with H29: Training hidden layers with standard loss breaks ELM properties
- Solution: Train hidden layer to minimize a proxy objective, then ELM solve

STRUCTURE:
Phase 1: Train first hidden layer using gradient descent on DT-PINN loss (few epochs)
Phase 2: Freeze first layer, add random second layer, ELM solve for output

This creates a TRUE multi-layer network:
- Layer 1: x -> h1 = tanh(W1 @ x + b1) [TRAINED via gradient]
- Layer 2: h1 -> h2 = tanh(W2 @ h1 + b2) [RANDOM, fixed]
- Output: h2 -> u = W_out @ h2 [ELM solved via Newton iteration]

The key difference from H29 is that we're not training the whole network,
just the first layer, then adding more random layers on top.

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


class TrainableHiddenLayer(torch.nn.Module):
    """First hidden layer that is trained via gradient descent"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim, dtype=TORCH_PRECISION)
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.tanh(self.linear(x))


def train_first_layer(X, L, B, f, g, ib_idx, hidden_dim, n_epochs=100, lr=0.01):
    """Train the first hidden layer using DT-PINN gradient descent"""

    # Convert to torch
    X_t = torch.tensor(X, dtype=TORCH_PRECISION, device=device)
    f_t = torch.tensor(f, dtype=TORCH_PRECISION, device=device)
    g_t = torch.tensor(g, dtype=TORCH_PRECISION, device=device)

    # Convert scipy sparse to cupy sparse
    L_cupy = cupy_csr(L.astype(np.float64))
    B_cupy = cupy_csr(B.astype(np.float64))
    L_t_cupy = cupy_csr(L.T.astype(np.float64))
    B_t_cupy = cupy_csr(B.T.astype(np.float64))

    # Custom sparse matmul with autograd
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

    # Create trainable first layer + output layer
    layer1 = TrainableHiddenLayer(2, hidden_dim).to(device)
    W_out = torch.nn.Linear(hidden_dim, 1, bias=False, dtype=TORCH_PRECISION).to(device)

    optimizer = optim.Adam(list(layer1.parameters()) + list(W_out.parameters()), lr=lr)

    print(f"  Training first hidden layer ({hidden_dim} neurons) for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Forward pass
        h1 = layer1(X_t)
        u = W_out(h1)

        # PDE loss
        Lu = L_mul(u)[:ib_idx]
        exp_u = torch.exp(u[:ib_idx])
        pde_residual = Lu - f_t[:ib_idx] - exp_u
        pde_loss = torch.mean(pde_residual**2)

        # BC loss
        Bu = B_mul(u)
        bc_residual = Bu - g_t
        bc_loss = torch.mean(bc_residual**2)

        loss = pde_loss + bc_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: loss = {loss.item():.4e}")

    # Extract trained weights
    W1 = layer1.linear.weight.detach().cpu().numpy().T  # (2, hidden_dim)
    b1 = layer1.linear.bias.detach().cpu().numpy()      # (hidden_dim,)

    return W1, b1


class HybridTrainableELM:
    """Multi-layer network with trained first layer and ELM output"""

    def __init__(self, X, L, B, f, g, ib_idx, W1_trained, b1_trained,
                 hidden_dims_random, n_random_layers=1):
        """
        W1_trained, b1_trained: pre-trained first layer weights
        hidden_dims_random: list of hidden dimensions for random layers
        """
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]

        self.L = L
        self.B = B
        self.f = f.flatten()
        self.g = g.flatten()
        self.X = X

        self.n_layers = 1 + n_random_layers  # trained layer + random layers

        # Compute output of trained first layer
        h1 = np.tanh(X @ W1_trained + b1_trained)

        # Add random layers on top
        self.random_weights = []
        self.random_biases = []
        h = h1
        input_dim = h1.shape[1]

        for i, hidden_dim in enumerate(hidden_dims_random):
            np.random.seed(42 + i * 1000)
            W = np.random.randn(input_dim, hidden_dim).astype(PRECISION) * np.sqrt(2.0 / input_dim)
            b = np.random.randn(hidden_dim).astype(PRECISION) * 0.1
            self.random_weights.append(W)
            self.random_biases.append(b)
            h = np.tanh(h @ W + b)
            input_dim = hidden_dim

        self.H = h  # Final hidden representation
        self.final_hidden_dim = self.H.shape[1]
        print(f"  Final hidden representation: {self.H.shape}")

        # Output weights (trainable via ELM)
        self.W_out = np.zeros(self.final_hidden_dim, dtype=PRECISION)

    def solve_nonlinear_newton(self, max_iter=20, tol=1e-8):
        """Solve for output weights using Newton iteration (ELM-style)"""
        LH_full = self.L @ self.H
        LH = LH_full[:self.N_ib, :]
        BH = self.B @ self.H

        # Initialize with linear solution
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f + 1.0, self.g])
        self.W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = self.H @ self.W_out
        u_ib = u[:self.N_ib]

        for k in range(max_iter):
            Lu = (self.L @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f - exp_u
            F_bc = self.B @ u - self.g

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if residual < tol:
                print(f"  Converged at iteration {k+1}")
                break

            H_ib = self.H[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            self.W_out = self.W_out + delta_W

            u = self.H @ self.W_out
            u_ib = u[:self.N_ib]

        return u

    def compute_l2_error(self, u_pred, u_true):
        u_pred_ib = u_pred[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment(hidden_dim_trained=50, hidden_dims_random=[50], train_epochs=100, lr=0.01):
    """Run Hybrid Trainable ELM experiment"""

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

    n_total_layers = 1 + len(hidden_dims_random)
    print(f"Points: {X_full.shape[0]} total, {ib_idx} interior+boundary")
    print(f"Architecture: {n_total_layers} layers ({hidden_dim_trained} trained + {hidden_dims_random} random)")

    start = time.perf_counter()

    # Phase 1: Train first hidden layer
    W1, b1 = train_first_layer(
        X_full, L, B, f[:ib_idx], g, ib_idx,
        hidden_dim=hidden_dim_trained,
        n_epochs=train_epochs,
        lr=lr
    )

    # Phase 2: Create hybrid network and ELM solve
    solver = HybridTrainableELM(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx,
        W1_trained=W1, b1_trained=b1,
        hidden_dims_random=hidden_dims_random,
        n_random_layers=len(hidden_dims_random)
    )

    print("\n  Phase 2: ELM solve for output weights...")
    u_pred = solver.solve_nonlinear_newton(max_iter=20)

    total_time = time.perf_counter() - start

    l2_error = solver.compute_l2_error(u_pred, u_true)

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Architecture: {hidden_dim_trained} trained + {hidden_dims_random} random")
    print(f"  Total layers: {n_total_layers}")
    print(f"  Time: {total_time:.2f}s")
    print(f"  L2 error: {l2_error:.4e}")
    print(f"\nTarget: L2 ≤ 6.5e-03 with ≥2 layers")
    print(f"{'='*60}")

    return {
        'time': total_time,
        'l2_error': l2_error,
        'hidden_dim_trained': hidden_dim_trained,
        'hidden_dims_random': hidden_dims_random,
        'n_layers': n_total_layers,
        'train_epochs': train_epochs,
    }


if __name__ == "__main__":
    print("="*70)
    print("H33: Hybrid Trainable ELM")
    print("="*70)

    results = []

    # Test different configurations
    configs = [
        # (trained_hidden, [random_hidden_dims], train_epochs, lr)
        (50, [50], 50, 0.01),      # 2 layers, 50 epochs training
        (50, [50], 100, 0.01),     # 2 layers, 100 epochs training
        (50, [50, 50], 100, 0.01), # 3 layers
        (100, [50], 100, 0.01),    # 2 layers, larger trained
        (100, [100], 100, 0.01),   # 2 layers, all 100
        (50, [50], 200, 0.01),     # 2 layers, more training
    ]

    for hidden_trained, hidden_random, epochs, lr in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {hidden_trained} trained + {hidden_random} random, {epochs} epochs")
        print(f"{'='*60}")

        r = run_experiment(
            hidden_dim_trained=hidden_trained,
            hidden_dims_random=hidden_random,
            train_epochs=epochs,
            lr=lr
        )
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Hybrid Trainable ELM")
    print("="*70)
    print(f"{'Config':<35} | {'Time':>10} | {'L2 Error':>12} | {'Status':<10}")
    print("-"*75)

    for r in results:
        config = f"{r['hidden_dim_trained']}T+{r['hidden_dims_random']}R ({r['train_epochs']}ep)"
        status = "PASS" if r['l2_error'] <= 6.5e-03 else "FAIL"
        print(f"{config:<35} | {r['time']:>9.2f}s | {r['l2_error']:>12.4e} | {status:<10}")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/hybrid_trainable_elm", exist_ok=True)
    with open("/workspace/dt-pinn/results/hybrid_trainable_elm/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
