"""
H47: ELM Warm-Start Hybrid Training

HYPOTHESIS: The ELM accuracy floor (~6.72e-03) exists because:
1. Random features can't perfectly represent the solution
2. Only output weights are optimized (input weights are frozen)

INSIGHT FROM CYCLE 46:
- Gradient-trained DT-PINN achieves L2=6.54e-03 (better than ELM's 6.72e-03)
- This proves the floor is NOT from RBF-FD discretization
- The gap is from ELM's restricted optimization (only output weights)

PROPOSED SOLUTION: Hybrid approach
1. Use ELM for FAST initialization (gets to ~6.72e-03 in <1s)
2. Apply FEW gradient steps to fine-tune ALL weights (breaks the floor)

This is a GENUINE third contribution because:
- It's NOT just "combining DT-PINN + ELM" (that's contribution 1)
- It's NOT just "skip connections" (that's contribution 2)
- It discovers that ELM initialization + minimal gradient refinement
  achieves BOTH speed (from ELM) and accuracy (from gradients)

Target: L2 < 6.5e-03 in < 5 seconds (faster than pure gradient, more accurate than pure ELM)
"""

import json
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from scipy.io import loadmat
from scipy.sparse import csr_matrix as scipy_csr
import time

try:
    import cupy
    from cupy.sparse import csr_matrix as cupy_csr
    from torch.utils.dlpack import to_dlpack, from_dlpack
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)
torch.manual_seed(42)

PRECISION_NP = np.float64
PRECISION_TORCH = torch.float64


class ELMWarmStartHybrid:
    """
    Hybrid approach: ELM initialization + gradient fine-tuning.

    Phase 1 (ELM): Fast direct solve to get good initialization
    Phase 2 (Gradient): Few optimization steps to break accuracy floor
    """

    def __init__(self, X, L, B, f, g, ib_idx, hidden_sizes, device='cuda'):
        self.device = device
        self.N_total = X.shape[0]
        self.N_ib = ib_idx
        self.N_bc = g.shape[0]

        # Store operators
        self.L_scipy = L
        self.B_scipy = B
        self.f_np = f.flatten()
        self.g_np = g.flatten()
        self.X_np = X

        # Convert to PyTorch
        self.X = torch.tensor(X, dtype=PRECISION_TORCH, device=device)
        self.f = torch.tensor(f.flatten()[:ib_idx], dtype=PRECISION_TORCH, device=device)
        self.g = torch.tensor(g.flatten(), dtype=PRECISION_TORCH, device=device)

        self.hidden_sizes = hidden_sizes

        # Initialize network weights (will be set by ELM)
        self.weights = []
        self.biases = []
        self.W_out = None

        # Setup sparse matrix multiplication for GPU
        self._setup_sparse_ops()

    def _setup_sparse_ops(self):
        """Setup sparse matrix operations with proper autograd support"""
        if HAS_CUPY and self.device == 'cuda':
            # Store cupy sparse matrices for forward pass
            self.L_gpu = cupy_csr(self.L_scipy.astype(np.float64))
            self.B_gpu = cupy_csr(self.B_scipy.astype(np.float64))
            # Store transposed matrices for backward pass
            self.L_t_gpu = cupy_csr(self.L_scipy.T.tocsr().astype(np.float64))
            self.B_t_gpu = cupy_csr(self.B_scipy.T.tocsr().astype(np.float64))
            self.use_cupy = True

            # Create autograd functions for proper gradient flow
            self._setup_autograd_functions()
        else:
            self.L_dense = torch.tensor(self.L_scipy.toarray(), dtype=PRECISION_TORCH, device=self.device)
            self.B_dense = torch.tensor(self.B_scipy.toarray(), dtype=PRECISION_TORCH, device=self.device)
            self.use_cupy = False

    def _setup_autograd_functions(self):
        """Create autograd functions for sparse matrix multiply"""
        L_gpu = self.L_gpu
        L_t_gpu = self.L_t_gpu
        B_gpu = self.B_gpu
        B_t_gpu = self.B_t_gpu

        class SparseMulL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, u):
                u_cupy = cupy.from_dlpack(to_dlpack(u.contiguous()))
                result = L_gpu.dot(u_cupy)
                return from_dlpack(result.toDlpack())

            @staticmethod
            def backward(ctx, grad_output):
                grad_cupy = cupy.from_dlpack(to_dlpack(grad_output.contiguous()))
                grad_input = L_t_gpu.dot(grad_cupy)
                return from_dlpack(grad_input.toDlpack())

        class SparseMulB(torch.autograd.Function):
            @staticmethod
            def forward(ctx, u):
                u_cupy = cupy.from_dlpack(to_dlpack(u.contiguous()))
                result = B_gpu.dot(u_cupy)
                return from_dlpack(result.toDlpack())

            @staticmethod
            def backward(ctx, grad_output):
                grad_cupy = cupy.from_dlpack(to_dlpack(grad_output.contiguous()))
                grad_input = B_t_gpu.dot(grad_cupy)
                return from_dlpack(grad_input.toDlpack())

        self.SparseMulL = SparseMulL
        self.SparseMulB = SparseMulB

    def L_mul(self, u):
        """Multiply by Laplacian operator"""
        if self.use_cupy:
            return self.SparseMulL.apply(u)
        else:
            return self.L_dense @ u

    def B_mul(self, u):
        """Multiply by boundary operator"""
        if self.use_cupy:
            return self.SparseMulB.apply(u)
        else:
            return self.B_dense @ u

    def phase1_elm(self, max_iter=20, tol=1e-10):
        """
        Phase 1: ELM direct solve with Newton iteration.
        Returns the ELM solution and initializes network weights.
        """
        print("Phase 1: ELM Initialization...")
        start = time.perf_counter()

        # Build multi-layer hidden representation
        np.random.seed(42)
        H = self.X_np
        all_H = []
        self.weights = []
        self.biases = []

        input_dim = 2
        for hidden_size in self.hidden_sizes:
            W = np.random.randn(input_dim, hidden_size).astype(PRECISION_NP) * np.sqrt(2.0 / input_dim)
            b = np.random.randn(hidden_size).astype(PRECISION_NP) * 0.1
            H = np.tanh(H @ W + b)
            all_H.append(H)
            self.weights.append(W)
            self.biases.append(b)
            input_dim = hidden_size

        # Concatenate all layer outputs (skip connections)
        H_concat = np.hstack(all_H)

        # Precompute operator products
        LH_full = self.L_scipy @ H_concat
        LH = LH_full[:self.N_ib, :]
        BH = self.B_scipy @ H_concat

        # Initialize with linear approximation
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([self.f_np + 1.0, self.g_np])
        self.W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        u = H_concat @ self.W_out
        u_ib = u[:self.N_ib]

        # Newton iteration
        for k in range(max_iter):
            Lu = (self.L_scipy @ u)[:self.N_ib]
            exp_u = np.exp(u_ib)
            F_pde = Lu - self.f_np - exp_u
            F_bc = self.B_scipy @ u - self.g_np

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            if residual < tol:
                print(f"  Newton converged at iteration {k+1}")
                break

            H_ib = H_concat[:self.N_ib, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            self.W_out = self.W_out + delta_W

            u = H_concat @ self.W_out
            u_ib = u[:self.N_ib]

        elm_time = time.perf_counter() - start
        print(f"  ELM time: {elm_time:.3f}s, final residual: {residual:.4e}")

        return u, elm_time

    def phase2_gradient(self, n_epochs=50, lr=0.01, optimizer_type='adam'):
        """
        Phase 2: Gradient-based fine-tuning of ALL weights.
        Starts from ELM-initialized weights.
        """
        print(f"Phase 2: Gradient Fine-tuning ({n_epochs} epochs)...")
        start = time.perf_counter()

        # Convert ELM weights to PyTorch parameters
        params = []
        torch_weights = []
        torch_biases = []

        for W, b in zip(self.weights, self.biases):
            W_t = torch.tensor(W, dtype=PRECISION_TORCH, device=self.device, requires_grad=True)
            b_t = torch.tensor(b, dtype=PRECISION_TORCH, device=self.device, requires_grad=True)
            torch_weights.append(W_t)
            torch_biases.append(b_t)
            params.extend([W_t, b_t])

        # Output weight initialization from ELM
        # Need to compute correct output weight dimension
        total_hidden = sum(self.hidden_sizes)
        W_out_t = torch.tensor(self.W_out, dtype=PRECISION_TORCH, device=self.device, requires_grad=True)
        params.append(W_out_t)

        # Setup optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(params, lr=lr)
        elif optimizer_type == 'lbfgs':
            optimizer = optim.LBFGS(params, lr=lr, max_iter=5)
        else:
            optimizer = optim.SGD(params, lr=lr)

        # Training loop
        for epoch in range(n_epochs):
            if optimizer_type == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    loss = self._compute_loss(torch_weights, torch_biases, W_out_t)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = self._compute_loss(torch_weights, torch_biases, W_out_t)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss = {loss.item():.6e}")

        # Get final solution
        with torch.no_grad():
            u_pred = self._forward(torch_weights, torch_biases, W_out_t)

        grad_time = time.perf_counter() - start
        print(f"  Gradient time: {grad_time:.3f}s")

        return u_pred, grad_time

    def _forward(self, weights, biases, W_out):
        """Forward pass through the network"""
        H_layers = []
        h = self.X

        for W, b in zip(weights, biases):
            h = torch.tanh(h @ W + b)
            H_layers.append(h)

        H_concat = torch.cat(H_layers, dim=1)
        u = H_concat @ W_out
        return u

    def _compute_loss(self, weights, biases, W_out):
        """Compute PDE + BC loss"""
        u_full = self._forward(weights, biases, W_out)

        # PDE residual
        Lu = self.L_mul(u_full)[:self.N_ib]
        exp_u = torch.exp(u_full[:self.N_ib])
        F_pde = Lu - self.f - exp_u

        # BC residual
        F_bc = self.B_mul(u_full) - self.g

        # Combined loss
        loss = torch.mean(F_pde**2) + torch.mean(F_bc**2)
        return loss

    def compute_l2_error(self, u_pred, u_true):
        """Compute L2 relative error"""
        if isinstance(u_pred, torch.Tensor):
            u_pred = u_pred.detach().cpu().numpy()

        u_pred_ib = u_pred.flatten()[:self.N_ib]
        u_true_flat = u_true.flatten()[:self.N_ib]
        return np.linalg.norm(u_pred_ib - u_true_flat) / np.linalg.norm(u_true_flat)


def run_experiment(hidden_sizes=[100], n_grad_epochs=50, grad_lr=0.01, optimizer_type='adam'):
    """Run hybrid ELM + gradient experiment"""

    data_path = "/workspace/dt-pinn/nonlinear"
    file_name = "2_2236"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading data: {file_name}")
    print(f"Device: {device}")

    X_i = loadmat(f"{data_path}/files_{file_name}/Xi.mat")["Xi"].astype(PRECISION_NP)
    X_b = loadmat(f"{data_path}/files_{file_name}/Xb.mat")["Xb"].astype(PRECISION_NP)
    X_g = loadmat(f"{data_path}/files_{file_name}/Xg.mat")["X_g"].astype(PRECISION_NP)
    u_true = loadmat(f"{data_path}/files_{file_name}/u.mat")["u"].astype(PRECISION_NP)
    f = loadmat(f"{data_path}/files_{file_name}/f.mat")["f"].astype(PRECISION_NP)
    g = loadmat(f"{data_path}/files_{file_name}/g.mat")["g"].astype(PRECISION_NP)
    L = scipy_csr(loadmat(f"{data_path}/files_{file_name}/L1.mat")["L1"], dtype=PRECISION_NP)
    B = scipy_csr(loadmat(f"{data_path}/files_{file_name}/B1.mat")["B1"], dtype=PRECISION_NP)

    X_full = np.vstack([X_i, X_b, X_g])
    ib_idx = X_i.shape[0] + X_b.shape[0]

    print(f"Points: {X_full.shape[0]} total, {ib_idx} interior+boundary")
    print(f"Architecture: {hidden_sizes}")

    # Create hybrid solver
    solver = ELMWarmStartHybrid(
        X=X_full, L=L, B=B, f=f[:ib_idx], g=g, ib_idx=ib_idx,
        hidden_sizes=hidden_sizes, device=device
    )

    # Phase 1: ELM
    u_elm, elm_time = solver.phase1_elm(max_iter=20)
    elm_error = solver.compute_l2_error(u_elm, u_true)
    print(f"  ELM L2 error: {elm_error:.4e}")

    # Phase 2: Gradient refinement
    u_hybrid, grad_time = solver.phase2_gradient(
        n_epochs=n_grad_epochs, lr=grad_lr, optimizer_type=optimizer_type
    )
    hybrid_error = solver.compute_l2_error(u_hybrid, u_true)

    total_time = elm_time + grad_time

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Architecture: {hidden_sizes}")
    print(f"  ELM only:     L2 = {elm_error:.4e}, time = {elm_time:.2f}s")
    print(f"  Hybrid:       L2 = {hybrid_error:.4e}, time = {total_time:.2f}s")
    print(f"  Improvement:  {(elm_error - hybrid_error) / elm_error * 100:.1f}%")
    print(f"{'='*60}")

    return {
        'hidden_sizes': hidden_sizes,
        'elm_error': elm_error,
        'hybrid_error': hybrid_error,
        'elm_time': elm_time,
        'grad_time': grad_time,
        'total_time': total_time,
        'n_grad_epochs': n_grad_epochs,
        'grad_lr': grad_lr,
        'optimizer': optimizer_type,
        'improvement_pct': (elm_error - hybrid_error) / elm_error * 100,
    }


if __name__ == "__main__":
    print("="*70)
    print("H47: ELM Warm-Start Hybrid Training")
    print("="*70)

    results = []

    # Test 1: Baseline - pure ELM (0 gradient epochs)
    print("\n" + "="*60)
    print("Test 1: Pure ELM (baseline)")
    print("="*60)
    r = run_experiment(hidden_sizes=[100], n_grad_epochs=0)
    results.append(r)

    # Test 2: Hybrid with varying gradient epochs
    print("\n" + "="*60)
    print("Test 2: Hybrid with varying gradient epochs")
    print("="*60)

    for n_epochs in [10, 25, 50, 100]:
        print(f"\n--- {n_epochs} gradient epochs ---")
        r = run_experiment(hidden_sizes=[100], n_grad_epochs=n_epochs, grad_lr=0.01)
        results.append(r)

    # Test 3: Different learning rates
    print("\n" + "="*60)
    print("Test 3: Different learning rates")
    print("="*60)

    for lr in [0.001, 0.005, 0.01, 0.05]:
        print(f"\n--- lr = {lr} ---")
        r = run_experiment(hidden_sizes=[100], n_grad_epochs=50, grad_lr=lr)
        results.append(r)

    # Test 4: Multi-layer hybrid
    print("\n" + "="*60)
    print("Test 4: Multi-layer hybrid")
    print("="*60)

    for hidden_sizes in [[100, 100], [100, 50], [50, 50]]:
        print(f"\n--- {hidden_sizes} ---")
        r = run_experiment(hidden_sizes=hidden_sizes, n_grad_epochs=50, grad_lr=0.01)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: ELM Warm-Start Hybrid")
    print("="*70)
    print(f"{'Config':<20} | {'ELM L2':>10} | {'Hybrid L2':>10} | {'Time':>8} | {'Improvement':>12}")
    print("-"*70)

    for r in results:
        config = str(r['hidden_sizes'])[:18]
        print(f"{config:<20} | {r['elm_error']:>10.4e} | {r['hybrid_error']:>10.4e} | {r['total_time']:>7.2f}s | {r['improvement_pct']:>10.1f}%")

    best = min(results, key=lambda x: x['hybrid_error'])
    print(f"\nBest result: L2 = {best['hybrid_error']:.4e}")
    print(f"  Config: {best}")
    print(f"\nTarget: L2 < 6.5e-03 in < 5s")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/elm_warmstart_hybrid", exist_ok=True)
    with open("/workspace/dt-pinn/results/elm_warmstart_hybrid/results.json", "w") as f_out:
        json.dump(results, f_out, indent=2, default=str)
