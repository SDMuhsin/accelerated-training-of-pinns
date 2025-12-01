"""
H29: Hybrid-DT-ELM-PINN (Brief Hidden Layer Training + ELM Finish)

HYPOTHESIS: Combine brief gradient descent training (to shape hidden features)
with ELM's direct solve (to quickly find optimal output weights).

TWO-PHASE TRAINING:
1. WARM-UP PHASE: Train full network with DT-PINN (L-BFGS) for K epochs
   - This shapes the hidden layer features to be meaningful for the PDE
   - Uses sparse operators (no autodiff for derivatives)

2. ELM PHASE: Freeze hidden layers, solve for W_out via Newton iteration
   - Recompute H = hidden(X) with trained weights
   - Solve (J @ H) @ W_out = -F via least squares

KEY INSIGHT: Even a few epochs of gradient-based training can produce much
better features than random initialization, while ELM finishing is much
faster than continuing gradient descent.

Target: Time < 30s, L2 ≤ 6.5e-03, Speedup ≥10x over vanilla PINN
"""

import json
import os
import sys
import torch
from torch import optim
import numpy as np
from scipy.io import loadmat
import cupy
from cupy.sparse import csr_matrix
from torch.utils.dlpack import to_dlpack, from_dlpack
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import W

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    pytorch_device = torch.device('cuda')
    device_string = "cuda"
    torch.cuda.manual_seed_all(42)
else:
    pytorch_device = torch.device('cpu')
    device_string = "cpu"

PRECISION = torch.float64
PRECISION_NP = np.float64

L_t, B_t = None, None


class Cupy_mul_L(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_pred_, sparse):
        return from_dlpack(sparse.dot(cupy.from_dlpack(to_dlpack(u_pred_))).toDlpack())
    @staticmethod
    def backward(ctx, grad_output):
        return from_dlpack(L_t.dot(cupy.from_dlpack(to_dlpack(grad_output))).toDlpack()), None


class Cupy_mul_B(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_pred_, sparse):
        return from_dlpack(sparse.dot(cupy.from_dlpack(to_dlpack(u_pred_))).toDlpack())
    @staticmethod
    def backward(ctx, grad_output):
        return from_dlpack(B_t.dot(cupy.from_dlpack(to_dlpack(grad_output))).toDlpack()), None


class HybridDTELMPINN:
    """Hybrid DT-ELM-PINN: Brief training + ELM finish"""

    def __init__(self, config, X_full, X_tilde, L, B, f, g, u_true, ib_idx):
        self.config = config
        self.X_full = X_full
        self.X_tilde = X_tilde
        self.L = L
        self.B = B
        self.f = f
        self.g = g
        self.u_true = u_true
        self.ib_idx = ib_idx

        # Create network
        self.w = W(config)
        n_params = sum(p.numel() for p in self.w.parameters())
        print(f"Network: {config['layers']} layers, {config['nodes']} nodes, {n_params} params")

    @staticmethod
    def compute_l2(a, b):
        diff = torch.subtract(torch.flatten(a).detach().cpu(), torch.flatten(b).detach().cpu())
        return (torch.linalg.norm(diff) / torch.linalg.norm(torch.flatten(b))).item()

    def train_warmup(self, warmup_epochs, lr=0.02):
        """Phase 1: Brief gradient-based training with L-BFGS"""
        global L_t, B_t

        # Initialize sparse matrices
        rand_vec = cupy.from_dlpack(to_dlpack(torch.rand(self.L.shape[1], 2).to(torch.float64).to(device_string)))
        self.L.dot(rand_vec)
        self.B.dot(rand_vec)

        L_t = csr_matrix(self.L.transpose().astype(np.float64))
        B_t = csr_matrix(self.B.transpose().astype(np.float64))

        rand_L_vec = cupy.from_dlpack(to_dlpack(torch.rand(self.L.shape[0], 2).to(torch.float64).to(device_string)))
        rand_B_vec = cupy.from_dlpack(to_dlpack(torch.rand(self.B.shape[0], 2).to(torch.float64).to(device_string)))
        L_t.dot(rand_L_vec)
        B_t.dot(rand_B_vec)

        L_mul = Cupy_mul_L.apply
        B_mul = Cupy_mul_B.apply

        optimizer = optim.LBFGS(self.w.parameters(), lr=lr)

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(1, warmup_epochs + 1):
            def closure():
                optimizer.zero_grad()
                u_pred_full = self.w.forward(self.X_full)
                lap_u = L_mul(u_pred_full, self.L)
                f_pred = lap_u[:self.ib_idx] - self.f[:self.ib_idx] - torch.exp(u_pred_full[:self.ib_idx])
                boundary_loss_term = B_mul(u_pred_full, self.B) - self.g
                l2_loss = torch.mean(torch.square(torch.flatten(f_pred)))
                bc_loss = torch.mean(torch.square(torch.flatten(boundary_loss_term)))
                train_loss = l2_loss + bc_loss
                train_loss.backward(retain_graph=True)
                return train_loss.item()

            loss = optimizer.step(closure)

            if epoch % 5 == 0 or epoch == 1:
                with torch.no_grad():
                    pred = self.w.forward(self.X_tilde)
                    l2_error = self.compute_l2(pred, self.u_true[:self.ib_idx])
                print(f"  Warmup epoch {epoch}: loss={loss:.4e}, L2={l2_error:.4e}")

        if device_string == "cuda":
            torch.cuda.synchronize()
        warmup_time = time.perf_counter() - start

        # Get L2 after warmup
        with torch.no_grad():
            pred = self.w.forward(self.X_tilde)
            warmup_l2 = self.compute_l2(pred, self.u_true[:self.ib_idx])

        return warmup_time, warmup_l2

    def elm_finish(self, max_iter=20, tol=1e-8):
        """Phase 2: ELM-style direct solve for output layer"""

        # Extract hidden layer features (freeze network except output)
        # Get the hidden layer output H by running forward pass without the last layer
        with torch.no_grad():
            # We need to manually extract hidden features
            # The network structure is: input -> hidden layers -> output
            # We'll compute hidden output by running through all but the last layer

            X_np = self.X_full.cpu().numpy()
            N_total = X_np.shape[0]

            # Extract weights from trained network - iterate through named modules
            # Network structure: linear_1, activation, linear_2, activation, ..., linear_n (output)
            linear_layers = []
            for name, module in self.w.net.named_modules():
                if isinstance(module, torch.nn.Linear):
                    W = module.weight.detach().cpu().numpy()  # Shape: (out, in)
                    b = module.bias.detach().cpu().numpy()    # Shape: (out,)
                    linear_layers.append((W.T, b))  # Transpose W for h @ W + b

            # Forward pass through hidden layers only (all except last linear layer)
            h = X_np
            n_layers = len(linear_layers)

            # Apply all hidden layers (everything except last layer)
            for i in range(n_layers - 1):
                W, b = linear_layers[i]
                h = np.tanh(h @ W + b)

            # h is now the hidden representation (N_total, hidden_dim)
            # The last layer is just W_out @ h + b_out
            # We'll solve for W_out directly

            H = h  # Shape: (N_total, hidden_dim)
            hidden_dim = H.shape[1]

        # Convert L, B to scipy sparse for CPU operations
        from scipy.sparse import csr_matrix as scipy_csr
        L_scipy = scipy_csr(self.L.get())
        B_scipy = scipy_csr(self.B.get())

        # Get f, g as numpy
        f_np = self.f.detach().cpu().numpy().flatten()[:self.ib_idx]
        g_np = self.g.detach().cpu().numpy().flatten()

        # Newton iteration for ELM
        LH_full = L_scipy @ H
        LH = LH_full[:self.ib_idx, :]
        BH = B_scipy @ H

        # Initialize W_out with linear solve (ignoring exp)
        A_init = np.vstack([LH, BH])
        b_init = np.concatenate([f_np + 1.0, g_np])
        W_out, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

        start = time.perf_counter()

        residual_history = []
        for k in range(max_iter):
            u = H @ W_out
            u_ib = u[:self.ib_idx]

            # Residual
            Lu_full = L_scipy @ u
            Lu = Lu_full[:self.ib_idx]
            exp_u = np.exp(u_ib)
            F_pde = Lu - f_np - exp_u

            Bu = B_scipy @ u
            F_bc = Bu - g_np

            residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            residual_history.append(residual)

            if residual < tol:
                print(f"  ELM converged at iteration {k+1} with residual {residual:.4e}")
                break

            # Jacobian
            H_ib = H[:self.ib_idx, :]
            JH = LH - exp_u[:, np.newaxis] * H_ib

            A = np.vstack([JH, BH])
            F = np.concatenate([-F_pde, -F_bc])

            delta_W, *_ = np.linalg.lstsq(A, F, rcond=None)
            W_out = W_out + delta_W

            if (k + 1) % 5 == 0:
                print(f"  ELM iteration {k+1}: residual = {residual:.4e}")

        elm_time = time.perf_counter() - start

        # Compute final L2 error
        u_final = H @ W_out
        u_true_np = self.u_true.cpu().numpy().flatten()[:self.ib_idx]
        l2_error = np.linalg.norm(u_final[:self.ib_idx] - u_true_np) / np.linalg.norm(u_true_np)

        return elm_time, l2_error, residual_history


def load_mat_cupy(mat):
    return csr_matrix(mat, dtype=np.float64)


def run_experiment(warmup_epochs=10, layers=2, nodes=50, lr=0.02):
    """Run Hybrid-DT-ELM-PINN experiment"""
    cupy.cuda.Device(0).use()

    data_path = "/workspace/dt-pinn/nonlinear"
    file_name = "2_2236"

    print(f"\nLoading data: {file_name}")

    X_i = torch.tensor(loadmat(f"{data_path}/files_{file_name}/Xi.mat")["Xi"],
                       dtype=PRECISION, requires_grad=True).to(device_string)
    X_b = torch.tensor(loadmat(f"{data_path}/files_{file_name}/Xb.mat")["Xb"],
                       dtype=PRECISION, requires_grad=True).to(device_string)
    X_g = torch.tensor(loadmat(f"{data_path}/files_{file_name}/Xg.mat")["X_g"],
                       dtype=PRECISION, requires_grad=True).to(device_string)
    u_true = torch.tensor(loadmat(f"{data_path}/files_{file_name}/u.mat")["u"],
                          dtype=PRECISION).to(device_string)
    f = torch.tensor(loadmat(f"{data_path}/files_{file_name}/f.mat")["f"],
                     dtype=PRECISION, requires_grad=True).to(device_string)
    g = torch.tensor(loadmat(f"{data_path}/files_{file_name}/g.mat")["g"],
                     dtype=PRECISION, requires_grad=True).to(device_string)
    L = load_mat_cupy(loadmat(f"{data_path}/files_{file_name}/L1.mat")["L1"])
    B = load_mat_cupy(loadmat(f"{data_path}/files_{file_name}/B1.mat")["B1"])

    x_i = X_i[:, 0].unsqueeze(dim=1)
    y_i = X_i[:, 1].unsqueeze(dim=1)
    x_b = X_b[:, 0].unsqueeze(dim=1)
    y_b = X_b[:, 1].unsqueeze(dim=1)
    x_g = X_g[:, 0].unsqueeze(dim=1)
    y_g = X_g[:, 1].unsqueeze(dim=1)

    x_full = torch.vstack([x_i, x_b, x_g])
    y_full = torch.vstack([y_i, y_b, y_g])
    X_full = torch.hstack([x_full, y_full])

    x_tilde = torch.vstack([x_i, x_b])
    y_tilde = torch.vstack([y_i, y_b])
    X_tilde = torch.hstack([x_tilde, y_tilde])

    ib_idx = X_i.shape[0] + X_b.shape[0]

    config = {
        'spatial_dim': 2,
        'precision': 'float64',
        'activation': 'tanh',
        'order': 2,
        'network_device': device_string,
        'layers': layers,
        'nodes': nodes,
    }

    print(f"Points: {X_full.shape[0]} total, {ib_idx} interior+boundary")
    print(f"Architecture: {layers} layers x {nodes} nodes")
    print(f"Warmup epochs: {warmup_epochs}")

    # Create trainer
    trainer = HybridDTELMPINN(
        config=config,
        X_full=X_full,
        X_tilde=X_tilde,
        L=L,
        B=B,
        f=f,
        g=g,
        u_true=u_true,
        ib_idx=ib_idx
    )

    # Phase 1: Warmup training
    print("\n--- Phase 1: Warmup Training (L-BFGS) ---")
    warmup_time, warmup_l2 = trainer.train_warmup(warmup_epochs, lr=lr)
    print(f"Warmup complete: time={warmup_time:.2f}s, L2={warmup_l2:.4e}")

    # Phase 2: ELM finish
    print("\n--- Phase 2: ELM Finish ---")
    elm_time, elm_l2, residuals = trainer.elm_finish(max_iter=30)
    print(f"ELM complete: time={elm_time:.2f}s, L2={elm_l2:.4e}")

    total_time = warmup_time + elm_time

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Warmup time: {warmup_time:.2f}s, L2 after warmup: {warmup_l2:.4e}")
    print(f"  ELM time: {elm_time:.2f}s, L2 after ELM: {elm_l2:.4e}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Final L2 error: {elm_l2:.4e}")
    print(f"\nTarget: Time < 30s, L2 ≤ 6.5e-03")
    print(f"Vanilla PINN baseline: 304s")
    print(f"Speedup vs vanilla: {304/total_time:.1f}x")
    print(f"{'='*60}")

    return {
        'warmup_epochs': warmup_epochs,
        'warmup_time': warmup_time,
        'warmup_l2': warmup_l2,
        'elm_time': elm_time,
        'elm_l2': elm_l2,
        'total_time': total_time,
        'layers': layers,
        'nodes': nodes,
    }


if __name__ == "__main__":
    print("="*70)
    print("H29: Hybrid-DT-ELM-PINN (Brief Hidden Layer Training + ELM Finish)")
    print("="*70)

    # Test different warmup epoch counts with 2-layer network
    all_results = []

    for warmup_epochs in [5, 10, 20, 30]:
        print(f"\n{'='*60}")
        print(f"Testing warmup_epochs={warmup_epochs}")
        print(f"{'='*60}")

        results = run_experiment(
            warmup_epochs=warmup_epochs,
            layers=2,
            nodes=50,
            lr=0.02
        )
        all_results.append(results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Hybrid-DT-ELM-PINN (2-layer x 50 nodes)")
    print("="*70)
    print(f"{'Warmup':<10} | {'Warmup T':>10} | {'Warmup L2':>12} | {'ELM T':>8} | {'Final L2':>12} | {'Total T':>10}")
    print("-"*70)

    for r in all_results:
        print(f"{r['warmup_epochs']:<10} | {r['warmup_time']:>9.2f}s | {r['warmup_l2']:>12.4e} | {r['elm_time']:>7.2f}s | {r['elm_l2']:>12.4e} | {r['total_time']:>9.2f}s")

    # Save results
    os.makedirs("/workspace/dt-pinn/results/hybrid_dt_elm_pinn", exist_ok=True)
    with open("/workspace/dt-pinn/results/hybrid_dt_elm_pinn/results.json", "w") as f_out:
        json.dump(all_results, f_out, indent=2)
