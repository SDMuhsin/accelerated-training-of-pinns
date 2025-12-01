"""
H25: ENGD-DT-PINN (Energy Natural Gradient Descent for Discrete-Trained PINN)

HYPOTHESIS: Combining DT-PINN's cheap discrete operators with exact Gauss-Newton
optimization could achieve faster convergence than L-BFGS while maintaining
the speedup from avoiding autodiff.

KEY INSIGHT: DT-PINN with small networks (2×20, ~500 params) allows exact
Gauss-Newton computation:
- Jacobian J: (N_residuals × N_params) ≈ (2236 × 501)
- Gramian G = J^T @ J: (501 × 501) - easily invertible

The update rule: θ_{k+1} = θ_k - (J^T J + λI)^{-1} J^T r

Where r is the residual vector and λ is a regularization parameter.

NOVELTY CHECK:
- KFAC-PINN (NeurIPS 2024) uses Taylor-mode autodiff for derivatives
- DT-PINN uses L-BFGS optimizer
- This combines DT-PINN's discrete operators with exact GN - NOT FOUND IN LITERATURE

Target: Time < 100s, L2 ≤ 3.0e-02
"""

import json
from collections import defaultdict
import os
import sys
from math import isnan
import torch
from torch.autograd import grad
from torch import optim
from torch.nn import MSELoss
import numpy as np
from scipy.io import loadmat
import cupy
from cupy.sparse import csr_matrix
from torch.utils.dlpack import to_dlpack, from_dlpack
import time
from torch.autograd.functional import jacobian

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network import W

torch.manual_seed(0)

# CUDA setup
if torch.cuda.is_available():
    pytorch_device = torch.device('cuda')
    torch.cuda.init()
    device_string = "cuda"
    torch.cuda.manual_seed_all(0)
else:
    pytorch_device = torch.device('cpu')
    device_string = "cpu"
print(f"Device: {device_string}")

PRECISION = torch.float64

# Global transposes
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


def get_params_flat(model):
    """Get all parameters as a flat vector."""
    return torch.cat([p.view(-1) for p in model.parameters()])


def set_params_flat(model, flat_params):
    """Set parameters from a flat vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
        offset += numel


def compute_jacobian_manual(model, X_full, L, B, f, g, ib_idx, L_mul, B_mul):
    """
    Compute the Jacobian of residuals w.r.t. network parameters.

    Residuals: r = [r_pde; r_bc] where
    - r_pde = L@u - f - exp(u) (for interior+boundary points)
    - r_bc = B@u - g (for boundary condition)

    Returns: J (n_residuals × n_params), r (n_residuals,)
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)

    # Forward pass
    u_pred_full = model.forward(X_full)

    # Compute residuals
    lap_u = L_mul(u_pred_full, L)
    r_pde = lap_u[:ib_idx] - f[:ib_idx] - torch.exp(u_pred_full[:ib_idx])
    r_bc = B_mul(u_pred_full, B) - g

    r_pde_flat = r_pde.flatten()
    r_bc_flat = r_bc.flatten()

    n_pde = r_pde_flat.shape[0]
    n_bc = r_bc_flat.shape[0]
    n_total = n_pde + n_bc

    # Compute Jacobian row by row using backward passes
    J = torch.zeros(n_total, n_params, dtype=PRECISION, device=device_string)

    # For PDE residuals
    for i in range(n_pde):
        model.zero_grad()
        if r_pde_flat.grad is not None:
            r_pde_flat.grad.zero_()

        # Recompute forward
        u_pred_full = model.forward(X_full)
        lap_u = L_mul(u_pred_full, L)
        r_pde_i = lap_u[i] - f[i] - torch.exp(u_pred_full[i])

        r_pde_i.backward(retain_graph=True)

        # Collect gradients
        grad_flat = torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device_string, dtype=PRECISION) for p in params])
        J[i] = grad_flat

    # For BC residuals
    for i in range(n_bc):
        model.zero_grad()

        u_pred_full = model.forward(X_full)
        r_bc_full = B_mul(u_pred_full, B) - g
        r_bc_i = r_bc_full[i]

        r_bc_i.backward(retain_graph=True)

        grad_flat = torch.cat([p.grad.view(-1) if p.grad is not None else torch.zeros(p.numel(), device=device_string, dtype=PRECISION) for p in params])
        J[n_pde + i] = grad_flat

    # Get final residual vector
    model.zero_grad()
    u_pred_full = model.forward(X_full)
    lap_u = L_mul(u_pred_full, L)
    r_pde = (lap_u[:ib_idx] - f[:ib_idx] - torch.exp(u_pred_full[:ib_idx])).flatten()
    r_bc = (B_mul(u_pred_full, B) - g).flatten()
    r = torch.cat([r_pde, r_bc])

    return J, r


def compute_jacobian_efficient(model, X_full, L, B, f, g, ib_idx, L_mul, B_mul):
    """
    Efficient Jacobian computation using torch.autograd.grad with create_graph=False.

    Instead of computing row by row, compute column by column (one per parameter).
    But actually for Gauss-Newton we can use the J^T @ r directly via backprop.
    """
    # Forward pass
    u_pred_full = model.forward(X_full)

    # Compute residuals
    lap_u = L_mul(u_pred_full, L)
    r_pde = (lap_u[:ib_idx] - f[:ib_idx] - torch.exp(u_pred_full[:ib_idx])).flatten()
    r_bc = (B_mul(u_pred_full, B) - g).flatten()
    r = torch.cat([r_pde, r_bc])

    # For GN, we need J^T @ J and J^T @ r
    # J^T @ r can be computed as the gradient of 0.5 * ||r||^2
    loss = 0.5 * torch.sum(r ** 2)

    # Get J^T @ r via backprop
    model.zero_grad()
    loss.backward(retain_graph=True)
    JTr = torch.cat([p.grad.view(-1).clone() for p in model.parameters()])

    # For J^T @ J, we need to compute J explicitly or use an approximation
    # Let's compute J explicitly by computing gradient of each residual element

    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    n_residuals = r.shape[0]

    J = torch.zeros(n_residuals, n_params, dtype=PRECISION, device=device_string)

    # Vectorized computation using functorch/vmap would be ideal
    # For now, use a loop (can be optimized later)
    for i in range(n_residuals):
        model.zero_grad()

        # Recompute to get fresh graph
        u_pred_full = model.forward(X_full)
        lap_u = L_mul(u_pred_full, L)
        r_pde = (lap_u[:ib_idx] - f[:ib_idx] - torch.exp(u_pred_full[:ib_idx])).flatten()
        r_bc = (B_mul(u_pred_full, B) - g).flatten()
        r_full = torch.cat([r_pde, r_bc])

        r_full[i].backward(retain_graph=True)

        grad_flat = torch.cat([p.grad.view(-1).clone() if p.grad is not None
                              else torch.zeros(p.numel(), device=device_string, dtype=PRECISION)
                              for p in params])
        J[i] = grad_flat

    return J, r, JTr


class ENGDDTPINNTrainer:
    """Energy Natural Gradient Descent DT-PINN Trainer"""

    def __init__(self, config, **kwargs):
        self.lr = config.get('lr', 1.0)  # GN typically uses lr=1
        self.epochs = config['epochs']
        self.damping = config.get('damping', 1e-4)  # Levenberg-Marquardt damping

        self.__dict__.update(kwargs)
        self.logged_results = defaultdict(list)

        # Network
        self.w = W(config)
        n_params = sum(p.numel() for p in self.w.parameters())
        print(f"Network: {config['layers']} layers, {config['nodes']} nodes, {n_params} params")
        print(f"Damping: {self.damping}, LR: {self.lr}")

        # Points setup
        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

    @staticmethod
    def compute_l2(a, b):
        diff = torch.subtract(torch.flatten(a).detach().cpu(), torch.flatten(b).detach().cpu())
        return (torch.linalg.norm(diff) / torch.linalg.norm(torch.flatten(b))).item()

    def train(self):
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

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            # Compute Jacobian and residuals
            J, r, JTr = compute_jacobian_efficient(
                self.w, self.X_full, self.L, self.B,
                self.f, self.g, self.ib_idx, L_mul, B_mul
            )

            # Compute Gramian G = J^T @ J
            G = J.T @ J

            # Add Levenberg-Marquardt damping
            n_params = G.shape[0]
            G_damped = G + self.damping * torch.eye(n_params, dtype=PRECISION, device=device_string)

            # Solve for update: delta = -(G + λI)^{-1} J^T r = -(G + λI)^{-1} JTr
            # Note: JTr is the gradient, so delta = -(G + λI)^{-1} JTr
            try:
                delta = -self.lr * torch.linalg.solve(G_damped, JTr)
            except Exception as e:
                print(f"Linear solve failed: {e}, using pseudo-inverse")
                delta = -self.lr * torch.linalg.lstsq(G_damped, JTr).solution

            # Update parameters
            params_flat = get_params_flat(self.w)
            new_params = params_flat + delta
            set_params_flat(self.w, new_params)

            # Compute loss and L2 error
            loss_value = 0.5 * torch.sum(r ** 2).item()

            if device_string == "cuda":
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - start

            # Compute L2 error
            with torch.no_grad():
                pred = self.w.forward(self.X_tilde)
                l2_error = self.compute_l2(pred, self.u_true[:self.ib_idx])

            # Log
            self.logged_results['training_losses'].append(loss_value)
            self.logged_results['training_l2_losses'].append(l2_error)
            self.logged_results['epochs_list'].append(epoch)
            self.logged_results['epoch_time'].append(epoch_time)
            self.logged_results['delta_norm'].append(torch.norm(delta).item())

            if isnan(loss_value) or loss_value > 500:
                print(f"Loss exploded: {loss_value}")
                return None

            if epoch % 5 == 0 or epoch <= 10:
                print(f"Epoch {epoch}: loss={loss_value:.4e}, L2={l2_error:.4e}, "
                      f"|delta|={torch.norm(delta).item():.4e}, time={epoch_time:.1f}s")

        final_time = time.perf_counter() - start
        final_l2 = self.logged_results['training_l2_losses'][-1]

        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"  Total time: {final_time:.2f}s")
        print(f"  Final L2 error: {final_l2:.4e}")
        print(f"  Target: Time < 100s, L2 ≤ 3.0e-02")
        print(f"{'='*60}")

        return dict(self.logged_results)


def load_mat_cupy(mat):
    return csr_matrix(mat, dtype=np.float64)


def run_experiment(epochs=50, layers=2, nodes=20, damping=1e-4, lr=1.0):
    """Run ENGD-DT-PINN experiment"""

    cupy.cuda.Device(0).use()

    # Load data (nonlinear Poisson, size=2236, order=2)
    order = 2
    size = 2236
    file_name = f"{order}_{size}"
    data_path = "/workspace/dt-pinn/nonlinear"

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

    ib_idx = X_i.shape[0] + X_b.shape[0]

    config = {
        'spatial_dim': 2,
        'precision': 'float64',
        'activation': 'tanh',
        'order': 2,
        'network_device': device_string,
        'layers': layers,
        'nodes': nodes,
        'epochs': epochs,
        'damping': damping,
        'lr': lr,
    }

    vars_dict = {
        'x_i': x_i, 'y_i': y_i,
        'x_b': x_b, 'y_b': y_b,
        'x_g': x_g, 'y_g': y_g,
        'ib_idx': ib_idx,
        'u_true': u_true,
        'L': L, 'B': B,
        'f': f, 'g': g,
    }

    trainer = ENGDDTPINNTrainer(config=config, **vars_dict)
    results = trainer.train()

    return results


if __name__ == "__main__":
    print("="*70)
    print("H25: ENGD-DT-PINN (Energy Natural Gradient Descent)")
    print("="*70)

    # Test with small network (same as best DT-PINN config)
    results = run_experiment(
        epochs=50,
        layers=2,
        nodes=20,
        damping=1e-4,
        lr=1.0
    )

    if results:
        os.makedirs("/workspace/dt-pinn/results/engd_dtpinn", exist_ok=True)
        with open("/workspace/dt-pinn/results/engd_dtpinn/experiment_results.json", "w") as f_out:
            json.dump(results, f_out, indent=2)
        print("\nResults saved to /workspace/dt-pinn/results/engd_dtpinn/experiment_results.json")
