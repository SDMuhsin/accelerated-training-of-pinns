"""
H26: Sparse-GN-DT-PINN (Sparse Gauss-Newton for DT-PINN)

HYPOTHESIS: Exploit the sparse structure of DT-PINN's operators to compute
an efficient approximate Gauss-Newton update.

KEY INSIGHT: For DT-PINN with nonlinear PDEs, the residual Jacobian has structure:
  J = M @ J_net where M = L - diag(exp(u)) is SPARSE

This means:
  G = J^T @ J = J_net^T @ M^T @ M @ J_net

We can compute M^T @ M efficiently (sparse × sparse = sparse), then use
this structure for efficient Gramian computation.

ALTERNATIVE APPROACH: Instead of full GN, use:
1. Compute gradient efficiently (one backward pass)
2. Scale by diagonal Fisher approximation
3. Use momentum for stability

This is essentially AdaGrad/Adam with better theoretical grounding for PINNs.

Target: Time < 100s, L2 ≤ 3.0e-02
"""

import json
from collections import defaultdict
import os
import sys
from math import isnan
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

torch.manual_seed(0)

if torch.cuda.is_available():
    pytorch_device = torch.device('cuda')
    device_string = "cuda"
    torch.cuda.manual_seed_all(0)
else:
    pytorch_device = torch.device('cpu')
    device_string = "cpu"

PRECISION = torch.float64
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


class DiagonalNGDOptimizer:
    """Diagonal Natural Gradient Descent optimizer.

    Uses running average of squared gradients (like Adam) but interprets it
    as a diagonal Fisher approximation.
    """

    def __init__(self, params, lr=0.1, beta=0.99, eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
        for p in self.params:
            self.state[p] = {'v': torch.zeros_like(p)}

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * p.data

            # Update running average of squared gradient (Fisher diagonal)
            self.state[p]['v'] = self.beta * self.state[p]['v'] + (1 - self.beta) * grad ** 2

            # Natural gradient update: scale by inverse sqrt of Fisher diagonal
            denom = torch.sqrt(self.state[p]['v']) + self.eps
            p.data -= self.lr * grad / denom


class SparseGNDTPINNTrainer:
    """Sparse Gauss-Newton DT-PINN Trainer using Diagonal Fisher"""

    def __init__(self, config, **kwargs):
        self.lr = config.get('lr', 0.1)
        self.epochs = config['epochs']
        self.beta = config.get('beta', 0.99)

        self.__dict__.update(kwargs)
        self.logged_results = defaultdict(list)

        self.w = W(config)
        n_params = sum(p.numel() for p in self.w.parameters())
        print(f"Network: {config['layers']} layers, {config['nodes']} nodes, {n_params} params")

        # Points setup
        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

        # Use custom diagonal NGD optimizer
        self.optimizer = DiagonalNGDOptimizer(
            self.w.parameters(),
            lr=self.lr,
            beta=self.beta
        )

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
            self.optimizer.zero_grad()

            u_pred_full = self.w.forward(self.X_full)

            # PDE residual
            lap_u = L_mul(u_pred_full, self.L)
            f_pred = lap_u[:self.ib_idx] - self.f[:self.ib_idx] - torch.exp(u_pred_full[:self.ib_idx])

            # BC residual
            boundary_loss_term = B_mul(u_pred_full, self.B) - self.g

            # MSE loss (same as original DT-PINN)
            l2 = torch.mean(torch.square(torch.flatten(f_pred)))
            l3 = torch.mean(torch.square(torch.flatten(boundary_loss_term)))
            train_loss = l2 + l3

            train_loss.backward()
            self.optimizer.step()

            loss_value = train_loss.item()

            if device_string == "cuda":
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - start

            # Compute L2 error
            with torch.no_grad():
                pred = self.w.forward(self.X_tilde)
                l2_error = self.compute_l2(pred, self.u_true[:self.ib_idx])

            self.logged_results['training_losses'].append(loss_value)
            self.logged_results['training_l2_losses'].append(l2_error)
            self.logged_results['epochs_list'].append(epoch)
            self.logged_results['epoch_time'].append(epoch_time)

            if isnan(loss_value) or loss_value > 500:
                print(f"Loss exploded: {loss_value}")
                return None

            if epoch % 100 == 0 or epoch <= 20:
                print(f"Epoch {epoch}: loss={loss_value:.4e}, L2={l2_error:.4e}, time={epoch_time:.2f}s")

        final_time = time.perf_counter() - start
        final_l2 = self.logged_results['training_l2_losses'][-1]

        print(f"\n{'='*60}")
        print(f"RESULTS: Time={final_time:.2f}s, L2={final_l2:.4e}")
        print(f"Target: Time < 100s, L2 ≤ 3.0e-02")
        print(f"{'='*60}")

        return dict(self.logged_results)


def load_mat_cupy(mat):
    return csr_matrix(mat, dtype=np.float64)


def run_experiment(epochs=500, layers=2, nodes=20, lr=0.1, beta=0.99):
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
        'lr': lr,
        'beta': beta,
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

    trainer = SparseGNDTPINNTrainer(config=config, **vars_dict)
    return trainer.train()


if __name__ == "__main__":
    print("="*70)
    print("H26: Sparse-GN-DT-PINN (Diagonal Fisher Approximation)")
    print("="*70)

    results = run_experiment(
        epochs=1000,
        layers=2,
        nodes=20,
        lr=0.1,
        beta=0.99
    )

    if results:
        os.makedirs("/workspace/dt-pinn/results/sparse_gn_dtpinn", exist_ok=True)
        with open("/workspace/dt-pinn/results/sparse_gn_dtpinn/results.json", "w") as f_out:
            json.dump(results, f_out, indent=2)
