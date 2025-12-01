"""
H24: NA-DT-PINN (Nonlinearity-Annealed DT-PINN) - Version 2

Revised approach: Instead of changing the PDE (which changes the solution),
we use a SOFT START for the exp(u) term in LOSS WEIGHTING only.

The PDE is always: ∇²u = f + exp(u)
But during training we weight: loss = w_pde * ||∇²u - f - exp(u)||² + w_bc * ||BC||²

Early epochs: Lower weight on full residual, allowing network to find rough structure
Later epochs: Full residual weight for accuracy

Alternative interpretation: Gradient clipping / loss scaling during early training.

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


class NADTPINNTrainer:
    """Nonlinearity-Annealed DT-PINN Trainer"""

    def __init__(self, config, **kwargs):
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.lambda_schedule = config.get('lambda_schedule', [0.0, 0.5, 1.0])
        self.epochs_per_phase = config.get('epochs_per_phase', None)

        self.__dict__.update(kwargs)
        self.logged_results = defaultdict(list)

        # Network
        self.w = W(config)
        print(f"Network: {config['layers']} layers, {config['nodes']} nodes")
        print(f"Lambda schedule: {self.lambda_schedule}")

        # Points setup
        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

        self.X_b = torch.hstack([self.x_b.clone(), self.y_b.clone()])

        # Optimizer
        self.optimizer = optim.LBFGS(self.w.parameters(), lr=self.lr)

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

        # Determine epochs per phase
        n_phases = len(self.lambda_schedule)
        if self.epochs_per_phase is None:
            self.epochs_per_phase = [self.epochs // n_phases] * n_phases
            # Add remaining epochs to last phase
            self.epochs_per_phase[-1] += self.epochs % n_phases

        print(f"Epochs per phase: {self.epochs_per_phase}")

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        total_epoch = 0
        for phase_idx, current_lambda in enumerate(self.lambda_schedule):
            phase_epochs = self.epochs_per_phase[phase_idx]
            print(f"\n{'='*60}")
            print(f"Phase {phase_idx + 1}/{n_phases}: λ = {current_lambda}, epochs = {phase_epochs}")
            print(f"{'='*60}")

            for local_epoch in range(1, phase_epochs + 1):
                total_epoch += 1

                def closure():
                    self.optimizer.zero_grad()

                    u_pred_full = self.w.forward(self.X_full)

                    # The stored f is such that for true u*: L@u* = f + exp(u*)
                    # We train with: L@u = f + λ·exp(u)
                    # When λ=1, this matches the original problem
                    # When λ<1, we're solving a "softer" nonlinear problem

                    # Compute Laplacian (on all points)
                    lap_u = L_mul(u_pred_full, self.L)

                    # PDE residual only on interior+boundary points (slice to ib_idx)
                    f_pred = lap_u[:self.ib_idx] - self.f[:self.ib_idx] - current_lambda * torch.exp(u_pred_full[:self.ib_idx])

                    boundary_loss_term = B_mul(u_pred_full, self.B) - self.g

                    l2 = torch.mean(torch.square(torch.flatten(f_pred)))
                    l3 = torch.mean(torch.square(torch.flatten(boundary_loss_term)))

                    train_loss = l2 + l3
                    train_loss.backward(retain_graph=True)
                    return train_loss.item()

                loss_value = self.optimizer.step(closure)

                if device_string == "cuda":
                    torch.cuda.synchronize()
                epoch_time = time.perf_counter() - start

                # Compute L2 error (against true nonlinear solution)
                # u_true is for interior+boundary points only (ib_idx points)
                with torch.no_grad():
                    pred = self.w.forward(self.X_tilde)
                    l2_error = self.compute_l2(pred, self.u_true[:self.ib_idx])

                # Log
                self.logged_results['training_losses'].append(loss_value)
                self.logged_results['training_l2_losses'].append(l2_error)
                self.logged_results['epochs_list'].append(total_epoch)
                self.logged_results['epoch_time'].append(epoch_time)
                self.logged_results['lambda_value'].append(current_lambda)

                if isnan(loss_value) or loss_value > 500:
                    print(f"Loss exploded: {loss_value}")
                    return None

                if local_epoch % 50 == 0 or local_epoch == phase_epochs:
                    print(f"  Epoch {total_epoch} (phase {local_epoch}/{phase_epochs}): "
                          f"loss={loss_value:.4e}, L2={l2_error:.4e}, time={epoch_time:.1f}s")

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


def run_experiment(lambda_schedule, epochs_per_phase, total_epochs=500):
    """Run NA-DT-PINN experiment with given lambda schedule"""

    # cupy setup
    cupy.cuda.Device(0).use()

    # Load data (nonlinear Poisson, size=2236, order=2)
    order = 2
    size = 2236
    file_name = f"{order}_{size}"
    test_name = f"{order}_21748_test"
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
        'layers': 4,
        'nodes': 50,
        'epochs': total_epochs,
        'optimizer': 'lbfgs',
        'lr': 0.04,  # Standard DT-PINN lr
        'lambda_schedule': lambda_schedule,
        'epochs_per_phase': epochs_per_phase,
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

    trainer = NADTPINNTrainer(config=config, **vars_dict)
    results = trainer.train()

    return results


if __name__ == "__main__":
    print("="*70)
    print("H24: NA-DT-PINN (Nonlinearity-Annealed DT-PINN)")
    print("="*70)

    # Test different annealing schedules
    # NOTE: λ=0 doesn't work because f was computed for the nonlinear problem
    # Start from small λ instead
    schedules = [
        # (lambda_schedule, epochs_per_phase, description)
        ([0.1, 0.5, 1.0], [100, 150, 250], "3-phase: 0.1→0.5→1.0"),
        ([0.25, 0.5, 0.75, 1.0], [75, 100, 125, 200], "4-phase: 0.25→1.0"),
        ([0.5, 1.0], [150, 350], "2-phase: 0.5→1.0"),
    ]

    # Run first schedule as primary test
    lambda_schedule, epochs_per_phase, desc = schedules[0]
    total_epochs = sum(epochs_per_phase)

    print(f"\nTesting: {desc}")
    print(f"Total epochs: {total_epochs}")

    results = run_experiment(lambda_schedule, epochs_per_phase, total_epochs)

    if results:
        # Save results
        os.makedirs("/workspace/dt-pinn/results/na_dtpinn", exist_ok=True)
        with open("/workspace/dt-pinn/results/na_dtpinn/experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nResults saved to ../results/na_dtpinn/experiment_results.json")
