"""
Hypothesis 1: Linearized Residual Loss (LRL-PINN)

Theory: By linearizing exp(u) around u_ref and updating u_ref periodically,
we decouple RBF-FD discretization error from nonlinearity amplification.

Original:  loss = MSE(L@u - f - exp(u))
LRL:       loss = MSE(L@u - exp(u_ref)*u - f_eff)
           where f_eff = f + exp(u_ref)*(1 - u_ref)

Expected: 2-10x improvement in L2 error with similar speedup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import optim
from torch.nn import MSELoss
import numpy as np
import cupy
import cupyx.scipy.sparse as cupy_sparse
from scipy.sparse import csr_matrix
from scipy.io import loadmat
import time
from collections import defaultdict
from math import isnan

from network import W

device_string = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

# Global sparse matrix transposes (for autograd)
L_t = None
B_t = None


class Cupy_mul_L(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, L):
        ctx.save_for_backward(u)
        u_cupy = cupy.asarray(u.detach())
        result = L.dot(u_cupy)
        return torch.as_tensor(result, device=u.device)

    @staticmethod
    def backward(ctx, grad_output):
        global L_t
        grad_cupy = cupy.asarray(grad_output.detach())
        grad_u = L_t.dot(grad_cupy)
        return torch.as_tensor(grad_u, device=grad_output.device), None


class Cupy_mul_B(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, B):
        ctx.save_for_backward(u)
        u_cupy = cupy.asarray(u.detach())
        result = B.dot(u_cupy)
        return torch.as_tensor(result, device=u.device)

    @staticmethod
    def backward(ctx, grad_output):
        global B_t
        grad_cupy = cupy.asarray(grad_output.detach())
        grad_u = B_t.dot(grad_cupy)
        return torch.as_tensor(grad_u, device=grad_output.device), None


def load_mat_cupy(mat):
    """Load scipy matrix and convert to cupy sparse"""
    scipy_csr = csr_matrix(mat, dtype=np.float64)
    return cupy_sparse.csr_matrix(scipy_csr)


class LRLDTPINNTrainer:
    """
    Linearized Residual Loss DT-PINN Trainer

    Key innovation: Replace exp(u) in loss with linearized form:
    ∇²u - exp(u_ref)*u = f + exp(u_ref)*(1 - u_ref)
    """

    def __init__(self, config, data_dict, u_ref_update_freq=50, damping=0.5, clip_u_ref=5.0):
        self.config = config
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.u_ref_update_freq = u_ref_update_freq
        self.damping = damping
        self.clip_u_ref = clip_u_ref

        # Unpack data
        for key, val in data_dict.items():
            setattr(self, key, val)

        self.logged_results = defaultdict(list)
        self.w = W(config)

        print(f"Network: {config['layers']} layers × {config['nodes']} nodes")
        print(f"LRL Parameters: u_ref_update_freq={u_ref_update_freq}, damping={damping}")

        # Setup points
        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        self.X_b = torch.hstack([self.x_b.clone(), self.y_b.clone()])

        self.optimizer = optim.LBFGS(self.w.parameters(), lr=self.lr)

        # Initialize u_ref to zeros (so exp(u_ref)=1 initially, like linear Poisson)
        self.u_ref = torch.zeros((self.ib_idx, 1), device=device,
                                  dtype=torch.float64 if config['precision'] == 'float64' else torch.float32)

    @staticmethod
    def compute_mse(a, b):
        return MSELoss()(torch.flatten(a), torch.flatten(b)).item()

    @staticmethod
    def compute_l2(a, b):
        diff = torch.subtract(torch.flatten(a).detach().cpu(), torch.flatten(b).detach().cpu())
        return (torch.linalg.norm(diff) / torch.linalg.norm(torch.flatten(b))).item()

    def train(self):
        global L_t, B_t
        epochs = self.epochs
        self.u_tilde = self.u[:self.ib_idx]

        # Initialize sparse matrices on GPU
        rand_vec = cupy.random.rand(self.L.shape[1], 2).astype(cupy.float64)
        self.L.dot(rand_vec)
        self.B.dot(rand_vec)

        L_t = self.L.transpose()
        B_t = self.B.transpose()

        rand_L_vec = cupy.random.rand(self.L.shape[0], 2).astype(cupy.float64)
        rand_B_vec = cupy.random.rand(self.B.shape[0], 2).astype(cupy.float64)
        L_t.dot(rand_L_vec)
        B_t.dot(rand_B_vec)

        L_mul = Cupy_mul_L.apply
        B_mul = Cupy_mul_B.apply

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for i in range(1, epochs + 1):
            # Update u_ref periodically with damping
            if i > 1 and i % self.u_ref_update_freq == 0:
                with torch.no_grad():
                    u_current = self.w.forward(self.X_full)[:self.ib_idx]
                    # Damped update
                    self.u_ref = self.damping * u_current + (1 - self.damping) * self.u_ref
                    # Clip to prevent exp overflow
                    self.u_ref = torch.clamp(self.u_ref, -self.clip_u_ref, self.clip_u_ref)

            def closure():
                self.optimizer.zero_grad()

                u_pred_full = self.w.forward(self.X_full)
                u_ib = u_pred_full[:self.ib_idx]

                # === LINEARIZED RESIDUAL LOSS ===
                # Original: ∇²u - f - exp(u) = 0
                # Linearized: ∇²u - exp(u_ref)*u = f + exp(u_ref)*(1 - u_ref)

                # Compute exp(u_ref) - detached to not track gradients
                exp_u_ref = torch.exp(self.u_ref.detach())

                # Effective RHS: f + exp(u_ref)*(1 - u_ref)
                f_eff = self.f[:self.ib_idx] + exp_u_ref * (1.0 - self.u_ref.detach())

                # Laplacian via RBF-FD
                lap_u = L_mul(u_pred_full, self.L)

                # Diagonal term: exp(u_ref) * u
                diag_term = exp_u_ref * u_ib

                # Linearized PDE residual: (L@u - exp(u_ref)*u) - f_eff
                f_pred = lap_u[:self.ib_idx] - diag_term - f_eff

                # Boundary loss (unchanged)
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

            # Logging
            training_pred = self.w.forward(self.X_tilde)
            test_pred = self.w.forward(self.test_X_tilde)

            training_mse = self.compute_mse(training_pred, self.u_true)
            test_mse = self.compute_mse(test_pred, self.test_u_true)
            training_l2 = self.compute_l2(training_pred, self.u_true)
            test_l2 = self.compute_l2(test_pred, self.test_u_true)

            self.logged_results['training_losses'].append(loss_value)
            self.logged_results['training_mse_losses'].append(training_mse)
            self.logged_results['training_l2_losses'].append(training_l2)
            self.logged_results['test_mse_losses'].append(test_mse)
            self.logged_results['test_l2_losses'].append(test_l2)
            self.logged_results['epochs_list'].append(i)
            self.logged_results['epoch_time'].append(epoch_time)

            # Check for explosion
            if i > 30 and (isnan(loss_value) or loss_value > 500):
                print(f"Loss exploded to: {loss_value}")
                return False

            if i % 100 == 0:
                u_ref_norm = torch.linalg.norm(self.u_ref).item()
                exp_u_ref_max = torch.exp(self.u_ref).max().item()
                print(f'Epoch {i}/{self.epochs}, Loss: {loss_value:.6f}, '
                      f'Train L2: {training_l2:.6e}, Test L2: {test_l2:.6e}, '
                      f'||u_ref||: {u_ref_norm:.4f}, max(exp(u_ref)): {exp_u_ref_max:.2f}, '
                      f'Time: {epoch_time:.2f}s')

        return dict(self.logged_results)


def load_nonlinear_data(size, precision_str='float64'):
    """Load nonlinear Poisson dataset"""
    precision = torch.float64 if precision_str == 'float64' else torch.float32
    file_name = f"2_{size}"
    test_name = "2_21748_test"

    # Training data
    X_i = torch.tensor(loadmat(f"nonlinear/files_{file_name}/Xi.mat")["Xi"],
                      dtype=precision, requires_grad=True).to(device)
    X_b = torch.tensor(loadmat(f"nonlinear/files_{file_name}/Xb.mat")["Xb"],
                      dtype=precision, requires_grad=True).to(device)
    X_g = torch.tensor(loadmat(f"nonlinear/files_{file_name}/Xg.mat")["X_g"],
                      dtype=precision, requires_grad=True).to(device)
    n = torch.tensor(loadmat(f"nonlinear/files_{file_name}/n.mat")["n"],
                    dtype=precision, requires_grad=True).to(device)
    u_true = torch.tensor(loadmat(f"nonlinear/files_{file_name}/u.mat")["u"], dtype=precision).to(device)
    f = torch.tensor(loadmat(f"nonlinear/files_{file_name}/f.mat")["f"],
                    dtype=precision, requires_grad=True).to(device)
    g = torch.tensor(loadmat(f"nonlinear/files_{file_name}/g.mat")["g"],
                    dtype=precision, requires_grad=True).to(device)
    alpha = torch.tensor(loadmat(f"nonlinear/files_{file_name}/alpha.mat")["Neucoeff"],
                        dtype=precision, requires_grad=True).to(device)
    beta = torch.tensor(loadmat(f"nonlinear/files_{file_name}/beta.mat")["Dircoeff"],
                       dtype=precision, requires_grad=True).to(device)

    # DT-PINN matrices
    L = load_mat_cupy(loadmat(f"nonlinear/files_{file_name}/L1.mat")["L1"])
    B = load_mat_cupy(loadmat(f"nonlinear/files_{file_name}/B1.mat")["B1"])

    # Test data (use linear test set since structure is same)
    X_i_test = torch.tensor(loadmat(f"scai/files_{test_name}/Xi.mat")["Xi"],
                           dtype=precision, requires_grad=True).to(device)
    X_b_test = torch.tensor(loadmat(f"scai/files_{test_name}/Xb.mat")["Xb"],
                           dtype=precision, requires_grad=True).to(device)
    test_u_true = torch.tensor(loadmat(f"scai/files_{test_name}/u.mat")["u"], dtype=precision).to(device)

    test_x_i = X_i_test[:, 0].unsqueeze(dim=1)
    test_y_i = X_i_test[:, 1].unsqueeze(dim=1)
    test_x_b = X_b_test[:, 0].unsqueeze(dim=1)
    test_y_b = X_b_test[:, 1].unsqueeze(dim=1)
    test_x_tilde = torch.vstack([test_x_i, test_x_b])
    test_y_tilde = torch.vstack([test_y_i, test_y_b])
    test_X_tilde = torch.hstack([test_x_tilde, test_y_tilde])

    # Separate spatial dimensions
    x_i = X_i[:, 0].unsqueeze(dim=1)
    y_i = X_i[:, 1].unsqueeze(dim=1)
    x_b = X_b[:, 0].unsqueeze(dim=1)
    y_b = X_b[:, 1].unsqueeze(dim=1)
    x_g = X_g[:, 0].unsqueeze(dim=1)
    y_g = X_g[:, 1].unsqueeze(dim=1)

    ib_idx = X_i.shape[0] + X_b.shape[0]

    return {
        'x_i': x_i, 'y_i': y_i, 'x_b': x_b, 'y_b': y_b, 'x_g': x_g, 'y_g': y_g,
        'n': n, 'u': u_true, 'u_true': u_true[:ib_idx],
        'f': f, 'g': g, 'alpha': alpha, 'beta': beta,
        'L': L, 'B': B,
        'test_X_tilde': test_X_tilde,
        'test_u_true': test_u_true[:X_i_test.shape[0] + X_b_test.shape[0]],
        'ib_idx': ib_idx
    }


def main():
    """Run LRL-PINN experiment"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset_size', type=int, default=2236)
    parser.add_argument('--u_ref_update_freq', type=int, default=50,
                        help='How often to update u_ref (default: 50)')
    parser.add_argument('--damping', type=float, default=0.5,
                        help='Damping for u_ref update (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.04)
    args = parser.parse_args()

    print("=" * 70)
    print("LRL-PINN (Linearized Residual Loss) Experiment")
    print("=" * 70)
    print(f"Dataset size: {args.dataset_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"u_ref update frequency: {args.u_ref_update_freq}")
    print(f"Damping: {args.damping}")
    print("=" * 70)

    # Load data
    data_dict = load_nonlinear_data(args.dataset_size)

    # Config
    config = {
        'precision': 'float64',
        'activation': 'tanh',
        'layers': 4,
        'nodes': 100,
        'spatial_dim': 2,
        'network_device': device_string,
        'lr': args.lr,
        'epochs': args.epochs
    }

    trainer = LRLDTPINNTrainer(
        config=config,
        data_dict=data_dict,
        u_ref_update_freq=args.u_ref_update_freq,
        damping=args.damping
    )

    results = trainer.train()

    if results is False:
        print("\nTraining failed (loss exploded)")
        return

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Final Train L2: {results['training_l2_losses'][-1]:.6e}")
    print(f"Final Test L2: {results['test_l2_losses'][-1]:.6e}")
    print(f"Total Time: {results['epoch_time'][-1]:.2f}s")

    # Compare to baseline
    baseline_l2 = 6.76e-03
    baseline_time = 216.07
    vanilla_l2 = 7.47e-05

    improvement = baseline_l2 / results['training_l2_losses'][-1]
    speedup = baseline_time / results['epoch_time'][-1]
    vs_vanilla = results['training_l2_losses'][-1] / vanilla_l2

    print(f"\nComparison:")
    print(f"  vs Baseline DT-PINN: {improvement:.1f}x better L2, {speedup:.2f}x speedup")
    print(f"  vs Vanilla PINN: {vs_vanilla:.1f}x worse L2")
    print(f"  Target L2: 1.49e-04 (need to be ≤{1.49e-04/results['training_l2_losses'][-1]:.1f}x better)")


if __name__ == '__main__':
    main()
