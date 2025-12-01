"""
Hypothesis 9: Hybrid-PINN (Warm-Start + Fine-Tune)

Theory: Use DT-PINN (fast RBF-FD) for initial training to get close to the solution,
then switch to vanilla PINN (exact autodiff) for final fine-tuning.

Key insight:
- DT-PINN is fast but has discretization error ceiling
- Vanilla PINN is slow but converges to accurate solution
- Hybrid: Fast convergence (DT-PINN) + Accurate final solution (Vanilla)

Expected: Near-vanilla accuracy with significant speedup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import optim
from torch.autograd import grad
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

# Global sparse matrix transposes
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
    scipy_csr = csr_matrix(mat, dtype=np.float64)
    return cupy_sparse.csr_matrix(scipy_csr)


class HybridPINNTrainer:
    """
    Hybrid PINN Trainer

    Two-phase training:
    1. DT-PINN phase: Fast training with RBF-FD (N1 epochs)
    2. Vanilla phase: Fine-tuning with exact autodiff (N2 epochs)

    Total time should be less than pure vanilla PINN.
    """

    def __init__(self, config, data_dict, dtpinn_epochs=400, vanilla_epochs=100):
        self.config = config
        self.lr = config['lr']
        self.dtpinn_epochs = dtpinn_epochs
        self.vanilla_epochs = vanilla_epochs

        # Unpack data
        for key, val in data_dict.items():
            setattr(self, key, val)

        self.logged_results = defaultdict(list)
        self.w = W(config)

        print(f"Network: {config['layers']} layers x {config['nodes']} nodes")
        print(f"DT-PINN epochs: {dtpinn_epochs}, Vanilla epochs: {vanilla_epochs}")

        # Setup points for DT-PINN
        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        self.X_b = torch.hstack([self.x_b.clone(), self.y_b.clone()])

        # Setup points for Vanilla (with requires_grad)
        self.x_interior_v = self.x_i.clone().detach().requires_grad_(True)
        self.y_interior_v = self.y_i.clone().detach().requires_grad_(True)
        self.X_interior_v = torch.hstack([self.x_interior_v, self.y_interior_v])

        self.x_b_v = self.x_b.clone().detach().requires_grad_(True)
        self.y_b_v = self.y_b.clone().detach().requires_grad_(True)
        self.X_b_v = torch.hstack([self.x_b_v, self.y_b_v])

    @staticmethod
    def compute_mse(a, b):
        return MSELoss()(torch.flatten(a), torch.flatten(b)).item()

    @staticmethod
    def compute_l2(a, b):
        diff = torch.subtract(torch.flatten(a).detach().cpu(), torch.flatten(b).detach().cpu())
        return (torch.linalg.norm(diff) / torch.linalg.norm(torch.flatten(b))).item()

    def train_phase1_dtpinn(self):
        """Phase 1: Train with RBF-FD (fast)"""
        global L_t, B_t

        # Initialize sparse matrices
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

        optimizer = optim.LBFGS(self.w.parameters(), lr=self.lr)

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        print("\n=== PHASE 1: DT-PINN Training (RBF-FD) ===")

        for i in range(1, self.dtpinn_epochs + 1):
            def closure():
                optimizer.zero_grad()

                u_pred_full = self.w.forward(self.X_full)

                # Nonlinear Poisson
                lap_u = L_mul(u_pred_full, self.L)
                f_pred = lap_u[:self.ib_idx] - self.f[:self.ib_idx] - torch.exp(u_pred_full[:self.ib_idx])

                boundary_loss_term = B_mul(u_pred_full, self.B) - self.g

                l2 = torch.mean(torch.square(torch.flatten(f_pred)))
                l3 = torch.mean(torch.square(torch.flatten(boundary_loss_term)))

                train_loss = l2 + l3
                train_loss.backward(retain_graph=True)
                return train_loss.item()

            loss_value = optimizer.step(closure)

            if device_string == "cuda":
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - start

            if i % 100 == 0:
                training_pred = self.w.forward(self.X_tilde)
                training_l2 = self.compute_l2(training_pred, self.u_true)
                print(f'Epoch {i}/{self.dtpinn_epochs}, Loss: {loss_value:.6f}, '
                      f'L2: {training_l2:.6e}, Time: {epoch_time:.2f}s')

            if i > 30 and (isnan(loss_value) or loss_value > 500):
                print(f"Loss exploded to: {loss_value}")
                return False, epoch_time

        phase1_time = time.perf_counter() - start
        training_pred = self.w.forward(self.X_tilde)
        training_l2 = self.compute_l2(training_pred, self.u_true)
        print(f"Phase 1 complete. Time: {phase1_time:.2f}s, L2: {training_l2:.6e}")

        return True, phase1_time

    def train_phase2_vanilla(self):
        """Phase 2: Fine-tune with exact autodiff"""

        optimizer = optim.LBFGS(self.w.parameters(), lr=self.lr * 0.5)

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        print("\n=== PHASE 2: Vanilla PINN Fine-tuning (Autodiff) ===")

        f_interior = self.f[:self.b_starts]

        for i in range(1, self.vanilla_epochs + 1):
            def closure():
                optimizer.zero_grad()

                # Interior PDE residual with autodiff
                u_pred_interior = self.w.forward(self.X_interior_v)

                u_x = grad(u_pred_interior, self.x_interior_v,
                           grad_outputs=torch.ones_like(u_pred_interior),
                           create_graph=True, retain_graph=True)[0]
                u_xx = grad(u_x, self.x_interior_v,
                            grad_outputs=torch.ones_like(u_x),
                            create_graph=True, retain_graph=True)[0]

                u_y = grad(u_pred_interior, self.y_interior_v,
                           grad_outputs=torch.ones_like(u_pred_interior),
                           create_graph=True, retain_graph=True)[0]
                u_yy = grad(u_y, self.y_interior_v,
                            grad_outputs=torch.ones_like(u_y),
                            create_graph=True, retain_graph=True)[0]

                # Nonlinear Poisson: nabla^2 u - f - exp(u) = 0
                f_pred = (u_xx + u_yy) - f_interior - torch.exp(u_pred_interior)

                # Boundary residual with autodiff
                boundary_pred = self.w.forward(self.X_b_v)
                l2_w_x = grad(boundary_pred, self.x_b_v,
                              grad_outputs=torch.ones_like(boundary_pred),
                              create_graph=True, retain_graph=True)[0]
                l2_w_y = grad(boundary_pred, self.y_b_v,
                              grad_outputs=torch.ones_like(boundary_pred),
                              create_graph=True, retain_graph=True)[0]

                w_xy = torch.hstack([l2_w_x, l2_w_y])
                gradient_n = torch.multiply(self.n, w_xy).sum(dim=1).unsqueeze(dim=1)
                boundary_loss_term = torch.multiply(self.alpha, gradient_n) + \
                                     torch.multiply(self.beta, boundary_pred) - self.g

                l2 = torch.mean(torch.square(torch.flatten(f_pred)))
                l3 = torch.mean(torch.square(torch.flatten(boundary_loss_term)))

                train_loss = l2 + l3
                train_loss.backward(retain_graph=True)
                return train_loss.item()

            loss_value = optimizer.step(closure)

            if device_string == "cuda":
                torch.cuda.synchronize()
            epoch_time = time.perf_counter() - start

            if i % 25 == 0:
                training_pred = self.w.forward(self.X_tilde)
                training_l2 = self.compute_l2(training_pred, self.u_true)
                print(f'Epoch {i}/{self.vanilla_epochs}, Loss: {loss_value:.6f}, '
                      f'L2: {training_l2:.6e}, Time: {epoch_time:.2f}s')

            if i > 30 and (isnan(loss_value) or loss_value > 500):
                print(f"Loss exploded to: {loss_value}")
                return False, epoch_time

        phase2_time = time.perf_counter() - start
        training_pred = self.w.forward(self.X_tilde)
        training_l2 = self.compute_l2(training_pred, self.u_true)
        test_pred = self.w.forward(self.test_X_tilde)
        test_l2 = self.compute_l2(test_pred, self.test_u_true)
        print(f"Phase 2 complete. Time: {phase2_time:.2f}s, Train L2: {training_l2:.6e}, Test L2: {test_l2:.6e}")

        return True, phase2_time, training_l2, test_l2

    def train(self):
        """Run both phases"""
        # Phase 1
        success, phase1_time = self.train_phase1_dtpinn()
        if not success:
            return None

        # Record post-DT-PINN L2
        training_pred = self.w.forward(self.X_tilde)
        post_dtpinn_l2 = self.compute_l2(training_pred, self.u_true)

        # Phase 2
        success, phase2_time, final_train_l2, final_test_l2 = self.train_phase2_vanilla()
        if not success:
            return None

        total_time = phase1_time + phase2_time

        return {
            'phase1_time': phase1_time,
            'phase2_time': phase2_time,
            'total_time': total_time,
            'post_dtpinn_l2': post_dtpinn_l2,
            'final_train_l2': final_train_l2,
            'final_test_l2': final_test_l2
        }


def load_nonlinear_data(size, precision_str='float64'):
    """Load nonlinear Poisson dataset"""
    precision = torch.float64 if precision_str == 'float64' else torch.float32
    file_name = f"2_{size}"
    test_name = "2_21748_test"

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

    L = load_mat_cupy(loadmat(f"nonlinear/files_{file_name}/L1.mat")["L1"])
    B = load_mat_cupy(loadmat(f"nonlinear/files_{file_name}/B1.mat")["B1"])

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

    x_i = X_i[:, 0].unsqueeze(dim=1)
    y_i = X_i[:, 1].unsqueeze(dim=1)
    x_b = X_b[:, 0].unsqueeze(dim=1)
    y_b = X_b[:, 1].unsqueeze(dim=1)
    x_g = X_g[:, 0].unsqueeze(dim=1)
    y_g = X_g[:, 1].unsqueeze(dim=1)

    ib_idx = X_i.shape[0] + X_b.shape[0]
    b_starts = X_i.shape[0]

    return {
        'x_i': x_i, 'y_i': y_i, 'x_b': x_b, 'y_b': y_b, 'x_g': x_g, 'y_g': y_g,
        'n': n, 'u': u_true, 'u_true': u_true[:ib_idx],
        'f': f, 'g': g, 'alpha': alpha, 'beta': beta,
        'L': L, 'B': B,
        'test_X_tilde': test_X_tilde,
        'test_u_true': test_u_true[:X_i_test.shape[0] + X_b_test.shape[0]],
        'ib_idx': ib_idx,
        'b_starts': b_starts
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtpinn_epochs', type=int, default=400)
    parser.add_argument('--vanilla_epochs', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=2236)
    parser.add_argument('--lr', type=float, default=0.04)
    args = parser.parse_args()

    print("=" * 70)
    print("Hybrid-PINN (DT-PINN Warm-Start + Vanilla Fine-Tune) Experiment")
    print("=" * 70)
    print(f"Dataset size: {args.dataset_size}")
    print(f"DT-PINN epochs: {args.dtpinn_epochs}")
    print(f"Vanilla epochs: {args.vanilla_epochs}")
    print(f"Learning rate: {args.lr}")
    print("=" * 70)

    data_dict = load_nonlinear_data(args.dataset_size)

    config = {
        'precision': 'float64',
        'activation': 'tanh',
        'layers': 4,
        'nodes': 100,
        'spatial_dim': 2,
        'network_device': device_string,
        'lr': args.lr,
        'epochs': args.dtpinn_epochs + args.vanilla_epochs
    }

    trainer = HybridPINNTrainer(
        config=config,
        data_dict=data_dict,
        dtpinn_epochs=args.dtpinn_epochs,
        vanilla_epochs=args.vanilla_epochs
    )

    results = trainer.train()

    if results is None:
        print("\nTraining failed")
        return

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Phase 1 (DT-PINN) Time: {results['phase1_time']:.2f}s")
    print(f"Phase 2 (Vanilla) Time: {results['phase2_time']:.2f}s")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"\nPost DT-PINN L2: {results['post_dtpinn_l2']:.6e}")
    print(f"Final Train L2: {results['final_train_l2']:.6e}")
    print(f"Final Test L2: {results['final_test_l2']:.6e}")

    # Baselines
    baseline_l2 = 6.76e-03
    baseline_time = 216.07
    vanilla_l2 = 7.47e-05
    vanilla_time = 304.59
    target_l2 = 1.49e-04

    improvement = baseline_l2 / results['final_train_l2']
    speedup_vs_vanilla = vanilla_time / results['total_time']
    vs_vanilla = results['final_train_l2'] / vanilla_l2

    print(f"\nComparison:")
    print(f"  vs Baseline DT-PINN: {improvement:.2f}x L2 improvement")
    print(f"  vs Vanilla PINN: {speedup_vs_vanilla:.2f}x speedup, {vs_vanilla:.1f}x {'worse' if vs_vanilla > 1 else 'better'} L2")
    print(f"  Pure Vanilla time: {vanilla_time:.2f}s, Our time: {results['total_time']:.2f}s")

    if results['final_train_l2'] <= target_l2:
        print(f"\n TARGET ACHIEVED! L2={results['final_train_l2']:.6e} <= {target_l2:.6e}")
    elif results['final_train_l2'] < baseline_l2:
        gap = results['final_train_l2'] / target_l2
        print(f"\n IMPROVED over baseline! {gap:.1f}x from target.")
    else:
        print(f"\n No improvement over baseline.")


if __name__ == '__main__':
    main()
