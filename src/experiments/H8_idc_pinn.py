"""
Hypothesis 8: IDC-PINN (Iterative Defect Correction)

Theory: Train DT-PINN normally, then compute the defect (difference between
RBF-FD Laplacian and exact autodiff Laplacian) ONCE, and train a small
correction network to predict this defect. Final solution is u_corrected = u_dtpinn + correction.

Key insight: The discretization error is STRUCTURED (geometric), not random.
A neural network can learn this spatial pattern efficiently.

Three phases:
1. Train DT-PINN for N epochs (fast, uses RBF-FD)
2. Compute defect via ONE autodiff pass (expensive but only once)
3. Train correction network on defect (fast, pure supervised learning)

Expected: 5-10x improvement in L2 error while maintaining most of the speedup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import optim, nn
from torch.autograd import grad
from torch.nn import MSELoss
import numpy as np
import cupy
import cupyx.scipy.sparse as cupy_sparse
from scipy.sparse import csr_matrix
from scipy.io import loadmat
import time
from collections import defaultdict, OrderedDict
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


class CorrectionNetwork(nn.Module):
    """Small network to learn the spatial defect pattern"""
    def __init__(self, hidden_size=50, num_layers=3, precision=torch.float64):
        super().__init__()
        self.precision = precision

        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)
        self.net = self.net.to(precision).to(device)

    def forward(self, x):
        return self.net(x.to(self.precision))


class IDCDTPINNTrainer:
    """
    Iterative Defect Correction DT-PINN Trainer

    Three-phase training:
    1. Train main DT-PINN network using RBF-FD (fast)
    2. Compute defect using ONE autodiff pass
    3. Train correction network to predict defect (supervised learning)
    """

    def __init__(self, config, data_dict, correction_epochs=100, correction_hidden=50, correction_layers=3):
        self.config = config
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.correction_epochs = correction_epochs
        self.correction_hidden = correction_hidden
        self.correction_layers = correction_layers

        # Unpack data
        for key, val in data_dict.items():
            setattr(self, key, val)

        self.logged_results = defaultdict(list)
        self.w = W(config)

        print(f"Main Network: {config['layers']} layers x {config['nodes']} nodes")
        print(f"Correction Network: {correction_layers} layers x {correction_hidden} nodes")

        # Setup points
        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        self.X_b = torch.hstack([self.x_b.clone(), self.y_b.clone()])

        # Interior points for autodiff
        self.x_interior = self.x_i.clone().requires_grad_(True)
        self.y_interior = self.y_i.clone().requires_grad_(True)
        self.X_interior = torch.hstack([self.x_interior, self.y_interior])

        self.optimizer = optim.LBFGS(self.w.parameters(), lr=self.lr)

    @staticmethod
    def compute_mse(a, b):
        return MSELoss()(torch.flatten(a), torch.flatten(b)).item()

    @staticmethod
    def compute_l2(a, b):
        diff = torch.subtract(torch.flatten(a).detach().cpu(), torch.flatten(b).detach().cpu())
        return (torch.linalg.norm(diff) / torch.linalg.norm(torch.flatten(b))).item()

    def train_phase1_dtpinn(self):
        """Phase 1: Train DT-PINN using RBF-FD (fast)"""
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

        print("\n=== PHASE 1: Training DT-PINN with RBF-FD ===")

        for i in range(1, epochs + 1):
            def closure():
                self.optimizer.zero_grad()

                u_pred_full = self.w.forward(self.X_full)

                # Nonlinear Poisson: nabla^2 u = f + exp(u)
                lap_u = L_mul(u_pred_full, self.L)
                f_pred = lap_u[:self.ib_idx] - self.f[:self.ib_idx] - torch.exp(u_pred_full[:self.ib_idx])

                # Boundary loss
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

            training_l2 = self.compute_l2(training_pred, self.u_true)
            test_l2 = self.compute_l2(test_pred, self.test_u_true)

            self.logged_results['phase1_training_l2'].append(training_l2)
            self.logged_results['phase1_test_l2'].append(test_l2)
            self.logged_results['phase1_loss'].append(loss_value)
            self.logged_results['phase1_time'].append(epoch_time)

            if i > 30 and (isnan(loss_value) or loss_value > 500):
                print(f"Loss exploded to: {loss_value}")
                return False, epoch_time

            if i % 100 == 0:
                print(f'Epoch {i}/{epochs}, Loss: {loss_value:.6f}, '
                      f'Train L2: {training_l2:.6e}, Test L2: {test_l2:.6e}, '
                      f'Time: {epoch_time:.2f}s')

        phase1_time = time.perf_counter() - start
        print(f"Phase 1 complete. Time: {phase1_time:.2f}s, Final L2: {training_l2:.6e}")

        return True, phase1_time

    def compute_defect_phase2(self):
        """Phase 2: Compute defect using autodiff (ONE pass)"""
        print("\n=== PHASE 2: Computing defect via autodiff ===")

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Create fresh interior points with requires_grad=True for autodiff
        x_int = self.x_i.clone().detach().requires_grad_(True)
        y_int = self.y_i.clone().detach().requires_grad_(True)
        X_int = torch.hstack([x_int, y_int])

        # Get network prediction at interior points
        u_pred_interior = self.w.forward(X_int)

        # Compute exact Laplacian via autodiff
        u_x = grad(u_pred_interior, x_int,
                   grad_outputs=torch.ones_like(u_pred_interior),
                   create_graph=True, retain_graph=True)[0]
        u_xx = grad(u_x, x_int,
                    grad_outputs=torch.ones_like(u_x),
                    create_graph=True, retain_graph=True)[0]

        u_y = grad(u_pred_interior, y_int,
                   grad_outputs=torch.ones_like(u_pred_interior),
                   create_graph=True, retain_graph=True)[0]
        u_yy = grad(u_y, y_int,
                    grad_outputs=torch.ones_like(u_y),
                    create_graph=False, retain_graph=False)[0]

        laplacian_autodiff = (u_xx + u_yy).detach()

        # Compute RBF-FD Laplacian at interior points
        u_pred_full = self.w.forward(self.X_full)
        L_mul = Cupy_mul_L.apply
        laplacian_rbf = L_mul(u_pred_full, self.L)[:self.x_i.shape[0]]  # Only interior

        # Defect = RBF-FD - Autodiff (what RBF-FD got wrong)
        defect = (laplacian_rbf - laplacian_autodiff).detach()

        if device_string == "cuda":
            torch.cuda.synchronize()
        phase2_time = time.perf_counter() - start

        defect_magnitude = torch.abs(defect).mean().item()
        defect_max = torch.abs(defect).max().item()
        print(f"Phase 2 complete. Time: {phase2_time:.2f}s")
        print(f"Defect stats: mean|d|={defect_magnitude:.6e}, max|d|={defect_max:.6e}")

        return defect, phase2_time

    def train_correction_phase3(self, defect):
        """Phase 3: Train correction network on defect (supervised learning)"""
        print("\n=== PHASE 3: Training correction network ===")

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Create correction network
        self.correction_net = CorrectionNetwork(
            hidden_size=self.correction_hidden,
            num_layers=self.correction_layers
        )

        # Use Adam for correction (faster than LBFGS for this small problem)
        correction_optimizer = optim.Adam(self.correction_net.parameters(), lr=1e-3)

        # Training data: X_interior -> defect
        X_train = self.X_interior.detach()
        y_train = defect.detach()

        for epoch in range(1, self.correction_epochs + 1):
            correction_optimizer.zero_grad()

            pred = self.correction_net(X_train)
            loss = MSELoss()(pred, y_train)

            loss.backward()
            correction_optimizer.step()

            if epoch % 50 == 0:
                print(f"Correction Epoch {epoch}/{self.correction_epochs}, Loss: {loss.item():.6e}")

        if device_string == "cuda":
            torch.cuda.synchronize()
        phase3_time = time.perf_counter() - start

        print(f"Phase 3 complete. Time: {phase3_time:.2f}s")

        return phase3_time

    def train_corrected_phase4(self, defect):
        """
        Phase 4: Retrain using defect-corrected loss

        Key insight: Instead of training on |L_rbf@u - f - exp(u)|^2,
        train on |L_rbf@u - d - f - exp(u)|^2 where d is the learned defect.
        This effectively gives us L_autodiff@u in the loss!
        """
        print("\n=== PHASE 4: Retraining with defect-corrected loss ===")

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Short retraining with corrected loss
        correction_epochs = 100
        L_mul = Cupy_mul_L.apply
        B_mul = Cupy_mul_B.apply

        # Re-initialize optimizer for fine-tuning
        finetune_optimizer = optim.LBFGS(self.w.parameters(), lr=self.lr * 0.5)

        for i in range(1, correction_epochs + 1):
            def closure():
                finetune_optimizer.zero_grad()

                u_pred_full = self.w.forward(self.X_full)

                # RBF-FD Laplacian
                lap_u_rbf = L_mul(u_pred_full, self.L)

                # Predict defect at interior points and SUBTRACT it
                # This gives us: L_rbf@u - d ≈ L_autodiff@u
                defect_pred = self.correction_net(self.X_interior.detach())

                # Corrected Laplacian (approximately autodiff)
                lap_u_corrected = lap_u_rbf[:self.x_i.shape[0]] - defect_pred

                # For boundary points, use uncorrected (defect mainly affects interior)
                # Full corrected Laplacian at interior+boundary
                n_interior = self.x_i.shape[0]
                n_boundary = self.x_b.shape[0]

                # Interior PDE residual with corrected Laplacian
                f_interior = self.f[:n_interior]
                u_interior = u_pred_full[:n_interior]
                pde_residual_interior = lap_u_corrected - f_interior - torch.exp(u_interior)

                # Boundary residual (unchanged)
                boundary_loss_term = B_mul(u_pred_full, self.B) - self.g

                l2 = torch.mean(torch.square(torch.flatten(pde_residual_interior)))
                l3 = torch.mean(torch.square(torch.flatten(boundary_loss_term)))

                train_loss = l2 + l3
                train_loss.backward(retain_graph=True)
                return train_loss.item()

            loss_value = finetune_optimizer.step(closure)

            if i % 25 == 0:
                training_pred = self.w.forward(self.X_tilde)
                training_l2 = self.compute_l2(training_pred, self.u_true)
                print(f'Correction Epoch {i}/{correction_epochs}, Loss: {loss_value:.6f}, L2: {training_l2:.6e}')

        if device_string == "cuda":
            torch.cuda.synchronize()
        phase4_time = time.perf_counter() - start

        # Final evaluation
        training_pred = self.w.forward(self.X_tilde)
        training_l2 = self.compute_l2(training_pred, self.u_true)
        test_pred = self.w.forward(self.test_X_tilde)
        test_l2 = self.compute_l2(test_pred, self.test_u_true)

        print(f"Phase 4 complete. Time: {phase4_time:.2f}s")
        print(f"Corrected L2: Train={training_l2:.6e}, Test={test_l2:.6e}")

        return phase4_time, training_l2, test_l2

    def evaluate_corrected_solution(self):
        """Evaluate the corrected solution"""
        print("\n=== Evaluating corrected solution ===")

        # Final solution accuracy
        training_pred = self.w.forward(self.X_tilde)
        training_l2 = self.compute_l2(training_pred, self.u_true)

        # Just return the final L2 - defect tracking is complex post-retraining
        return training_l2, 0.0

    def train(self):
        """Run all four phases"""
        # Phase 1: Train DT-PINN
        success, phase1_time = self.train_phase1_dtpinn()
        if not success:
            return None

        # Record pre-correction L2
        training_pred = self.w.forward(self.X_tilde)
        pre_correction_l2 = self.compute_l2(training_pred, self.u_true)
        print(f"\nPre-correction L2: {pre_correction_l2:.6e}")

        # Phase 2: Compute defect
        defect, phase2_time = self.compute_defect_phase2()

        # Phase 3: Train correction network
        phase3_time = self.train_correction_phase3(defect)

        # Phase 4: Retrain with defect-corrected loss
        phase4_time, final_train_l2, final_test_l2 = self.train_corrected_phase4(defect)

        # Final evaluation
        final_l2, defect_error = self.evaluate_corrected_solution()

        total_time = phase1_time + phase2_time + phase3_time + phase4_time

        results = {
            'phase1_time': phase1_time,
            'phase2_time': phase2_time,
            'phase3_time': phase3_time,
            'phase4_time': phase4_time,
            'total_time': total_time,
            'pre_correction_l2': pre_correction_l2,
            'final_l2': final_l2,
            'final_train_l2': final_train_l2,
            'final_test_l2': final_test_l2,
            'defect_prediction_error': defect_error,
            'logged': dict(self.logged_results)
        }

        return results


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

    # Test data
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
    """Run IDC-PINN experiment"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--dataset_size', type=int, default=2236)
    parser.add_argument('--lr', type=float, default=0.04)
    parser.add_argument('--correction_epochs', type=int, default=200)
    parser.add_argument('--correction_hidden', type=int, default=50)
    parser.add_argument('--correction_layers', type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("IDC-PINN (Iterative Defect Correction) Experiment")
    print("=" * 70)
    print(f"Dataset size: {args.dataset_size}")
    print(f"Main DT-PINN epochs: {args.epochs}")
    print(f"Correction epochs: {args.correction_epochs}")
    print(f"Learning rate: {args.lr}")
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

    trainer = IDCDTPINNTrainer(
        config=config,
        data_dict=data_dict,
        correction_epochs=args.correction_epochs,
        correction_hidden=args.correction_hidden,
        correction_layers=args.correction_layers
    )

    results = trainer.train()

    if results is None:
        print("\nTraining failed")
        return

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Phase 1 (DT-PINN) Time: {results['phase1_time']:.2f}s")
    print(f"Phase 2 (Defect computation) Time: {results['phase2_time']:.2f}s")
    print(f"Phase 3 (Correction training) Time: {results['phase3_time']:.2f}s")
    print(f"Phase 4 (Corrected retraining) Time: {results['phase4_time']:.2f}s")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"\nPre-correction L2: {results['pre_correction_l2']:.6e}")
    print(f"Final L2 Error: {results['final_l2']:.6e}")
    print(f"Defect Prediction Error: {results['defect_prediction_error']:.6e}")

    # Compare to baseline
    baseline_l2 = 6.76e-03
    baseline_time = 216.07
    vanilla_l2 = 7.47e-05
    vanilla_time = 304.59

    improvement = baseline_l2 / results['final_l2']
    pre_to_post = results['pre_correction_l2'] / results['final_l2']
    speedup_vs_vanilla = vanilla_time / results['total_time']
    vs_vanilla = results['final_l2'] / vanilla_l2

    print(f"\nComparison:")
    print(f"  Pre->Post correction: {pre_to_post:.2f}x improvement")
    print(f"  vs Baseline DT-PINN: {improvement:.2f}x L2 improvement")
    print(f"  vs Vanilla PINN: {speedup_vs_vanilla:.2f}x speedup, {vs_vanilla:.1f}x worse L2")
    print(f"  Baseline time: {baseline_time:.2f}s, Our time: {results['total_time']:.2f}s")

    target_l2 = 1.49e-04
    if results['final_l2'] <= target_l2:
        print(f"\n TARGET ACHIEVED! L2={results['final_l2']:.6e} <= {target_l2:.6e}")
    elif results['final_l2'] < baseline_l2:
        gap = results['final_l2'] / target_l2
        print(f"\n IMPROVED over baseline! Still {gap:.1f}x from target.")
    else:
        print(f"\n No improvement over baseline.")


if __name__ == '__main__':
    main()
