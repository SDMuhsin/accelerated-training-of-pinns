"""
Simplified DT-PINN training script for smallest task (582 points).
This script runs DT-PINN on the linear Poisson equation using meshless discretizations.
"""

import json
import os
import time
import torch
import cupy
from torch.autograd import grad
from torch import optim
from torch.nn import MSELoss
from torch.utils.dlpack import to_dlpack, from_dlpack
from network import W
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from collections import defaultdict
from math import isnan

torch.manual_seed(0)

# CUDA support
if torch.cuda.is_available():
    device_string = "cuda"
    torch.cuda.init()
    torch.cuda.manual_seed_all(0)
else:
    device_string = "cpu"
print(f"Device being used: {device_string}")

# cupy setup
if device_string == "cuda":
    device = cupy.cuda.Device(0)
    cupy.cuda.Device(0).use()

PRECISION = torch.float64
precision_string = "float64"

# global transposes of the sparse matrices
L_t, B_t = None, None


class Cupy_mul_L(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_pred_, sparse):
        """u_pred is the network's prediction"""
        return from_dlpack(sparse.dot(cupy.from_dlpack(to_dlpack(u_pred_))).toDlpack())

    @staticmethod
    def backward(ctx, grad_output):
        """grad_output is with respect to u_pred"""
        return from_dlpack(L_t.dot(cupy.from_dlpack(to_dlpack(grad_output))).toDlpack()), None


class Cupy_mul_B(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u_pred_, sparse):
        """u_pred is the network's prediction"""
        return from_dlpack(sparse.dot(cupy.from_dlpack(to_dlpack(u_pred_))).toDlpack())

    @staticmethod
    def backward(ctx, grad_output):
        """grad_output is with respect to u_pred"""
        return from_dlpack(B_t.dot(cupy.from_dlpack(to_dlpack(grad_output))).toDlpack()), None


class Trainer:
    def __init__(self, config=None, **kwargs):
        self.lr = config['lr']
        self.network_precision_string = config['precision']
        self.network_precision_dtype = torch.float32 if self.network_precision_string == "float32" else torch.float64
        self.epochs = config['epochs']
        self.__dict__.update(kwargs)
        self.logged_results = defaultdict(list)
        self.f_pred_ = None
        self.boundary_loss_term_ = None

        self.w = W(config)
        print(self.w, '\n')

        # interior and boundary points
        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

        # interior, boundary, and ghost points
        self.x_full = torch.vstack([self.x_i.clone(), self.x_b.clone(), self.x_g.clone()])
        self.y_full = torch.vstack([self.y_i.clone(), self.y_b.clone(), self.y_g.clone()])
        self.X_full = torch.hstack([self.x_full, self.y_full])

        # boundary points
        self.X_b = torch.hstack([self.x_b.clone(), self.y_b.clone()])

        self.optimizer = optim.LBFGS(self.w.parameters(), lr=self.lr)

    @staticmethod
    def compute_mse(a, b):
        mse = MSELoss()(torch.flatten(a), torch.flatten(b))
        return mse.item()

    @staticmethod
    def compute_l2(a, b):
        diff = torch.subtract(torch.flatten(a).detach().cpu(), torch.flatten(b).detach().cpu())
        relative_l2_error = torch.linalg.norm(diff) / torch.linalg.norm(torch.flatten(b))
        return relative_l2_error.item()

    @staticmethod
    def compute_linf(a):
        return torch.linalg.norm(a.to(PRECISION), ord=float('inf')).item()

    def train(self):
        global L_t, B_t
        epochs = self.epochs
        self.u_tilde = self.u[:self.ib_idx]

        # multiplying L and B with random vectors to "generate a kernel" and move them to the GPU
        rand_vec = cupy.random.rand(self.L.shape[1], 2).astype(cupy.float64)
        self.L.dot(rand_vec)
        self.B.dot(rand_vec)

        L_t = self.L.transpose()
        B_t = self.B.transpose()

        # initializing the matvec kernel for the sparse matrices' transposes
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
            try:
                def closure():
                    self.optimizer.zero_grad()

                    # u_pred_full on all points
                    u_pred_full = self.w.forward(self.X_full)
                    assert u_pred_full.dtype == self.network_precision_dtype

                    # pde being enforced on interior and boundary
                    f_pred = L_mul(u_pred_full, self.L) - self.f
                    assert f_pred.dtype == torch.float64
                    self.f_pred_ = f_pred

                    # boundary condition enforced on boundary
                    boundary_loss_term = B_mul(u_pred_full, self.B) - self.g
                    assert boundary_loss_term.dtype == torch.float64
                    self.boundary_loss_term_ = boundary_loss_term

                    l2 = torch.mean(torch.square(torch.flatten(f_pred)))
                    l3 = torch.mean(torch.square(torch.flatten(boundary_loss_term)))

                    train_loss = l2 + l3

                    train_loss.backward(retain_graph=True)
                    return train_loss.item()

                loss_value = self.optimizer.step(closure)
                if device_string == "cuda":
                    torch.cuda.synchronize()
                epoch_time = time.perf_counter() - start

                # Compute metrics
                training_pred = self.w.forward(self.X_tilde)
                test_pred = self.w.forward(self.test_X_tilde)

                training_mse = self.compute_mse(training_pred, self.u_true)
                test_mse = self.compute_mse(test_pred, self.test_u_true)
                training_l2 = self.compute_l2(training_pred, self.u_true)
                test_l2 = self.compute_l2(test_pred, self.test_u_true)

                training_discrete_pde_residual = self.compute_linf(self.f_pred_)
                training_discrete_boundary_residual = self.compute_linf(self.boundary_loss_term_)

                # logging
                self.logged_results['training_losses'].append(loss_value)
                self.logged_results['training_mse_losses'].append(training_mse)
                self.logged_results['training_l2_losses'].append(training_l2)
                self.logged_results['test_mse_losses'].append(test_mse)
                self.logged_results['test_l2_losses'].append(test_l2)
                self.logged_results['epochs_list'].append(i)
                self.logged_results['epoch_time'].append(epoch_time)
                self.logged_results['training_discrete_pde_residual'].append(training_discrete_pde_residual)
                self.logged_results['training_discrete_boundary_residual'].append(training_discrete_boundary_residual)

                if i > 30 and (isnan(loss_value) or loss_value > 500):
                    print(f"Loss exploded to: {loss_value}")
                    return False

                if i % 100 == 0:
                    print(f'Epoch {i}/{epochs}, Loss: {loss_value:.6f}, '
                          f'Train L2: {training_l2:.6f}, Test L2: {test_l2:.6f}, '
                          f'Time: {epoch_time:.2f}s')

            except KeyboardInterrupt:
                print('Keyboard Interrupt. Ending training.')
                return dict(self.logged_results)

        return dict(self.logged_results)


def load_mat_cupy(mat):
    import cupyx.scipy.sparse as cupy_sparse
    scipy_csr = csr_matrix(mat, dtype=np.float64)
    # Convert to cupy sparse matrix
    cupy_csr = cupy_sparse.csr_matrix(scipy_csr)
    return cupy_csr


def save_results(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # Configuration - use larger problem for better speedup demonstration
    order = 2
    size = 2236  # Larger problem: 4x more points
    test_name = "2_21748_test"
    file_name = f"{order}_{size}"

    lr = 0.04
    epochs = 500  # Reduced for faster comparison on larger problem
    activation_function = 'tanh'
    network_precision = "float64"

    save_folder = f'results/dtpinn/{order}/{size}/'
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = save_folder + 'results.json'

    print('=' * 70)
    print(f'Running {network_precision} DT-PINN on {file_name} with {activation_function}')
    print('=' * 70)

    # read mat files
    X_i = torch.tensor(loadmat(f"scai/files_{file_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device_string)
    X_b = torch.tensor(loadmat(f"scai/files_{file_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device_string)
    X_g = torch.tensor(loadmat(f"scai/files_{file_name}/Xg.mat")["X_g"], dtype=PRECISION, requires_grad=True).to(device_string)
    n = torch.tensor(loadmat(f"scai/files_{file_name}/n.mat")["n"], dtype=PRECISION, requires_grad=True).to(device_string)
    u_true = torch.tensor(loadmat(f"scai/files_{file_name}/u.mat")["u"], dtype=PRECISION).to(device_string)
    f = torch.tensor(loadmat(f"scai/files_{file_name}/f.mat")["f"], dtype=PRECISION, requires_grad=True).to(device_string)
    g = torch.tensor(loadmat(f"scai/files_{file_name}/g.mat")["g"], dtype=PRECISION, requires_grad=True).to(device_string)
    alpha = torch.tensor(loadmat(f"scai/files_{file_name}/alpha.mat")["Neucoeff"], dtype=PRECISION, requires_grad=True).to(device_string)
    beta = torch.tensor(loadmat(f"scai/files_{file_name}/beta.mat")["Dircoeff"], dtype=PRECISION, requires_grad=True).to(device_string)
    L = load_mat_cupy(loadmat(f"scai/files_{file_name}/L1.mat")["L1"])
    B = load_mat_cupy(loadmat(f"scai/files_{file_name}/B1.mat")["B1"])

    b_starts = X_i.shape[0]
    b_end = b_starts + X_b.shape[0]

    # test files
    X_i_test = torch.tensor(loadmat(f"scai/files_{test_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device_string)
    X_b_test = torch.tensor(loadmat(f"scai/files_{test_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device_string)
    test_u_true = torch.tensor(loadmat(f"scai/files_{test_name}/u.mat")["u"], dtype=PRECISION).to(device_string)

    test_x_i = X_i_test[:, 0].unsqueeze(dim=1)
    test_y_i = X_i_test[:, 1].unsqueeze(dim=1)
    test_x_b = X_b_test[:, 0].unsqueeze(dim=1)
    test_y_b = X_b_test[:, 1].unsqueeze(dim=1)

    test_x_tilde = torch.vstack([test_x_i, test_x_b])
    test_y_tilde = torch.vstack([test_y_i, test_y_b])
    test_X_tilde = torch.hstack([test_x_tilde, test_y_tilde])

    # separate spatial dimensions
    x_i = X_i[:, 0].unsqueeze(dim=1)
    y_i = X_i[:, 1].unsqueeze(dim=1)
    x_b = X_b[:, 0].unsqueeze(dim=1)
    y_b = X_b[:, 1].unsqueeze(dim=1)
    x_g = X_g[:, 0].unsqueeze(dim=1)
    y_g = X_g[:, 1].unsqueeze(dim=1)

    ib_idx = X_i.shape[0] + X_b.shape[0]

    config = {
        'spatial_dim': 2,
        'precision': network_precision,
        'activation': activation_function,
        'order': 2,
        'network_device': device_string,
        'layers': 4,
        'nodes': 50,
        'epochs': epochs,
        'optimizer': 'lbfgs',
        'lr': lr,
    }

    vars = {
        'n': n,
        'x_i': x_i,
        'x_b': x_b,
        'x_g': x_g,
        'y_i': y_i,
        'y_b': y_b,
        'y_g': y_g,
        'ib_idx': ib_idx,
        'u': u_true,
        'u_true': u_true[:ib_idx],  # Only interior + boundary points
        'L': L,
        'B': B,
        'test_X_tilde': test_X_tilde,
        'test_u_true': test_u_true[:X_i_test.shape[0] + X_b_test.shape[0]],  # Only interior + boundary
        'f': f,
        'g': g,
        'alpha': alpha,
        'beta': beta,
        'b_end': b_end,
        'b_starts': b_starts,
    }

    print(f"Learning rate: {config['lr']}")
    print(f"Training for {epochs} epochs...")

    flag = True
    while flag:
        trainer = Trainer(config=config, **vars)
        logged_results = trainer.train()
        if type(logged_results) == bool:
            config['lr'] /= 2.0
            print(f"Restarting with learning rate = {config['lr']}")
            continue
        else:
            flag = False

    logged_results = logged_results | config
    save_results(logged_results, save_file_name)

    print(f"\nTraining completed!")
    print(f"Final training L2 error: {logged_results['training_l2_losses'][-1]:.6f}")
    print(f"Final test L2 error: {logged_results['test_l2_losses'][-1]:.6f}")
    print(f"Total training time: {logged_results['epoch_time'][-1]:.2f}s")
    print(f"Results saved to: {save_file_name}")
