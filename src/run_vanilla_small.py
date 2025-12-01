"""
Simplified vanilla PINN training script for smallest task (582 points).
This script runs vanilla PINN on the linear Poisson equation.
"""

import json
import os
import time
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch import optim
from torch.nn import MSELoss
from network import W
import numpy as np
from scipy.io import loadmat
from collections import defaultdict
from math import isnan

torch.manual_seed(0)

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_string = "cuda"
    torch.cuda.manual_seed_all(0)
else:
    device = torch.device('cpu')
    device_string = "cpu"
print(f"Device being used: {device}")

PRECISION = torch.float32
precision_string = "float32"


class Trainer:
    def __init__(self, config=None, **kwargs):
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.__dict__.update(kwargs)
        self.logged_results = defaultdict(list)

        self.w = W(config)
        print(self.w, '\n')

        # interior points
        self.x_interior = torch.vstack([self.x_i.clone()])
        self.y_interior = torch.vstack([self.y_i.clone()])
        self.X_interior = torch.hstack([self.x_interior, self.y_interior])

        # interior and boundary points
        self.x_tilde = torch.vstack([self.x_i.clone(), self.x_b.clone()])
        self.y_tilde = torch.vstack([self.y_i.clone(), self.y_b.clone()])
        self.X_tilde = torch.hstack([self.x_tilde, self.y_tilde])

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
        epochs = self.epochs
        self.u_tilde = self.u[:self.ib_idx]
        self.f_interior = self.f[:self.b_starts]

        if device_string == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for i in range(1, epochs + 1):
            try:
                def closure():
                    self.optimizer.zero_grad()

                    # residual term on interior points only
                    u_pred_interior = self.w.forward(self.X_interior)
                    u_x = grad(u_pred_interior, self.x_interior, grad_outputs=torch.ones_like(
                               u_pred_interior), create_graph=True, retain_graph=True)[0]
                    u_xx = grad(u_x, self.x_interior, grad_outputs=torch.ones_like(
                                u_pred_interior), create_graph=True, retain_graph=True)[0]

                    u_y = grad(u_pred_interior, self.y_interior, grad_outputs=torch.ones_like(
                               u_pred_interior), create_graph=True, retain_graph=True)[0]
                    u_yy = grad(u_y, self.y_interior, grad_outputs=torch.ones_like(
                                u_pred_interior), create_graph=True, retain_graph=True)[0]

                    # Poisson residual
                    f_pred = (u_xx + u_yy) - self.f_interior

                    boundary_pred = self.w.forward(self.X_b)

                    # first partial derivatives on the boundary
                    l2_w_x = grad(boundary_pred, self.x_b, grad_outputs=torch.ones_like(boundary_pred),
                                  create_graph=True, retain_graph=True)[0]
                    l2_w_y = grad(boundary_pred, self.y_b, grad_outputs=torch.ones_like(boundary_pred),
                                  create_graph=True, retain_graph=True)[0]

                    # combining the partial derivatives
                    w_xy = torch.hstack([l2_w_x, l2_w_y])

                    # element wise dot product between the normal vectors and first partial derivatives
                    gradient_n = torch.multiply(self.n, w_xy).sum(dim=1).unsqueeze(dim=1)

                    # loss on the boundary
                    boundary_loss_term = torch.multiply(self.alpha, gradient_n) + \
                                         torch.multiply(self.beta, boundary_pred) - self.g

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

                # logging
                self.logged_results['training_losses'].append(loss_value)
                self.logged_results['training_mse_losses'].append(training_mse)
                self.logged_results['training_l2_losses'].append(training_l2)
                self.logged_results['test_mse_losses'].append(test_mse)
                self.logged_results['test_l2_losses'].append(test_l2)
                self.logged_results['epochs_list'].append(i)
                self.logged_results['epoch_time'].append(epoch_time)

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


def save_results(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # Configuration - use larger problem for better speedup demonstration
    order = 2
    size = 2236  # Larger problem: 4x more points
    test_name = "2_21748_test"
    file_name = f"{order}_{size}"

    lr = 0.2
    epochs = 500  # Reduced for faster comparison on larger problem
    activation_function = 'tanh'

    save_folder = f'results/vanilla_pinn/{order}/{size}/'
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = save_folder + 'results.json'

    print('=' * 70)
    print(f'Running {precision_string} vanilla-PINN on {file_name} with {activation_function}')
    print('=' * 70)

    # read mat files
    X_i = torch.tensor(loadmat(f"scai/files_{file_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device)
    X_b = torch.tensor(loadmat(f"scai/files_{file_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device)
    X_g = torch.tensor(loadmat(f"scai/files_{file_name}/Xg.mat")["X_g"], dtype=PRECISION, requires_grad=True).to(device)
    n = torch.tensor(loadmat(f"scai/files_{file_name}/n.mat")["n"], dtype=PRECISION, requires_grad=True).to(device)
    u_true = torch.tensor(loadmat(f"scai/files_{file_name}/u.mat")["u"], dtype=PRECISION).to(device)
    f = torch.tensor(loadmat(f"scai/files_{file_name}/f.mat")["f"], dtype=PRECISION, requires_grad=True).to(device)
    g = torch.tensor(loadmat(f"scai/files_{file_name}/g.mat")["g"], dtype=PRECISION, requires_grad=True).to(device)
    alpha = torch.tensor(loadmat(f"scai/files_{file_name}/alpha.mat")["Neucoeff"], dtype=PRECISION, requires_grad=True).to(device)
    beta = torch.tensor(loadmat(f"scai/files_{file_name}/beta.mat")["Dircoeff"], dtype=PRECISION, requires_grad=True).to(device)

    b_starts = X_i.shape[0]
    b_end = b_starts + X_b.shape[0]

    # test files
    X_i_test = torch.tensor(loadmat(f"scai/files_{test_name}/Xi.mat")["Xi"], dtype=PRECISION, requires_grad=True).to(device)
    X_b_test = torch.tensor(loadmat(f"scai/files_{test_name}/Xb.mat")["Xb"], dtype=PRECISION, requires_grad=True).to(device)
    test_u_true = torch.tensor(loadmat(f"scai/files_{test_name}/u.mat")["u"], dtype=PRECISION).to(device)

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
        'precision': precision_string,
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
