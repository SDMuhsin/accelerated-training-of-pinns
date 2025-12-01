"""
Vanilla PINN: Standard Physics-Informed Neural Network

Uses automatic differentiation to compute derivatives.
This is the baseline model without discrete operator acceleration.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import time
from typing import Dict, Any, List, Optional

from .base import BaseModel, TrainResult


class VanillaPINN(BaseModel):
    """
    Standard PINN solver using automatic differentiation.

    For nonlinear Poisson: ∇²u = f + exp(u)
    Uses autograd to compute Laplacian from network outputs.
    """

    name = "vanilla-pinn"

    def __init__(
        self,
        task,
        layers: int = 4,
        nodes: int = 50,
        activation: str = 'tanh',
        optimizer: str = 'lbfgs',
        lr: float = 0.01,
        epochs: int = 1000,
        use_cuda: bool = True,
        pde_weight: float = 1.0,
        bc_weight: float = 1.0,
        seed: int = 0,
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE data
            layers: Number of hidden layers
            nodes: Nodes per hidden layer
            activation: Activation function ('tanh', 'relu', 'sin')
            optimizer: 'lbfgs' or 'adam'
            lr: Learning rate
            epochs: Number of training epochs
            use_cuda: Whether to use GPU
            pde_weight: Weight for PDE loss
            bc_weight: Weight for BC loss
            seed: Random seed
        """
        super().__init__(task, **kwargs)

        self.layers = layers
        self.nodes = nodes
        self.activation = activation
        self.optimizer_name = optimizer
        self.lr = lr
        self.epochs = epochs
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.seed = seed

        # Will be set during setup
        self.network = None
        self.device = None

    def _build_network(self, input_dim: int, precision: torch.dtype) -> nn.Module:
        """Build MLP network."""
        torch.manual_seed(self.seed)

        layers_list = []
        in_features = input_dim

        for i in range(self.layers):
            layers_list.append(nn.Linear(in_features, self.nodes))
            if self.activation == 'tanh':
                layers_list.append(nn.Tanh())
            elif self.activation == 'relu':
                layers_list.append(nn.ReLU())
            elif self.activation == 'sin':
                class Sin(nn.Module):
                    def forward(self, x):
                        return torch.sin(x)
                layers_list.append(Sin())
            in_features = self.nodes

        layers_list.append(nn.Linear(in_features, 1))

        network = nn.Sequential(*layers_list)
        network = network.to(precision)
        return network

    def setup(self):
        """Initialize network."""
        torch.manual_seed(self.seed)

        data = self.task.data
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        input_dim = data.spatial_dim
        self.network = self._build_network(input_dim, precision)
        self.network = self.network.to(self.device)

        self._is_setup = True

    def _compute_laplacian(self, u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian ∇²u using automatic differentiation.

        For 2D: ∇²u = ∂²u/∂x² + ∂²u/∂y²
        """
        # First derivatives
        u_grad = grad(u, X, grad_outputs=torch.ones_like(u),
                     create_graph=True, retain_graph=True)[0]

        laplacian = torch.zeros_like(u)

        # Second derivatives for each spatial dimension
        for i in range(X.shape[1]):
            u_i = u_grad[:, i:i+1]
            u_ii = grad(u_i, X, grad_outputs=torch.ones_like(u_i),
                       create_graph=True, retain_graph=True)[0][:, i:i+1]
            laplacian = laplacian + u_ii

        return laplacian

    def _compute_robin_bc(self, u: torch.Tensor, X_b: torch.Tensor,
                          alpha: torch.Tensor, beta: torch.Tensor,
                          n: torch.Tensor) -> torch.Tensor:
        """
        Compute Robin boundary condition residual.

        Robin BC: α * ∂u/∂n + β * u = g
        where ∂u/∂n is the normal derivative.
        """
        # Compute gradient
        u_grad = grad(u, X_b, grad_outputs=torch.ones_like(u),
                     create_graph=True, retain_graph=True)[0]

        # Normal derivative: dot product of gradient and normal vector
        u_n = torch.sum(u_grad * n, dim=1, keepdim=True)

        # Robin condition
        return alpha * u_n + beta * u

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Train using gradient-based optimization with autodiff.
        """
        if not self._is_setup:
            self.setup()

        data = self.task.data
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        # Prepare interior points (need requires_grad for Laplacian)
        X_interior = torch.tensor(data.X_interior, dtype=precision,
                                  device=self.device, requires_grad=True)
        f_interior = torch.tensor(data.f[:data.N_interior], dtype=precision,
                                  device=self.device).unsqueeze(1)

        # Prepare boundary points
        X_boundary = torch.tensor(data.X_boundary, dtype=precision,
                                  device=self.device, requires_grad=True)
        g = torch.tensor(data.g, dtype=precision, device=self.device).unsqueeze(1)

        # Robin BC coefficients (if available)
        if data.alpha is not None:
            alpha = torch.tensor(data.alpha, dtype=precision, device=self.device).unsqueeze(1)
            beta = torch.tensor(data.beta, dtype=precision, device=self.device).unsqueeze(1)
            n = torch.tensor(data.n, dtype=precision, device=self.device)
            use_robin = True
        else:
            use_robin = False

        # Setup optimizer
        if self.optimizer_name == 'lbfgs':
            optimizer = torch.optim.LBFGS(self.network.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        loss_history = []

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            def closure():
                optimizer.zero_grad()

                # PDE loss on interior points
                u_interior = self.network(X_interior)
                laplacian_u = self._compute_laplacian(u_interior, X_interior)

                # Nonlinear Poisson: ∇²u - f - exp(u) = 0
                pde_residual = laplacian_u - f_interior - torch.exp(u_interior)
                pde_loss = torch.mean(pde_residual ** 2)

                # BC loss
                u_boundary = self.network(X_boundary)
                if use_robin:
                    bc_pred = self._compute_robin_bc(u_boundary, X_boundary, alpha, beta, n)
                    bc_residual = bc_pred - g
                else:
                    bc_residual = u_boundary - g
                bc_loss = torch.mean(bc_residual ** 2)

                loss = self.pde_weight * pde_loss + self.bc_weight * bc_loss
                loss.backward(retain_graph=True)
                return loss

            if self.optimizer_name == 'lbfgs':
                loss = optimizer.step(closure)
            else:
                loss = closure()
                optimizer.step()

            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
            loss_history.append(loss_value)

            if verbose and (epoch <= 5 or epoch % 100 == 0):
                print(f"  Epoch {epoch}: loss = {loss_value:.4e}")

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        train_time = time.perf_counter() - start_time

        # Get final predictions
        with torch.no_grad():
            X_ib = torch.tensor(data.X_ib, dtype=precision, device=self.device)
            u_pred = self.network(X_ib).cpu().numpy().flatten()

        # Compute L2 error
        l2_error = None
        if data.u_true is not None:
            u_true_ib = data.u_true[:data.N_ib]
            l2_error = self.compute_l2_error(u_pred, u_true_ib)

        return TrainResult(
            u_pred=u_pred,
            train_time=train_time,
            l2_error=l2_error,
            final_loss=loss_history[-1] if loss_history else None,
            loss_history=loss_history,
            n_iterations=len(loss_history),
            extra={
                'layers': self.layers,
                'nodes': self.nodes,
                'optimizer': self.optimizer_name,
                'method': 'autodiff',
            }
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions at given points."""
        if self.network is None:
            raise RuntimeError("Model not trained. Call train() first.")

        precision = torch.float64 if X.dtype == np.float64 else torch.float32
        X_tensor = torch.tensor(X, dtype=precision, device=self.device)

        with torch.no_grad():
            u_pred = self.network(X_tensor)

        return u_pred.cpu().numpy().flatten()

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        return {
            'layers': 4,
            'nodes': 50,
            'activation': 'tanh',
            'optimizer': 'lbfgs',
            'lr': 0.01,
            'epochs': 1000,
            'use_cuda': True,
            'pde_weight': 1.0,
            'bc_weight': 1.0,
            'seed': 0,
        }

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--layers', type=int, default=4,
                           help='Number of hidden layers')
        parser.add_argument('--nodes', type=int, default=50,
                           help='Nodes per hidden layer')
        parser.add_argument('--activation', type=str, default='tanh',
                           choices=['tanh', 'relu', 'sin'],
                           help='Activation function')
        parser.add_argument('--optimizer', type=str, default='lbfgs',
                           choices=['lbfgs', 'adam'],
                           help='Optimizer')
        parser.add_argument('--lr', type=float, default=0.01,
                           help='Learning rate')
        parser.add_argument('--epochs', type=int, default=1000,
                           help='Number of training epochs')
        parser.add_argument('--no-cuda', action='store_true',
                           help='Disable CUDA')
        parser.add_argument('--pde-weight', type=float, default=1.0,
                           help='PDE loss weight')
        parser.add_argument('--bc-weight', type=float, default=1.0,
                           help='BC loss weight')
        parser.add_argument('--seed', type=int, default=0,
                           help='Random seed')
