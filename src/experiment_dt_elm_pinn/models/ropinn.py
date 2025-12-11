"""
RoPINN: Region-Optimized Physics-Informed Neural Network

Extends standard PINN training by optimizing over continuous neighborhood
regions instead of individual collocation points.

Paper: "RoPINN: Region Optimized Physics-Informed Neural Networks"
Key innovation: Trust region calibration based on gradient variance.

Reference implementation: /workspace/dt-pinn/temp/RoPINN/
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import time
from typing import Dict, Any, List, Optional

from .base import BaseModel, TrainResult


class RoPINN(BaseModel):
    """
    RoPINN: Region-Optimized Physics-Informed Neural Network.

    Extends VanillaPINN with region optimization:
    - Samples perturbed points within a neighborhood region
    - Calibrates region size based on gradient variance
    - Uses L-BFGS with strong Wolfe line search

    For Poisson equations: nabla^2 u = f (linear) or nabla^2 u = f + exp(u) (nonlinear)
    """

    name = "ropinn"

    def __init__(
        self,
        task,
        layers: int = 4,
        nodes: int = 50,
        activation: str = 'tanh',
        epochs: int = 1000,
        use_cuda: bool = True,
        pde_weight: float = 1.0,
        bc_weight: float = 1.0,
        seed: int = 0,
        # RoPINN-specific parameters (paper defaults)
        initial_region: float = 1e-4,
        sample_num: int = 1,
        past_iterations: int = 10,
        region_max: float = 0.01,
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE data
            layers: Number of hidden layers
            nodes: Nodes per hidden layer
            activation: Activation function ('tanh', 'relu', 'sin')
            epochs: Number of training epochs
            use_cuda: Whether to use GPU
            pde_weight: Weight for PDE loss
            bc_weight: Weight for BC loss
            seed: Random seed

            # RoPINN-specific (paper defaults)
            initial_region: Base region radius (default: 1e-4)
            sample_num: Number of region samples per iteration (default: 1)
            past_iterations: Window for gradient variance computation (default: 10)
            region_max: Maximum region radius after calibration (default: 0.01)
        """
        super().__init__(task, **kwargs)

        self.layers = layers
        self.nodes = nodes
        self.activation = activation
        self.epochs = epochs
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.seed = seed

        # RoPINN parameters
        self.initial_region = initial_region
        self.sample_num = sample_num
        self.past_iterations = past_iterations
        self.region_max = region_max

        # Will be set during setup
        self.network = None
        self.device = None

    def _build_network(self, input_dim: int, precision: torch.dtype) -> nn.Module:
        """Build MLP network with Xavier initialization."""
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

        # Xavier initialization (from RoPINN code)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        network.apply(init_weights)
        return network

    def setup(self):
        """Initialize network and device."""
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
        Compute Laplacian nabla^2 u using automatic differentiation.

        For 2D: nabla^2 u = d^2u/dx^2 + d^2u/dy^2
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
        Compute Robin boundary condition: alpha * du/dn + beta * u
        """
        u_grad = grad(u, X_b, grad_outputs=torch.ones_like(u),
                     create_graph=True, retain_graph=True)[0]

        # Normal derivative: dot product of gradient and normal vector
        u_n = torch.sum(u_grad * n, dim=1, keepdim=True)

        return alpha * u_n + beta * u

    def _compute_gradient_variance(
        self,
        gradient_list: List[np.ndarray]
    ) -> float:
        """
        Compute normalized gradient variance for trust region calibration.

        From RoPINN paper:
        variance = mean(std(gradients) / (mean(|gradients|) + eps))

        Args:
            gradient_list: List of flattened gradient arrays from past iterations

        Returns:
            Normalized gradient variance (scalar)
        """
        if len(gradient_list) < 2:
            return 1.0  # Default variance before enough history

        gradient_array = np.array(gradient_list)

        # Normalized variance: std / (mean_abs + eps)
        std_grad = np.std(gradient_array, axis=0)
        mean_abs_grad = np.mean(np.abs(gradient_array), axis=0) + 1e-6

        variance = (std_grad / mean_abs_grad).mean()

        if variance == 0:
            variance = 1.0  # Numerical stability

        return variance

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Train using RoPINN region optimization with L-BFGS.

        Key differences from VanillaPINN:
        1. Sample perturbed points within calibrated region
        2. Track gradient history for trust region calibration
        3. Adjust region radius based on gradient variance
        """
        if not self._is_setup:
            self.setup()

        data = self.task.data
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        # Prepare interior points (base points for region sampling)
        X_interior_base = torch.tensor(data.X_interior, dtype=precision,
                                       device=self.device)
        f_interior = torch.tensor(data.f[:data.N_interior], dtype=precision,
                                  device=self.device).unsqueeze(1)

        # Prepare boundary points (NOT perturbed - fixed)
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

        # Check if task is linear
        is_linear = hasattr(self.task, 'is_linear') and self.task.is_linear()

        # Setup L-BFGS optimizer (RoPINN always uses L-BFGS with strong Wolfe)
        optimizer = torch.optim.LBFGS(
            self.network.parameters(),
            line_search_fn='strong_wolfe'
        )

        # RoPINN state variables
        gradient_list_overall = []  # History of average gradients
        gradient_list_temp = []     # Gradients within current iteration (closure calls)
        gradient_variance = 1.0     # Current variance estimate

        loss_history = []

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            # Calculate current region radius
            current_region = np.clip(
                self.initial_region / gradient_variance,
                a_min=0,
                a_max=self.region_max
            )

            def closure():
                optimizer.zero_grad()

                # === Region Optimization: Sample perturbed interior points ===
                # Sample points in neighborhood region (RoPINN core innovation)
                X_samples = []
                for _ in range(self.sample_num):
                    perturbation = torch.rand_like(X_interior_base) * current_region
                    X_samples.append(X_interior_base + perturbation)

                X_interior_perturbed = torch.cat(X_samples, dim=0)
                X_interior_perturbed.requires_grad_(True)

                # Replicate source term for sample_num copies
                f_perturbed = f_interior.repeat(self.sample_num, 1)

                # Forward pass on perturbed interior points
                u_interior = self.network(X_interior_perturbed)
                laplacian_u = self._compute_laplacian(u_interior, X_interior_perturbed)

                # PDE residual
                if is_linear:
                    pde_residual = laplacian_u - f_perturbed
                else:
                    # Nonlinear Poisson: nabla^2 u - f - exp(u) = 0
                    u_clamped = torch.clamp(u_interior, max=50.0)
                    pde_residual = laplacian_u - f_perturbed - torch.exp(u_clamped)

                pde_loss = torch.mean(pde_residual ** 2)

                # Boundary condition loss (NOT perturbed - fixed boundary)
                u_boundary = self.network(X_boundary)
                if use_robin:
                    bc_pred = self._compute_robin_bc(u_boundary, X_boundary, alpha, beta, n)
                    bc_residual = bc_pred - g
                else:
                    bc_residual = u_boundary - g

                bc_loss = torch.mean(bc_residual ** 2)

                # Total loss
                loss = self.pde_weight * pde_loss + self.bc_weight * bc_loss
                loss.backward(retain_graph=True)

                # === Gradient tracking for trust region calibration ===
                # (L-BFGS may call closure multiple times per step)
                gradients = []
                for p in self.network.parameters():
                    if p.grad is not None:
                        gradients.append(p.grad.view(-1))
                    else:
                        gradients.append(torch.zeros(1, device=self.device))

                flat_grad = torch.cat(gradients).cpu().numpy()
                gradient_list_temp.append(flat_grad)

                return loss

            # L-BFGS step (may call closure multiple times)
            loss = optimizer.step(closure)

            # === Trust Region Calibration ===
            # Average gradients from all closure calls in this iteration
            if gradient_list_temp:
                avg_gradient = np.mean(np.array(gradient_list_temp), axis=0)
                gradient_list_overall.append(avg_gradient)

                # Keep only past_iterations history
                gradient_list_overall = gradient_list_overall[-self.past_iterations:]

                # Update variance estimate
                gradient_variance = self._compute_gradient_variance(gradient_list_overall)

                # Clear temp list for next iteration
                gradient_list_temp.clear()

            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
            loss_history.append(loss_value)

            if verbose and (epoch <= 5 or epoch % 100 == 0):
                print(f"  Epoch {epoch}: loss = {loss_value:.4e}, "
                      f"region = {current_region:.2e}, "
                      f"grad_var = {gradient_variance:.4f}")

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
                'method': 'region_optimization',
                'initial_region': self.initial_region,
                'sample_num': self.sample_num,
                'past_iterations': self.past_iterations,
                'final_gradient_variance': gradient_variance,
                'final_region': np.clip(self.initial_region / gradient_variance, 0, self.region_max),
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
            'epochs': 1000,
            'use_cuda': True,
            'pde_weight': 1.0,
            'bc_weight': 1.0,
            'seed': 0,
            # RoPINN defaults from paper
            'initial_region': 1e-4,
            'sample_num': 1,
            'past_iterations': 10,
            'region_max': 0.01,
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
        # RoPINN-specific
        parser.add_argument('--initial-region', type=float, default=1e-4,
                           help='RoPINN: Initial region radius')
        parser.add_argument('--sample-num', type=int, default=1,
                           help='RoPINN: Number of region samples per iteration')
        parser.add_argument('--past-iterations', type=int, default=10,
                           help='RoPINN: Window for gradient variance computation')


class RoPINNLarge(RoPINN):
    """
    RoPINN with larger architecture (4 layers x 512 nodes).

    Matches the default architecture in the original RoPINN paper
    for fair comparison.
    """

    name = "ropinn-large"

    def __init__(self, task, **kwargs):
        # Override defaults for large variant
        kwargs.setdefault('layers', 4)
        kwargs.setdefault('nodes', 512)
        super().__init__(task, **kwargs)

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        defaults = RoPINN.get_default_args()
        defaults['nodes'] = 512
        return defaults
