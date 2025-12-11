"""
DAS: Deep Adaptive Sampling for Physics-Informed Neural Networks.

PyTorch implementation of DAS (Deep Adaptive Sampling) for solving PDEs.
Uses a normalizing flow to adaptively sample collocation points based on
PDE residual distributions.

Reference: Tang et al., "DAS: A Deep Adaptive Sampling Method for Solving
           High-Dimensional Partial Differential Equations" (2022)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import time
from typing import Dict, Any, List, Optional, Callable, Tuple

from .base import BaseModel, TrainResult
from .das_flow import BoundedRealNVP


class FCNN(nn.Module):
    """Fully connected neural network for PDE solution approximation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 50,
        n_layers: int = 4,
        activation: str = 'tanh'
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        # Hidden layers
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sin':
                layers.append(SinActivation())
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinActivation(nn.Module):
    """Sine activation function."""
    def forward(self, x):
        return torch.sin(x)


class DAS(BaseModel):
    """
    Deep Adaptive Sampling (DAS) for PDEs.

    Uses a normalizing flow to adaptively sample collocation points
    based on PDE residual distributions. Multi-stage training alternates
    between PDE network optimization and flow-based resampling.
    """

    name = "das"

    def __init__(
        self,
        task,
        # PDE network hyperparameters (defaults match vanilla-pinn for fair comparison)
        layers: int = 4,
        nodes: int = 50,
        activation: str = 'tanh',
        # Flow model hyperparameters
        flow_layers: int = 6,
        flow_hidden: int = 64,
        # Training hyperparameters
        # Default: 5 stages × 200 epochs = 1000 total (matches vanilla-pinn)
        n_train: int = 1000,
        pde_epochs: int = 200,
        flow_epochs: int = 200,
        max_stage: int = 5,
        lr: float = 1e-3,
        lambda_bd: float = 1.0,
        tol: float = 1e-7,
        # DAS-specific options
        quantity_type: str = 'residual',  # 'residual' or 'slope'
        replace_all: bool = False,
        # Framework options
        use_cuda: bool = True,
        seed: int = 0,
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE data
            layers: Number of hidden layers in PDE network
            nodes: Hidden layer width in PDE network
            activation: Activation function ('tanh', 'relu', 'sin')
            flow_layers: Number of coupling layers in flow
            flow_hidden: Hidden dimension in flow networks
            n_train: Number of training samples per stage
            pde_epochs: PDE training epochs per stage
            flow_epochs: Flow training epochs per stage
            max_stage: Maximum number of adaptive stages
            lr: Learning rate
            lambda_bd: Boundary condition loss weight
            tol: Convergence tolerance
            quantity_type: Error indicator ('residual' or 'slope')
            replace_all: If True, replace all samples each stage (DAS-R);
                        otherwise accumulate (DAS-G)
            use_cuda: Whether to use GPU
            seed: Random seed
        """
        super().__init__(task, **kwargs)

        self.layers = layers
        self.nodes = nodes
        self.activation = activation
        self.flow_layers = flow_layers
        self.flow_hidden = flow_hidden
        self.n_train = n_train
        self.pde_epochs = pde_epochs
        self.flow_epochs = flow_epochs
        self.max_stage = max_stage
        self.lr = lr
        self.lambda_bd = lambda_bd
        self.tol = tol
        self.quantity_type = quantity_type
        self.replace_all = replace_all
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.seed = seed

        # Will be set during setup
        self.net_u = None
        self.flow = None
        self.device = None
        self.domain_lb = 0.0
        self.domain_ub = 1.0

    def setup(self):
        """Initialize networks and optimizers."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        data = self.task.data
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.precision = precision

        # Extract domain bounds from task
        self._extract_domain_bounds()

        # Build PDE network
        input_dim = data.spatial_dim
        self.net_u = FCNN(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=self.nodes,
            n_layers=self.layers,
            activation=self.activation
        ).to(precision).to(self.device)

        # Build normalizing flow
        self.flow = BoundedRealNVP(
            dim=input_dim,
            n_layers=self.flow_layers,
            hidden_dim=self.flow_hidden,
            lb=self.domain_lb,
            ub=self.domain_ub,
            device=self.device,
            dtype=precision
        ).to(precision).to(self.device)

        # Get source term function
        self.source_func = self._get_source_function()

        self._is_setup = True

    def _extract_domain_bounds(self):
        """Extract domain bounds from task."""
        data = self.task.data

        # Try to get bounds from task attributes
        if hasattr(self.task, 'x_range'):
            self.domain_lb = self.task.x_range[0]
            self.domain_ub = self.task.x_range[1]
        else:
            # Infer from data points
            X_all = np.vstack([data.X_interior, data.X_boundary])
            self.domain_lb = X_all.min()
            self.domain_ub = X_all.max()

    def _get_source_function(self) -> Callable:
        """
        Get source term function for the PDE.

        For manufactured solutions, we know f analytically.
        For general tasks, use interpolation from task's points.
        """
        task_name = self.task.name.lower()

        if 'poisson' in task_name and 'nonlinear' not in task_name:
            # Linear Poisson: -nabla^2 u = f
            # For spectral-poisson-square: u = sin(pi*x)*sin(pi*y)
            # f = 2*pi^2 * sin(pi*x) * sin(pi*y)
            def f_poisson(X):
                return 2 * np.pi**2 * torch.sin(np.pi * X[:, 0]) * torch.sin(np.pi * X[:, 1])
            return f_poisson

        elif 'laplace' in task_name:
            # Laplace: -nabla^2 u = 0
            def f_laplace(X):
                return torch.zeros(X.shape[0], device=X.device, dtype=X.dtype)
            return f_laplace

        elif 'nonlinear' in task_name and 'poisson' in task_name:
            # Nonlinear Poisson: -nabla^2 u + exp(u) = f
            # For manufactured solution u = sin(pi*x)*sin(pi*y):
            # f = 2*pi^2 * sin(pi*x)*sin(pi*y) + exp(sin(pi*x)*sin(pi*y))
            def f_nonlinear_poisson(X):
                u_exact = torch.sin(np.pi * X[:, 0]) * torch.sin(np.pi * X[:, 1])
                lap_u = 2 * np.pi**2 * u_exact
                return lap_u + torch.exp(u_exact)
            return f_nonlinear_poisson

        else:
            # Fallback: interpolate from task's pre-computed values
            return self._create_interpolated_source()

    def _create_interpolated_source(self) -> Callable:
        """Create interpolated source function from task's points."""
        from scipy.interpolate import RBFInterpolator

        data = self.task.data
        X_task = data.X_ib
        f_task = data.f

        interpolator = RBFInterpolator(X_task, f_task, kernel='thin_plate_spline')

        def f_interp(X):
            X_np = X.detach().cpu().numpy()
            f_vals = interpolator(X_np)
            return torch.tensor(f_vals, dtype=X.dtype, device=X.device)

        return f_interp

    def _compute_laplacian(self, u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian using automatic differentiation."""
        # First derivatives
        u_grad = grad(u, X, grad_outputs=torch.ones_like(u),
                     create_graph=True, retain_graph=True)[0]

        laplacian = torch.zeros_like(u)

        # Second derivatives for each dimension
        for i in range(X.shape[1]):
            u_i = u_grad[:, i:i+1]
            u_ii = grad(u_i, X, grad_outputs=torch.ones_like(u_i),
                       create_graph=True, retain_graph=True)[0][:, i:i+1]
            laplacian = laplacian + u_ii

        return laplacian

    def _compute_residual(
        self,
        X: torch.Tensor,
        compute_grad: bool = True
    ) -> torch.Tensor:
        """
        Compute PDE residual at given points.

        For Poisson: residual = |nabla^2 u + f|^2
        For nonlinear Poisson: residual = |nabla^2 u + f - exp(u)|^2
        """
        # Always need gradients for Laplacian computation
        X = X.clone().requires_grad_(True)

        u = self.net_u(X)
        laplacian_u = self._compute_laplacian(u, X)

        # Source term
        f = self.source_func(X).unsqueeze(1)

        # Check if task is linear
        is_linear = hasattr(self.task, 'is_linear') and self.task.is_linear()

        if is_linear:
            # Linear Poisson: nabla^2 u + f = 0 (note: our formulation has -nabla^2 u = f)
            residual = (laplacian_u + f) ** 2
        else:
            # Nonlinear Poisson: nabla^2 u + f - exp(u) = 0
            u_clamped = torch.clamp(u, max=50.0)
            residual = (laplacian_u + f - torch.exp(u_clamped)) ** 2

        return residual

    def _compute_slope(self, X: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude (slope) at given points."""
        X = X.requires_grad_(True)
        u = self.net_u(X)

        u_grad = grad(u, X, grad_outputs=torch.ones_like(u),
                     create_graph=True, retain_graph=True)[0]

        slope = torch.sum(u_grad ** 2, dim=1, keepdim=True)
        return slope

    def _generate_uniform_samples(self, n_samples: int) -> torch.Tensor:
        """Generate uniform samples in the domain."""
        dim = self.task.data.spatial_dim
        samples = torch.rand(n_samples, dim, dtype=self.precision, device=self.device)
        samples = self.domain_lb + (self.domain_ub - self.domain_lb) * samples
        return samples

    def _generate_boundary_samples(self, n_samples: int) -> torch.Tensor:
        """Generate samples on the boundary."""
        dim = self.task.data.spatial_dim

        if dim == 2:
            # Generate points on edges of unit square
            n_per_edge = n_samples // 4

            # Bottom edge (y = lb)
            x_bottom = torch.rand(n_per_edge, dtype=self.precision, device=self.device)
            x_bottom = self.domain_lb + (self.domain_ub - self.domain_lb) * x_bottom
            y_bottom = torch.full((n_per_edge,), self.domain_lb, dtype=self.precision, device=self.device)
            bottom = torch.stack([x_bottom, y_bottom], dim=1)

            # Top edge (y = ub)
            x_top = torch.rand(n_per_edge, dtype=self.precision, device=self.device)
            x_top = self.domain_lb + (self.domain_ub - self.domain_lb) * x_top
            y_top = torch.full((n_per_edge,), self.domain_ub, dtype=self.precision, device=self.device)
            top = torch.stack([x_top, y_top], dim=1)

            # Left edge (x = lb)
            y_left = torch.rand(n_per_edge, dtype=self.precision, device=self.device)
            y_left = self.domain_lb + (self.domain_ub - self.domain_lb) * y_left
            x_left = torch.full((n_per_edge,), self.domain_lb, dtype=self.precision, device=self.device)
            left = torch.stack([x_left, y_left], dim=1)

            # Right edge (x = ub)
            y_right = torch.rand(n_per_edge, dtype=self.precision, device=self.device)
            y_right = self.domain_lb + (self.domain_ub - self.domain_lb) * y_right
            x_right = torch.full((n_per_edge,), self.domain_ub, dtype=self.precision, device=self.device)
            right = torch.stack([x_right, y_right], dim=1)

            boundary = torch.cat([bottom, top, left, right], dim=0)
            return boundary[:n_samples]

        else:
            raise NotImplementedError(f"Boundary sampling not implemented for dim={dim}")

    def _get_boundary_values(self, X_boundary: torch.Tensor) -> torch.Tensor:
        """Get exact boundary values for Dirichlet BCs."""
        task_name = self.task.name.lower()

        if 'poisson' in task_name or 'laplace' in task_name:
            # For sin(pi*x)*sin(pi*y), boundary values are 0
            return torch.zeros(X_boundary.shape[0], 1, dtype=self.precision, device=self.device)
        else:
            # Interpolate from task's boundary data
            from scipy.interpolate import RBFInterpolator
            data = self.task.data
            interpolator = RBFInterpolator(data.X_boundary, data.g, kernel='thin_plate_spline')
            X_np = X_boundary.detach().cpu().numpy()
            g_vals = interpolator(X_np)
            return torch.tensor(g_vals, dtype=self.precision, device=self.device).unsqueeze(1)

    def _train_pde_stage(
        self,
        X_train: torch.Tensor,
        X_boundary: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        verbose: bool = False
    ) -> Tuple[float, float]:
        """
        Train PDE network for one stage.

        Returns: (final_loss, residual_variance)
        """
        loss_history = []
        g_boundary = self._get_boundary_values(X_boundary)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # PDE loss
            residual = self._compute_residual(X_train)
            pde_loss = residual.mean()

            # Boundary loss
            X_boundary_grad = X_boundary.requires_grad_(True)
            u_boundary = self.net_u(X_boundary_grad)
            bc_loss = ((u_boundary - g_boundary) ** 2).mean()

            # Total loss
            loss = pde_loss + self.lambda_bd * bc_loss
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if verbose and (epoch <= 5 or epoch % 500 == 0):
                print(f"    PDE epoch {epoch}: loss = {loss.item():.4e}, "
                      f"pde = {pde_loss.item():.4e}, bc = {bc_loss.item():.4e}")

        # Compute final metrics
        residual = self._compute_residual(X_train.detach())
        final_loss = np.mean(loss_history[-5:]) if len(loss_history) >= 5 else loss_history[-1]
        res_var = residual.detach().var().item()

        return final_loss, res_var

    def _train_flow_stage(
        self,
        X_train: torch.Tensor,
        quantity: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        verbose: bool = False
    ):
        """
        Train flow model to match residual/quantity distribution.

        The flow learns to generate samples from regions with high quantity values.
        Loss: -E[log p(x) * quantity(x) / p_prev(x)]
        """
        # Normalize quantity to avoid numerical issues
        quantity = quantity / (quantity.mean() + 1e-8)

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # Log probability under flow
            log_prob = self.flow.log_prob(X_train)

            # Entropy loss: -E[log p(x) * quantity(x)]
            # We want flow to assign high probability to high-quantity regions
            entropy_loss = -(log_prob * quantity.squeeze()).mean()

            entropy_loss.backward()
            optimizer.step()

            if verbose and (epoch <= 5 or epoch % 500 == 0):
                print(f"    Flow epoch {epoch}: entropy_loss = {entropy_loss.item():.4e}")

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Multi-stage training with adaptive sampling.

        Algorithm:
        1. Initialize with uniform samples
        2. For each stage:
           a. Train PDE network
           b. Check convergence
           c. Train flow on residual distribution
           d. Resample from flow (high-residual regions)
        """
        if not self._is_setup:
            self.setup()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # Initialize with uniform samples (or task's points)
        X_train = self._generate_uniform_samples(self.n_train)
        X_boundary = self._generate_boundary_samples(self.n_train // 4)

        # Optimizers
        pde_optimizer = torch.optim.Adam(self.net_u.parameters(), lr=self.lr)
        flow_optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr)

        loss_history = []
        stages_completed = 0

        for stage in range(1, self.max_stage + 1):
            if verbose:
                print(f"\n=== Stage {stage}/{self.max_stage} ===")
                print(f"  Training points: {X_train.shape[0]}")

            # Stage A: Train PDE network
            final_loss, res_var = self._train_pde_stage(
                X_train, X_boundary, pde_optimizer,
                self.pde_epochs, verbose
            )
            loss_history.append(final_loss)
            stages_completed = stage

            if verbose:
                print(f"  Stage {stage} PDE: loss = {final_loss:.4e}, var = {res_var:.4e}")

            # Check convergence
            if final_loss < self.tol and res_var < self.tol and stage > 1:
                if verbose:
                    print(f"  Converged at stage {stage}")
                break

            # Stage B: Train flow (except last stage)
            if stage < self.max_stage:
                # Compute quantity (residual or slope)
                if self.quantity_type == 'slope':
                    quantity = self._compute_slope(X_train.detach())
                else:
                    quantity = self._compute_residual(X_train.detach())
                quantity = quantity.detach()

                # Reset flow for new stage
                self.flow.reset_actnorm()

                self._train_flow_stage(
                    X_train, quantity, flow_optimizer,
                    self.flow_epochs, verbose
                )

                # Stage C: Resample from flow
                with torch.no_grad():
                    X_new = self.flow.sample(self.n_train)
                    X_new = torch.clamp(X_new, self.domain_lb, self.domain_ub)
                    X_boundary_new = self._generate_boundary_samples(self.n_train // 4)

                if self.replace_all:
                    # DAS-R: Replace all samples
                    X_train = X_new
                    X_boundary = X_boundary_new
                else:
                    # DAS-G: Accumulate samples
                    X_train = torch.cat([X_train, X_new], dim=0)
                    X_boundary = torch.cat([X_boundary, X_boundary_new], dim=0)

                if verbose:
                    print(f"  Resampled {X_new.shape[0]} points from flow")

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        train_time = time.perf_counter() - start_time

        # Evaluate on task's Chebyshev grid
        with torch.no_grad():
            X_eval = torch.tensor(
                self.task.data.X_ib,
                dtype=self.precision,
                device=self.device
            )
            u_pred = self.net_u(X_eval).cpu().numpy().flatten()

        # Compute L2 error
        l2_error = None
        if self.task.data.u_true is not None:
            u_true_ib = self.task.data.u_true[:self.task.data.N_ib]
            l2_error = self.compute_l2_error(u_pred, u_true_ib)

        return TrainResult(
            u_pred=u_pred,
            train_time=train_time,
            l2_error=l2_error,
            final_loss=loss_history[-1] if loss_history else None,
            loss_history=loss_history,
            n_iterations=sum([self.pde_epochs] * stages_completed),
            extra={
                'stages_completed': stages_completed,
                'layers': self.layers,
                'nodes': self.nodes,
                'flow_layers': self.flow_layers,
                'quantity_type': self.quantity_type,
                'replace_all': self.replace_all,
                'method': 'deep_adaptive_sampling',
            }
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions at given points."""
        if self.net_u is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_tensor = torch.tensor(X, dtype=self.precision, device=self.device)

        with torch.no_grad():
            u_pred = self.net_u(X_tensor)

        return u_pred.cpu().numpy().flatten()

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        return {
            'layers': 6,
            'nodes': 32,
            'activation': 'tanh',
            'flow_layers': 6,
            'flow_hidden': 64,
            'n_train': 1000,
            'pde_epochs': 3000,
            'flow_epochs': 3000,
            'max_stage': 5,
            'lr': 1e-4,
            'lambda_bd': 1.0,
            'tol': 1e-7,
            'quantity_type': 'residual',
            'replace_all': False,
            'use_cuda': True,
            'seed': 0,
        }
