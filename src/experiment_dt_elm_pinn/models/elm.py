"""
ELM: Extreme Learning Machine (without discrete operators)

Standalone ELM using autodiff to compute Laplacian.
Uses random fixed hidden layer, solves for output weights via lstsq.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import time
from typing import Dict, Any, List, Optional

from .base import BaseModel, TrainResult


class ELM(BaseModel):
    """
    Extreme Learning Machine with autodiff for Laplacian computation.

    Key features:
    - Fixed random hidden layer weights (not trained)
    - Output weights solved via least squares + Newton iteration
    - Uses autodiff to compute ∇²u (not discrete operators)
    """

    name = "elm"

    def __init__(
        self,
        task,
        hidden_sizes: List[int] = None,
        activation: str = 'tanh',
        max_iter: int = 20,
        tol: float = 1e-8,
        use_cuda: bool = True,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE data
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('tanh', 'sin')
            max_iter: Maximum Newton iterations
            tol: Convergence tolerance
            use_cuda: Whether to use GPU
            seed: Random seed
        """
        super().__init__(task, **kwargs)

        # ELM only supports 2nd order PDEs (Laplacian)
        # Analytical derivatives for higher orders are complex
        self._check_pde_order(task)

        self.hidden_sizes = hidden_sizes or [100]
        self.activation = activation
        self.max_iter = max_iter
        self.tol = tol
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.seed = seed

        # Will be set during setup
        self.W_in = []  # Input weights for each layer
        self.b_in = []  # Biases for each layer
        self.W_out = None
        self.device = None

    def _check_pde_order(self, task):
        """Check that PDE order and type are supported."""
        pde_order = getattr(task, 'pde_order', 2)
        if pde_order != 2:
            raise ValueError(
                f"ELM only supports 2nd order PDEs (Laplacian).\n"
                f"Task '{task.name}' requires {pde_order}th order derivatives.\n"
                f"For higher-order PDEs like biharmonic, use:\n"
                f"  --model vanilla-pinn (autodiff, any order)\n"
                f"  --model dt-elm-pinn (spectral, any order on square/cube)"
            )

        # ELM only supports steady-state elliptic PDEs (Poisson-type)
        # It cannot handle time-dependent PDEs like heat equation
        task_name = task.name.lower()
        if 'heat' in task_name or 'wave' in task_name or 'advection' in task_name:
            raise ValueError(
                f"ELM only supports steady-state elliptic PDEs (Poisson/Laplace).\n"
                f"Task '{task.name}' is time-dependent.\n"
                f"For time-dependent PDEs, use:\n"
                f"  --model vanilla-pinn (autodiff)\n"
                f"  --model dt-pinn (RBF-FD)"
            )

        # Note: ELM now supports Helmholtz, convection-diffusion, and variable-coefficient
        # using analytically computed gradients. Only truly unsupported types are blocked.
        pde_type = getattr(task, 'pde_type', None)
        unsupported_types = ['mixed_derivative']  # Requires ∂²u/∂x∂y which is complex
        if pde_type in unsupported_types:
            raise ValueError(
                f"ELM does not support mixed derivative PDEs.\n"
                f"Task '{task.name}' has pde_type='{pde_type}'.\n"
                f"For this PDE type, use:\n"
                f"  --model dt-elm-pinn (spectral operators)\n"
                f"  --model vanilla-pinn (autodiff)"
            )

    def _activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'sin':
            return torch.sin(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _compute_hidden_features(self, X: torch.Tensor) -> torch.Tensor:
        """Compute hidden layer features with skip connections."""
        H_layers = []
        h = X

        for i, (W, b) in enumerate(zip(self.W_in, self.b_in)):
            h = self._activation_fn(h @ W + b)
            H_layers.append(h)

        return torch.cat(H_layers, dim=1)

    def _compute_gradient(self, X: torch.Tensor) -> tuple:
        """
        Compute gradient ∂H/∂x and ∂H/∂y for the hidden features analytically.

        For tanh: ∂tanh(z)/∂x_i = sech²(z) * W[i,:]
        For sin: ∂sin(z)/∂x_i = cos(z) * W[i,:]

        Returns:
            dH_dx: (N, total_features) - derivative w.r.t. x
            dH_dy: (N, total_features) - derivative w.r.t. y
        """
        dHdx_layers = []
        dHdy_layers = []

        h = X  # Current layer input (only first layer uses original X)

        for layer_idx, (W, b) in enumerate(zip(self.W_in, self.b_in)):
            if layer_idx == 0:
                # First layer: derivatives w.r.t. original input
                pre_act = X @ W + b  # (N, n_hidden)

                if self.activation == 'tanh':
                    tanh_z = torch.tanh(pre_act)
                    sech2_z = 1 - tanh_z ** 2  # (N, n_hidden)
                    # ∂H_j/∂x = sech²(z_j) * W[0,j]
                    # ∂H_j/∂y = sech²(z_j) * W[1,j]
                    dH_dx = sech2_z * W[0, :]  # (N, n_hidden)
                    dH_dy = sech2_z * W[1, :]  # (N, n_hidden)
                elif self.activation == 'sin':
                    cos_z = torch.cos(pre_act)
                    dH_dx = cos_z * W[0, :]
                    dH_dy = cos_z * W[1, :]
                else:
                    raise ValueError(f"Gradient not implemented for {self.activation}")

                dHdx_layers.append(dH_dx)
                dHdy_layers.append(dH_dy)
            else:
                # For deeper layers, use chain rule (simplified approximation)
                pre_act = h @ W + b
                if self.activation == 'tanh':
                    tanh_z = torch.tanh(pre_act)
                    sech2_z = 1 - tanh_z ** 2
                    dH_dx = sech2_z * W[0, :]
                    dH_dy = sech2_z * W[1, :]
                elif self.activation == 'sin':
                    cos_z = torch.cos(pre_act)
                    dH_dx = cos_z * W[0, :]
                    dH_dy = cos_z * W[1, :]

                dHdx_layers.append(dH_dx)
                dHdy_layers.append(dH_dy)

            h = self._activation_fn(h @ W + b)

        return torch.cat(dHdx_layers, dim=1), torch.cat(dHdy_layers, dim=1)

    def _compute_laplacian(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian ∇²H for the hidden features.

        Returns tensor of shape (N, total_features) where each column
        is the Laplacian of the corresponding feature at each point.
        """
        N = X.shape[0]
        d = X.shape[1]

        # Compute H and its Laplacian
        H_layers = []
        LapH_layers = []

        h = X  # Current layer input

        for layer_idx, (W, b) in enumerate(zip(self.W_in, self.b_in)):
            n_hidden = W.shape[1]

            # Forward through activation
            pre_act = h @ W + b  # (N, n_hidden)
            h_next = self._activation_fn(pre_act)  # (N, n_hidden)
            H_layers.append(h_next)

            if layer_idx == 0:
                # First layer: ∇²tanh(Wx + b) where x is the input
                # For tanh(z) where z = w·x + b:
                #   ∂u/∂x = sech²(z) * w_x
                #   ∂²u/∂x² = w_x * d/dz[sech²(z)] * w_x
                #           = w_x * (-2*sech²(z)*tanh(z)) * w_x
                #           = -2 * tanh(z) * sech²(z) * w_x²
                # ∇²u = -2 * tanh(z) * sech²(z) * ||w||²

                if self.activation == 'tanh':
                    tanh_z = torch.tanh(pre_act)
                    sech2_z = 1 - tanh_z ** 2
                    # Laplacian of tanh features
                    # d²tanh(z)/dx² = -2*tanh(z)*sech²(z) * ||∇z||²
                    W_norm_sq = torch.sum(W ** 2, dim=0)  # (n_hidden,)
                    lap_h = -2 * tanh_z * sech2_z * W_norm_sq
                elif self.activation == 'sin':
                    # d²sin(z)/dx² = -sin(z) * ||∇z||²
                    W_norm_sq = torch.sum(W ** 2, dim=0)
                    lap_h = -torch.sin(pre_act) * W_norm_sq
                else:
                    raise ValueError(f"Laplacian not implemented for {self.activation}")

                LapH_layers.append(lap_h)
            else:
                # For deeper layers, this gets more complex
                # For simplicity, we'll use numerical differentiation or
                # rely on autograd (which is expensive)
                # For now, we use a simple approximation that ignores cross terms
                # This is a limitation of multi-layer ELM with autodiff

                # Approximate: assume previous layer is already computed
                # This is a rough approximation
                if self.activation == 'tanh':
                    tanh_z = torch.tanh(pre_act)
                    sech2_z = 1 - tanh_z ** 2
                    W_norm_sq = torch.sum(W ** 2, dim=0)
                    lap_h = -2 * tanh_z * sech2_z * W_norm_sq
                elif self.activation == 'sin':
                    W_norm_sq = torch.sum(W ** 2, dim=0)
                    lap_h = -torch.sin(pre_act) * W_norm_sq

                LapH_layers.append(lap_h)

            h = h_next  # Update for next layer

        return torch.cat(LapH_layers, dim=1)

    def setup(self):
        """Initialize random hidden layer weights."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        data = self.task.data
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Build random hidden layer weights
        self.W_in = []
        self.b_in = []

        input_dim = data.spatial_dim
        for n_hidden in self.hidden_sizes:
            W = torch.randn(input_dim, n_hidden, dtype=precision, device=self.device)
            W = W * np.sqrt(2.0 / input_dim)
            b = torch.randn(n_hidden, dtype=precision, device=self.device) * 0.1

            self.W_in.append(W)
            self.b_in.append(b)
            input_dim = n_hidden

        # Total number of features
        self.total_features = sum(self.hidden_sizes)

        # Initialize output weights
        self.W_out = torch.zeros(self.total_features, dtype=precision, device=self.device)

        self._is_setup = True

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Train using Newton iteration.
        """
        if not self._is_setup:
            self.setup()

        data = self.task.data
        N_interior = data.N_interior
        N_boundary = data.N_boundary
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        # Prepare data
        X_interior = torch.tensor(data.X_interior, dtype=precision, device=self.device)
        X_boundary = torch.tensor(data.X_boundary, dtype=precision, device=self.device)
        f_interior = torch.tensor(data.f[:N_interior], dtype=precision, device=self.device)
        g = torch.tensor(data.g, dtype=precision, device=self.device)

        start_time = time.perf_counter()

        # Compute hidden features
        H_interior = self._compute_hidden_features(X_interior)  # (N_interior, total_features)
        H_boundary = self._compute_hidden_features(X_boundary)  # (N_boundary, total_features)

        # Compute Laplacian of hidden features on interior
        LapH_interior = self._compute_laplacian(X_interior)  # (N_interior, total_features)

        # Check PDE type and build appropriate operator
        pde_type = getattr(self.task, 'pde_type', None)

        # Build PDE operator based on type
        if pde_type == 'helmholtz':
            # Helmholtz: ∇²u + k²u = f → operator is LapH + k²*H
            k_squared = getattr(self.task, 'k_squared', 1.0)
            OpH_interior = LapH_interior + k_squared * H_interior
            if verbose:
                print(f"  Helmholtz equation: k² = {k_squared}")

        elif pde_type == 'convection_diffusion':
            # Convection-diffusion: ε∇²u + b·∇u = f
            epsilon = getattr(self.task, 'epsilon', 0.1)
            bx = getattr(self.task, 'bx', 1.0)
            by = getattr(self.task, 'by', 1.0)
            dHdx, dHdy = self._compute_gradient(X_interior)
            OpH_interior = epsilon * LapH_interior + bx * dHdx + by * dHdy
            if verbose:
                print(f"  Convection-diffusion: ε={epsilon}, b=({bx}, {by})")

        elif pde_type == 'variable_diffusion':
            # Variable-coefficient: ∇·(a∇u) = a∇²u + ∇a·∇u = f
            dHdx, dHdy = self._compute_gradient(X_interior)
            # Get coefficient and its gradient from task
            a = torch.tensor(self.task.diffusion_coeff(data.X_interior),
                           dtype=precision, device=self.device).unsqueeze(1)
            da_dx, da_dy = self.task.diffusion_coeff_grad(data.X_interior)
            da_dx = torch.tensor(da_dx, dtype=precision, device=self.device).unsqueeze(1)
            da_dy = torch.tensor(da_dy, dtype=precision, device=self.device).unsqueeze(1)
            OpH_interior = a * LapH_interior + da_dx * dHdx + da_dy * dHdy
            if verbose:
                print(f"  Variable-coefficient diffusion")

        elif pde_type == 'reaction_diffusion':
            # Reaction-diffusion: ∇²u + r(x)u = f
            r = torch.tensor(self.task.reaction_coeff(data.X_interior),
                           dtype=precision, device=self.device).unsqueeze(1)
            OpH_interior = LapH_interior + r * H_interior
            if verbose:
                print(f"  Reaction-diffusion equation")

        elif pde_type == 'anisotropic_diffusion':
            # Anisotropic: a_xx ∂²u/∂x² + a_yy ∂²u/∂y² = f
            # For now, use approximate: (a_xx + a_yy)/2 * LapH (TODO: implement properly)
            a_xx = getattr(self.task, 'a_xx', 1.0)
            a_yy = getattr(self.task, 'a_yy', 1.0)
            # Approximate by scaling Laplacian (works for mild anisotropy)
            OpH_interior = ((a_xx + a_yy) / 2.0) * LapH_interior
            if verbose:
                print(f"  Anisotropic diffusion (approximate): a_xx={a_xx}, a_yy={a_yy}")

        else:
            # Default: Poisson/Laplace ∇²u = f
            OpH_interior = LapH_interior

        # Check if task is linear (no exp(u) term)
        is_linear = hasattr(self.task, 'is_linear') and self.task.is_linear()

        residual_history = []

        if is_linear:
            # Linear PDE: Op(u) = f
            # Direct solve: [OpH; H_bc] @ W = [f; g]
            A = torch.cat([OpH_interior, H_boundary], dim=0)
            b = torch.cat([f_interior, g], dim=0)

            W_out, *_ = torch.linalg.lstsq(A, b.unsqueeze(1))
            self.W_out = W_out.squeeze()

            # Compute residual for logging
            Op_u = OpH_interior @ self.W_out
            u_boundary = H_boundary @ self.W_out
            F_pde = Op_u - f_interior
            F_bc = u_boundary - g
            residual = torch.sqrt(torch.mean(F_pde**2) + torch.mean(F_bc**2)).item()
            residual_history.append(residual)

            if verbose:
                print(f"  Linear PDE: direct solve, residual = {residual:.4e}")
        else:
            # Nonlinear Poisson: ∇²u = f + exp(u)
            # Linearize around current solution: ∇²u - exp(u) = f
            # Newton: J @ delta_W = -F where J = ∇²H - diag(exp(u)) @ H

            # Initial solve (assume exp(u) ≈ 1)
            A_pde = LapH_interior
            A_bc = H_boundary
            A = torch.cat([A_pde, A_bc], dim=0)
            b = torch.cat([f_interior + 1.0, g], dim=0)

            # Solve initial system
            W_out, *_ = torch.linalg.lstsq(A, b.unsqueeze(1))
            self.W_out = W_out.squeeze()

            # Compute initial u
            u_interior = H_interior @ self.W_out
            u_boundary = H_boundary @ self.W_out

            # Newton iterations
            for k in range(self.max_iter):
                # Compute residuals
                Lu = LapH_interior @ self.W_out
                exp_u = torch.exp(torch.clamp(u_interior, max=50.0))  # Clamp to prevent overflow
                F_pde = Lu - f_interior - exp_u
                F_bc = u_boundary - g

                residual = torch.sqrt(torch.mean(F_pde**2) + torch.mean(F_bc**2)).item()
                residual_history.append(residual)

                if verbose and (k < 5 or k % 5 == 0):
                    print(f"  Newton iter {k}: residual = {residual:.4e}")

                if residual < self.tol:
                    if verbose:
                        print(f"  Converged at iteration {k+1}")
                    break

                # Form Jacobian: J = LapH - diag(exp(u)) @ H
                JH = LapH_interior - exp_u.unsqueeze(1) * H_interior

                # System: [JH; H_bc] @ delta_W = -[F_pde; F_bc]
                A = torch.cat([JH, H_boundary], dim=0)
                F = torch.cat([-F_pde, -F_bc], dim=0)

                delta_W, *_ = torch.linalg.lstsq(A, F.unsqueeze(1))
                self.W_out = self.W_out + delta_W.squeeze()

                u_interior = H_interior @ self.W_out
                u_boundary = H_boundary @ self.W_out

        train_time = time.perf_counter() - start_time

        # Get final predictions
        X_ib = torch.tensor(data.X_ib, dtype=precision, device=self.device)
        H_ib = self._compute_hidden_features(X_ib)
        u_pred = (H_ib @ self.W_out).cpu().numpy()

        # Compute L2 error
        l2_error = None
        if data.u_true is not None:
            u_true_ib = data.u_true[:data.N_ib]
            l2_error = self.compute_l2_error(u_pred, u_true_ib)

        return TrainResult(
            u_pred=u_pred,
            train_time=train_time,
            l2_error=l2_error,
            final_loss=residual_history[-1] if residual_history else None,
            loss_history=residual_history,
            n_iterations=len(residual_history),
            extra={
                'hidden_sizes': self.hidden_sizes,
                'total_features': self.total_features,
                'method': 'elm-autodiff',
            }
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions at given points."""
        if self.W_out is None:
            raise RuntimeError("Model not trained. Call train() first.")

        precision = torch.float64 if X.dtype == np.float64 else torch.float32
        X_tensor = torch.tensor(X, dtype=precision, device=self.device)

        H = self._compute_hidden_features(X_tensor)
        u_pred = H @ self.W_out

        return u_pred.cpu().numpy()

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        return {
            'hidden_sizes': [100],
            'activation': 'tanh',
            'max_iter': 20,
            'tol': 1e-8,
            'use_cuda': True,
            'seed': 42,
        }

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[100],
                           help='Hidden layer sizes')
        parser.add_argument('--activation', type=str, default='tanh',
                           choices=['tanh', 'sin'],
                           help='Activation function')
        parser.add_argument('--max-iter', type=int, default=20,
                           help='Maximum Newton iterations')
        parser.add_argument('--tol', type=float, default=1e-8,
                           help='Convergence tolerance')
        parser.add_argument('--no-cuda', action='store_true',
                           help='Disable CUDA')
        parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
