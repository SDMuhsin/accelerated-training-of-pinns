"""
PIELM: Physics-Informed Extreme Learning Machine

Implementation based on:
Dwivedi & Srinivasan (2020) "Physics Informed Extreme Learning Machine (PIELM) -
A rapid method for the numerical solution of partial differential equations"
Neurocomputing, https://arxiv.org/abs/1907.03507

Key differences from DT-ELM-PINN:
- Uses analytical derivatives of hidden layer (not discrete operators)
- Single hidden layer with sigmoid activation (paper's default)
- Uniform random weight initialization in [-1, 1]
- Direct least-squares solve for linear PDEs
- Newton iteration for nonlinear PDEs
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional

from .base import BaseModel, TrainResult


class PIELM(BaseModel):
    """
    Physics-Informed Extreme Learning Machine (PIELM).

    For 2D Poisson: ∇²u = f (linear) or ∇²u = f + g(u) (nonlinear)

    Network: u(x,y) = sum_j β_j * σ(w_j · [x,y] + b_j)

    Where:
    - σ is the activation function (sigmoid by default)
    - w_j, b_j are random, FIXED (not trained)
    - β_j are output weights solved via least squares

    The Laplacian is computed analytically using the chain rule.
    """

    name = "pielm"

    def __init__(
        self,
        task,
        n_hidden: int = 100,
        activation: str = 'sigmoid',
        weight_range: float = 1.0,
        max_iter: int = 20,
        tol: float = 1e-8,
        seed: int = 42,
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE data
            n_hidden: Number of hidden neurons
            activation: Activation function ('sigmoid', 'tanh', 'sin')
            weight_range: Random weights sampled from [-weight_range, weight_range]
            max_iter: Maximum Newton iterations (for nonlinear PDEs)
            tol: Convergence tolerance
            seed: Random seed for reproducibility
        """
        super().__init__(task, **kwargs)

        self.n_hidden = n_hidden
        self.activation = activation
        self.weight_range = weight_range
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        # Will be set during setup
        self.W = None  # Input weights (2, n_hidden) for 2D
        self.b = None  # Biases (n_hidden,)
        self.beta = None  # Output weights (n_hidden,)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))

    def _sigmoid_derivative(self, z: np.ndarray) -> np.ndarray:
        """First derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
        s = self._sigmoid(z)
        return s * (1 - s)

    def _sigmoid_second_derivative(self, z: np.ndarray) -> np.ndarray:
        """Second derivative of sigmoid: σ''(z) = σ'(z)(1 - 2σ(z))"""
        s = self._sigmoid(z)
        sp = s * (1 - s)
        return sp * (1 - 2 * s)

    def _activation_fn(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sin':
            return np.sin(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_second_derivative(self, z: np.ndarray) -> np.ndarray:
        """Second derivative of activation function."""
        if self.activation == 'sigmoid':
            return self._sigmoid_second_derivative(z)
        elif self.activation == 'tanh':
            # d²tanh(z)/dz² = -2 * tanh(z) * sech²(z) = -2 * tanh(z) * (1 - tanh²(z))
            t = np.tanh(z)
            return -2 * t * (1 - t**2)
        elif self.activation == 'sin':
            return -np.sin(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute hidden layer features H.

        H[i,j] = σ(W[:,j] · X[i,:] + b[j])

        Args:
            X: Input points (N, 2)

        Returns:
            H: Hidden features (N, n_hidden)
        """
        z = X @ self.W + self.b  # (N, n_hidden)
        return self._activation_fn(z)

    def _compute_laplacian_features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Laplacian of hidden layer features analytically.

        For u_j(x,y) = σ(w1_j * x + w2_j * y + b_j):

        ∂u_j/∂x = σ'(z_j) * w1_j
        ∂²u_j/∂x² = σ''(z_j) * w1_j²

        ∇²u_j = σ''(z_j) * (w1_j² + w2_j²)

        Args:
            X: Input points (N, 2)

        Returns:
            LapH: Laplacian of features (N, n_hidden)
        """
        z = X @ self.W + self.b  # (N, n_hidden)
        sigma_pp = self._activation_second_derivative(z)  # (N, n_hidden)

        # ||w_j||² for each hidden neuron
        W_norm_sq = np.sum(self.W**2, axis=0)  # (n_hidden,)

        # ∇²H[i,j] = σ''(z[i,j]) * ||w_j||²
        return sigma_pp * W_norm_sq

    def setup(self):
        """Initialize random hidden layer weights."""
        np.random.seed(self.seed)

        data = self.task.data
        spatial_dim = data.spatial_dim
        precision = data.X_full.dtype

        # PIELM uses uniform distribution in [-weight_range, weight_range]
        self.W = np.random.uniform(
            -self.weight_range, self.weight_range,
            (spatial_dim, self.n_hidden)
        ).astype(precision)

        self.b = np.random.uniform(
            -self.weight_range, self.weight_range,
            self.n_hidden
        ).astype(precision)

        # Initialize output weights
        self.beta = np.zeros(self.n_hidden, dtype=precision)

        self._is_setup = True

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Train PIELM using analytical derivatives.

        For linear Poisson (∇²u = f):
            Solve [LapH; H_bc] @ beta = [f; g] directly

        For nonlinear Poisson (∇²u = f + exp(u)):
            Use Newton iteration to handle nonlinearity
        """
        if not self._is_setup:
            self.setup()

        data = self.task.data
        N_interior = data.N_interior
        N_boundary = data.N_boundary
        N_ib = data.N_ib

        # Get coordinates
        X_interior = data.X_interior
        X_boundary = data.X_boundary
        f_interior = data.f[:N_interior]
        g_boundary = data.g

        start_time = time.perf_counter()

        # Compute features and their Laplacians
        H_interior = self._compute_features(X_interior)
        H_boundary = self._compute_features(X_boundary)
        LapH_interior = self._compute_laplacian_features(X_interior)

        # Check if this is a nonlinear problem (has exp(u) term)
        # For our nonlinear Poisson: ∇²u = f + exp(u)
        is_nonlinear = True  # Assume nonlinear for this task

        residual_history = []

        if is_nonlinear:
            # Newton iteration for nonlinear PDE
            # Initial solve: assume exp(u) ≈ 1
            A_init = np.vstack([LapH_interior, H_boundary])
            b_init = np.concatenate([f_interior + 1.0, g_boundary])
            self.beta, *_ = np.linalg.lstsq(A_init, b_init, rcond=None)

            u_interior = H_interior @ self.beta

            for k in range(self.max_iter):
                # Compute residuals
                Lu = LapH_interior @ self.beta
                exp_u = np.exp(u_interior)
                F_pde = Lu - f_interior - exp_u
                F_bc = H_boundary @ self.beta - g_boundary

                residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
                residual_history.append(residual)

                if verbose and (k < 5 or k % 5 == 0):
                    print(f"  Newton iter {k}: residual = {residual:.4e}")

                if residual < self.tol:
                    if verbose:
                        print(f"  Converged at iteration {k+1}")
                    break

                # Form Jacobian: J = LapH - diag(exp(u)) @ H
                JH = LapH_interior - exp_u[:, np.newaxis] * H_interior

                # Solve: [JH; H_bc] @ delta_beta = -[F_pde; F_bc]
                A = np.vstack([JH, H_boundary])
                F = np.concatenate([-F_pde, -F_bc])

                delta_beta, *_ = np.linalg.lstsq(A, F, rcond=None)
                self.beta = self.beta + delta_beta

                u_interior = H_interior @ self.beta
        else:
            # Direct solve for linear PDE
            A = np.vstack([LapH_interior, H_boundary])
            b = np.concatenate([f_interior, g_boundary])
            self.beta, *_ = np.linalg.lstsq(A, b, rcond=None)

        train_time = time.perf_counter() - start_time

        # Compute predictions at interior + boundary
        X_ib = data.X_ib
        H_ib = self._compute_features(X_ib)
        u_pred = H_ib @ self.beta

        # Compute L2 error
        l2_error = None
        if data.u_true is not None:
            u_true_ib = data.u_true[:N_ib]
            l2_error = self.compute_l2_error(u_pred, u_true_ib)

        return TrainResult(
            u_pred=u_pred,
            train_time=train_time,
            l2_error=l2_error,
            final_loss=residual_history[-1] if residual_history else None,
            loss_history=residual_history,
            n_iterations=len(residual_history),
            extra={
                'n_hidden': self.n_hidden,
                'activation': self.activation,
                'method': 'pielm-analytical',
            }
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions at given points."""
        if self.beta is None:
            raise RuntimeError("Model not trained. Call train() first.")

        H = self._compute_features(X)
        return H @ self.beta

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        return {
            'n_hidden': 100,
            'activation': 'sigmoid',
            'weight_range': 1.0,
            'max_iter': 20,
            'tol': 1e-8,
            'seed': 42,
        }

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--n-hidden', type=int, default=100,
                           help='Number of hidden neurons')
        parser.add_argument('--activation', type=str, default='sigmoid',
                           choices=['sigmoid', 'tanh', 'sin'],
                           help='Activation function')
        parser.add_argument('--weight-range', type=float, default=1.0,
                           help='Random weights in [-range, range]')
        parser.add_argument('--max-iter', type=int, default=20,
                           help='Maximum Newton iterations')
        parser.add_argument('--tol', type=float, default=1e-8,
                           help='Convergence tolerance')
        parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
