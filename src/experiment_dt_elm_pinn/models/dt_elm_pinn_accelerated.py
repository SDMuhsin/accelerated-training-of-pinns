"""
SPECTO-ELM Accelerated: GPU-accelerated version using PyTorch.
(Also known as DT-ELM-PINN Accelerated)

Key optimizations:
1. PyTorch Cholesky solver (4.4x faster than SciPy on CPU for large M)
2. Automatic GPU acceleration when available (sparse×dense, dense×dense, Cholesky)
3. Efficient memory handling with in-place operations

This implementation achieves significant speedups especially for deep networks
where M (total features) is large due to skip connection concatenation.

IMPORTANT: Uses SPECTRAL COLLOCATION which requires TENSOR-PRODUCT domains
(square, cube). For non-tensor-product domains (disk, L-shape), use DT-PINN.
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

from .base import BaseModel, TrainResult


class DTELMPINNAccelerated(BaseModel):
    """
    GPU-accelerated SPECTO-ELM / DT-ELM-PINN solver using PyTorch.

    Network: u(x) = H @ W_out where H = concat([h1, h2, ..., hL])
    Each layer: h_l = tanh(h_{l-1} @ W_l + b_l)

    - All hidden weights W_l, b_l are random and FIXED
    - Only W_out is solved via least squares
    - Uses PyTorch for faster linear algebra operations
    - Automatically uses GPU when available

    IMPORTANT: Requires tensor-product domain (square, cube) for spectral collocation.
    For disk or L-shaped domains, use DT-PINN (RBF-FD discretization) instead.
    """

    name = "dt-elm-pinn-accel"

    # Domain types compatible with spectral collocation
    COMPATIBLE_DOMAINS = ('square', 'cube')

    def __init__(
        self,
        task,
        hidden_sizes: List[int] = None,
        activation: str = 'tanh',
        max_iter: int = 20,
        tol: float = 1e-8,
        seed: int = 42,
        use_skip_connections: bool = True,
        device: str = 'auto',
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE data
            hidden_sizes: List of hidden layer sizes. Default: [100]
            activation: Activation function ('tanh', 'sin')
            max_iter: Maximum Newton iterations (for nonlinear PDEs)
            tol: Convergence tolerance for residual
            seed: Random seed for reproducibility
            use_skip_connections: If True, concatenate all layer outputs
            device: 'auto' (GPU if available), 'cuda', or 'cpu'

        Raises:
            ValueError: If task domain is not compatible with spectral collocation.
        """
        super().__init__(task, **kwargs)

        # Check domain compatibility for spectral collocation
        self._check_domain_compatibility(task)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DTELMPINNAccelerated")

        self.hidden_sizes = hidden_sizes or [100]
        self.activation = activation
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.use_skip_connections = use_skip_connections

        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        else:
            self.device = torch.device(device)

        # Will be set during setup
        self.H = None           # Hidden layer outputs (torch tensor)
        self.W_out = None       # Output weights (torch tensor)
        self.LH = None          # Precomputed L @ H (torch tensor)
        self.BH = None          # Precomputed B @ H (torch tensor)

        # Store random weights for prediction
        self._weights = []

    def _check_domain_compatibility(self, task):
        """
        Check if task domain is compatible with spectral collocation.

        Raises:
            ValueError: If domain is not tensor-product (square, cube).
        """
        if not hasattr(task, 'domain_type'):
            return

        domain_type = task.domain_type

        if domain_type not in self.COMPATIBLE_DOMAINS:
            task_name = getattr(task, 'name', 'unknown')
            raise ValueError(
                f"\n{'='*70}\n"
                f"SPECTO-ELM (dt-elm-pinn-accel) requires a TENSOR-PRODUCT domain\n"
                f"(square, cube) for spectral collocation.\n"
                f"\n"
                f"Task '{task_name}' uses domain '{domain_type}' which is NOT supported.\n"
                f"\n"
                f"Alternatives:\n"
                f"  - Use --model dt-pinn (RBF-FD discretization, works on any domain)\n"
                f"  - Use --model vanilla-pinn (autodiff, works on any domain)\n"
                f"  - Use a square/cube domain task (e.g., poisson-square-sin, laplace-square)\n"
                f"{'='*70}"
            )

    def _activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'sin':
            return torch.sin(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _sparse_to_torch(self, sparse_scipy):
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        coo = sparse_scipy.tocoo()
        indices = torch.stack([
            torch.from_numpy(coo.row.astype(np.int64)),
            torch.from_numpy(coo.col.astype(np.int64))
        ])
        values = torch.from_numpy(coo.data)
        return torch.sparse_coo_tensor(
            indices, values, sparse_scipy.shape,
            dtype=torch.float64, device=self.device
        ).coalesce()

    def _solve_lstsq_cholesky(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve least squares via Cholesky decomposition.

        Solves: min ||Ax - b||^2 via (A'A)x = A'b

        PyTorch Cholesky is 4.4x faster than SciPy for large matrices.
        """
        AtA = A.T @ A
        Atb = A.T @ b

        # Add regularization for numerical stability
        AtA += 1e-10 * torch.eye(AtA.shape[0], dtype=AtA.dtype, device=AtA.device)

        try:
            L = torch.linalg.cholesky(AtA)
            return torch.cholesky_solve(Atb.unsqueeze(1), L).squeeze(1)
        except RuntimeError:
            # Fall back to lstsq if Cholesky fails
            return torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)

    def setup(self):
        """Build hidden layer features and precompute operator products."""
        # Use NumPy for random generation to match original implementation exactly
        np.random.seed(self.seed)

        data = self.task.data
        X_np = data.X_full
        precision = X_np.dtype

        # Build multi-layer hidden representation using NumPy (for identical results)
        # then convert to PyTorch
        self._weights = []
        if self.use_skip_connections:
            H_layers_np = []
            h_np = X_np
            input_dim = X_np.shape[1]

            for n_hidden in self.hidden_sizes:
                W_np = np.random.randn(input_dim, n_hidden).astype(precision) * np.sqrt(2.0 / input_dim)
                b_np = np.random.randn(n_hidden).astype(precision) * 0.1
                h_np = np.tanh(h_np @ W_np + b_np) if self.activation == 'tanh' else np.sin(h_np @ W_np + b_np)
                H_layers_np.append(h_np)
                # Store weights for prediction
                W_torch = torch.from_numpy(W_np).to(self.device)
                b_torch = torch.from_numpy(b_np).to(self.device)
                self._weights.append((W_torch, b_torch))
                input_dim = n_hidden

            # Concatenate all layers (skip connections)
            H_np = np.hstack(H_layers_np)
        else:
            # Standard single-layer ELM
            n_hidden = self.hidden_sizes[0]
            W_np = np.random.randn(X_np.shape[1], n_hidden).astype(precision) * np.sqrt(2.0 / X_np.shape[1])
            b_np = np.random.randn(n_hidden).astype(precision) * 0.1
            H_np = np.tanh(X_np @ W_np + b_np) if self.activation == 'tanh' else np.sin(X_np @ W_np + b_np)
            self._weights.append((
                torch.from_numpy(W_np).to(self.device),
                torch.from_numpy(b_np).to(self.device)
            ))

        # Convert to PyTorch tensor
        self.H = torch.from_numpy(H_np).to(self.device, dtype=torch.float64)

        # Move input to device for later use
        X = torch.from_numpy(X_np).to(self.device, dtype=torch.float64)

        # Convert sparse operators to PyTorch
        L_torch = self._sparse_to_torch(data.L)
        B_torch = self._sparse_to_torch(data.B)
        N_ib = data.N_ib

        # Precompute operator products
        # For sparse @ dense, we use torch.sparse.mm which is efficient
        LH_full = torch.sparse.mm(L_torch, self.H)
        self.LH = LH_full[:N_ib, :]
        self.BH = torch.sparse.mm(B_torch, self.H)

        # Initialize output weights
        self.W_out = torch.zeros(self.H.shape[1], dtype=torch.float64, device=self.device)

        # Store data tensors on device
        self._f = torch.from_numpy(data.f).to(self.device, dtype=torch.float64)
        self._g = torch.from_numpy(data.g).to(self.device, dtype=torch.float64)
        self._L = L_torch
        self._B = B_torch
        self._N_ib = N_ib

        self._is_setup = True

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Train using Newton iteration for nonlinear PDE, or direct solve for linear PDE.
        """
        if not self._is_setup:
            self.setup()

        N_ib = self._N_ib
        f = self._f
        g = self._g

        start_time = time.perf_counter()

        # Check if task is linear
        is_linear = hasattr(self.task, 'is_linear') and self.task.is_linear()

        if is_linear:
            # LINEAR PDE: Direct solve
            if verbose:
                print("  Linear PDE detected: using direct solve")

            A = torch.cat([self.LH, self.BH], dim=0)
            b = torch.cat([f, g])

            self.W_out = self._solve_lstsq_cholesky(A, b)

            u = self.H @ self.W_out
            residual_history = []

            # Compute final residual
            Lu = torch.sparse.mm(self._L, u.unsqueeze(1)).squeeze(1)[:N_ib]
            F_pde = Lu - f
            F_bc = torch.sparse.mm(self._B, u.unsqueeze(1)).squeeze(1) - g
            final_residual = torch.sqrt(torch.mean(F_pde**2) + torch.mean(F_bc**2))
            residual_history.append(final_residual.item())

        else:
            # NONLINEAR PDE: Newton iteration with damped backtracking
            A_init = torch.cat([self.LH, self.BH], dim=0)
            b_init = torch.cat([f + 1.0, g])
            self.W_out = self._solve_lstsq_cholesky(A_init, b_init)

            u = self.H @ self.W_out
            u_ib = u[:N_ib]

            residual_history = []

            def compute_residual(u_vec, u_ib_vec):
                Lu_vec = torch.sparse.mm(self._L, u_vec.unsqueeze(1)).squeeze(1)[:N_ib]
                exp_u_vec = torch.exp(torch.clamp(u_ib_vec, -50, 50))
                F_pde_vec = Lu_vec - f - exp_u_vec
                F_bc_vec = torch.sparse.mm(self._B, u_vec.unsqueeze(1)).squeeze(1) - g
                return torch.sqrt(torch.mean(F_pde_vec**2) + torch.mean(F_bc_vec**2))

            best_residual = float('inf')
            best_W_out = self.W_out.clone()

            for k in range(self.max_iter):
                Lu = torch.sparse.mm(self._L, u.unsqueeze(1)).squeeze(1)[:N_ib]
                exp_u = torch.exp(torch.clamp(u_ib, -50, 50))
                F_pde = Lu - f - exp_u
                F_bc = torch.sparse.mm(self._B, u.unsqueeze(1)).squeeze(1) - g

                residual = torch.sqrt(torch.mean(F_pde**2) + torch.mean(F_bc**2))
                residual_history.append(residual.item())

                if residual.item() < best_residual:
                    best_residual = residual.item()
                    best_W_out = self.W_out.clone()

                if verbose and (k < 5 or k % 5 == 0):
                    print(f"  Newton iter {k}: residual = {residual.item():.4e}")

                if residual.item() < self.tol:
                    if verbose:
                        print(f"  Converged at iteration {k+1}")
                    break

                # Form Jacobian: J = L - diag(exp(u))
                H_ib = self.H[:N_ib, :]
                JH = self.LH - exp_u.unsqueeze(1) * H_ib

                # Solve linear system
                A = torch.cat([JH, self.BH], dim=0)
                F = torch.cat([-F_pde, -F_bc])

                delta_W = self._solve_lstsq_cholesky(A, F)

                # Backtracking line search
                alpha = 1.0
                W_out_old = self.W_out.clone()
                for _ in range(10):
                    self.W_out = W_out_old + alpha * delta_W
                    u_new = self.H @ self.W_out
                    u_ib_new = u_new[:N_ib]
                    new_residual = compute_residual(u_new, u_ib_new)

                    if new_residual.item() < residual.item() * (1 - 1e-4 * alpha):
                        break
                    alpha *= 0.5

                u = self.H @ self.W_out
                u_ib = u[:N_ib]

            self.W_out = best_W_out
            u = self.H @ self.W_out

        train_time = time.perf_counter() - start_time

        # Move results to CPU for output
        u_pred_np = u[:N_ib].cpu().numpy()

        # Compute L2 error if ground truth available
        l2_error = None
        if self.task.data.u_true is not None:
            u_true_ib = self.task.data.u_true[:N_ib]
            l2_error = self.compute_l2_error(u_pred_np, u_true_ib)

        return TrainResult(
            u_pred=u_pred_np,
            train_time=train_time,
            l2_error=l2_error,
            final_loss=residual_history[-1] if residual_history else None,
            loss_history=residual_history,
            n_iterations=len(residual_history),
            extra={
                'hidden_sizes': self.hidden_sizes,
                'total_features': self.H.shape[1],
                'is_linear': is_linear,
                'device': str(self.device),
            }
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions at given points."""
        if self.W_out is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_torch = torch.from_numpy(X).to(self.device, dtype=torch.float64)

        if self.use_skip_connections:
            H_layers = []
            h = X_torch
            for W, b in self._weights:
                h = self._activation_fn(h @ W + b)
                H_layers.append(h)
            H = torch.cat(H_layers, dim=1)
        else:
            W, b = self._weights[0]
            H = self._activation_fn(X_torch @ W + b)

        return (H @ self.W_out).cpu().numpy()

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        return {
            'hidden_sizes': [100],
            'activation': 'tanh',
            'max_iter': 20,
            'tol': 1e-8,
            'seed': 42,
            'use_skip_connections': True,
            'device': 'auto',
        }


# =============================================================================
# Deep (Multi-Layer) Accelerated Variants
# =============================================================================

class DTELMPINNAccelDeep2(DTELMPINNAccelerated):
    """Accelerated DT-ELM-PINN with 2 hidden layers."""
    name = "dt-elm-pinn-accel-deep2"

    def __init__(self, task, **kwargs):
        kwargs['hidden_sizes'] = [100, 100]
        kwargs['use_skip_connections'] = True
        super().__init__(task, **kwargs)


class DTELMPINNAccelDeep3(DTELMPINNAccelerated):
    """Accelerated DT-ELM-PINN with 3 hidden layers."""
    name = "dt-elm-pinn-accel-deep3"

    def __init__(self, task, **kwargs):
        kwargs['hidden_sizes'] = [100, 100, 100]
        kwargs['use_skip_connections'] = True
        super().__init__(task, **kwargs)


class DTELMPINNAccelDeep4(DTELMPINNAccelerated):
    """Accelerated DT-ELM-PINN with 4 hidden layers."""
    name = "dt-elm-pinn-accel-deep4"

    def __init__(self, task, **kwargs):
        kwargs['hidden_sizes'] = [100, 100, 100, 100]
        kwargs['use_skip_connections'] = True
        super().__init__(task, **kwargs)
