"""
SPECTO-ELM: SPECtral Collocation Extreme Learning Machine PINN
(Also known as DT-ELM-PINN: Discrete-Trained Extreme Learning Machine PINN)

Combines spectral collocation discretization with ELM's direct solve.
- No autodiff (uses precomputed L, B matrices from Chebyshev spectral methods)
- No iterative optimization (direct lstsq + Newton iteration)
- Achieves 27x speedup over DT-PINN, 585x over vanilla PINN

IMPORTANT: This model uses SPECTRAL COLLOCATION which requires TENSOR-PRODUCT
domains (square, cube). For non-tensor-product domains (disk, L-shape), use
DT-PINN instead which uses RBF-FD discretization.

NOTE: This model builds its own spectral operators using SpectralDiscretizer.
It does NOT use operators provided by the task.
"""

import numpy as np
import scipy.linalg
import time
from typing import Dict, Any, List, Optional

from .base import BaseModel, TrainResult

# Import discretizer - handle both relative and absolute import
try:
    from ..discretization import SpectralDiscretizer
except ImportError:
    from src.experiment_dt_elm_pinn.discretization import SpectralDiscretizer


def _solve_lstsq_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve least squares via Cholesky decomposition of normal equations.

    Solves: min ||Ax - b||^2  via  (A'A)x = A'b

    This is ~2x faster than np.linalg.lstsq for our problem sizes,
    with identical accuracy. Works because A'A is symmetric positive definite.

    Args:
        A: Design matrix (m, n) with m > n
        b: Right-hand side (m,)

    Returns:
        x: Solution (n,)
    """
    AtA = A.T @ A
    Atb = A.T @ b
    # Use Cholesky factorization (faster than general solve for SPD)
    c, low = scipy.linalg.cho_factor(AtA)
    return scipy.linalg.cho_solve((c, low), Atb)


def _solve_lstsq_svd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve least squares via SVD (standard np.linalg.lstsq).

    Solves: min ||Ax - b||^2 using SVD decomposition.

    This is the standard, numerically stable approach but slower than Cholesky.
    Useful for comparison and for ill-conditioned problems.

    Args:
        A: Design matrix (m, n) with m > n
        b: Right-hand side (m,)

    Returns:
        x: Solution (n,)
    """
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


def _solve_lstsq_robust(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve least squares with robust fallback: try Cholesky first, fall back to SVD.

    This is ideal for deep networks where the normal equations matrix may become
    ill-conditioned due to concatenation of many layer outputs.

    Args:
        A: Design matrix (m, n) with m > n
        b: Right-hand side (m,)

    Returns:
        x: Solution (n,)
    """
    try:
        AtA = A.T @ A
        Atb = A.T @ b
        # Add regularization for numerical stability (1e-8 ensures positive definiteness
        # for deep networks with skip connections, which can have negative eigenvalues ~1e-10)
        AtA += 1e-8 * np.eye(AtA.shape[0])
        c, low = scipy.linalg.cho_factor(AtA)
        return scipy.linalg.cho_solve((c, low), Atb)
    except np.linalg.LinAlgError:
        # Fall back to SVD if Cholesky fails
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return x


class DTELMPINN(BaseModel):
    """
    SPECTO-ELM / DT-ELM-PINN solver using spectral collocation.

    Network: u(x) = H @ W_out where H = tanh(X @ W_in + b_in)
    - W_in, b_in: Random, FIXED (not trained)
    - W_out: Solved via least squares + Newton iteration

    For nonlinear PDEs (e.g., ∇²u = f + exp(u)):
    Uses Newton linearization to solve iteratively.

    IMPORTANT: Requires tensor-product domain (square, cube) for spectral collocation.
    For disk or L-shaped domains, use DT-PINN (RBF-FD discretization) instead.
    """

    name = "dt-elm-pinn"

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
        solver: str = 'cholesky',
        N: int = 25,
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE definition (not operators)
            hidden_sizes: List of hidden layer sizes. Default: [100]
            activation: Activation function ('tanh', 'sin')
            max_iter: Maximum Newton iterations
            tol: Convergence tolerance for residual
            seed: Random seed for reproducibility
            use_skip_connections: If True, concatenate all layer outputs (recommended)
            solver: Least squares solver ('cholesky' or 'svd'). Cholesky is ~2x faster,
                   SVD is more numerically stable for ill-conditioned problems.
            N: Number of Chebyshev points per dimension for spectral discretization.

        Raises:
            ValueError: If task domain is not compatible with spectral collocation.
        """
        super().__init__(task, **kwargs)

        self.hidden_sizes = hidden_sizes or [100]
        self.activation = activation
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.use_skip_connections = use_skip_connections
        self.solver = solver
        self.N = N

        # Create discretizer and check domain compatibility
        self.discretizer = SpectralDiscretizer(N=N)
        self._check_domain_compatibility(task)

        # Select solver function
        if solver == 'cholesky':
            self._solve_lstsq = _solve_lstsq_cholesky
        elif solver == 'svd':
            self._solve_lstsq = _solve_lstsq_svd
        elif solver == 'robust':
            self._solve_lstsq = _solve_lstsq_robust
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'cholesky', 'svd', or 'robust'.")

        # Will be set during setup
        self.H = None           # Hidden layer outputs
        self.W_out = None       # Output weights
        self.LH = None          # Precomputed L @ H
        self.BH = None          # Precomputed B @ H

        # Discretized data (built by model, not task)
        self.L = None           # Laplacian operator
        self.L_pde = None       # PDE operator (L for 2nd order, L² for 4th order)
        self.B = None           # Boundary operator
        self.X_interior = None  # Interior points (Chebyshev grid)
        self.X_boundary = None  # Boundary points (Chebyshev grid)
        self.f = None           # Source term at interior+boundary
        self.g = None           # BC values at boundary
        self.u_true = None      # True solution (if available)

        # Get PDE order from task (2 for Laplacian, 4 for biharmonic)
        self.pde_order = getattr(task, 'pde_order', 2)

    def _check_domain_compatibility(self, task):
        """
        Check if task domain is compatible with spectral collocation.

        Raises:
            ValueError: If domain is not tensor-product (square, cube).
        """
        # Check if task has domain_type property
        if not hasattr(task, 'domain_type'):
            # If no domain_type property, assume compatible (legacy tasks)
            return

        domain_type = task.domain_type
        task_name = getattr(task, 'name', 'unknown')

        # Use discretizer's compatibility check
        self.discretizer.check_compatibility(domain_type, task_name)

    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sin':
            return np.sin(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def setup(self):
        """
        Build discretization and hidden layer features.

        This method:
        1. Generates Chebyshev grid using SpectralDiscretizer
        2. Builds L, B operators using SpectralDiscretizer
        3. Evaluates f, g, u_true at Chebyshev points using task methods
        4. Builds multi-layer hidden features
        5. Precomputes L @ H and B @ H for efficient training
        """
        np.random.seed(self.seed)

        # Get domain info from task
        domain_type = getattr(self.task, 'domain_type', 'square')
        domain_bounds = getattr(self.task, 'domain_bounds', {'x': (0.0, 1.0), 'y': (0.0, 1.0)})

        # Generate Chebyshev grid
        self.X_interior, self.X_boundary, perm = self.discretizer.generate_grid(
            domain_type, domain_bounds
        )

        # Build operators using discretizer
        self.L, self.B, _ = self.discretizer.build_operators(
            self.X_interior, self.X_boundary,
            domain_type=domain_type,
            domain_bounds=domain_bounds
        )

        # For 4th order PDEs (biharmonic), compute L² = L @ L
        if self.pde_order == 4:
            # L is sparse, L² is also sparse
            self.L_pde = self.L @ self.L
        else:
            self.L_pde = self.L

        # Compute N_interior, N_boundary, N_ib
        N_interior = self.X_interior.shape[0]
        N_boundary = self.X_boundary.shape[0]
        N_ib = N_interior + N_boundary

        # Stack points: interior first, then boundary
        X_ib = np.vstack([self.X_interior, self.X_boundary])
        precision = X_ib.dtype

        # Evaluate source term, BC values, and true solution at Chebyshev points
        if hasattr(self.task, 'evaluate_source'):
            self.f = self.task.evaluate_source(X_ib)
        else:
            # Fallback to task's pre-computed f (may be at different points)
            self.f = self.task.data.f[:N_ib] if hasattr(self.task, 'data') else np.zeros(N_ib)

        if hasattr(self.task, 'evaluate_bc'):
            self.g = self.task.evaluate_bc(self.X_boundary)
        else:
            self.g = self.task.data.g if hasattr(self.task, 'data') else np.zeros(N_boundary)

        if hasattr(self.task, 'evaluate_exact'):
            self.u_true = self.task.evaluate_exact(X_ib)
        else:
            self.u_true = self.task.data.u_true[:N_ib] if hasattr(self.task, 'data') and self.task.data.u_true is not None else None

        # Store dimensions for later use
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.N_ib = N_ib

        # Build X_full for ELM (no ghost points in spectral)
        X = X_ib

        # Build multi-layer hidden representation
        if self.use_skip_connections:
            H_layers = []
            h = X
            input_dim = X.shape[1]

            for n_hidden in self.hidden_sizes:
                W = np.random.randn(input_dim, n_hidden).astype(precision) * np.sqrt(2.0 / input_dim)
                b = np.random.randn(n_hidden).astype(precision) * 0.1
                h = self._activation_fn(h @ W + b)
                H_layers.append(h)
                input_dim = n_hidden

            # Concatenate all layers (skip connections)
            self.H = np.hstack(H_layers)
        else:
            # Standard single-layer ELM
            n_hidden = self.hidden_sizes[0]
            W = np.random.randn(X.shape[1], n_hidden).astype(precision) * np.sqrt(2.0 / X.shape[1])
            b = np.random.randn(n_hidden).astype(precision) * 0.1
            self.H = self._activation_fn(X @ W + b)

        # Precompute operator products
        # L_pde @ H gives (N_total, n_features), we need first N_ib rows for PDE
        # L_pde is L for 2nd order PDEs, L² for 4th order (biharmonic)
        LH_full = self.L_pde @ self.H
        self.LH = LH_full[:N_ib, :]  # (N_ib, n_features)
        self.BH = self.B @ self.H    # (N_bc, n_features)

        # Initialize output weights
        self.W_out = np.zeros(self.H.shape[1], dtype=precision)

        self._is_setup = True

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Train using Newton iteration for nonlinear PDE, or direct solve for linear PDE.

        Supports both 2nd order PDEs (Poisson/Laplace) and 4th order PDEs (biharmonic).
        For 4th order PDEs, uses L² operator (biharmonic = Laplacian squared).

        For nonlinear PDE (∇²u = f + exp(u)):
        - Linearize exp(u) around current solution
        - Solve linear system via lstsq
        - Iterate until convergence

        For linear PDE (∇²u = f or ∇⁴u = f):
        - Direct least-squares solve (no Newton iteration)
        """
        if not self._is_setup:
            self.setup()

        # Use model's own discretized data (not task's data)
        N_ib = self.N_ib
        f = self.f
        g = self.g
        L = self.L_pde  # Use L for 2nd order, L² for 4th order (biharmonic)
        B = self.B

        start_time = time.perf_counter()

        # Check if task is linear (no nonlinear terms like exp(u))
        is_linear = hasattr(self.task, 'is_linear') and self.task.is_linear()

        if is_linear:
            # LINEAR PDE: Direct solve with single lstsq
            # PDE: L @ u = f (no exp(u) term)
            # BC:  B @ u = g
            if verbose:
                print("  Linear PDE detected: using direct solve")

            A = np.vstack([self.LH, self.BH])
            b = np.concatenate([f, g])

            try:
                self.W_out = self._solve_lstsq(A, b)
            except np.linalg.LinAlgError:
                # Fallback to SVD if Cholesky fails
                if verbose:
                    print("  Cholesky failed, falling back to SVD solver")
                self.W_out = _solve_lstsq_svd(A, b)

            u = self.H @ self.W_out
            residual_history = []

            # Compute final residual for reporting
            Lu = (L @ u)[:N_ib]
            F_pde = Lu - f
            F_bc = (B @ u) - g
            final_residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
            residual_history.append(final_residual)

        else:
            # NONLINEAR PDE: Damped Newton iteration with backtracking line search
            # Initial solve: approximate exp(u) ≈ exp(0) = 1
            A_init = np.vstack([self.LH, self.BH])
            b_init = np.concatenate([f + 1.0, g])
            self.W_out = self._solve_lstsq(A_init, b_init)

            u = self.H @ self.W_out
            u_ib = u[:N_ib]

            residual_history = []

            def compute_residual(u_vec, u_ib_vec):
                """Compute PDE + BC residual."""
                Lu_vec = (L @ u_vec)[:N_ib]
                exp_u_vec = np.exp(np.clip(u_ib_vec, -50, 50))  # Clip to avoid overflow
                F_pde_vec = Lu_vec - f - exp_u_vec
                F_bc_vec = (B @ u_vec) - g
                return np.sqrt(np.mean(F_pde_vec**2) + np.mean(F_bc_vec**2))

            # Track best solution found
            best_residual = float('inf')
            best_W_out = self.W_out.copy()

            # Newton iterations with damping
            for k in range(self.max_iter):
                # Compute residual: F = L @ u - f - exp(u)
                Lu = (L @ u)[:N_ib]
                exp_u = np.exp(np.clip(u_ib, -50, 50))  # Clip to avoid overflow
                F_pde = Lu - f - exp_u
                F_bc = (B @ u) - g

                residual = np.sqrt(np.mean(F_pde**2) + np.mean(F_bc**2))
                residual_history.append(residual)

                # Track best solution
                if residual < best_residual:
                    best_residual = residual
                    best_W_out = self.W_out.copy()

                if verbose and (k < 5 or k % 5 == 0):
                    print(f"  Newton iter {k}: residual = {residual:.4e}")

                if residual < self.tol:
                    if verbose:
                        print(f"  Converged at iteration {k+1}")
                    break

                # Form Jacobian: J = L - diag(exp(u))
                # J @ H = LH - diag(exp(u)) @ H_ib
                H_ib = self.H[:N_ib, :]
                JH = self.LH - exp_u[:, np.newaxis] * H_ib

                # Solve linear system: [JH; BH] @ delta_W = -[F_pde; F_bc]
                A = np.vstack([JH, self.BH])
                F = np.concatenate([-F_pde, -F_bc])

                delta_W = self._solve_lstsq(A, F)

                # Backtracking line search
                alpha = 1.0
                W_out_old = self.W_out.copy()
                for _ in range(10):  # Max 10 backtracking steps
                    self.W_out = W_out_old + alpha * delta_W
                    u_new = self.H @ self.W_out
                    u_ib_new = u_new[:N_ib]
                    new_residual = compute_residual(u_new, u_ib_new)

                    if new_residual < residual * (1 - 1e-4 * alpha):
                        # Sufficient decrease achieved
                        break
                    alpha *= 0.5

                u = self.H @ self.W_out
                u_ib = u[:N_ib]

            # Restore best solution
            self.W_out = best_W_out
            u = self.H @ self.W_out

        train_time = time.perf_counter() - start_time

        # Compute L2 error if ground truth available
        u_pred = u[:N_ib]
        l2_error = None
        if self.u_true is not None:
            l2_error = self.compute_l2_error(u_pred, self.u_true)

        return TrainResult(
            u_pred=u_pred,
            train_time=train_time,
            l2_error=l2_error,
            final_loss=residual_history[-1] if residual_history else None,
            loss_history=residual_history,
            n_iterations=len(residual_history),
            extra={
                'hidden_sizes': self.hidden_sizes,
                'total_features': self.H.shape[1],
                'is_linear': is_linear,
            }
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions at given points.

        Note: For points not in the training set, this requires
        recomputing hidden features. For best accuracy, use points
        from the original collocation set.
        """
        if self.W_out is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Recompute hidden features for new points
        np.random.seed(self.seed)
        precision = X.dtype

        if self.use_skip_connections:
            H_layers = []
            h = X
            input_dim = X.shape[1]

            for n_hidden in self.hidden_sizes:
                W = np.random.randn(input_dim, n_hidden).astype(precision) * np.sqrt(2.0 / input_dim)
                b = np.random.randn(n_hidden).astype(precision) * 0.1
                h = self._activation_fn(h @ W + b)
                H_layers.append(h)
                input_dim = n_hidden

            H = np.hstack(H_layers)
        else:
            n_hidden = self.hidden_sizes[0]
            W = np.random.randn(X.shape[1], n_hidden).astype(precision) * np.sqrt(2.0 / X.shape[1])
            b = np.random.randn(n_hidden).astype(precision) * 0.1
            H = self._activation_fn(X @ W + b)

        return H @ self.W_out

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        return {
            'hidden_sizes': [100],
            'activation': 'tanh',
            'max_iter': 20,
            'tol': 1e-8,
            'seed': 42,
            'use_skip_connections': True,
            'solver': 'cholesky',
        }

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[100],
                           help='Hidden layer sizes (default: [100])')
        parser.add_argument('--activation', type=str, default='tanh',
                           choices=['tanh', 'sin'], help='Activation function')
        parser.add_argument('--max-iter', type=int, default=20,
                           help='Maximum Newton iterations')
        parser.add_argument('--tol', type=float, default=1e-8,
                           help='Convergence tolerance')
        parser.add_argument('--seed', type=int, default=42, help='Random seed')
        parser.add_argument('--no-skip-connections', action='store_true',
                           help='Disable skip connections in multi-layer ELM')


class DTELMPINNCholesky(DTELMPINN):
    """DT-ELM-PINN with Cholesky solver (~2x faster)."""

    name = "dt-elm-pinn-cholesky"

    def __init__(self, task, **kwargs):
        kwargs['solver'] = 'cholesky'
        super().__init__(task, **kwargs)


class DTELMPINNSVD(DTELMPINN):
    """DT-ELM-PINN with SVD solver (more numerically stable)."""

    name = "dt-elm-pinn-svd"

    def __init__(self, task, **kwargs):
        kwargs['solver'] = 'svd'
        super().__init__(task, **kwargs)


# =============================================================================
# Deep (Multi-Layer) Variants
# =============================================================================
# These use skip connections (concatenate all layer outputs) to preserve
# information across layers. This is what enables ELM to work with multiple
# layers - unlike PIELM which requires analytical derivatives σ''(z).

class DTELMPINNDeep2(DTELMPINN):
    """DT-ELM-PINN with 2 hidden layers [100, 100] and skip connections."""

    name = "dt-elm-pinn-deep2"

    def __init__(self, task, **kwargs):
        # Force 2-layer architecture with skip connections
        # Use robust solver to handle potential ill-conditioning from concatenated layers
        kwargs['hidden_sizes'] = [100, 100]
        kwargs['use_skip_connections'] = True
        kwargs['solver'] = 'robust'
        super().__init__(task, **kwargs)


class DTELMPINNDeep3(DTELMPINN):
    """DT-ELM-PINN with 3 hidden layers [100, 100, 100] and skip connections."""

    name = "dt-elm-pinn-deep3"

    def __init__(self, task, **kwargs):
        # Force 3-layer architecture with skip connections
        # Use robust solver to handle potential ill-conditioning from concatenated layers
        kwargs['hidden_sizes'] = [100, 100, 100]
        kwargs['use_skip_connections'] = True
        kwargs['solver'] = 'robust'
        super().__init__(task, **kwargs)


class DTELMPINNDeep4(DTELMPINN):
    """DT-ELM-PINN with 4 hidden layers [100, 100, 100, 100] and skip connections."""

    name = "dt-elm-pinn-deep4"

    def __init__(self, task, **kwargs):
        # Force 4-layer architecture with skip connections
        # Use robust solver to handle potential ill-conditioning from concatenated layers
        kwargs['hidden_sizes'] = [100, 100, 100, 100]
        kwargs['use_skip_connections'] = True
        kwargs['solver'] = 'robust'
        super().__init__(task, **kwargs)
