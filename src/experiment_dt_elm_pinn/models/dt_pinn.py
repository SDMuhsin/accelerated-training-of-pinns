"""
DT-PINN: Discrete-Trained Physics-Informed Neural Network

Uses RBF-FD (Radial Basis Function - Finite Difference) discretization.
Sparse operators (L, B) replace autodiff for computing spatial derivatives.
Gradient-based training (L-BFGS or Adam) optimizes network parameters.

Key properties:
- Works on ANY domain geometry (disk, square, L-shape, etc.)
- Builds its own RBF-FD operators using RBFFDDiscretizer
- Network predicts u values at collocation points
- Spatial derivatives computed via sparse matrix-vector products

For tensor-product domains (square, cube), consider SPECTO-ELM which uses
spectral collocation for potentially higher accuracy.

NOTE: This model builds its own RBF-FD operators using RBFFDDiscretizer.
It does NOT use operators provided by the task.
"""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, Any, List, Optional

from .base import BaseModel, TrainResult

# Import discretizer
try:
    from ..discretization import RBFFDDiscretizer
except ImportError:
    from src.experiment_dt_elm_pinn.discretization import RBFFDDiscretizer


class DTPINN(BaseModel):
    """
    DT-PINN solver using precomputed discrete operators.

    Key features:
    - Uses sparse L (Laplacian) and B (boundary) operators
    - Gradient-based optimization (L-BFGS or Adam)
    - Supports GPU acceleration via CuPy for sparse operations
    """

    name = "dt-pinn"

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
        seed: int = 0,
        stencil_size: int = 21,
        poly_degree: int = 3,
        rbf_order: int = 5,
        **kwargs
    ):
        """
        Args:
            task: Task object providing PDE definition (not operators)
            layers: Number of hidden layers
            nodes: Nodes per hidden layer
            activation: Activation function ('tanh', 'relu', 'sin')
            optimizer: 'lbfgs' or 'adam'
            lr: Learning rate
            epochs: Number of training epochs
            use_cuda: Whether to use GPU acceleration
            seed: Random seed
            stencil_size: Number of neighbors for RBF-FD stencil
            poly_degree: Polynomial augmentation degree
            rbf_order: Polyharmonic spline order
        """
        super().__init__(task, **kwargs)

        self.layers = layers
        self.nodes = nodes
        self.activation = activation
        self.optimizer_name = optimizer
        self.lr = lr
        self.epochs = epochs
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.seed = seed

        # Create RBF-FD discretizer
        self.discretizer = RBFFDDiscretizer(
            stencil_size=stencil_size,
            poly_degree=poly_degree,
            rbf_order=rbf_order,
        )

        # Will be set during setup
        self.network = None
        self.device = None
        self.L_sparse = None
        self.B_sparse = None
        self.L_t = None
        self.B_t = None

        # Discretized data (built by model, not task)
        self.L = None           # Laplacian operator (scipy sparse)
        self.L_pde = None       # PDE operator (L for 2nd order, L² for 4th order)
        self.B = None           # Boundary operator (scipy sparse)
        self.X_ghost = None     # Ghost points
        self.f = None           # Source term
        self.g = None           # BC values
        self.u_true = None      # True solution

        # Get PDE order from task (2 for Poisson/Laplace, 4 for biharmonic)
        self.pde_order = getattr(task, 'pde_order', 2)

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
                # Custom sin activation
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
        """
        Initialize network and build RBF-FD operators.

        This method:
        1. Gets X_interior, X_boundary from task
        2. Builds L, B operators using RBFFDDiscretizer
        3. Evaluates f, g, u_true using task methods
        4. Prepares sparse operators for training
        """
        torch.manual_seed(self.seed)

        # Get points from task (task provides points, model builds operators)
        data = self.task.data
        X_interior = data.X_interior
        X_boundary = data.X_boundary
        precision = torch.float64 if X_interior.dtype == np.float64 else torch.float32

        # Build RBF-FD operators using discretizer
        self.L, self.B, self.X_ghost = self.discretizer.build_operators(
            X_interior, X_boundary
        )

        # For 4th order PDEs (biharmonic), compute L² = L @ L
        if self.pde_order == 4:
            self.L_pde = self.L @ self.L
        else:
            self.L_pde = self.L

        # Compute dimensions
        N_interior = X_interior.shape[0]
        N_boundary = X_boundary.shape[0]
        N_ghost = self.X_ghost.shape[0] if self.X_ghost is not None else 0
        N_ib = N_interior + N_boundary
        N_full = N_ib + N_ghost

        # Store dimensions
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.N_ghost = N_ghost
        self.N_ib = N_ib
        self.N_full = N_full

        # Identify well-conditioned interior rows (filter out ill-conditioned stencils)
        # This is critical for RBF-FD: near-boundary stencils can have extreme weights
        # IMPORTANT: Use L_pde (not L) because for biharmonic, L² has much larger weights
        L_pde_dense = self.L_pde.toarray()
        row_max_abs = np.abs(L_pde_dense[:N_interior]).max(axis=1)
        # For 4th order PDEs, use much higher threshold since L² amplifies weights exponentially
        # L weights can reach 1e19 near boundaries, L² can reach 1e26
        weight_threshold = 1e4 if self.pde_order == 2 else 1e10
        self.valid_interior_mask = row_max_abs < weight_threshold
        self.N_valid_interior = self.valid_interior_mask.sum()

        if self.N_valid_interior < N_interior * 0.5:
            import warnings
            warnings.warn(
                f"Only {self.N_valid_interior}/{N_interior} interior points have well-conditioned "
                f"stencils for {self.pde_order}th order PDE. Consider using more interior points "
                f"or a different method (e.g., dt-elm-pinn for spectral discretization)."
            )
        elif self.pde_order == 4:
            import warnings
            warnings.warn(
                f"RBF-FD is not well-suited for 4th order PDEs like biharmonic. "
                f"Consider using dt-elm-pinn with spectral discretization for better accuracy."
            )

        # Build X_full: [interior, boundary, ghost]
        if self.X_ghost is not None and len(self.X_ghost) > 0:
            self.X_full = np.vstack([X_interior, X_boundary, self.X_ghost])
        else:
            self.X_full = np.vstack([X_interior, X_boundary])

        # Evaluate source term, BC values, and true solution
        X_ib = np.vstack([X_interior, X_boundary])

        # Note: f is only needed at valid interior points (PDE is enforced there, not at boundary)
        # Use the valid_interior_mask to filter out ill-conditioned points
        if hasattr(self.task, 'evaluate_source'):
            f_all = self.task.evaluate_source(X_interior)
            self.f = f_all[self.valid_interior_mask]
        else:
            f_all = data.f[:N_interior] if hasattr(data, 'f') else np.zeros(N_interior)
            self.f = f_all[self.valid_interior_mask]

        if hasattr(self.task, 'evaluate_bc'):
            self.g = self.task.evaluate_bc(X_boundary)
        else:
            self.g = data.g if hasattr(data, 'g') else np.zeros(N_boundary)

        if hasattr(self.task, 'evaluate_exact'):
            self.u_true = self.task.evaluate_exact(X_ib)
        else:
            self.u_true = data.u_true[:N_ib] if hasattr(data, 'u_true') and data.u_true is not None else None

        # Set device
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Build network
        input_dim = X_interior.shape[1]
        self.network = self._build_network(input_dim, precision)
        self.network = self.network.to(self.device)

        # Setup sparse operators for torch
        if self.use_cuda:
            self._setup_cuda_operators_from_scipy(precision)
        else:
            self._setup_cpu_operators_from_scipy(precision)

        self._is_setup = True

    def _setup_cpu_operators_from_scipy(self, precision):
        """Setup sparse operators for CPU computation using torch sparse tensors."""
        from scipy.sparse import coo_matrix, csr_matrix

        # Convert to COO format for torch sparse tensor creation
        # Use L_pde (L for 2nd order, L² for 4th order biharmonic)
        L_coo = coo_matrix(self.L_pde, dtype=np.float64)
        B_coo = coo_matrix(self.B, dtype=np.float64)

        # Create torch sparse tensors (enables proper autograd for L-BFGS)
        L_indices = torch.tensor(np.vstack([L_coo.row, L_coo.col]), dtype=torch.long)
        L_values = torch.tensor(L_coo.data, dtype=precision)
        self.L_torch = torch.sparse_coo_tensor(
            L_indices, L_values, L_coo.shape
        ).coalesce()

        B_indices = torch.tensor(np.vstack([B_coo.row, B_coo.col]), dtype=torch.long)
        B_values = torch.tensor(B_coo.data, dtype=precision)
        self.B_torch = torch.sparse_coo_tensor(
            B_indices, B_values, B_coo.shape
        ).coalesce()

        # Keep scipy sparse for fallback
        # Use L_pde (L for 2nd order, L² for 4th order biharmonic)
        self.L_sparse = csr_matrix(self.L_pde, dtype=np.float64)
        self.B_sparse = csr_matrix(self.B, dtype=np.float64)

    def _setup_cuda_operators_from_scipy(self, precision):
        """Setup sparse operators for GPU computation with CuPy."""
        try:
            import cupy
            from cupy.sparse import csr_matrix as cupy_csr

            # Convert to CuPy sparse matrices
            # Use L_pde (L for 2nd order, L² for 4th order biharmonic)
            self.L_sparse = cupy_csr(self.L_pde, dtype=np.float64)
            self.B_sparse = cupy_csr(self.B, dtype=np.float64)

            # Initialize kernel by doing a dummy multiplication
            dummy = cupy.zeros((self.L_sparse.shape[1], 1), dtype=np.float64)
            self.L_sparse.dot(dummy)
            self.B_sparse.dot(dummy)

            # Setup transposes for backward pass
            self.L_t = cupy_csr(self.L_sparse.transpose().toarray(), dtype=np.float64)
            self.B_t = cupy_csr(self.B_sparse.transpose().toarray(), dtype=np.float64)

            # Initialize transpose kernels
            dummy_L = cupy.zeros((self.L_sparse.shape[0], 1), dtype=np.float64)
            dummy_B = cupy.zeros((self.B_sparse.shape[0], 1), dtype=np.float64)
            self.L_t.dot(dummy_L)
            self.B_t.dot(dummy_B)

        except ImportError:
            print("CuPy not available, falling back to CPU")
            self.use_cuda = False
            self.device = torch.device('cpu')
            self.network = self.network.to(self.device)
            self._setup_cpu_operators_from_scipy(precision)

    def _sparse_matmul(self, sparse_mat, tensor, sparse_t=None, use_torch_sparse=None):
        """Multiply sparse matrix with torch tensor."""
        if self.use_cuda:
            return self._cuda_sparse_matmul(sparse_mat, tensor, sparse_t)
        else:
            return self._cpu_sparse_matmul(sparse_mat, tensor, use_torch_sparse)

    def _cpu_sparse_matmul(self, sparse_mat, tensor, use_torch_sparse=None):
        """CPU sparse matrix multiplication with autograd support.

        Uses torch.sparse.mm which properly supports autograd for L-BFGS.
        """
        # Determine which torch sparse tensor to use
        if use_torch_sparse is not None:
            sparse_torch = use_torch_sparse
        elif sparse_mat is self.L_sparse:
            sparse_torch = self.L_torch
        elif sparse_mat is self.B_sparse:
            sparse_torch = self.B_torch
        else:
            # Fallback: create torch sparse tensor on the fly
            from scipy.sparse import coo_matrix
            coo = coo_matrix(sparse_mat)
            indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
            values = torch.tensor(coo.data, dtype=tensor.dtype)
            sparse_torch = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()

        # Use torch.sparse.mm for proper autograd support
        # This enables L-BFGS to work correctly
        return torch.sparse.mm(sparse_torch, tensor)

    def _cuda_sparse_matmul(self, sparse_mat, tensor, sparse_t):
        """GPU sparse matrix multiplication with autograd support."""
        import cupy
        from torch.utils.dlpack import to_dlpack, from_dlpack

        # Create custom autograd function with closure over transpose
        class SparseMul(torch.autograd.Function):
            @staticmethod
            def forward(ctx, u_pred):
                cupy_tensor = cupy.from_dlpack(to_dlpack(u_pred))
                result = sparse_mat.dot(cupy_tensor)
                return from_dlpack(result.toDlpack())

            @staticmethod
            def backward(ctx, grad_output):
                cupy_grad = cupy.from_dlpack(to_dlpack(grad_output))
                result = sparse_t.dot(cupy_grad)
                return from_dlpack(result.toDlpack())

        return SparseMul.apply(tensor)

    def train(self, verbose: bool = False, **kwargs) -> TrainResult:
        """
        Train using gradient-based optimization.
        """
        if not self._is_setup:
            self.setup()

        # Use model's own discretized data
        N_interior = self.N_interior
        N_ib = self.N_ib
        precision = torch.float64 if self.X_full.dtype == np.float64 else torch.float32

        # Prepare data tensors
        X_full = torch.tensor(self.X_full, dtype=precision, device=self.device)
        f = torch.tensor(self.f, dtype=precision, device=self.device).unsqueeze(1)
        g = torch.tensor(self.g, dtype=precision, device=self.device).unsqueeze(1)

        # Mask for valid interior points (well-conditioned stencils)
        valid_mask = torch.tensor(self.valid_interior_mask, dtype=torch.bool, device=self.device)

        # Check if task is linear (no exp(u) term)
        is_linear = hasattr(self.task, 'is_linear') and self.task.is_linear()

        # Setup optimizer
        # Using torch.sparse.mm now enables proper autograd for L-BFGS
        effective_optimizer = self.optimizer_name
        effective_lr = self.lr
        effective_epochs = self.epochs

        if effective_optimizer == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                self.network.parameters(),
                lr=effective_lr,
                line_search_fn='strong_wolfe'  # More robust line search
            )
        else:
            optimizer = torch.optim.Adam(self.network.parameters(), lr=effective_lr)

        loss_history = []

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for epoch in range(1, effective_epochs + 1):
            def closure():
                optimizer.zero_grad()

                # Forward pass
                u_pred = self.network(X_full)

                # PDE loss depends on whether task is linear or nonlinear
                # IMPORTANT: Only apply PDE at valid interior points
                # (near-boundary stencils are ill-conditioned and must be filtered out)
                Lu = self._sparse_matmul(self.L_sparse, u_pred, self.L_t)
                Lu_valid = Lu[:N_interior][valid_mask]  # Filter to valid interior points
                if is_linear:
                    # Linear Poisson: L @ u - f = 0 at valid interior points only
                    pde_residual = Lu_valid - f
                else:
                    # Nonlinear Poisson: L @ u - f - exp(u) = 0 at valid interior points
                    # Clamp u to prevent exp overflow
                    u_valid = u_pred[:N_interior][valid_mask]
                    u_clamped = torch.clamp(u_valid, max=50.0)
                    pde_residual = Lu_valid - f - torch.exp(u_clamped)
                pde_loss = torch.mean(pde_residual ** 2)

                # BC loss: B @ u - g = 0
                Bu = self._sparse_matmul(self.B_sparse, u_pred, self.B_t)
                bc_residual = Bu - g
                bc_loss = torch.mean(bc_residual ** 2)

                loss = pde_loss + bc_loss
                loss.backward(retain_graph=True)
                return loss

            if effective_optimizer == 'lbfgs':
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
            X_ib_np = self.X_full[:N_ib]  # Interior + boundary points
            X_ib_t = torch.tensor(X_ib_np, dtype=precision, device=self.device)
            u_pred = self.network(X_ib_t).cpu().numpy().flatten()

        # Compute L2 error
        l2_error = None
        if self.u_true is not None:
            l2_error = self.compute_l2_error(u_pred, self.u_true)

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
        parser.add_argument('--seed', type=int, default=0,
                           help='Random seed')
