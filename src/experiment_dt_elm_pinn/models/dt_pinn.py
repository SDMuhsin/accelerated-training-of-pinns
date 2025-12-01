"""
DT-PINN: Discrete-Trained Physics-Informed Neural Network

Uses precomputed RBF-FD sparse operators (L, B) instead of autodiff.
Gradient-based training (L-BFGS or Adam) optimizes network parameters.
"""

import numpy as np
import torch
import torch.nn as nn
import time
from typing import Dict, Any, List, Optional

from .base import BaseModel, TrainResult


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
            use_cuda: Whether to use GPU acceleration
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
        self.seed = seed

        # Will be set during setup
        self.network = None
        self.device = None
        self.L_sparse = None
        self.B_sparse = None
        self.L_t = None
        self.B_t = None

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
        """Initialize network and prepare sparse operators."""
        torch.manual_seed(self.seed)

        data = self.task.data
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        # Set device
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        # Build network
        input_dim = data.spatial_dim
        self.network = self._build_network(input_dim, precision)
        self.network = self.network.to(self.device)

        # Setup sparse operators
        if self.use_cuda:
            self._setup_cuda_operators(data, precision)
        else:
            self._setup_cpu_operators(data, precision)

        self._is_setup = True

    def _setup_cpu_operators(self, data, precision):
        """Setup sparse operators for CPU computation."""
        from scipy.sparse import csr_matrix

        self.L_sparse = csr_matrix(data.L, dtype=np.float64)
        self.B_sparse = csr_matrix(data.B, dtype=np.float64)

    def _setup_cuda_operators(self, data, precision):
        """Setup sparse operators for GPU computation with CuPy."""
        try:
            import cupy
            from cupy.sparse import csr_matrix as cupy_csr

            # Convert to CuPy sparse matrices
            self.L_sparse = cupy_csr(data.L, dtype=np.float64)
            self.B_sparse = cupy_csr(data.B, dtype=np.float64)

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
            self._setup_cpu_operators(data, precision)

    def _sparse_matmul(self, sparse_mat, tensor, sparse_t=None):
        """Multiply sparse matrix with torch tensor."""
        if self.use_cuda:
            return self._cuda_sparse_matmul(sparse_mat, tensor, sparse_t)
        else:
            return self._cpu_sparse_matmul(sparse_mat, tensor)

    def _cpu_sparse_matmul(self, sparse_mat, tensor):
        """CPU sparse matrix multiplication."""
        # Convert to numpy, multiply, convert back
        tensor_np = tensor.detach().cpu().numpy()
        result_np = sparse_mat.dot(tensor_np)
        return torch.tensor(result_np, dtype=tensor.dtype, device=tensor.device)

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

        data = self.task.data
        N_ib = data.N_ib
        precision = torch.float64 if data.X_full.dtype == np.float64 else torch.float32

        # Prepare data tensors
        X_full = torch.tensor(data.X_full, dtype=precision, device=self.device)
        f = torch.tensor(data.f, dtype=precision, device=self.device).unsqueeze(1)
        g = torch.tensor(data.g, dtype=precision, device=self.device).unsqueeze(1)

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

                # Forward pass
                u_pred = self.network(X_full)

                # PDE loss: L @ u - f - exp(u) = 0
                Lu = self._sparse_matmul(self.L_sparse, u_pred, self.L_t)
                pde_residual = Lu[:N_ib] - f - torch.exp(u_pred[:N_ib])
                pde_loss = torch.mean(pde_residual ** 2)

                # BC loss: B @ u - g = 0
                Bu = self._sparse_matmul(self.B_sparse, u_pred, self.B_t)
                bc_residual = Bu - g
                bc_loss = torch.mean(bc_residual ** 2)

                loss = pde_loss + bc_loss
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
            u_true_ib = data.u_true[:N_ib]
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
