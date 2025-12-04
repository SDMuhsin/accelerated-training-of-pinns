"""
Task with Python-generated RBF-FD operators.

Generates discrete operators (L, B) from point clouds using pure Python,
eliminating the MATLAB dependency.
"""

import os
import sys
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Any, Optional, Tuple

# Add rbf_fd module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rbf_fd import (
    RBFFDOperators,
    GhostPointGenerator,
    BoundaryOperatorBuilder,
    estimate_normals_radial,
)
from .base import BaseTask, TaskData


class RBFFDTask(BaseTask):
    """
    Base class for tasks using Python-generated RBF-FD operators.

    Subclasses implement specific PDEs by defining:
    - Domain geometry (point cloud generation or loading)
    - PDE equation (source term, nonlinearity)
    - Boundary conditions (Dirichlet, Neumann, Robin)
    - Exact solution (if available)
    """

    name = "rbf-fd-task"

    def __init__(
        self,
        stencil_size: int = 21,
        poly_degree: int = 3,
        rbf_order: int = 5,
        precision: str = "float64",
        **kwargs
    ):
        """
        Args:
            stencil_size: Number of neighbors for Laplacian stencil
            poly_degree: Polynomial augmentation degree
            rbf_order: PHS order (odd integer, typically 3, 5, or 7)
            precision: 'float32' or 'float64'
        """
        super().__init__(**kwargs)
        self.stencil_size = stencil_size
        self.poly_degree = poly_degree
        self.rbf_order = rbf_order
        self.precision = np.float64 if precision == "float64" else np.float32

        self._L_builder = RBFFDOperators(
            stencil_size=stencil_size,
            poly_degree=poly_degree,
            rbf_order=rbf_order,
        )
        self._B_builder = BoundaryOperatorBuilder(
            stencil_size=13,  # Typically smaller than Laplacian stencil
            poly_degree=poly_degree,
            rbf_order=rbf_order,
        )
        self._ghost_gen = GhostPointGenerator(normal_method='interior')

    def generate_operators(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        normals: Optional[np.ndarray] = None,
        bc_type: str = 'dirichlet',
    ) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
        """
        Generate discrete operators from point cloud.

        Args:
            X_interior: Interior points (N_i, d)
            X_boundary: Boundary points (N_b, d)
            normals: Normal vectors at boundary (for Neumann BCs)
            bc_type: 'dirichlet', 'neumann', or 'robin'

        Returns:
            L: Laplacian operator
            B: Boundary operator
            X_ghost: Generated ghost points
        """
        # Generate ghost points
        X_ghost, gen_normals = self._ghost_gen.generate(X_interior, X_boundary)

        if normals is None:
            normals = gen_normals

        # Stack points: [interior, boundary, ghost]
        X_all = np.vstack([X_interior, X_boundary, X_ghost])

        # Build Laplacian operator
        L = self._L_builder.build_laplacian(X_all)

        # Build boundary operator based on BC type
        if bc_type == 'dirichlet':
            B = self._B_builder.build_dirichlet(
                X_interior, X_boundary, X_ghost,
                method='extraction'
            )
        elif bc_type == 'neumann':
            B = self._B_builder.build_neumann(
                X_interior, X_boundary, normals, X_ghost
            )
        else:
            # For Robin BCs, we need both - use extraction for now
            B = self._B_builder.build_dirichlet(
                X_interior, X_boundary, X_ghost,
                method='extraction'
            )

        return L, B, X_ghost


class PoissonRBFFDTask(RBFFDTask):
    """
    Poisson equation with Python-generated RBF-FD operators.

    PDE: ∇²u = f   in Ω
    BC:  u = g     on ∂Ω (Dirichlet)

    Supports unit disk domain with known analytic solution.
    """

    name = "poisson-rbf-fd"

    def __init__(
        self,
        n_interior: int = 500,
        n_boundary: int = 100,
        domain: str = 'disk',  # 'disk', 'square', or custom
        radius: float = 1.0,
        source_func: str = 'constant',  # 'constant', 'sin', 'quadratic'
        **kwargs
    ):
        """
        Args:
            n_interior: Number of interior collocation points
            n_boundary: Number of boundary points
            domain: Domain type ('disk', 'square')
            radius: Radius for disk domain
            source_func: Source term type
        """
        super().__init__(**kwargs)
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.domain = domain
        self.radius = radius
        self.source_func = source_func

    def _generate_disk_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points in a unit disk."""
        # Interior: random points in disk using rejection sampling
        n_samples = self.n_interior * 2
        X = np.random.uniform(-self.radius, self.radius, (n_samples, 2))
        mask = np.sum(X**2, axis=1) < self.radius**2 * 0.95  # Leave margin
        X_interior = X[mask][:self.n_interior].astype(self.precision)

        # Boundary: uniform points on circle
        theta = np.linspace(0, 2*np.pi, self.n_boundary, endpoint=False)
        X_boundary = self.radius * np.column_stack([
            np.cos(theta), np.sin(theta)
        ]).astype(self.precision)

        return X_interior, X_boundary

    def _generate_square_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points in a unit square [-1,1]^2."""
        # Interior: grid + random perturbation
        n_side = int(np.sqrt(self.n_interior))
        x = np.linspace(-0.9, 0.9, n_side)
        xx, yy = np.meshgrid(x, x)
        X_interior = np.column_stack([xx.ravel(), yy.ravel()]).astype(self.precision)

        # Boundary: uniform on edges
        n_per_side = self.n_boundary // 4
        t = np.linspace(-1, 1, n_per_side, endpoint=False)

        bottom = np.column_stack([t, -np.ones(n_per_side)])
        top = np.column_stack([t, np.ones(n_per_side)])
        left = np.column_stack([-np.ones(n_per_side), t])
        right = np.column_stack([np.ones(n_per_side), t])

        X_boundary = np.vstack([bottom, right, top, left]).astype(self.precision)

        return X_interior, X_boundary

    def _compute_source_and_solution(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute source term and exact solution.

        Returns:
            f: Source term at all points
            g: Boundary values
            u_exact: Exact solution (if available)
        """
        x, y = X[:, 0], X[:, 1]

        if self.source_func == 'constant':
            # u = (r² - R²)/4, ∇²u = 1
            f = np.ones(len(X), dtype=self.precision)
            u_exact = (x**2 + y**2 - self.radius**2) / 4

        elif self.source_func == 'quadratic':
            # u = x² + y², ∇²u = 4
            f = 4 * np.ones(len(X), dtype=self.precision)
            u_exact = x**2 + y**2

        elif self.source_func == 'sin':
            # u = sin(πx)sin(πy), ∇²u = -2π²u
            u_exact = np.sin(np.pi * x) * np.sin(np.pi * y)
            f = -2 * np.pi**2 * u_exact
        else:
            raise ValueError(f"Unknown source_func: {self.source_func}")

        return f.astype(self.precision), u_exact.astype(self.precision)

    def load_data(self) -> TaskData:
        """Generate task data with Python RBF-FD operators."""
        # Generate domain points
        if self.domain == 'disk':
            X_interior, X_boundary = self._generate_disk_points()
        elif self.domain == 'square':
            X_interior, X_boundary = self._generate_square_points()
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

        # Generate operators
        L, B, X_ghost = self.generate_operators(
            X_interior, X_boundary, bc_type='dirichlet'
        )

        # Compute source and exact solution
        X_ib = np.vstack([X_interior, X_boundary])
        f, u_exact_ib = self._compute_source_and_solution(X_ib)

        # Boundary values
        _, u_exact_boundary = self._compute_source_and_solution(X_boundary)
        g = u_exact_boundary

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=X_ghost,
            f=f[:len(X_interior) + len(X_boundary)],
            g=g,
            u_true=u_exact_ib,
            L=L,
            B=B,
        )

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """PDE residual: ∇²u - f = 0"""
        return laplacian_u - self.data.f

    def compute_bc_residual(self, u: np.ndarray, u_full: Optional[np.ndarray] = None) -> np.ndarray:
        """BC residual: u - g = 0 on boundary"""
        if u_full is not None and self.data.B is not None:
            return (self.data.B @ u_full).flatten() - self.data.g
        else:
            u_boundary = u[self.data.N_interior:]
            return u_boundary - self.data.g

    def is_linear(self) -> bool:
        """Linear Poisson equation (no nonlinear term)."""
        return True

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        return {
            'n_interior': 500,
            'n_boundary': 100,
            'domain': 'disk',
            'stencil_size': 21,
            'poly_degree': 3,
            'rbf_order': 5,
        }


class NonlinearPoissonRBFFDTask(PoissonRBFFDTask):
    """
    Nonlinear Poisson with Python-generated RBF-FD operators.

    PDE: ∇²u = f + exp(u)   in Ω
    BC:  u = g              on ∂Ω

    This matches the original DT-PINN benchmark but uses Python operators.
    """

    name = "nonlinear-poisson-rbf-fd"

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """PDE residual: ∇²u - f - exp(u) = 0"""
        return laplacian_u - self.data.f - np.exp(u)

    def compute_jacobian_exp(self, u: np.ndarray) -> np.ndarray:
        """Diagonal of Jacobian for exp(u) term."""
        return np.exp(u)

    def is_linear(self) -> bool:
        """Nonlinear Poisson (has exp(u) term)."""
        return False
