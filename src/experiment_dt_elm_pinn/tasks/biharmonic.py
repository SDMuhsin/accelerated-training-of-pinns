"""
Biharmonic Equation Tasks (4th Order PDE)

The biharmonic equation is:
    ∇⁴u = f  where ∇⁴ = (∂²/∂x² + ∂²/∂y²)²

This is a 4th order PDE commonly used in:
- Plate bending problems
- Stream function formulation of Stokes flow
- Image processing (thin plate splines)

For PIELM, this would require computing σ''''(z) - the 4th derivative of the
activation function - which is complex and error-prone.

For SPECTO-ELM, we just compute ∇⁴ = L² where L is the Laplacian operator.
This demonstrates SPECTO-ELM's advantage for higher-order PDEs.
"""

import numpy as np
from typing import Optional
from .base import BaseTask, TaskData


class BiharmonicSquareTask(BaseTask):
    """
    Biharmonic equation on [0,1]² with smooth sinusoidal solution.

    PDE: ∇⁴u = f
    BC: u = g on boundary (Dirichlet)

    Exact solution: u(x,y) = sin(πx)sin(πy)
    Source: f = 4π⁴ sin(πx)sin(πy)
    """

    name = "biharmonic-square"
    domain_type = "square"
    pde_order = 4  # 4th order PDE

    def __init__(self, N_interior: int = 400, N_boundary: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        # domain_bounds uses default [0,1]² from base class

    def load_data(self) -> TaskData:
        """Generate grid points and data."""
        np.random.seed(42)

        # Interior points
        X_interior = np.random.rand(self.N_interior, 2)

        # Boundary points (on edges)
        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        # Evaluate at all points
        X_ib = np.vstack([X_interior, X_boundary])
        f = self.evaluate_source(X_ib)
        g = self.evaluate_bc(X_boundary)
        u_true = self.evaluate_exact(X_ib)

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=f,
            g=g,
            u_true=u_true
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        """Exact solution: u = sin(πx)sin(πy)"""
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """
        Source term for biharmonic equation.

        u = sin(πx)sin(πy)
        ∇²u = -2π² sin(πx)sin(πy)
        ∇⁴u = 4π⁴ sin(πx)sin(πy)
        """
        x, y = X[:, 0], X[:, 1]
        return 4 * np.pi**4 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        """BC: u = 0 on boundary (sin vanishes at 0 and 1)"""
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, biharmonic_u: np.ndarray) -> np.ndarray:
        """Compute PDE residual: ∇⁴u - f"""
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        return biharmonic_u - f

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        """Compute BC residual: u - g on boundary"""
        g = self.evaluate_bc(self.data.X_boundary)
        return u_boundary - g

    def is_linear(self) -> bool:
        return True


class BiharmonicSquarePolyTask(BaseTask):
    """
    Biharmonic equation with polynomial solution (simpler).

    Exact solution: u(x,y) = x²(1-x)²y²(1-y)²
    This vanishes on boundary along with its first derivatives.
    """

    name = "biharmonic-square-poly"
    domain_type = "square"
    pde_order = 4

    def __init__(self, N_interior: int = 400, N_boundary: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        # domain_bounds uses default [0,1]² from base class

    def load_data(self) -> TaskData:
        """Generate grid points and data."""
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        """u = x²(1-x)²y²(1-y)²"""
        x, y = X[:, 0], X[:, 1]
        return (x**2 * (1-x)**2) * (y**2 * (1-y)**2)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ∇⁴u analytically.

        Let φ(t) = t²(1-t)² = t² - 2t³ + t⁴
        φ''(t) = 2 - 12t + 12t²
        φ''''(t) = 24

        u = φ(x)φ(y)
        ∇⁴u = φ''''(x)φ(y) + 2φ''(x)φ''(y) + φ(x)φ''''(y)
        """
        x, y = X[:, 0], X[:, 1]

        phi_x = x**2 * (1-x)**2
        phi_y = y**2 * (1-y)**2
        phi_pp_x = 2 - 12*x + 12*x**2
        phi_pp_y = 2 - 12*y + 12*y**2

        d4u_dx4 = 24 * phi_y
        d4u_dy4 = 24 * phi_x
        d4u_dx2dy2 = phi_pp_x * phi_pp_y

        return d4u_dx4 + 2 * d4u_dx2dy2 + d4u_dy4

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, biharmonic_u: np.ndarray) -> np.ndarray:
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        return biharmonic_u - f

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        g = self.evaluate_bc(self.data.X_boundary)
        return u_boundary - g

    def is_linear(self) -> bool:
        return True
