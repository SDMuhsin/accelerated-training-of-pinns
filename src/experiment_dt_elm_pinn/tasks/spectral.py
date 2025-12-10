"""
Spectral Collocation Tasks for DISCO-ELM (SPECTO-ELM)

Uses Chebyshev spectral collocation instead of RBF-FD for discrete operators.
This provides exponential convergence for smooth solutions on regular domains.

References:
- Trefethen, "Spectral Methods in MATLAB" (2000)
- Boyd, "Chebyshev and Fourier Spectral Methods" (2001)
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye as sparse_eye
from typing import Tuple, Optional
from .base import BaseTask, TaskData, TaskRegistry


def chebyshev_points(N: int) -> np.ndarray:
    """
    Generate Chebyshev-Gauss-Lobatto collocation points on [-1, 1].

    These points cluster near the boundaries, which is optimal for
    spectral accuracy and avoiding Runge phenomenon.

    Args:
        N: Number of points

    Returns:
        x: Array of N points, x[0] = 1, x[N-1] = -1
    """
    if N < 2:
        raise ValueError("Need at least 2 points")
    i = np.arange(N)
    x = np.cos(np.pi * i / (N - 1))
    return x


def chebyshev_differentiation_matrix(N: int) -> np.ndarray:
    """
    Compute the Chebyshev spectral differentiation matrix D.

    D is an N×N matrix such that (D @ u)_i ≈ du/dx at x_i,
    where x_i are Chebyshev-Gauss-Lobatto points.

    Uses the standard formula from Trefethen's book.

    Args:
        N: Number of collocation points

    Returns:
        D: N×N differentiation matrix (dense)
    """
    if N < 2:
        raise ValueError("Need at least 2 points")

    x = chebyshev_points(N)

    # Weights c_i: c_0 = c_{N-1} = 2, c_i = 1 otherwise
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0

    # Build differentiation matrix
    D = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j)) / (x[i] - x[j])

    # Diagonal entries: row sum must be zero (derivative of constant = 0)
    # But use explicit formula for numerical stability
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])

    # Alternative explicit diagonal formula (more stable at endpoints):
    # D[0, 0] = (2*(N-1)**2 + 1) / 6
    # D[N-1, N-1] = -(2*(N-1)**2 + 1) / 6
    # For interior: D[i,i] = -x[i] / (2*(1-x[i]**2))

    return D


def chebyshev_second_derivative_matrix(N: int) -> np.ndarray:
    """
    Compute the Chebyshev spectral second derivative matrix D2 = D @ D.

    Args:
        N: Number of collocation points

    Returns:
        D2: N×N second derivative matrix (dense)
    """
    D = chebyshev_differentiation_matrix(N)
    D2 = D @ D
    return D2


def chebyshev_laplacian_2d(Nx: int, Ny: int) -> np.ndarray:
    """
    Compute the 2D Laplacian on a tensor-product Chebyshev grid.

    L = I_y ⊗ D2_x + D2_y ⊗ I_x

    where ⊗ is Kronecker product.

    The grid ordering is: for each y value, sweep through all x values.
    Total points: Nx * Ny, indexed as k = j*Nx + i (x varies fastest).

    Args:
        Nx: Number of points in x direction
        Ny: Number of points in y direction

    Returns:
        L: (Nx*Ny) × (Nx*Ny) Laplacian matrix (dense)
    """
    D2x = chebyshev_second_derivative_matrix(Nx)
    D2y = chebyshev_second_derivative_matrix(Ny)

    Ix = np.eye(Nx)
    Iy = np.eye(Ny)

    # L = I_y ⊗ D2_x + D2_y ⊗ I_x
    L = np.kron(Iy, D2x) + np.kron(D2y, Ix)

    return L


def chebyshev_grid_2d(Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D Chebyshev tensor-product grid on [-1,1] × [-1,1].

    Args:
        Nx: Number of points in x direction
        Ny: Number of points in y direction

    Returns:
        X: (Nx*Ny, 2) array of (x, y) coordinates
        boundary_mask: boolean array indicating boundary points
    """
    x = chebyshev_points(Nx)
    y = chebyshev_points(Ny)

    # Create meshgrid (y varies slowest, x varies fastest)
    xx, yy = np.meshgrid(x, y, indexing='xy')

    # Flatten to (N_total, 2) with x varying fastest
    X = np.column_stack([xx.ravel(), yy.ravel()])

    # Identify boundary points (on edges of [-1,1]²)
    eps = 1e-10
    boundary_mask = (
        (np.abs(X[:, 0] - 1.0) < eps) |   # right edge
        (np.abs(X[:, 0] + 1.0) < eps) |   # left edge
        (np.abs(X[:, 1] - 1.0) < eps) |   # top edge
        (np.abs(X[:, 1] + 1.0) < eps)     # bottom edge
    )

    return X, boundary_mask


def scale_domain(X: np.ndarray, x_range: Tuple[float, float],
                 y_range: Tuple[float, float]) -> np.ndarray:
    """
    Scale points from [-1,1]² to [x0,x1] × [y0,y1].

    Args:
        X: Points in [-1,1]² domain
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)

    Returns:
        X_scaled: Points in physical domain
    """
    X_scaled = X.copy()

    # x: [-1, 1] -> [x0, x1]
    x0, x1 = x_range
    X_scaled[:, 0] = 0.5 * (x1 - x0) * (X[:, 0] + 1) + x0

    # y: [-1, 1] -> [y0, y1]
    y0, y1 = y_range
    X_scaled[:, 1] = 0.5 * (y1 - y0) * (X[:, 1] + 1) + y0

    return X_scaled


def scale_laplacian(L: np.ndarray, x_range: Tuple[float, float],
                    y_range: Tuple[float, float]) -> np.ndarray:
    """
    Scale Laplacian operator for domain transformation.

    For scaling x: [-1,1] -> [a,b], the chain rule gives:
    d/dx_physical = (2/(b-a)) * d/dx_reference
    d²/dx² = (2/(b-a))² * d²/dx_ref²

    Args:
        L: Laplacian in reference domain [-1,1]²
        x_range: Physical x range (a, b)
        y_range: Physical y range (c, d)

    Returns:
        L_scaled: Laplacian in physical domain
    """
    x0, x1 = x_range
    y0, y1 = y_range

    # Scale factors (from chain rule)
    scale_x = (2.0 / (x1 - x0)) ** 2
    scale_y = (2.0 / (y1 - y0)) ** 2

    # For uniform scaling (square domain): both scales are same
    # For non-uniform: need to separate Laplacian components
    # Here we assume the Laplacian was built for [-1,1]² and scale uniformly
    # This is exact only for square domains
    if np.abs(scale_x - scale_y) < 1e-10:
        return L * scale_x
    else:
        # For non-square domains, we'd need to rebuild L with separate components
        # For now, use geometric mean (approximate)
        import warnings
        warnings.warn("Non-square domain: using approximate Laplacian scaling")
        return L * np.sqrt(scale_x * scale_y)


class SpectralPoissonSquareTask(BaseTask):
    """
    Poisson equation on a square domain using Chebyshev spectral collocation.

    PDE: -∇²u = f  on Ω = [0, 1]²
    BC:  u = g     on ∂Ω (Dirichlet)

    True solution: u(x,y) = sin(πx) * sin(πy)
    Source term:   f(x,y) = 2π² * sin(πx) * sin(πy)
    BC:           g = 0 (homogeneous)
    """

    name = "spectral-poisson-square"

    def __init__(self, N: int = 25, **kwargs):
        """
        Args:
            N: Number of Chebyshev points per dimension (total points = N²)
        """
        super().__init__(**kwargs)
        self.N = N
        self.x_range = (0.0, 1.0)
        self.y_range = (0.0, 1.0)

    def load_data(self) -> TaskData:
        """Generate spectral collocation data."""
        N = self.N

        # Generate Chebyshev grid on reference domain [-1, 1]²
        X_ref, boundary_mask = chebyshev_grid_2d(N, N)
        interior_mask = ~boundary_mask

        # Scale to physical domain [0, 1]²
        X = scale_domain(X_ref, self.x_range, self.y_range)

        # Separate interior and boundary points
        X_interior = X[interior_mask]
        X_boundary = X[boundary_mask]

        N_interior = X_interior.shape[0]
        N_boundary = X_boundary.shape[0]
        N_total = N_interior + N_boundary

        # Build permutation to reorder: interior first, then boundary
        interior_idx = np.where(interior_mask)[0]
        boundary_idx = np.where(boundary_mask)[0]
        perm = np.concatenate([interior_idx, boundary_idx])
        inv_perm = np.argsort(perm)

        # Reordered points
        X_reordered = X[perm]
        X_ib = X_reordered[:N_total]  # All points (no ghost in spectral)

        # Build Laplacian on reference domain
        L_ref = chebyshev_laplacian_2d(N, N)

        # Scale Laplacian for physical domain
        L_full = scale_laplacian(L_ref, self.x_range, self.y_range)

        # Reorder rows and columns according to permutation
        L_reordered = L_full[perm][:, perm]

        # Extract interior rows of Laplacian (for PDE residual)
        # L operates on all points, but PDE is enforced only at interior
        L_interior = L_reordered[:N_interior, :]

        # Boundary operator: just extracts boundary values
        # B @ u_full = u at boundary points
        B = np.zeros((N_boundary, N_total))
        B[:, N_interior:] = np.eye(N_boundary)

        # True solution: u(x,y) = sin(πx) * sin(πy)
        u_true = np.sin(np.pi * X_ib[:, 0]) * np.sin(np.pi * X_ib[:, 1])

        # Source term: f at all N_ib points (interior + boundary)
        # For Poisson with L being the positive Laplacian ∇²:
        # We solve L @ u = f, where f = ∇²u_exact
        # For u = sin(πx)sin(πy): ∇²u = -2π² sin(πx)sin(πy)
        # So f = -2π² sin(πx)sin(πy)
        X_ib_ordered = X_reordered[:N_total]
        f = -2 * np.pi**2 * np.sin(np.pi * X_ib_ordered[:, 0]) * np.sin(np.pi * X_ib_ordered[:, 1])

        # Boundary values: g = 0 (homogeneous Dirichlet)
        g = np.zeros(N_boundary)

        # For DISCO-ELM, we need:
        # L: maps u_full -> Laplacian at interior points (for PDE: ∇²u - f = 0)
        # B: maps u_full -> u at boundary points (for BC: u - g = 0)
        #
        # But DISCO-ELM expects:
        # - L @ u_full = f at interior+boundary? Need to check model expectations

        # Looking at RBF-FD tasks, L has shape (N_ib, N_full)
        # and represents the PDE operator applied at all interior+boundary points
        # Let's keep L as full Laplacian rows for interior+boundary
        L_ib = L_reordered[:N_total, :]

        # Convert to sparse for consistency with RBF-FD interface
        L_sparse = csr_matrix(L_ib)
        B_sparse = csr_matrix(B)

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=None,  # No ghost points in spectral methods
            f=f,
            g=g,
            u_true=u_true,
            L=L_sparse,
            B=B_sparse,
        )

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """Compute PDE residual: ∇²u - f (Poisson equation)."""
        # For -∇²u = f, residual is ∇²u + f
        # But if L represents ∇², then residual = L@u + f (want = 0)
        # Actually depends on sign convention in L
        return laplacian_u - self.data.f

    def compute_bc_residual(self, u: np.ndarray) -> np.ndarray:
        """Compute BC residual: u - g (Dirichlet)."""
        return u - self.data.g

    def is_linear(self) -> bool:
        """Poisson is a linear PDE."""
        return True


class SpectralLaplaceSquareTask(BaseTask):
    """
    Laplace equation on a square domain using Chebyshev spectral collocation.

    PDE: ∇²u = 0  on Ω = [0, 1]²
    BC:  u = g    on ∂Ω (Dirichlet, non-homogeneous)

    True solution: u(x,y) = (e^(πx) + e^(-πx)) * cos(πy) / (e^π + e^(-π))
                         = cosh(πx) / cosh(π) * cos(πy)

    This is a harmonic function satisfying Laplace with:
    - u(0, y) = cos(πy) / cosh(π)
    - u(1, y) = cos(πy)
    - u(x, 0) = cosh(πx) / cosh(π)
    - u(x, 1) = -cosh(πx) / cosh(π)
    """

    name = "spectral-laplace-square"

    def __init__(self, N: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.x_range = (0.0, 1.0)
        self.y_range = (0.0, 1.0)

    def _true_solution(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute true solution at (x, y) points."""
        return np.cosh(np.pi * x) / np.cosh(np.pi) * np.cos(np.pi * y)

    def load_data(self) -> TaskData:
        """Generate spectral collocation data."""
        N = self.N

        # Generate Chebyshev grid
        X_ref, boundary_mask = chebyshev_grid_2d(N, N)
        interior_mask = ~boundary_mask

        # Scale to physical domain
        X = scale_domain(X_ref, self.x_range, self.y_range)

        X_interior = X[interior_mask]
        X_boundary = X[boundary_mask]

        N_interior = X_interior.shape[0]
        N_boundary = X_boundary.shape[0]
        N_total = N_interior + N_boundary

        # Permutation: interior first, then boundary
        interior_idx = np.where(interior_mask)[0]
        boundary_idx = np.where(boundary_mask)[0]
        perm = np.concatenate([interior_idx, boundary_idx])

        X_reordered = X[perm]

        # Build and reorder Laplacian
        L_ref = chebyshev_laplacian_2d(N, N)
        L_full = scale_laplacian(L_ref, self.x_range, self.y_range)
        L_reordered = L_full[perm][:, perm]
        L_ib = L_reordered[:N_total, :]

        # Boundary operator
        B = np.zeros((N_boundary, N_total))
        B[:, N_interior:] = np.eye(N_boundary)

        # True solution at all reordered points
        u_true = self._true_solution(X_reordered[:, 0], X_reordered[:, 1])

        # Source term: f = 0 (Laplace equation) at all N_ib points
        f = np.zeros(N_total)

        # Boundary values from true solution
        g = self._true_solution(X_boundary[:, 0], X_boundary[:, 1])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=None,
            f=f,
            g=g,
            u_true=u_true[:N_total],
            L=csr_matrix(L_ib),
            B=csr_matrix(B),
        )

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """Laplace: ∇²u = 0, residual = ∇²u."""
        return laplacian_u

    def compute_bc_residual(self, u: np.ndarray) -> np.ndarray:
        return u - self.data.g

    def is_linear(self) -> bool:
        return True


class SpectralNonlinearPoissonSquareTask(BaseTask):
    """
    Nonlinear Poisson equation on a square domain using Chebyshev spectral collocation.

    PDE: ∇²u - exp(u) = f  on Ω = [0, 1]²
    BC:  u = g             on ∂Ω (Dirichlet)

    We use manufactured solution approach:
    True solution: u(x,y) = sin(πx) * sin(πy) (same as linear Poisson)
    Then f = ∇²u - exp(u) = -2π² sin(πx)sin(πy) - exp(sin(πx)sin(πy))
    BC: g = 0 (homogeneous since sin = 0 at boundaries)
    """

    name = "spectral-nonlinear-poisson-square"

    def __init__(self, N: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.x_range = (0.0, 1.0)
        self.y_range = (0.0, 1.0)

    def load_data(self) -> TaskData:
        """Generate spectral collocation data for nonlinear Poisson."""
        N = self.N

        # Generate Chebyshev grid on reference domain [-1, 1]²
        X_ref, boundary_mask = chebyshev_grid_2d(N, N)
        interior_mask = ~boundary_mask

        # Scale to physical domain [0, 1]²
        X = scale_domain(X_ref, self.x_range, self.y_range)

        X_interior = X[interior_mask]
        X_boundary = X[boundary_mask]

        N_interior = X_interior.shape[0]
        N_boundary = X_boundary.shape[0]
        N_total = N_interior + N_boundary

        # Permutation: interior first, then boundary
        interior_idx = np.where(interior_mask)[0]
        boundary_idx = np.where(boundary_mask)[0]
        perm = np.concatenate([interior_idx, boundary_idx])

        X_reordered = X[perm]
        X_ib = X_reordered[:N_total]

        # Build and scale Laplacian
        L_ref = chebyshev_laplacian_2d(N, N)
        L_full = scale_laplacian(L_ref, self.x_range, self.y_range)
        L_reordered = L_full[perm][:, perm]
        L_ib = L_reordered[:N_total, :]

        # Boundary operator
        B = np.zeros((N_boundary, N_total))
        B[:, N_interior:] = np.eye(N_boundary)

        # True solution: u(x,y) = sin(πx) * sin(πy)
        u_true = np.sin(np.pi * X_ib[:, 0]) * np.sin(np.pi * X_ib[:, 1])

        # Source term for nonlinear Poisson: f = ∇²u - exp(u)
        # ∇²u = -2π² sin(πx)sin(πy)
        # f = -2π² sin(πx)sin(πy) - exp(sin(πx)sin(πy))
        X_ib_ordered = X_reordered[:N_total]
        lap_u = -2 * np.pi**2 * np.sin(np.pi * X_ib_ordered[:, 0]) * np.sin(np.pi * X_ib_ordered[:, 1])
        f = lap_u - np.exp(u_true)

        # Boundary values: g = 0 (homogeneous Dirichlet)
        g = np.zeros(N_boundary)

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=None,
            f=f,
            g=g,
            u_true=u_true,
            L=csr_matrix(L_ib),
            B=csr_matrix(B),
        )

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """Nonlinear Poisson: ∇²u - exp(u) - f."""
        return laplacian_u - np.exp(np.clip(u, -50, 50)) - self.data.f

    def compute_bc_residual(self, u: np.ndarray) -> np.ndarray:
        return u - self.data.g

    def is_linear(self) -> bool:
        return False  # Nonlinear due to exp(u) term


# Register spectral tasks
TaskRegistry.register("spectral-poisson-square", SpectralPoissonSquareTask)
TaskRegistry.register("spectral-laplace-square", SpectralLaplaceSquareTask)
TaskRegistry.register("spectral-nonlinear-poisson-square", SpectralNonlinearPoissonSquareTask)
