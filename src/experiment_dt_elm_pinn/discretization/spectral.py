"""
Spectral Collocation discretization using Chebyshev polynomials.

ONLY works on tensor-product domains (square, cube).
Used by SPECTO-ELM (dt-elm-pinn) models.

References:
- Trefethen, "Spectral Methods in MATLAB" (2000)
- Boyd, "Chebyshev and Fourier Spectral Methods" (2001)
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy.sparse import csr_matrix

from .base import Discretizer


# =============================================================================
# Chebyshev Infrastructure Functions
# =============================================================================

def chebyshev_points(N: int) -> np.ndarray:
    """
    Generate Chebyshev-Gauss-Lobatto collocation points on [-1, 1].

    Points cluster near boundaries for optimal spectral accuracy.

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
    Compute Chebyshev spectral differentiation matrix D.

    D @ u gives du/dx at Chebyshev-Gauss-Lobatto points.

    Args:
        N: Number of collocation points

    Returns:
        D: N x N differentiation matrix
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

    # Diagonal: row sum must be zero
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])

    return D


def chebyshev_second_derivative_matrix(N: int) -> np.ndarray:
    """Compute second derivative matrix D2 = D @ D."""
    D = chebyshev_differentiation_matrix(N)
    return D @ D


def chebyshev_laplacian_2d(Nx: int, Ny: int) -> np.ndarray:
    """
    2D Laplacian on tensor-product Chebyshev grid.

    L = I_y (x) D2_x + D2_y (x) I_x  (Kronecker products)

    Args:
        Nx, Ny: Points per dimension

    Returns:
        L: (Nx*Ny) x (Nx*Ny) Laplacian matrix
    """
    D2x = chebyshev_second_derivative_matrix(Nx)
    D2y = chebyshev_second_derivative_matrix(Ny)

    Ix = np.eye(Nx)
    Iy = np.eye(Ny)

    L = np.kron(Iy, D2x) + np.kron(D2y, Ix)
    return L


def chebyshev_laplacian_3d(Nx: int, Ny: int, Nz: int) -> np.ndarray:
    """
    3D Laplacian on tensor-product Chebyshev grid.

    Args:
        Nx, Ny, Nz: Points per dimension

    Returns:
        L: (Nx*Ny*Nz) x (Nx*Ny*Nz) Laplacian matrix
    """
    D2x = chebyshev_second_derivative_matrix(Nx)
    D2y = chebyshev_second_derivative_matrix(Ny)
    D2z = chebyshev_second_derivative_matrix(Nz)

    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    Iz = np.eye(Nz)

    L = (np.kron(np.kron(Iz, Iy), D2x) +
         np.kron(np.kron(Iz, D2y), Ix) +
         np.kron(np.kron(D2z, Iy), Ix))
    return L


def chebyshev_gradient_2d(Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D gradient operators on tensor-product Chebyshev grid.

    Dx = I_y (x) D_x  (derivative in x direction)
    Dy = D_y (x) I_x  (derivative in y direction)

    Args:
        Nx, Ny: Points per dimension

    Returns:
        Dx, Dy: (Nx*Ny) x (Nx*Ny) first derivative matrices
    """
    Dx_1d = chebyshev_differentiation_matrix(Nx)
    Dy_1d = chebyshev_differentiation_matrix(Ny)

    Ix = np.eye(Nx)
    Iy = np.eye(Ny)

    Dx = np.kron(Iy, Dx_1d)
    Dy = np.kron(Dy_1d, Ix)
    return Dx, Dy


def chebyshev_grid_2d(Nx: int, Ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D Chebyshev tensor-product grid on [-1,1]^2.

    Returns:
        X: (Nx*Ny, 2) coordinates
        boundary_mask: Boolean array for boundary points
    """
    x = chebyshev_points(Nx)
    y = chebyshev_points(Ny)

    xx, yy = np.meshgrid(x, y, indexing='xy')
    X = np.column_stack([xx.ravel(), yy.ravel()])

    # Boundary: on edges of [-1,1]^2
    eps = 1e-10
    boundary_mask = (
        (np.abs(X[:, 0] - 1.0) < eps) |
        (np.abs(X[:, 0] + 1.0) < eps) |
        (np.abs(X[:, 1] - 1.0) < eps) |
        (np.abs(X[:, 1] + 1.0) < eps)
    )

    return X, boundary_mask


def chebyshev_grid_3d(Nx: int, Ny: int, Nz: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D Chebyshev tensor-product grid on [-1,1]^3.

    Returns:
        X: (Nx*Ny*Nz, 3) coordinates
        boundary_mask: Boolean array for boundary points
    """
    x = chebyshev_points(Nx)
    y = chebyshev_points(Ny)
    z = chebyshev_points(Nz)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    X = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Boundary: on 6 faces of [-1,1]^3
    eps = 1e-10
    boundary_mask = (
        (np.abs(X[:, 0] - 1.0) < eps) |
        (np.abs(X[:, 0] + 1.0) < eps) |
        (np.abs(X[:, 1] - 1.0) < eps) |
        (np.abs(X[:, 1] + 1.0) < eps) |
        (np.abs(X[:, 2] - 1.0) < eps) |
        (np.abs(X[:, 2] + 1.0) < eps)
    )

    return X, boundary_mask


def scale_domain_2d(X: np.ndarray, x_range: Tuple[float, float],
                    y_range: Tuple[float, float]) -> np.ndarray:
    """Scale points from [-1,1]^2 to [x0,x1] x [y0,y1]."""
    X_scaled = X.copy()
    x0, x1 = x_range
    y0, y1 = y_range
    X_scaled[:, 0] = 0.5 * (x1 - x0) * (X[:, 0] + 1) + x0
    X_scaled[:, 1] = 0.5 * (y1 - y0) * (X[:, 1] + 1) + y0
    return X_scaled


def scale_domain_3d(X: np.ndarray, x_range: Tuple[float, float],
                    y_range: Tuple[float, float],
                    z_range: Tuple[float, float]) -> np.ndarray:
    """Scale points from [-1,1]^3 to physical domain."""
    X_scaled = X.copy()
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    X_scaled[:, 0] = 0.5 * (x1 - x0) * (X[:, 0] + 1) + x0
    X_scaled[:, 1] = 0.5 * (y1 - y0) * (X[:, 1] + 1) + y0
    X_scaled[:, 2] = 0.5 * (z1 - z0) * (X[:, 2] + 1) + z0
    return X_scaled


def scale_laplacian(L: np.ndarray, ranges: list) -> np.ndarray:
    """
    Scale Laplacian for domain transformation using chain rule.

    For x: [-1,1] -> [a,b]: d^2/dx^2 = (2/(b-a))^2 * d^2/dx_ref^2
    """
    scales = [(2.0 / (r[1] - r[0])) ** 2 for r in ranges]

    # For uniform scaling (square/cube), all scales are same
    if len(set(np.round(scales, 10))) == 1:
        return L * scales[0]
    else:
        # Non-uniform: use geometric mean (approximate)
        import warnings
        warnings.warn("Non-uniform domain scaling: using approximate Laplacian")
        return L * np.power(np.prod(scales), 1/len(scales))


def scale_gradient(D: np.ndarray, range_: Tuple[float, float]) -> np.ndarray:
    """
    Scale first derivative for domain transformation using chain rule.

    For x: [-1,1] -> [a,b]: d/dx = (2/(b-a)) * d/dx_ref
    """
    scale = 2.0 / (range_[1] - range_[0])
    return D * scale


# =============================================================================
# Spectral Discretizer Class
# =============================================================================

class SpectralDiscretizer(Discretizer):
    """
    Spectral collocation discretization using Chebyshev polynomials.

    ONLY compatible with tensor-product domains (square, cube).
    Provides exponential convergence for smooth solutions.

    This discretizer GENERATES ITS OWN Chebyshev points - it does not
    use the points provided by the task. The task provides domain bounds
    and functions to evaluate f, g, u_true at these new points.
    """

    # Only works on tensor-product domains
    COMPATIBLE_DOMAINS = ('square', 'cube')

    def __init__(self, N: int = 25):
        """
        Initialize spectral discretizer.

        Args:
            N: Number of Chebyshev points per dimension
        """
        self.N = N

    def is_compatible(self, domain_type: str) -> bool:
        """Check if domain supports spectral methods."""
        return domain_type in self.COMPATIBLE_DOMAINS

    def _incompatibility_message(self, domain_type: str, task_name: str) -> str:
        """Generate helpful error message for incompatible domain."""
        return (
            f"SPECTO-ELM (dt-elm-pinn) requires a tensor-product domain "
            f"(square, cube) for spectral collocation.\n\n"
            f"Task '{task_name}' uses domain '{domain_type}' which is NOT supported.\n\n"
            f"Alternatives:\n"
            f"  - Use --model dt-pinn (RBF-FD discretization, works on any domain)\n"
            f"  - Use --model vanilla-pinn (autodiff, works on any domain)\n"
            f"  - Use a square/cube domain task (e.g., poisson-square-sin, poisson-cube)"
        )

    def build_operators(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        domain_type: str = 'square',
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs
    ) -> Tuple[csr_matrix, csr_matrix, None]:
        """
        Build spectral collocation operators.

        Note: This method ignores X_interior and X_boundary!
        Spectral methods require structured Chebyshev grids, so we generate
        our own points. The returned operators work on these new points.

        Args:
            X_interior: Ignored (spectral uses Chebyshev grid)
            X_boundary: Ignored (spectral uses Chebyshev grid)
            domain_type: 'square' or 'cube'
            domain_bounds: Dict with 'x', 'y', (and 'z' for 3D) ranges

        Returns:
            L: Laplacian operator (sparse)
            B: Boundary operator (sparse)
            None: No ghost points for spectral methods
        """
        if domain_bounds is None:
            domain_bounds = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}

        if domain_type == 'square':
            return self._build_operators_2d(domain_bounds)
        elif domain_type == 'cube':
            return self._build_operators_3d(domain_bounds)
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")

    def _build_operators_2d(
        self,
        domain_bounds: Dict[str, Tuple[float, float]]
    ) -> Tuple[csr_matrix, csr_matrix, None]:
        """Build 2D spectral operators."""
        N = self.N
        x_range = domain_bounds.get('x', (0.0, 1.0))
        y_range = domain_bounds.get('y', (0.0, 1.0))

        # Generate Chebyshev grid on reference domain [-1, 1]^2
        X_ref, boundary_mask = chebyshev_grid_2d(N, N)
        interior_mask = ~boundary_mask

        N_interior = np.sum(interior_mask)
        N_boundary = np.sum(boundary_mask)
        N_total = N_interior + N_boundary

        # Permutation: interior first, then boundary
        interior_idx = np.where(interior_mask)[0]
        boundary_idx = np.where(boundary_mask)[0]
        perm = np.concatenate([interior_idx, boundary_idx])

        # Build Laplacian on reference domain
        L_ref = chebyshev_laplacian_2d(N, N)

        # Scale for physical domain
        L_full = scale_laplacian(L_ref, [x_range, y_range])

        # Reorder rows and columns
        L_reordered = L_full[perm][:, perm]
        L_ib = L_reordered[:N_total, :]

        # Boundary operator: extracts boundary values
        B = np.zeros((N_boundary, N_total))
        B[:, N_interior:] = np.eye(N_boundary)

        return csr_matrix(L_ib), csr_matrix(B), None

    def _build_operators_3d(
        self,
        domain_bounds: Dict[str, Tuple[float, float]]
    ) -> Tuple[csr_matrix, csr_matrix, None]:
        """Build 3D spectral operators."""
        N = self.N
        x_range = domain_bounds.get('x', (0.0, 1.0))
        y_range = domain_bounds.get('y', (0.0, 1.0))
        z_range = domain_bounds.get('z', (0.0, 1.0))

        # Generate Chebyshev grid
        X_ref, boundary_mask = chebyshev_grid_3d(N, N, N)
        interior_mask = ~boundary_mask

        N_interior = np.sum(interior_mask)
        N_boundary = np.sum(boundary_mask)
        N_total = N_interior + N_boundary

        # Permutation
        interior_idx = np.where(interior_mask)[0]
        boundary_idx = np.where(boundary_mask)[0]
        perm = np.concatenate([interior_idx, boundary_idx])

        # Build and scale Laplacian
        L_ref = chebyshev_laplacian_3d(N, N, N)
        L_full = scale_laplacian(L_ref, [x_range, y_range, z_range])

        # Reorder
        L_reordered = L_full[perm][:, perm]
        L_ib = L_reordered[:N_total, :]

        # Boundary operator
        B = np.zeros((N_boundary, N_total))
        B[:, N_interior:] = np.eye(N_boundary)

        return csr_matrix(L_ib), csr_matrix(B), None

    def generate_grid(
        self,
        domain_type: str,
        domain_bounds: Dict[str, Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Chebyshev grid for a domain.

        Args:
            domain_type: 'square' or 'cube'
            domain_bounds: Dict with range tuples

        Returns:
            X_interior: Interior points in physical coordinates
            X_boundary: Boundary points in physical coordinates
            perm: Permutation array to reorder points
        """
        N = self.N

        if domain_type == 'square':
            X_ref, boundary_mask = chebyshev_grid_2d(N, N)
            x_range = domain_bounds.get('x', (0.0, 1.0))
            y_range = domain_bounds.get('y', (0.0, 1.0))
            X_phys = scale_domain_2d(X_ref, x_range, y_range)
        elif domain_type == 'cube':
            X_ref, boundary_mask = chebyshev_grid_3d(N, N, N)
            x_range = domain_bounds.get('x', (0.0, 1.0))
            y_range = domain_bounds.get('y', (0.0, 1.0))
            z_range = domain_bounds.get('z', (0.0, 1.0))
            X_phys = scale_domain_3d(X_ref, x_range, y_range, z_range)
        else:
            raise ValueError(f"Unsupported domain type: {domain_type}")

        interior_mask = ~boundary_mask
        interior_idx = np.where(interior_mask)[0]
        boundary_idx = np.where(boundary_mask)[0]
        perm = np.concatenate([interior_idx, boundary_idx])

        X_interior = X_phys[interior_mask]
        X_boundary = X_phys[boundary_mask]

        return X_interior, X_boundary, perm
