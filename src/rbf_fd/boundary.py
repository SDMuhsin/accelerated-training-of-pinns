"""
Boundary Condition Operators for RBF-FD

Implements boundary operators for Dirichlet and Neumann boundary conditions.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix, eye
from typing import Tuple, Optional

from .kernels import phs
from .polynomial import polynomial_basis_2d, n_poly_terms_2d


def build_dirichlet_extraction(
    n_interior: int,
    n_boundary: int,
    n_ghost: int = 0,
) -> csr_matrix:
    """
    Build simple Dirichlet boundary extraction operator.

    For point ordering [interior, boundary, ghost], this extracts
    the boundary point values directly: B @ u = u_boundary.

    This is the simplest form of boundary operator for Dirichlet BCs.

    Args:
        n_interior: Number of interior points
        n_boundary: Number of boundary points
        n_ghost: Number of ghost points (default 0)

    Returns:
        B: Sparse extraction matrix (n_boundary, n_interior + n_boundary + n_ghost)
    """
    n_total = n_interior + n_boundary + n_ghost

    # B is identity on boundary portion
    B = lil_matrix((n_boundary, n_total), dtype=np.float64)
    for i in range(n_boundary):
        B[i, n_interior + i] = 1.0

    return B.tocsr()


def build_dirichlet_interpolation(
    X_all: np.ndarray,
    X_boundary: np.ndarray,
    n_interior: int,
    stencil_size: int = 13,
    poly_degree: int = 3,
    rbf_order: int = 5,
) -> csr_matrix:
    """
    Build Dirichlet interpolation operator using RBF-FD.

    Uses RBF interpolation to compute boundary values from nearby points.
    This is useful when boundary points are not explicitly in the solution vector.

    Args:
        X_all: All point coordinates (N, 2) - [interior, boundary, ghost]
        X_boundary: Boundary point coordinates (n_boundary, 2)
        n_interior: Number of interior points
        stencil_size: Number of neighbors for interpolation
        poly_degree: Polynomial augmentation degree
        rbf_order: PHS order

    Returns:
        B: Sparse interpolation matrix (n_boundary, N)
    """
    N = X_all.shape[0]
    n_boundary = X_boundary.shape[0]
    tree = cKDTree(X_all)
    n_poly = n_poly_terms_2d(poly_degree)

    B = lil_matrix((n_boundary, N), dtype=X_all.dtype)

    for i in range(n_boundary):
        target = X_boundary[i]

        # Find neighbors
        _, neighbor_idx = tree.query(target, k=stencil_size)
        neighbors = X_all[neighbor_idx]

        # Compute interpolation weights
        weights = _compute_interpolation_weights(
            target, neighbors, poly_degree, rbf_order
        )

        B[i, neighbor_idx] = weights

    return B.tocsr()


def _compute_interpolation_weights(
    target: np.ndarray,
    neighbors: np.ndarray,
    poly_degree: int,
    rbf_order: int,
) -> np.ndarray:
    """
    Compute RBF interpolation weights at a target point.

    Solves the augmented system to find weights that reproduce
    the value at target from values at neighbors.
    """
    k = len(neighbors)
    n_poly = n_poly_terms_2d(poly_degree)

    # Shift to local coordinates centered at target
    X_local = neighbors - target

    # Compute pairwise distances between neighbors
    r = np.sqrt(np.sum((X_local[:, np.newaxis, :] - X_local[np.newaxis, :, :]) ** 2, axis=2))

    # RBF matrix
    Phi = phs(r, m=rbf_order)

    # Polynomial matrix
    P = polynomial_basis_2d(X_local, poly_degree)

    # Build augmented system
    n_total = k + n_poly
    A = np.zeros((n_total, n_total), dtype=neighbors.dtype)
    A[:k, :k] = Phi
    A[:k, k:] = P
    A[k:, :k] = P.T

    # RHS: RBF and polynomial values at target (local origin = 0)
    rhs = np.zeros(n_total, dtype=neighbors.dtype)

    # RBF values from target to each neighbor
    r_from_target = np.linalg.norm(X_local, axis=1)
    rhs[:k] = phs(r_from_target, m=rbf_order)

    # Polynomial values at target (0,0)
    target_local = np.zeros((1, 2), dtype=neighbors.dtype)
    poly_at_target = polynomial_basis_2d(target_local, poly_degree)
    rhs[k:] = poly_at_target.flatten()

    # Solve
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    return sol[:k]


def build_neumann_operator(
    X_all: np.ndarray,
    X_boundary: np.ndarray,
    normals: np.ndarray,
    n_interior: int,
    stencil_size: int = 13,
    poly_degree: int = 3,
    rbf_order: int = 5,
) -> csr_matrix:
    """
    Build Neumann boundary operator using RBF-FD.

    Computes the directional derivative in the normal direction at boundary points.
    This is used for enforcing du/dn = g on the boundary.

    Args:
        X_all: All point coordinates (N, 2)
        X_boundary: Boundary point coordinates (n_boundary, 2)
        normals: Outward normal vectors at boundary (n_boundary, 2)
        n_interior: Number of interior points
        stencil_size: Number of neighbors
        poly_degree: Polynomial augmentation degree
        rbf_order: PHS order

    Returns:
        B_neumann: Normal derivative operator (n_boundary, N)
    """
    N = X_all.shape[0]
    n_boundary = X_boundary.shape[0]
    tree = cKDTree(X_all)

    B = lil_matrix((n_boundary, N), dtype=X_all.dtype)

    for i in range(n_boundary):
        target = X_boundary[i]
        normal = normals[i]

        # Find neighbors
        _, neighbor_idx = tree.query(target, k=stencil_size)
        neighbors = X_all[neighbor_idx]

        # Compute normal derivative weights
        weights = _compute_normal_derivative_weights(
            target, neighbors, normal, poly_degree, rbf_order
        )

        B[i, neighbor_idx] = weights

    return B.tocsr()


def _compute_normal_derivative_weights(
    target: np.ndarray,
    neighbors: np.ndarray,
    normal: np.ndarray,
    poly_degree: int,
    rbf_order: int,
) -> np.ndarray:
    """
    Compute RBF weights for normal derivative at target point.

    The normal derivative is: du/dn = n_x * du/dx + n_y * du/dy

    We compute gradient weights and combine with normal vector.
    """
    k = len(neighbors)
    n_poly = n_poly_terms_2d(poly_degree)

    # Shift to local coordinates
    X_local = neighbors - target

    # Compute pairwise distances
    r = np.sqrt(np.sum((X_local[:, np.newaxis, :] - X_local[np.newaxis, :, :]) ** 2, axis=2))

    # RBF matrix
    Phi = phs(r, m=rbf_order)

    # Polynomial matrix
    P = polynomial_basis_2d(X_local, poly_degree)

    # Build augmented system
    n_total = k + n_poly
    A = np.zeros((n_total, n_total), dtype=neighbors.dtype)
    A[:k, :k] = Phi
    A[:k, k:] = P
    A[k:, :k] = P.T

    # RHS for du/dx and du/dy
    rhs_x = np.zeros(n_total, dtype=neighbors.dtype)
    rhs_y = np.zeros(n_total, dtype=neighbors.dtype)

    # Gradient of PHS at target (derivatives w.r.t. target location)
    r_from_target = np.linalg.norm(X_local, axis=1)

    # For odd m: d(r^m)/dx = m * r^(m-2) * (x_j - x_target) at target
    # Since we're at target, this is m * r^(m-2) * (-dx_local)
    m = rbf_order
    if m % 2 == 1:
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_factor = m * np.power(r_from_target, m - 2)
            grad_factor[r_from_target == 0] = 0
        rhs_x[:k] = -grad_factor * X_local[:, 0]
        rhs_y[:k] = -grad_factor * X_local[:, 1]
    else:
        # Even m case (more complex, involves log terms)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_pow = np.power(r_from_target, m - 2)
            log_r = np.log(r_from_target)
            grad_factor = m * r_pow * (log_r + 1)
            grad_factor[r_from_target == 0] = 0
        rhs_x[:k] = -grad_factor * X_local[:, 0]
        rhs_y[:k] = -grad_factor * X_local[:, 1]

    # Gradient of polynomials at origin
    # d/dx(x^a * y^b) = a * x^(a-1) * y^b, evaluated at (0,0) is 0 unless a=1, b=0
    # d/dy(x^a * y^b) = b * x^a * y^(b-1), evaluated at (0,0) is 0 unless a=0, b=1
    # For degree d polynomials: term order is [1, x, y, x^2, xy, y^2, ...]
    # At origin: d/dx = [0, 1, 0, 0, 0, 0, ...]
    #           d/dy = [0, 0, 1, 0, 0, 0, ...]
    rhs_x[k + 1] = 1.0  # d/dx of x term
    rhs_y[k + 2] = 1.0  # d/dy of y term

    # Solve for x and y gradient weights
    try:
        sol_x = np.linalg.solve(A, rhs_x)
        sol_y = np.linalg.solve(A, rhs_y)
    except np.linalg.LinAlgError:
        sol_x, *_ = np.linalg.lstsq(A, rhs_x, rcond=None)
        sol_y, *_ = np.linalg.lstsq(A, rhs_y, rcond=None)

    # Normal derivative: n_x * du/dx + n_y * du/dy
    weights = normal[0] * sol_x[:k] + normal[1] * sol_y[:k]

    return weights


def build_robin_operator(
    X_all: np.ndarray,
    X_boundary: np.ndarray,
    normals: np.ndarray,
    n_interior: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    stencil_size: int = 13,
    poly_degree: int = 3,
    rbf_order: int = 5,
) -> csr_matrix:
    """
    Build Robin boundary operator using RBF-FD.

    Robin BC: alpha * du/dn + beta * u = g

    This combines Neumann (normal derivative) and Dirichlet (interpolation)
    operators into a single matrix.

    Args:
        X_all: All point coordinates (N, 2) - [interior, boundary, ghost]
        X_boundary: Boundary point coordinates (n_boundary, 2)
        normals: Outward normal vectors at boundary (n_boundary, 2)
        n_interior: Number of interior points
        alpha: Coefficient for Neumann term (du/dn)
        beta: Coefficient for Dirichlet term (u)
        stencil_size: Number of neighbors
        poly_degree: Polynomial augmentation degree
        rbf_order: PHS order

    Returns:
        B_robin: Robin operator (n_boundary, N) such that B @ u = alpha*du/dn + beta*u
    """
    N = X_all.shape[0]
    n_boundary = X_boundary.shape[0]
    tree = cKDTree(X_all)

    B = lil_matrix((n_boundary, N), dtype=X_all.dtype)

    for i in range(n_boundary):
        target = X_boundary[i]
        normal = normals[i]

        # Find neighbors
        _, neighbor_idx = tree.query(target, k=stencil_size)
        neighbors = X_all[neighbor_idx]

        # Compute Neumann (normal derivative) weights
        weights_neumann = _compute_normal_derivative_weights(
            target, neighbors, normal, poly_degree, rbf_order
        )

        # Compute Dirichlet (interpolation) weights
        weights_interp = _compute_interpolation_weights(
            target, neighbors, poly_degree, rbf_order
        )

        # Combine: alpha * du/dn + beta * u
        weights = alpha * weights_neumann + beta * weights_interp

        B[i, neighbor_idx] = weights

    return B.tocsr()


class BoundaryOperatorBuilder:
    """
    Builder for boundary condition operators.

    Supports Dirichlet (value), Neumann (derivative), and Robin boundary conditions.
    """

    def __init__(
        self,
        stencil_size: int = 13,
        poly_degree: int = 3,
        rbf_order: int = 5,
    ):
        self.stencil_size = stencil_size
        self.poly_degree = poly_degree
        self.rbf_order = rbf_order

    def build_dirichlet(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        X_ghost: Optional[np.ndarray] = None,
        method: str = 'extraction',
    ) -> csr_matrix:
        """
        Build Dirichlet boundary operator.

        Args:
            X_interior: Interior point coordinates
            X_boundary: Boundary point coordinates
            X_ghost: Ghost point coordinates (optional)
            method: 'extraction' (simple) or 'interpolation' (RBF-FD)

        Returns:
            B: Boundary operator
        """
        n_i = len(X_interior)
        n_b = len(X_boundary)
        n_g = len(X_ghost) if X_ghost is not None else 0

        if method == 'extraction':
            return build_dirichlet_extraction(n_i, n_b, n_g)
        else:
            X_all = np.vstack([X_interior, X_boundary] +
                             ([X_ghost] if X_ghost is not None else []))
            return build_dirichlet_interpolation(
                X_all, X_boundary, n_i,
                self.stencil_size, self.poly_degree, self.rbf_order
            )

    def build_neumann(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        normals: np.ndarray,
        X_ghost: Optional[np.ndarray] = None,
    ) -> csr_matrix:
        """
        Build Neumann boundary operator.

        Args:
            X_interior: Interior point coordinates
            X_boundary: Boundary point coordinates
            normals: Outward normal vectors at boundary
            X_ghost: Ghost point coordinates (optional)

        Returns:
            B: Normal derivative operator
        """
        n_i = len(X_interior)
        X_all = np.vstack([X_interior, X_boundary] +
                         ([X_ghost] if X_ghost is not None else []))

        return build_neumann_operator(
            X_all, X_boundary, normals, n_i,
            self.stencil_size, self.poly_degree, self.rbf_order
        )

    def build_robin(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        normals: np.ndarray,
        X_ghost: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> csr_matrix:
        """
        Build Robin boundary operator.

        Robin BC: alpha * du/dn + beta * u = g

        Args:
            X_interior: Interior point coordinates
            X_boundary: Boundary point coordinates
            normals: Outward normal vectors at boundary
            X_ghost: Ghost point coordinates (optional)
            alpha: Coefficient for Neumann term (du/dn)
            beta: Coefficient for Dirichlet term (u)

        Returns:
            B: Robin operator
        """
        n_i = len(X_interior)
        X_all = np.vstack([X_interior, X_boundary] +
                         ([X_ghost] if X_ghost is not None else []))

        return build_robin_operator(
            X_all, X_boundary, normals, n_i,
            alpha, beta,
            self.stencil_size, self.poly_degree, self.rbf_order
        )
