"""
RBF-FD Operator Generation

Main module for generating discrete differential operators (Laplacian, interpolation)
from scattered point clouds using RBF-FD method.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, lil_matrix
from typing import Tuple, Optional, Dict, Any

from .kernels import phs, phs_laplacian
from .polynomial import (
    polynomial_basis_2d, polynomial_laplacian_2d,
    n_poly_terms_2d, recommended_poly_degree
)


class RBFFDOperators:
    """
    RBF-FD Operator Generator.

    Generates sparse Laplacian (L) and boundary/interpolation (B) matrices
    from scattered point clouds.

    Usage:
        gen = RBFFDOperators(stencil_size=21, poly_degree=4, rbf_order=3)
        L, B = gen.build_operators(X_interior, X_boundary, X_ghost)

    Based on the MATLAB operators analysis:
    - Stencil size: 21 neighbors
    - L: (N_total, N_total) Laplacian operator
    - B: (N_b, N_total) Boundary interpolation operator
    """

    def __init__(
        self,
        stencil_size: int = 21,
        poly_degree: int = 4,
        rbf_order: int = 3,
        boundary_stencil_size: Optional[int] = None,
    ):
        """
        Initialize RBF-FD operator generator.

        Args:
            stencil_size: Number of neighbors for Laplacian stencil (default 21)
            poly_degree: Polynomial augmentation degree (default 4 for 15 terms in 2D)
            rbf_order: PHS order m for phi(r) = r^m (default 3)
            boundary_stencil_size: Stencil size for B matrix (default: stencil_size // 2 + 1)
        """
        self.stencil_size = stencil_size
        self.poly_degree = poly_degree
        self.rbf_order = rbf_order
        self.boundary_stencil_size = boundary_stencil_size or (stencil_size // 2 + 1)

        # Validate parameters
        n_poly = n_poly_terms_2d(poly_degree)
        if stencil_size < n_poly:
            raise ValueError(
                f"Stencil size {stencil_size} too small for poly degree {poly_degree} "
                f"({n_poly} terms). Need at least {n_poly} neighbors."
            )

    def _compute_laplacian_weights(
        self,
        center: np.ndarray,
        neighbors: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Laplacian stencil weights at a single point.

        Solves the augmented RBF-FD system:
        [Phi  P ] [w]   [Lap(phi)]
        [P^T  0 ] [c] = [Lap(p)  ]

        Args:
            center: Center point coordinates (2,)
            neighbors: Neighbor coordinates (k, 2) including center

        Returns:
            weights: Stencil weights for Laplacian (k,)
        """
        k = len(neighbors)
        n_poly = n_poly_terms_2d(self.poly_degree)

        # Shift coordinates to be relative to center (improves conditioning)
        X_local = neighbors - center

        # Compute pairwise distances
        r = np.sqrt(np.sum((X_local[:, np.newaxis, :] - X_local[np.newaxis, :, :]) ** 2, axis=2))

        # RBF matrix Phi
        Phi = phs(r, m=self.rbf_order)

        # Polynomial matrix P
        P = polynomial_basis_2d(X_local, self.poly_degree)

        # Build augmented system matrix
        n_total = k + n_poly
        A = np.zeros((n_total, n_total), dtype=neighbors.dtype)
        A[:k, :k] = Phi
        A[:k, k:] = P
        A[k:, :k] = P.T

        # Right-hand side: Laplacian of RBF and polynomials evaluated at center
        rhs = np.zeros(n_total, dtype=neighbors.dtype)

        # Laplacian of phi at center (r=0 for center, r>0 for others)
        r_from_center = np.linalg.norm(X_local, axis=1)
        rhs[:k] = phs_laplacian(r_from_center, m=self.rbf_order)

        # Laplacian of polynomials at center (0,0 in local coords)
        center_local = np.zeros((1, 2), dtype=neighbors.dtype)
        lap_poly_at_center = polynomial_laplacian_2d(center_local, self.poly_degree)
        rhs[k:] = lap_poly_at_center.flatten()

        # Solve the system
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Fall back to least squares if singular
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        # The first k entries are the stencil weights
        return sol[:k]

    def _compute_interpolation_weights(
        self,
        target: np.ndarray,
        neighbors: np.ndarray,
    ) -> np.ndarray:
        """
        Compute interpolation weights at a target point.

        Used for boundary operator B.

        Args:
            target: Target point coordinates (2,)
            neighbors: Neighbor coordinates (k, 2)

        Returns:
            weights: Interpolation weights (k,)
        """
        k = len(neighbors)
        n_poly = n_poly_terms_2d(self.poly_degree)

        # Shift to local coordinates centered at target
        X_local = neighbors - target

        # Compute pairwise distances between neighbors
        r = np.sqrt(np.sum((X_local[:, np.newaxis, :] - X_local[np.newaxis, :, :]) ** 2, axis=2))

        # RBF matrix
        Phi = phs(r, m=self.rbf_order)

        # Polynomial matrix
        P = polynomial_basis_2d(X_local, self.poly_degree)

        # Build augmented system
        n_total = k + n_poly
        A = np.zeros((n_total, n_total), dtype=neighbors.dtype)
        A[:k, :k] = Phi
        A[:k, k:] = P
        A[k:, :k] = P.T

        # RHS: RBF and polynomial values at target (local origin)
        rhs = np.zeros(n_total, dtype=neighbors.dtype)

        # RBF values from target to each neighbor
        r_from_target = np.linalg.norm(X_local, axis=1)
        rhs[:k] = phs(r_from_target, m=self.rbf_order)

        # Polynomial values at target (0,0)
        target_local = np.zeros((1, 2), dtype=neighbors.dtype)
        poly_at_target = polynomial_basis_2d(target_local, self.poly_degree)
        rhs[k:] = poly_at_target.flatten()

        # Solve
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

        return sol[:k]

    def build_laplacian(
        self,
        X_all: np.ndarray,
        verbose: bool = False,
    ) -> csr_matrix:
        """
        Build sparse Laplacian matrix L.

        L @ u gives Laplacian values at all points.

        Args:
            X_all: All point coordinates (N, 2) - interior + boundary + ghost
            verbose: Print progress

        Returns:
            L: Sparse Laplacian matrix (N, N)
        """
        N = X_all.shape[0]
        tree = cKDTree(X_all)

        # Use LIL format for efficient construction
        L = lil_matrix((N, N), dtype=X_all.dtype)

        for i in range(N):
            if verbose and i % 500 == 0:
                print(f"  Building L: {i}/{N}")

            # Find k nearest neighbors (includes self)
            _, neighbor_idx = tree.query(X_all[i], k=self.stencil_size)

            # Compute stencil weights
            neighbors = X_all[neighbor_idx]
            weights = self._compute_laplacian_weights(X_all[i], neighbors)

            # Fill row
            L[i, neighbor_idx] = weights

        return L.tocsr()

    def build_boundary_operator(
        self,
        X_all: np.ndarray,
        X_boundary: np.ndarray,
        verbose: bool = False,
    ) -> csr_matrix:
        """
        Build sparse boundary interpolation matrix B.

        B @ u_all gives interpolated values at boundary points.

        Note: This is used when boundary points need to be interpolated
        from nearby points (including ghost points for Neumann BCs).

        Args:
            X_all: All point coordinates (N, 2)
            X_boundary: Boundary point coordinates (N_b, 2)
            verbose: Print progress

        Returns:
            B: Sparse boundary operator (N_b, N)
        """
        N = X_all.shape[0]
        N_b = X_boundary.shape[0]
        tree = cKDTree(X_all)

        B = lil_matrix((N_b, N), dtype=X_all.dtype)

        for i in range(N_b):
            if verbose and i % 100 == 0:
                print(f"  Building B: {i}/{N_b}")

            # Find neighbors
            _, neighbor_idx = tree.query(X_boundary[i], k=self.boundary_stencil_size)

            # Compute interpolation weights
            neighbors = X_all[neighbor_idx]
            weights = self._compute_interpolation_weights(X_boundary[i], neighbors)

            B[i, neighbor_idx] = weights

        return B.tocsr()

    def build_operators(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        X_ghost: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[csr_matrix, csr_matrix]:
        """
        Build both L and B operators.

        Point ordering: [interior, boundary, ghost]

        Args:
            X_interior: Interior points (N_i, 2)
            X_boundary: Boundary points (N_b, 2)
            X_ghost: Ghost points (N_g, 2)
            verbose: Print progress

        Returns:
            L: Laplacian operator (N_total, N_total)
            B: Boundary operator (N_b, N_total)
        """
        X_all = np.vstack([X_interior, X_boundary, X_ghost])

        if verbose:
            print(f"Building operators for {len(X_all)} points:")
            print(f"  Interior: {len(X_interior)}")
            print(f"  Boundary: {len(X_boundary)}")
            print(f"  Ghost: {len(X_ghost)}")
            print(f"  Stencil size: {self.stencil_size}")
            print(f"  Poly degree: {self.poly_degree}")

        L = self.build_laplacian(X_all, verbose=verbose)
        B = self.build_boundary_operator(X_all, X_boundary, verbose=verbose)

        return L, B

    def validate_against_matlab(
        self,
        L_python: csr_matrix,
        L_matlab: csr_matrix,
        B_python: csr_matrix,
        B_matlab: csr_matrix,
    ) -> Dict[str, float]:
        """
        Validate Python operators against MATLAB operators.

        Args:
            L_python: Python-generated Laplacian
            L_matlab: MATLAB-generated Laplacian
            B_python: Python-generated boundary operator
            B_matlab: MATLAB-generated boundary operator

        Returns:
            Dict with relative errors
        """
        # Convert to dense for comparison (for small problems)
        L_py = L_python.toarray()
        L_mat = L_matlab.toarray()
        B_py = B_python.toarray()
        B_mat = B_matlab.toarray()

        L_rel_err = np.linalg.norm(L_py - L_mat) / np.linalg.norm(L_mat)
        B_rel_err = np.linalg.norm(B_py - B_mat) / np.linalg.norm(B_mat)

        return {
            'L_relative_error': L_rel_err,
            'B_relative_error': B_rel_err,
            'L_max_abs_diff': np.abs(L_py - L_mat).max(),
            'B_max_abs_diff': np.abs(B_py - B_mat).max(),
        }
