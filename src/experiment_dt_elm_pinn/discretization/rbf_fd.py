"""
RBF-FD (Radial Basis Function - Finite Difference) discretization.

Works on ANY domain geometry (disk, square, irregular shapes).
Used by DT-PINN model.
"""

import os
import sys
from typing import Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix

# Add rbf_fd module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'rbf_fd'))

from rbf_fd import (
    RBFFDOperators,
    GhostPointGenerator,
    BoundaryOperatorBuilder,
)

from .base import Discretizer


class RBFFDDiscretizer(Discretizer):
    """
    RBF-FD discretization using radial basis function stencils.

    This method works on ANY domain (disk, square, irregular).
    Builds sparse Laplacian (L) and boundary (B) operators from scattered points.
    """

    # RBF-FD works on all domain types
    COMPATIBLE_DOMAINS = ('disk', 'square', 'cube', 'lshape', 'any')

    def __init__(
        self,
        stencil_size: int = 21,
        poly_degree: int = 3,
        rbf_order: int = 5,
        boundary_stencil_size: int = 13,
    ):
        """
        Initialize RBF-FD discretizer.

        Args:
            stencil_size: Number of neighbors for Laplacian stencil
            poly_degree: Polynomial augmentation degree
            rbf_order: Polyharmonic spline order (odd integer)
            boundary_stencil_size: Stencil size for boundary operator
        """
        self.stencil_size = stencil_size
        self.poly_degree = poly_degree
        self.rbf_order = rbf_order
        self.boundary_stencil_size = boundary_stencil_size

        # Initialize RBF-FD infrastructure
        self._L_builder = RBFFDOperators(
            stencil_size=stencil_size,
            poly_degree=poly_degree,
            rbf_order=rbf_order,
        )
        self._B_builder = BoundaryOperatorBuilder(
            stencil_size=boundary_stencil_size,
            poly_degree=poly_degree,
            rbf_order=rbf_order,
        )
        self._ghost_gen = GhostPointGenerator(normal_method='interior')

    def is_compatible(self, domain_type: str) -> bool:
        """RBF-FD works on all domain types."""
        return True

    def build_operators(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        normals: Optional[np.ndarray] = None,
        bc_type: str = 'dirichlet',
        **kwargs
    ) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
        """
        Build RBF-FD operators from point cloud.

        Args:
            X_interior: Interior points (N_i, d)
            X_boundary: Boundary points (N_b, d)
            normals: Normal vectors at boundary (optional)
            bc_type: 'dirichlet', 'neumann', or 'robin'

        Returns:
            L: Laplacian operator (sparse)
            B: Boundary operator (sparse)
            X_ghost: Ghost points for accurate BC enforcement
        """
        # Generate ghost points for accurate boundary conditions
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
            # Robin or other - default to extraction
            B = self._B_builder.build_dirichlet(
                X_interior, X_boundary, X_ghost,
                method='extraction'
            )

        return L, B, X_ghost

    def _incompatibility_message(self, domain_type: str, task_name: str) -> str:
        # RBF-FD is always compatible, but override for completeness
        return (
            f"RBF-FD discretization is compatible with all domains. "
            f"If you see this error, something is wrong."
        )
