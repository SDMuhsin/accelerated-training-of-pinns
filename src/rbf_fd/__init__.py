"""
RBF-FD (Radial Basis Function - Finite Difference) Module

Generates discrete differential operators (Laplacian, boundary) from scattered point clouds.
Eliminates MATLAB dependency for DT-PINN.

Reference: Fornberg & Flyer, "A Primer on Radial Basis Functions" (2015)
"""

from .operators import RBFFDOperators
from .kernels import phs, phs_laplacian, gaussian, multiquadric
from .polynomial import polynomial_basis_2d, polynomial_laplacian_2d
from .ghost_points import (
    GhostPointGenerator,
    estimate_normals_radial,
    estimate_normals_from_curve,
    estimate_normals_from_interior,
    generate_ghost_points,
    estimate_ghost_offset,
)
from .boundary import (
    BoundaryOperatorBuilder,
    build_dirichlet_extraction,
    build_dirichlet_interpolation,
    build_neumann_operator,
    build_robin_operator,
)

__all__ = [
    'RBFFDOperators',
    'phs', 'phs_laplacian',
    'gaussian', 'multiquadric',
    'polynomial_basis_2d', 'polynomial_laplacian_2d',
    'GhostPointGenerator',
    'estimate_normals_radial',
    'estimate_normals_from_curve',
    'estimate_normals_from_interior',
    'generate_ghost_points',
    'estimate_ghost_offset',
    'BoundaryOperatorBuilder',
    'build_dirichlet_extraction',
    'build_dirichlet_interpolation',
    'build_neumann_operator',
    'build_robin_operator',
]
