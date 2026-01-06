"""
Discretization methods for building differential operators.

This module provides discretizers that convert point clouds into
discrete differential operators (L for Laplacian, B for boundary).

Available discretizers:
- RBFFDDiscretizer: RBF-FD method (works on ANY domain)
- SpectralDiscretizer: Spectral collocation (tensor-product domains only)

Usage:
    # In DT-PINN model (any domain)
    from discretization import RBFFDDiscretizer
    discretizer = RBFFDDiscretizer()
    L, B, X_ghost = discretizer.build_operators(X_interior, X_boundary)

    # In SPECTO-ELM model (square/cube only)
    from discretization import SpectralDiscretizer
    discretizer = SpectralDiscretizer(N=25)
    discretizer.check_compatibility(task.domain_type, task.name)
    L, B, _ = discretizer.build_operators(X_int, X_bnd, domain_type='square')
"""

from .base import Discretizer
from .rbf_fd import RBFFDDiscretizer
from .spectral import SpectralDiscretizer

__all__ = [
    'Discretizer',
    'RBFFDDiscretizer',
    'SpectralDiscretizer',
]
