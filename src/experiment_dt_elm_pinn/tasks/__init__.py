"""
Task registry for PDE problems.

Tasks define:
- Domain geometry and collocation points
- PDE equation (e.g., Laplacian, source terms)
- Boundary conditions
- Ground truth solution (if available)
- Precomputed operators (L, B) for discrete methods
"""

from .base import BaseTask, TaskRegistry
from .nonlinear_poisson import NonlinearPoissonTask

# Register all available tasks
TaskRegistry.register('nonlinear-poisson', NonlinearPoissonTask)

__all__ = ['BaseTask', 'TaskRegistry', 'NonlinearPoissonTask']
