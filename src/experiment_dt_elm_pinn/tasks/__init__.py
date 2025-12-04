"""
Task registry for PDE problems.

Tasks define:
- Domain geometry and collocation points
- PDE equation (e.g., Laplacian, source terms)
- Boundary conditions
- Ground truth solution (if available)
- Precomputed operators (L, B) for discrete methods

Supports both MATLAB-generated operators and Python RBF-FD operators.
"""

from functools import partial
from .base import BaseTask, TaskRegistry, TaskData
from .nonlinear_poisson import NonlinearPoissonTask
from .heat_equation import HeatEquationTask

# Import RBF-FD tasks (Python-generated operators)
try:
    from .rbf_fd_task import (
        RBFFDTask,
        PoissonRBFFDTask,
        NonlinearPoissonRBFFDTask,
    )
    _rbf_fd_available = True
except ImportError:
    _rbf_fd_available = False

# Register all available tasks
# 1. Nonlinear Poisson (MATLAB data) - L-shaped domain
TaskRegistry.register('nonlinear-poisson', NonlinearPoissonTask)

# 2. Heat/Laplace equation (MATLAB data) - Linear PDE
TaskRegistry.register('heat-equation', HeatEquationTask)

# Register RBF-FD tasks (Python-generated operators) if available
if _rbf_fd_available:
    # 3. Poisson on disk - constant source
    TaskRegistry.register('poisson-rbf-fd', PoissonRBFFDTask)

    # 4. Poisson on disk - sinusoidal source
    class PoissonDiskSinTask(PoissonRBFFDTask):
        name = "poisson-disk-sin"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'disk')
            kwargs.setdefault('source_func', 'sin')
            super().__init__(**kwargs)
    TaskRegistry.register('poisson-disk-sin', PoissonDiskSinTask)

    # 5. Poisson on disk - quadratic source
    class PoissonDiskQuadraticTask(PoissonRBFFDTask):
        name = "poisson-disk-quadratic"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'disk')
            kwargs.setdefault('source_func', 'quadratic')
            super().__init__(**kwargs)
    TaskRegistry.register('poisson-disk-quadratic', PoissonDiskQuadraticTask)

    # 6. Poisson on square - constant source
    class PoissonSquareConstantTask(PoissonRBFFDTask):
        name = "poisson-square-constant"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'square')
            kwargs.setdefault('source_func', 'constant')
            super().__init__(**kwargs)
    TaskRegistry.register('poisson-square-constant', PoissonSquareConstantTask)

    # 7. Poisson on square - sinusoidal source
    class PoissonSquareSinTask(PoissonRBFFDTask):
        name = "poisson-square-sin"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'square')
            kwargs.setdefault('source_func', 'sin')
            super().__init__(**kwargs)
    TaskRegistry.register('poisson-square-sin', PoissonSquareSinTask)

    # 8. Nonlinear Poisson on disk (Python operators)
    TaskRegistry.register('nonlinear-poisson-rbf-fd', NonlinearPoissonRBFFDTask)

    # 9. Nonlinear Poisson on disk - sinusoidal source
    class NonlinearPoissonDiskSinTask(NonlinearPoissonRBFFDTask):
        name = "nonlinear-poisson-disk-sin"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'disk')
            kwargs.setdefault('source_func', 'sin')
            super().__init__(**kwargs)
    TaskRegistry.register('nonlinear-poisson-disk-sin', NonlinearPoissonDiskSinTask)

    # 10. Nonlinear Poisson on square - constant source
    class NonlinearPoissonSquareConstantTask(NonlinearPoissonRBFFDTask):
        name = "nonlinear-poisson-square-constant"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'square')
            kwargs.setdefault('source_func', 'constant')
            super().__init__(**kwargs)
    TaskRegistry.register('nonlinear-poisson-square-constant', NonlinearPoissonSquareConstantTask)

    # 11. Nonlinear Poisson on square - sinusoidal source
    class NonlinearPoissonSquareSinTask(NonlinearPoissonRBFFDTask):
        name = "nonlinear-poisson-square-sin"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'square')
            kwargs.setdefault('source_func', 'sin')
            super().__init__(**kwargs)
    TaskRegistry.register('nonlinear-poisson-square-sin', NonlinearPoissonSquareSinTask)

__all__ = [
    'BaseTask',
    'TaskRegistry',
    'TaskData',
    'NonlinearPoissonTask',
    'HeatEquationTask',
]

if _rbf_fd_available:
    __all__.extend([
        'RBFFDTask',
        'PoissonRBFFDTask',
        'NonlinearPoissonRBFFDTask',
    ])
