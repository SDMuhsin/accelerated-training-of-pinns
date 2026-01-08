"""
Task registry for PDE problems.

Tasks define:
- Domain geometry and collocation points
- PDE equation (e.g., Laplacian, source terms)
- Boundary conditions
- Ground truth solution (if available)

Note: Discrete operators (L, B) are now built by discretizers, not tasks.
- DT-PINN uses RBF-FD discretization (works on any domain)
- SPECTO-ELM uses Spectral collocation (requires tensor-product domains: square, cube)
"""

from functools import partial
from .base import BaseTask, TaskRegistry, TaskData
from .nonlinear_poisson import NonlinearPoissonTask
from .heat_equation import HeatEquationTask
from .biharmonic import BiharmonicSquareTask, BiharmonicSquarePolyTask

# Import challenging PDE tasks (SPECTO-ELM advantages)
from .challenging_pdes import (
    HelmholtzSquareTask, HelmholtzHighFreqTask,
    VariableCoefficientDiffusionTask,
    ConvectionDiffusionTask, ConvectionDominatedTask,
    AnisotropicDiffusionTask, StronglyAnisotropicTask,
    HighFrequencyPoissonTask, VeryHighFrequencyPoissonTask,
    MixedDerivativeTask,
    ReactionDiffusionTask,
)

# Register biharmonic tasks (4th order PDEs)
TaskRegistry.register('biharmonic-square', BiharmonicSquareTask)
TaskRegistry.register('biharmonic-square-poly', BiharmonicSquarePolyTask)

# Register challenging PDE tasks where SPECTO-ELM excels
# Helmholtz (wave equation): ∇²u + k²u = f
TaskRegistry.register('helmholtz-square', HelmholtzSquareTask)
TaskRegistry.register('helmholtz-highfreq', HelmholtzHighFreqTask)

# Variable coefficient diffusion: ∇·(a(x,y)∇u) = f
TaskRegistry.register('variable-coeff-diffusion', VariableCoefficientDiffusionTask)

# Convection-diffusion: ε∇²u + b·∇u = f
TaskRegistry.register('convection-diffusion', ConvectionDiffusionTask)
TaskRegistry.register('convection-dominated', ConvectionDominatedTask)

# Anisotropic diffusion: a_xx ∂²u/∂x² + a_yy ∂²u/∂y² = f
TaskRegistry.register('anisotropic-diffusion', AnisotropicDiffusionTask)
TaskRegistry.register('strongly-anisotropic', StronglyAnisotropicTask)

# High-frequency Poisson (spectral advantage)
TaskRegistry.register('poisson-highfreq', HighFrequencyPoissonTask)
TaskRegistry.register('poisson-veryhighfreq', VeryHighFrequencyPoissonTask)

# Mixed derivatives: ∇²u + c·∂²u/∂x∂y = f
TaskRegistry.register('mixed-derivative', MixedDerivativeTask)

# Reaction-diffusion: ∇²u + r(x,y)·u = f
TaskRegistry.register('reaction-diffusion', ReactionDiffusionTask)

# Import RBF-FD tasks (Python-generated operators)
try:
    from .rbf_fd_task import (
        RBFFDTask,
        PoissonRBFFDTask,
        NonlinearPoissonRBFFDTask,
    )
    from .heat_equation_rbffd import (
        LaplaceEquationTask,
        HeatEquationSpaceTimeTask,
    )
    _rbf_fd_available = True
except ImportError:
    _rbf_fd_available = False

# Import Spectral tasks
try:
    from .spectral import (
        SpectralPoissonSquareTask,
        SpectralLaplaceSquareTask,
        SpectralNonlinearPoissonSquareTask,
        SpectralPoissonCubeTask,
        SpectralLaplaceCubeTask,
        SpectralNonlinearPoissonCubeTask,
        SpectralPoissonPeakedTask,
        SpectralBoundaryLayerTask,
        SpectralPoissonCornerTask,
    )
    _spectral_available = True
except ImportError as e:
    _spectral_available = False
    print(f"Warning: Spectral tasks not available: {e}")

# =============================================================================
# Register all available tasks with CLEAN names (no discretization prefix)
# =============================================================================

# -----------------------------------------------------------------------------
# Nonlinear Poisson on L-shaped domain (MATLAB data) - special case
# This is the only task that uses precomputed MATLAB operators
# -----------------------------------------------------------------------------
TaskRegistry.register('nonlinear-poisson-lshape', NonlinearPoissonTask)

# -----------------------------------------------------------------------------
# Disk domain tasks (RBF-FD only, no spectral support)
# -----------------------------------------------------------------------------
if _rbf_fd_available:
    # Poisson on disk
    TaskRegistry.register('poisson-disk', PoissonRBFFDTask)  # constant source

    class PoissonDiskSinTask(PoissonRBFFDTask):
        name = "poisson-disk-sin"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'disk')
            kwargs.setdefault('source_func', 'sin')
            super().__init__(**kwargs)
    TaskRegistry.register('poisson-disk-sin', PoissonDiskSinTask)

    class PoissonDiskQuadraticTask(PoissonRBFFDTask):
        name = "poisson-disk-quadratic"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'disk')
            kwargs.setdefault('source_func', 'quadratic')
            super().__init__(**kwargs)
    TaskRegistry.register('poisson-disk-quadratic', PoissonDiskQuadraticTask)

    # Nonlinear Poisson on disk
    TaskRegistry.register('nonlinear-poisson-disk', NonlinearPoissonRBFFDTask)

    class NonlinearPoissonDiskSinTask(NonlinearPoissonRBFFDTask):
        name = "nonlinear-poisson-disk-sin"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'disk')
            kwargs.setdefault('source_func', 'sin')
            super().__init__(**kwargs)
    TaskRegistry.register('nonlinear-poisson-disk-sin', NonlinearPoissonDiskSinTask)

    # Laplace on disk
    class LaplaceDiskTask(LaplaceEquationTask):
        name = "laplace-disk"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'disk')
            kwargs.setdefault('solution_type', 'harmonic')
            super().__init__(**kwargs)
    TaskRegistry.register('laplace-disk', LaplaceDiskTask)

# -----------------------------------------------------------------------------
# Square domain tasks (both RBF-FD and spectral support)
# Using spectral task implementations when available (better accuracy)
# -----------------------------------------------------------------------------
if _spectral_available:
    # Poisson on square (sin solution - spectral)
    TaskRegistry.register('poisson-square-sin', SpectralPoissonSquareTask)

    # Laplace on square (spectral)
    TaskRegistry.register('laplace-square', SpectralLaplaceSquareTask)

    # Nonlinear Poisson on square (spectral)
    TaskRegistry.register('nonlinear-poisson-square', SpectralNonlinearPoissonSquareTask)

    # Localized feature tasks
    TaskRegistry.register('poisson-peaked', SpectralPoissonPeakedTask)
    TaskRegistry.register('boundary-layer', SpectralBoundaryLayerTask)
    TaskRegistry.register('poisson-corner', SpectralPoissonCornerTask)

# Add RBF-FD square tasks (constant and sin source variants)
if _rbf_fd_available:
    class PoissonSquareConstantTask(PoissonRBFFDTask):
        name = "poisson-square-constant"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'square')
            kwargs.setdefault('source_func', 'constant')
            super().__init__(**kwargs)
    TaskRegistry.register('poisson-square-constant', PoissonSquareConstantTask)

    # Only register if spectral not available (avoid duplicate)
    if not _spectral_available:
        class PoissonSquareSinTask(PoissonRBFFDTask):
            name = "poisson-square-sin"
            def __init__(self, **kwargs):
                kwargs.setdefault('domain', 'square')
                kwargs.setdefault('source_func', 'sin')
                super().__init__(**kwargs)
        TaskRegistry.register('poisson-square-sin', PoissonSquareSinTask)

    # Nonlinear Poisson on square - constant source variant
    class NonlinearPoissonSquareConstantTask(NonlinearPoissonRBFFDTask):
        name = "nonlinear-poisson-square-constant"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'square')
            kwargs.setdefault('source_func', 'constant')
            super().__init__(**kwargs)
    TaskRegistry.register('nonlinear-poisson-square-constant', NonlinearPoissonSquareConstantTask)

    # Nonlinear Poisson on square - sin source variant
    class NonlinearPoissonSquareSinTask(NonlinearPoissonRBFFDTask):
        name = "nonlinear-poisson-square-sin"
        def __init__(self, **kwargs):
            kwargs.setdefault('domain', 'square')
            kwargs.setdefault('source_func', 'sin')
            super().__init__(**kwargs)
    TaskRegistry.register('nonlinear-poisson-square-sin', NonlinearPoissonSquareSinTask)

    # Only register if spectral not available
    if not _spectral_available:
        class LaplaceSquareTask(LaplaceEquationTask):
            name = "laplace-square"
            def __init__(self, **kwargs):
                kwargs.setdefault('domain', 'square')
                kwargs.setdefault('solution_type', 'harmonic')
                super().__init__(**kwargs)
        TaskRegistry.register('laplace-square', LaplaceSquareTask)

# -----------------------------------------------------------------------------
# Cube domain tasks (3D, spectral only)
# -----------------------------------------------------------------------------
if _spectral_available:
    TaskRegistry.register('poisson-cube', SpectralPoissonCubeTask)
    TaskRegistry.register('laplace-cube', SpectralLaplaceCubeTask)
    TaskRegistry.register('nonlinear-poisson-cube', SpectralNonlinearPoissonCubeTask)

# -----------------------------------------------------------------------------
# Heat equation tasks (time-dependent, square domain)
# -----------------------------------------------------------------------------
if _rbf_fd_available:
    TaskRegistry.register('heat-equation', HeatEquationSpaceTimeTask)

    class HeatEquationFastDecayTask(HeatEquationSpaceTimeTask):
        name = "heat-fast-decay"
        def __init__(self, **kwargs):
            kwargs.setdefault('k_x', 2)
            kwargs.setdefault('k_y', 2)
            kwargs.setdefault('T_final', 0.05)
            super().__init__(**kwargs)
    TaskRegistry.register('heat-fast-decay', HeatEquationFastDecayTask)


__all__ = [
    'BaseTask',
    'TaskRegistry',
    'TaskData',
    'NonlinearPoissonTask',
    'HeatEquationTask',
    # Biharmonic (4th order)
    'BiharmonicSquareTask',
    'BiharmonicSquarePolyTask',
    # Challenging PDEs where SPECTO-ELM excels
    'HelmholtzSquareTask',
    'HelmholtzHighFreqTask',
    'VariableCoefficientDiffusionTask',
    'ConvectionDiffusionTask',
    'ConvectionDominatedTask',
    'AnisotropicDiffusionTask',
    'StronglyAnisotropicTask',
    'HighFrequencyPoissonTask',
    'VeryHighFrequencyPoissonTask',
    'MixedDerivativeTask',
    'ReactionDiffusionTask',
]

if _rbf_fd_available:
    __all__.extend([
        'RBFFDTask',
        'PoissonRBFFDTask',
        'NonlinearPoissonRBFFDTask',
        'LaplaceEquationTask',
        'HeatEquationSpaceTimeTask',
    ])

if _spectral_available:
    __all__.extend([
        'SpectralPoissonSquareTask',
        'SpectralLaplaceSquareTask',
        'SpectralNonlinearPoissonSquareTask',
        'SpectralPoissonCubeTask',
        'SpectralLaplaceCubeTask',
        'SpectralNonlinearPoissonCubeTask',
        'SpectralPoissonPeakedTask',
        'SpectralBoundaryLayerTask',
        'SpectralPoissonCornerTask',
    ])
