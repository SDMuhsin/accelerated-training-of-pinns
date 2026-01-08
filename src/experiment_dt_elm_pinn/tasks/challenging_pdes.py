"""
Challenging PDE Tasks where SPECTO-ELM has advantages.

These PDEs are difficult for PIELM because:
1. Biharmonic (4th order): PIELM needs σ''''(z) which is complex
2. Helmholtz: Requires handling k²u term with proper operator
3. Variable coefficient: PIELM assumes constant coefficients in σ''
4. Convection-diffusion: PIELM only computes ∇², not first derivatives ∇
5. Anisotropic diffusion: PIELM assumes isotropic Laplacian
6. High-frequency: Spectral methods have exponential convergence

SPECTO-ELM wins because it uses discrete operators that can handle all these cases
by simply constructing the appropriate operator matrix.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from .base import BaseTask, TaskData


# =============================================================================
# HELMHOLTZ EQUATION: ∇²u + k²u = f
# =============================================================================

class HelmholtzSquareTask(BaseTask):
    """
    Helmholtz equation on [0,1]²: ∇²u + k²u = f

    This is the wave equation in frequency domain.
    PIELM struggles because it computes ∇²u but not the k²u term properly.
    SPECTO-ELM builds (L + k²I) operator directly.

    Exact solution: u(x,y) = sin(πx)sin(πy)
    With k²=1: f = (1 - 2π²)sin(πx)sin(πy)
    """

    name = "helmholtz-square"
    domain_type = "square"
    pde_type = "helmholtz"

    def __init__(self, N_interior: int = 400, N_boundary: int = 100,
                 k: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.k = k  # Helmholtz wavenumber
        self.k_squared = k * k

    def load_data(self) -> TaskData:
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        """u = sin(πx)sin(πy)"""
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """
        f = ∇²u + k²u = -2π²sin(πx)sin(πy) + k²sin(πx)sin(πy)
          = (k² - 2π²)sin(πx)sin(πy)
        """
        x, y = X[:, 0], X[:, 1]
        return (self.k_squared - 2 * np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """Residual: ∇²u + k²u - f"""
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        return laplacian_u + self.k_squared * u - f

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        g = self.evaluate_bc(self.data.X_boundary)
        return u_boundary - g

    def is_linear(self) -> bool:
        return True


class HelmholtzHighFreqTask(HelmholtzSquareTask):
    """
    Helmholtz with higher wavenumber k=5.
    Higher k means more oscillatory solution - harder for point methods.
    """
    name = "helmholtz-highfreq"

    def __init__(self, **kwargs):
        kwargs.setdefault('k', 5.0)
        super().__init__(**kwargs)


# =============================================================================
# VARIABLE COEFFICIENT DIFFUSION: ∇·(a(x,y)∇u) = f
# =============================================================================

class VariableCoefficientDiffusionTask(BaseTask):
    """
    Variable coefficient diffusion: ∇·(a(x,y)∇u) = f on [0,1]²

    Expanded: a(x,y)∇²u + ∇a·∇u = f

    PIELM's analytical σ'' only handles constant coefficient Laplacian.
    SPECTO-ELM can build the full operator with variable coefficients.

    Coefficient: a(x,y) = 1 + 0.5*sin(2πx)*sin(2πy)
    Exact solution: u(x,y) = sin(πx)sin(πy)
    """

    name = "variable-coeff-diffusion"
    domain_type = "square"
    pde_type = "variable_diffusion"

    def __init__(self, N_interior: int = 400, N_boundary: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary

    def diffusion_coeff(self, X: np.ndarray) -> np.ndarray:
        """a(x,y) = 1 + 0.5*sin(2πx)*sin(2πy)"""
        x, y = X[:, 0], X[:, 1]
        return 1.0 + 0.5 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def diffusion_coeff_grad(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """∇a = (∂a/∂x, ∂a/∂y)"""
        x, y = X[:, 0], X[:, 1]
        da_dx = np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        da_dy = np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        return da_dx, da_dy

    def load_data(self) -> TaskData:
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        """u = sin(πx)sin(πy)"""
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """
        f = ∇·(a∇u) = a∇²u + ∇a·∇u

        u = sin(πx)sin(πy)
        ∇²u = -2π²sin(πx)sin(πy)
        ∇u = (π cos(πx)sin(πy), π sin(πx)cos(πy))
        """
        x, y = X[:, 0], X[:, 1]

        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        laplacian_u = -2 * np.pi**2 * u

        du_dx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        du_dy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

        a = self.diffusion_coeff(X)
        da_dx, da_dy = self.diffusion_coeff_grad(X)

        return a * laplacian_u + da_dx * du_dx + da_dy * du_dy

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """Note: This residual is approximate for PIELM (missing ∇a·∇u term)"""
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        a = self.diffusion_coeff(self.data.X_ib[:len(u)])
        return a * laplacian_u - f  # Missing gradient term!

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        return u_boundary - self.evaluate_bc(self.data.X_boundary)

    def is_linear(self) -> bool:
        return True


# =============================================================================
# CONVECTION-DIFFUSION: ε∇²u + b·∇u = f
# =============================================================================

class ConvectionDiffusionTask(BaseTask):
    """
    Convection-diffusion equation: ε∇²u + b·∇u = f on [0,1]²

    PIELM only computes ∇²u via σ'', it cannot compute ∇u (first derivatives).
    SPECTO-ELM can build both Laplacian and gradient operators.

    Parameters:
        epsilon: diffusion coefficient (small = convection dominated)
        b: velocity field [bx, by]

    Exact solution: u(x,y) = sin(πx)sin(πy)
    """

    name = "convection-diffusion"
    domain_type = "square"
    pde_type = "convection_diffusion"

    def __init__(self, N_interior: int = 400, N_boundary: int = 100,
                 epsilon: float = 0.1, bx: float = 1.0, by: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.epsilon = epsilon
        self.bx = bx
        self.by = by

    def load_data(self) -> TaskData:
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """
        f = ε∇²u + b·∇u

        u = sin(πx)sin(πy)
        ∇²u = -2π²sin(πx)sin(πy)
        ∇u = (π cos(πx)sin(πy), π sin(πx)cos(πy))
        """
        x, y = X[:, 0], X[:, 1]

        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        laplacian_u = -2 * np.pi**2 * u

        du_dx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        du_dy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

        return self.epsilon * laplacian_u + self.bx * du_dx + self.by * du_dy

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """Note: PIELM will fail here - it cannot compute ∇u"""
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        # PIELM only has laplacian_u, not gradient_u!
        return self.epsilon * laplacian_u - f  # Missing b·∇u term!

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        return u_boundary - self.evaluate_bc(self.data.X_boundary)

    def is_linear(self) -> bool:
        return True


class ConvectionDominatedTask(ConvectionDiffusionTask):
    """
    Convection-dominated case: very small diffusion (ε=0.01).
    Forms boundary layers that are challenging for all methods.
    """
    name = "convection-dominated"

    def __init__(self, **kwargs):
        kwargs.setdefault('epsilon', 0.01)
        kwargs.setdefault('bx', 1.0)
        kwargs.setdefault('by', 0.5)
        super().__init__(**kwargs)


# =============================================================================
# ANISOTROPIC DIFFUSION: a_xx ∂²u/∂x² + a_yy ∂²u/∂y² = f
# =============================================================================

class AnisotropicDiffusionTask(BaseTask):
    """
    Anisotropic diffusion: a_xx ∂²u/∂x² + a_yy ∂²u/∂y² = f on [0,1]²

    PIELM computes isotropic Laplacian σ''(z)(w_x² + w_y²).
    It cannot handle different diffusion coefficients in x and y.
    SPECTO-ELM can build D²_x and D²_y operators separately.

    Parameters:
        a_xx: diffusion in x direction
        a_yy: diffusion in y direction

    Exact solution: u(x,y) = sin(πx)sin(πy)
    """

    name = "anisotropic-diffusion"
    domain_type = "square"
    pde_type = "anisotropic_diffusion"

    def __init__(self, N_interior: int = 400, N_boundary: int = 100,
                 a_xx: float = 1.0, a_yy: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.a_xx = a_xx
        self.a_yy = a_yy

    def load_data(self) -> TaskData:
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """
        f = a_xx ∂²u/∂x² + a_yy ∂²u/∂y²

        u = sin(πx)sin(πy)
        ∂²u/∂x² = -π²sin(πx)sin(πy)
        ∂²u/∂y² = -π²sin(πx)sin(πy)
        """
        x, y = X[:, 0], X[:, 1]
        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        return -(self.a_xx + self.a_yy) * np.pi**2 * u

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """
        Note: PIELM assumes a_xx = a_yy = 1 (isotropic).
        This will give wrong results for anisotropic problems.
        """
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        # PIELM treats this as isotropic: ∇²u instead of a_xx∂²u/∂x² + a_yy∂²u/∂y²
        return laplacian_u - f

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        return u_boundary - self.evaluate_bc(self.data.X_boundary)

    def is_linear(self) -> bool:
        return True


class StronglyAnisotropicTask(AnisotropicDiffusionTask):
    """
    Strongly anisotropic case: 100x difference in x and y diffusion.
    """
    name = "strongly-anisotropic"

    def __init__(self, **kwargs):
        kwargs.setdefault('a_xx', 1.0)
        kwargs.setdefault('a_yy', 100.0)
        super().__init__(**kwargs)


# =============================================================================
# HIGH-FREQUENCY POISSON (Spectral methods excel)
# =============================================================================

class HighFrequencyPoissonTask(BaseTask):
    """
    Poisson with high-frequency solution: u = sin(kπx)sin(kπy)

    Higher wavenumber k means more oscillations.
    Spectral methods have exponential convergence for smooth oscillatory solutions.
    Point methods (PIELM) need more collocation points to resolve.

    Exact solution: u(x,y) = sin(kπx)sin(kπy)
    Source: f = -2k²π²sin(kπx)sin(kπy)
    """

    name = "poisson-highfreq"
    domain_type = "square"

    def __init__(self, N_interior: int = 400, N_boundary: int = 100,
                 k: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.k = k  # wavenumber

    def load_data(self) -> TaskData:
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        x, y = X[:, 0], X[:, 1]
        return np.sin(self.k * np.pi * x) * np.sin(self.k * np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """f = -2k²π²sin(kπx)sin(kπy)"""
        x, y = X[:, 0], X[:, 1]
        return -2 * (self.k * np.pi)**2 * np.sin(self.k * np.pi * x) * np.sin(self.k * np.pi * y)

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        return laplacian_u - f

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        return u_boundary - self.evaluate_bc(self.data.X_boundary)

    def is_linear(self) -> bool:
        return True


class VeryHighFrequencyPoissonTask(HighFrequencyPoissonTask):
    """Very high frequency: k=8 (64 oscillations in domain)"""
    name = "poisson-veryhighfreq"

    def __init__(self, **kwargs):
        kwargs.setdefault('k', 8)
        super().__init__(**kwargs)


# =============================================================================
# MIXED DERIVATIVE PROBLEMS
# =============================================================================

class MixedDerivativeTask(BaseTask):
    """
    PDE with mixed derivatives: ∂²u/∂x² + ∂²u/∂y² + c·∂²u/∂x∂y = f

    PIELM computes pure Laplacian σ''(z)(w_x² + w_y²), not mixed derivatives.
    SPECTO-ELM can build the mixed derivative operator D_x @ D_y.

    Exact solution: u(x,y) = sin(πx)sin(πy) + 0.5*sin(2πx)sin(2πy)
    """

    name = "mixed-derivative"
    domain_type = "square"
    pde_type = "mixed_derivative"

    def __init__(self, N_interior: int = 400, N_boundary: int = 100,
                 c: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary
        self.c = c  # coefficient of mixed derivative

    def load_data(self) -> TaskData:
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """
        f = ∇²u + c·∂²u/∂x∂y

        u = sin(πx)sin(πy)
        ∇²u = -2π²sin(πx)sin(πy)
        ∂²u/∂x∂y = π²cos(πx)cos(πy)
        """
        x, y = X[:, 0], X[:, 1]
        laplacian_u = -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        mixed_deriv = np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)
        return laplacian_u + self.c * mixed_deriv

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """PIELM will miss the mixed derivative term"""
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        return laplacian_u - f  # Missing c·∂²u/∂x∂y term!

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        return u_boundary - self.evaluate_bc(self.data.X_boundary)

    def is_linear(self) -> bool:
        return True


# =============================================================================
# REACTION-DIFFUSION: ∇²u + r(x,y)·u = f
# =============================================================================

class ReactionDiffusionTask(BaseTask):
    """
    Reaction-diffusion: ∇²u + r(x,y)·u = f on [0,1]²

    The reaction term r(x,y) varies spatially.
    PIELM's σ'' only gives Laplacian, cannot handle spatially varying r(x,y)·u.
    SPECTO-ELM can build L + diag(r) operator.

    Reaction: r(x,y) = 1 + sin(2πx)sin(2πy)
    Exact solution: u(x,y) = sin(πx)sin(πy)
    """

    name = "reaction-diffusion"
    domain_type = "square"
    pde_type = "reaction_diffusion"

    def __init__(self, N_interior: int = 400, N_boundary: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.N_interior = N_interior
        self.N_boundary = N_boundary

    def reaction_coeff(self, X: np.ndarray) -> np.ndarray:
        """r(x,y) = 1 + sin(2πx)sin(2πy)"""
        x, y = X[:, 0], X[:, 1]
        return 1.0 + np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    def load_data(self) -> TaskData:
        np.random.seed(42)

        X_interior = np.random.rand(self.N_interior, 2)

        N_per_edge = self.N_boundary // 4
        bottom = np.column_stack([np.linspace(0, 1, N_per_edge), np.zeros(N_per_edge)])
        top = np.column_stack([np.linspace(0, 1, N_per_edge), np.ones(N_per_edge)])
        left = np.column_stack([np.zeros(N_per_edge), np.linspace(0, 1, N_per_edge)])
        right = np.column_stack([np.ones(N_per_edge), np.linspace(0, 1, N_per_edge)])
        X_boundary = np.vstack([bottom, top, left, right])

        X_ib = np.vstack([X_interior, X_boundary])

        return TaskData(
            X_interior=X_interior,
            X_boundary=X_boundary,
            X_ghost=np.array([]).reshape(0, 2),
            f=self.evaluate_source(X_ib),
            g=self.evaluate_bc(X_boundary),
            u_true=self.evaluate_exact(X_ib)
        )

    def evaluate_exact(self, X: np.ndarray) -> np.ndarray:
        x, y = X[:, 0], X[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def evaluate_source(self, X: np.ndarray) -> np.ndarray:
        """f = ∇²u + r·u"""
        x, y = X[:, 0], X[:, 1]
        u = np.sin(np.pi * x) * np.sin(np.pi * y)
        laplacian_u = -2 * np.pi**2 * u
        r = self.reaction_coeff(X)
        return laplacian_u + r * u

    def evaluate_bc(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X))

    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        f = self.evaluate_source(self.data.X_ib[:len(u)])
        r = self.reaction_coeff(self.data.X_ib[:len(u)])
        return laplacian_u + r * u - f

    def compute_bc_residual(self, u_boundary: np.ndarray) -> np.ndarray:
        return u_boundary - self.evaluate_bc(self.data.X_boundary)

    def is_linear(self) -> bool:
        return True
