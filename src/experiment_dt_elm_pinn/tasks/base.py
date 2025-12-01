"""
Base task interface and registry for PDE problems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any, Type
import numpy as np
import torch


@dataclass
class TaskData:
    """Container for task data."""
    # Collocation points
    X_interior: np.ndarray      # Interior points (N_i, d)
    X_boundary: np.ndarray      # Boundary points (N_b, d)

    # Source and boundary terms
    f: np.ndarray               # Source term at interior+boundary (N_ib,)
    g: np.ndarray               # Boundary condition values (N_b,)

    # Optional: Ghost points for RBF-FD
    X_ghost: Optional[np.ndarray] = None  # Ghost points (N_g, d)

    # Ground truth (if available)
    u_true: Optional[np.ndarray] = None   # True solution at interior+boundary (N_ib,)

    # Discrete operators (for DT methods)
    L: Optional[Any] = None     # Laplacian operator (sparse)
    B: Optional[Any] = None     # Boundary operator (sparse)

    # Robin boundary coefficients (optional)
    alpha: Optional[np.ndarray] = None  # Neumann coefficient
    beta: Optional[np.ndarray] = None   # Dirichlet coefficient
    n: Optional[np.ndarray] = None      # Normal vectors at boundary

    @property
    def X_full(self) -> np.ndarray:
        """Full collocation points: interior + boundary + ghost"""
        parts = [self.X_interior, self.X_boundary]
        if self.X_ghost is not None:
            parts.append(self.X_ghost)
        return np.vstack(parts)

    @property
    def X_ib(self) -> np.ndarray:
        """Interior + boundary points"""
        return np.vstack([self.X_interior, self.X_boundary])

    @property
    def N_interior(self) -> int:
        return self.X_interior.shape[0]

    @property
    def N_boundary(self) -> int:
        return self.X_boundary.shape[0]

    @property
    def N_ghost(self) -> int:
        return self.X_ghost.shape[0] if self.X_ghost is not None else 0

    @property
    def N_ib(self) -> int:
        """Number of interior + boundary points"""
        return self.N_interior + self.N_boundary

    @property
    def N_total(self) -> int:
        return self.N_interior + self.N_boundary + self.N_ghost

    @property
    def spatial_dim(self) -> int:
        return self.X_interior.shape[1]


class BaseTask(ABC):
    """
    Abstract base class for PDE tasks.

    Each task defines:
    - PDE equation and boundary conditions
    - Data loading/generation
    - Residual computation methods
    """

    name: str = "base"

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._data: Optional[TaskData] = None

    @abstractmethod
    def load_data(self) -> TaskData:
        """Load or generate task data."""
        pass

    @property
    def data(self) -> TaskData:
        """Lazy-load task data."""
        if self._data is None:
            self._data = self.load_data()
        return self._data

    @abstractmethod
    def compute_pde_residual(self, u: np.ndarray, laplacian_u: np.ndarray) -> np.ndarray:
        """
        Compute PDE residual: F(u) = 0

        Args:
            u: Solution values at interior+boundary points
            laplacian_u: Laplacian of u at interior+boundary points

        Returns:
            Residual at interior+boundary points
        """
        pass

    @abstractmethod
    def compute_bc_residual(self, u: np.ndarray) -> np.ndarray:
        """
        Compute boundary condition residual.

        Args:
            u: Solution values at boundary points

        Returns:
            BC residual at boundary points
        """
        pass

    def has_discrete_operators(self) -> bool:
        """Check if discrete operators (L, B) are available."""
        return self.data.L is not None and self.data.B is not None

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        """Return default arguments for this task."""
        return {}


class TaskRegistry:
    """Registry for available tasks."""

    _tasks: Dict[str, Type[BaseTask]] = {}

    @classmethod
    def register(cls, name: str, task_cls: Type[BaseTask]):
        """Register a task class."""
        cls._tasks[name] = task_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseTask]:
        """Get a task class by name."""
        if name not in cls._tasks:
            raise ValueError(f"Unknown task: {name}. Available: {list(cls._tasks.keys())}")
        return cls._tasks[name]

    @classmethod
    def list_tasks(cls) -> list:
        """List all registered task names."""
        return list(cls._tasks.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseTask:
        """Create a task instance."""
        task_cls = cls.get(name)
        return task_cls(**kwargs)
