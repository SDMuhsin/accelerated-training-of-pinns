"""
Base discretizer interface.

Discretizers build discrete differential operators (L, B) from point clouds.
Different models use different discretization methods:
- DT-PINN: RBF-FD (works on any domain)
- SPECTO-ELM: Spectral collocation (requires tensor-product domains)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix


class Discretizer(ABC):
    """
    Abstract base class for discretization methods.

    Discretizers convert point clouds into discrete differential operators.
    """

    @abstractmethod
    def build_operators(
        self,
        X_interior: np.ndarray,
        X_boundary: np.ndarray,
        **kwargs
    ) -> Tuple[csr_matrix, csr_matrix, Optional[np.ndarray]]:
        """
        Build discrete operators from point cloud.

        Args:
            X_interior: Interior collocation points (N_i, d)
            X_boundary: Boundary collocation points (N_b, d)
            **kwargs: Method-specific parameters

        Returns:
            L: Laplacian operator (sparse matrix)
            B: Boundary operator (sparse matrix)
            X_ghost: Ghost points if applicable, else None
        """
        pass

    @abstractmethod
    def is_compatible(self, domain_type: str) -> bool:
        """
        Check if this discretizer supports the given domain type.

        Args:
            domain_type: One of 'disk', 'square', 'cube', 'lshape'

        Returns:
            True if compatible, False otherwise
        """
        pass

    def check_compatibility(self, domain_type: str, task_name: str = "unknown") -> None:
        """
        Raise error if discretizer is incompatible with domain.

        Args:
            domain_type: Domain type string
            task_name: Name of task for error message

        Raises:
            ValueError: If incompatible
        """
        if not self.is_compatible(domain_type):
            raise ValueError(self._incompatibility_message(domain_type, task_name))

    def _incompatibility_message(self, domain_type: str, task_name: str) -> str:
        """Generate helpful error message for incompatible domain."""
        return (
            f"Discretizer '{self.__class__.__name__}' is incompatible with "
            f"domain type '{domain_type}' (task: '{task_name}')."
        )
