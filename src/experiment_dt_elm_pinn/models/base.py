"""
Base model interface and registry for PINN solvers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, List
import numpy as np
import time


@dataclass
class TrainResult:
    """Container for training results."""
    u_pred: np.ndarray                          # Predicted solution
    train_time: float                           # Total training time (seconds)
    l2_error: Optional[float] = None            # Relative L2 error (if u_true available)
    final_loss: Optional[float] = None          # Final training loss
    loss_history: List[float] = field(default_factory=list)  # Loss per epoch
    n_iterations: int = 0                       # Number of iterations/epochs
    extra: Dict[str, Any] = field(default_factory=dict)  # Model-specific metrics


class BaseModel(ABC):
    """
    Abstract base class for PINN models.

    All models must implement:
    - train(): Training procedure
    - predict(): Inference

    Models can optionally override:
    - setup(): Pre-training setup (e.g., building operators)
    """

    name: str = "base"

    def __init__(self, task, **kwargs):
        """
        Args:
            task: Task object providing PDE data
            **kwargs: Model-specific hyperparameters
        """
        self.task = task
        self.config = kwargs
        self._is_setup = False

    def setup(self):
        """Pre-training setup. Called once before train()."""
        self._is_setup = True

    @abstractmethod
    def train(self, **kwargs) -> TrainResult:
        """
        Train the model.

        Returns:
            TrainResult with predictions and metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions at given points.

        Args:
            X: Input points (N, d)

        Returns:
            Predictions (N,)
        """
        pass

    def compute_l2_error(self, u_pred: np.ndarray, u_true: np.ndarray) -> float:
        """Compute relative L2 error."""
        pred_flat = u_pred.flatten()
        true_flat = u_true.flatten()
        return np.linalg.norm(pred_flat - true_flat) / np.linalg.norm(true_flat)

    @classmethod
    def get_default_args(cls) -> Dict[str, Any]:
        """Return default hyperparameters for this model."""
        return {}

    @classmethod
    def add_argparse_args(cls, parser):
        """Add model-specific arguments to argparse."""
        pass


class ModelRegistry:
    """Registry for available models."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_cls: Type[BaseModel]):
        """Register a model class."""
        cls._models[name] = model_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """Get a model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def create(cls, name: str, task, **kwargs) -> BaseModel:
        """Create a model instance."""
        model_cls = cls.get(name)
        return model_cls(task, **kwargs)
