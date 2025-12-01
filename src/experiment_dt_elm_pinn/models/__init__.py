"""
Model registry for PINN solvers.

Models define:
- Network architecture
- Training paradigm (gradient-based, ELM, Newton iteration)
- Loss computation
"""

from .base import BaseModel, ModelRegistry
from .vanilla_pinn import VanillaPINN
from .dt_pinn import DTPINN
from .elm import ELM
from .dt_elm_pinn import DTELMPINN, DTELMPINNCholesky, DTELMPINNSVD

# Register all available models
ModelRegistry.register('vanilla-pinn', VanillaPINN)
ModelRegistry.register('dt-pinn', DTPINN)
ModelRegistry.register('elm', ELM)
ModelRegistry.register('dt-elm-pinn', DTELMPINN)  # Default (Cholesky)
ModelRegistry.register('dt-elm-pinn-cholesky', DTELMPINNCholesky)
ModelRegistry.register('dt-elm-pinn-svd', DTELMPINNSVD)

__all__ = [
    'BaseModel', 'ModelRegistry',
    'VanillaPINN', 'DTPINN', 'ELM', 'DTELMPINN',
    'DTELMPINNCholesky', 'DTELMPINNSVD'
]
