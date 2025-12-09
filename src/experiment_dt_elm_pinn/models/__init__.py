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
from .dt_elm_pinn import (
    DTELMPINN, DTELMPINNCholesky, DTELMPINNSVD,
    DTELMPINNDeep2, DTELMPINNDeep3, DTELMPINNDeep4
)
from .pielm import PIELM

# Register all available models
ModelRegistry.register('vanilla-pinn', VanillaPINN)
ModelRegistry.register('dt-pinn', DTPINN)
ModelRegistry.register('elm', ELM)
ModelRegistry.register('dt-elm-pinn', DTELMPINN)  # Default (Cholesky, single layer)
ModelRegistry.register('dt-elm-pinn-cholesky', DTELMPINNCholesky)
ModelRegistry.register('dt-elm-pinn-svd', DTELMPINNSVD)
ModelRegistry.register('pielm', PIELM)

# Deep (multi-layer) variants - uses skip connections
ModelRegistry.register('dt-elm-pinn-deep2', DTELMPINNDeep2)  # 2 layers [100, 100]
ModelRegistry.register('dt-elm-pinn-deep3', DTELMPINNDeep3)  # 3 layers [100, 100, 100]
ModelRegistry.register('dt-elm-pinn-deep4', DTELMPINNDeep4)  # 4 layers [100, 100, 100, 100]

__all__ = [
    'BaseModel', 'ModelRegistry',
    'VanillaPINN', 'DTPINN', 'ELM', 'DTELMPINN',
    'DTELMPINNCholesky', 'DTELMPINNSVD', 'PIELM',
    'DTELMPINNDeep2', 'DTELMPINNDeep3', 'DTELMPINNDeep4'
]
