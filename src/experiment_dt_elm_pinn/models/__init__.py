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
from .dt_elm_pinn_accelerated import (
    DTELMPINNAccelerated,
    DTELMPINNAccelDeep2, DTELMPINNAccelDeep3, DTELMPINNAccelDeep4
)
from .dt_elm_pinn_hybrid import (
    DTELMPINNHybrid,
    DTELMPINNHybridDeep2, DTELMPINNHybridDeep3, DTELMPINNHybridDeep4
)
from .pielm import PIELM
from .ropinn import RoPINN, RoPINNLarge
from .das import DAS

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

# GPU-accelerated variants (full PyTorch backend - best for GPU)
ModelRegistry.register('dt-elm-pinn-accel', DTELMPINNAccelerated)
ModelRegistry.register('dt-elm-pinn-accel-deep2', DTELMPINNAccelDeep2)
ModelRegistry.register('dt-elm-pinn-accel-deep3', DTELMPINNAccelDeep3)
ModelRegistry.register('dt-elm-pinn-accel-deep4', DTELMPINNAccelDeep4)

# Hybrid variants (SciPy sparse + PyTorch Cholesky - best for CPU with large M)
ModelRegistry.register('dt-elm-pinn-hybrid', DTELMPINNHybrid)
ModelRegistry.register('dt-elm-pinn-hybrid-deep2', DTELMPINNHybridDeep2)
ModelRegistry.register('dt-elm-pinn-hybrid-deep3', DTELMPINNHybridDeep3)
ModelRegistry.register('dt-elm-pinn-hybrid-deep4', DTELMPINNHybridDeep4)

# RoPINN: Region-Optimized PINN (2024 baseline)
ModelRegistry.register('ropinn', RoPINN)              # 4 layers x 50 nodes (matches vanilla-pinn)
ModelRegistry.register('ropinn-large', RoPINNLarge)   # 4 layers x 512 nodes (RoPINN paper default)

# DAS: Deep Adaptive Sampling (2022 baseline)
ModelRegistry.register('das', DAS)  # Multi-stage training with normalizing flow sampling

__all__ = [
    'BaseModel', 'ModelRegistry',
    'VanillaPINN', 'DTPINN', 'ELM', 'DTELMPINN',
    'DTELMPINNCholesky', 'DTELMPINNSVD', 'PIELM',
    'DTELMPINNDeep2', 'DTELMPINNDeep3', 'DTELMPINNDeep4',
    'DTELMPINNAccelerated', 'DTELMPINNAccelDeep2', 'DTELMPINNAccelDeep3', 'DTELMPINNAccelDeep4',
    'DTELMPINNHybrid', 'DTELMPINNHybridDeep2', 'DTELMPINNHybridDeep3', 'DTELMPINNHybridDeep4',
    'RoPINN', 'RoPINNLarge',
    'DAS',
]
