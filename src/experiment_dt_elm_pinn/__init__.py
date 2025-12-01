"""
Modular DT-ELM-PINN Experimental Framework

This package provides a modular framework for running PDE experiments with:
- Multiple tasks (PDEs from PINNacle benchmark)
- Multiple models (Vanilla PINN, DT-PINN, ELM, DT-ELM-PINN)
- Configurable hyperparameters via argparse
"""

from .train_pinn import main

__all__ = ['main']
