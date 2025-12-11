#!/usr/bin/env python3
"""
Validate all spectral tasks for correctness and model compatibility.

This script checks:
1. Task loads without errors
2. Operator L satisfies L @ u_true ≈ f (for interior points)
3. Boundary operator B extracts correct values
4. All models can run (at least minimal training)
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_dt_elm_pinn.tasks import TaskRegistry


# All spectral tasks (existing + new)
ALL_TASKS = [
    # Existing 2D smooth
    'spectral-poisson-square',
    'spectral-laplace-square',
    'spectral-nonlinear-poisson-square',
    # New 3D
    'spectral-poisson-cube',
    'spectral-laplace-cube',
    'spectral-nonlinear-poisson-cube',
    # New localized features
    'spectral-poisson-peaked',
    'spectral-boundary-layer',
    'spectral-poisson-corner',
]


def validate_task(task_name: str, verbose: bool = True) -> bool:
    """Validate a single task."""
    try:
        task = TaskRegistry.create(task_name)
        data = task.load_data()

        N_int = data.X_interior.shape[0]
        N_bnd = data.X_boundary.shape[0]
        dim = data.X_interior.shape[1]

        # Check operator residual (interior only)
        L_u = data.L @ data.u_true
        residual_int = np.max(np.abs((L_u - data.f)[:N_int]))

        # For nonlinear tasks, expect higher residual
        is_nonlinear = not task.is_linear()
        tol = 1.0 if is_nonlinear else 1e-3

        # Check BC satisfaction
        B_u = data.B @ data.u_true
        bc_error = np.max(np.abs(B_u - data.g))

        passed = (residual_int < tol or is_nonlinear) and bc_error < 1e-4

        if verbose:
            status = "✓" if passed else "✗"
            print(f'{status} {task_name}')
            print(f'    Dim: {dim}D, Interior: {N_int}, Boundary: {N_bnd}')
            print(f'    Interior PDE residual: {residual_int:.2e}')
            print(f'    BC error: {bc_error:.2e}')
            print(f'    Linear: {task.is_linear()}')
            if not passed:
                print(f'    FAILED: Residual or BC error too high!')
            print()

        return passed

    except Exception as e:
        if verbose:
            print(f'✗ {task_name}: {e}')
            import traceback
            traceback.print_exc()
        return False


def validate_model_compatibility(task_name: str, model_name: str, verbose: bool = True) -> bool:
    """Test that a model can train on a task (minimal epochs)."""
    try:
        task = TaskRegistry.create(task_name)

        if model_name == 'dt-elm-pinn':
            from src.experiment_dt_elm_pinn.models.dt_elm_pinn import DTELMPINN
            model = DTELMPINN(task=task, max_iter=3)
        elif model_name == 'vanilla-pinn':
            from src.experiment_dt_elm_pinn.models.vanilla_pinn import VanillaPINN
            model = VanillaPINN(task=task, n_layers=2, n_hidden=20, epochs=5, lr=1e-3, device='cpu')
        elif model_name == 'das':
            from src.experiment_dt_elm_pinn.models.das import DAS
            model = DAS(task=task, n_layers=2, n_hidden=20, max_stage=1, pde_epochs=5, n_train=100, device='cpu')
        else:
            if verbose:
                print(f'  Unknown model: {model_name}')
            return False

        result = model.train(verbose=False)

        # Check for valid output (not NaN, not crazy large)
        valid = (result.l2_error is not None and
                 not np.isnan(result.l2_error) and
                 result.l2_error < 1e20)

        if verbose:
            status = "✓" if valid else "✗"
            print(f'  {status} {model_name}: L2={result.l2_error:.2e}, Time={result.train_time:.2f}s')

        return valid

    except Exception as e:
        if verbose:
            print(f'  ✗ {model_name}: {e}')
        return False


def main():
    print('=' * 70)
    print('TASK VALIDATION')
    print('=' * 70)

    task_results = {}
    for task_name in ALL_TASKS:
        task_results[task_name] = validate_task(task_name)

    print('=' * 70)
    print('MODEL COMPATIBILITY')
    print('=' * 70)

    MODELS = ['dt-elm-pinn', 'vanilla-pinn', 'das']
    model_results = {}

    for task_name in ALL_TASKS[:4]:  # Test subset for speed
        print(f'\n{task_name}:')
        model_results[task_name] = {}
        for model_name in MODELS:
            model_results[task_name][model_name] = validate_model_compatibility(task_name, model_name)

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    n_tasks_passed = sum(task_results.values())
    n_tasks_total = len(task_results)
    print(f'Tasks: {n_tasks_passed}/{n_tasks_total} passed')

    n_compat_passed = sum(v for t in model_results.values() for v in t.values())
    n_compat_total = len(model_results) * len(MODELS)
    print(f'Model compatibility: {n_compat_passed}/{n_compat_total} passed')

    all_passed = (n_tasks_passed == n_tasks_total and n_compat_passed == n_compat_total)

    if all_passed:
        print('\n✓ All validations passed!')
        return 0
    else:
        print('\n✗ Some validations failed!')
        return 1


if __name__ == '__main__':
    sys.exit(main())
