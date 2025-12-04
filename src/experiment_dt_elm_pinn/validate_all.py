#!/usr/bin/env python3
"""
Comprehensive validation script for all tasks and models.

Tests each registered task with each registered model to ensure
broad coverage and identify any anomalies in the implementation.

Usage:
    python -m src.experiment_dt_elm_pinn.validate_all [--quick] [--verbose]
"""

import sys
import time
import traceback
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Set random seed for reproducibility
np.random.seed(42)

# Import tasks and models
from tasks import TaskRegistry, BaseTask
from models import ModelRegistry, BaseModel


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    task_name: str
    model_name: str
    success: bool
    l2_error: Optional[float] = None
    train_time: Optional[float] = None
    error_msg: Optional[str] = None
    is_linear: Optional[bool] = None
    n_points: Optional[int] = None


def validate_task_model(task_name: str, model_name: str,
                        verbose: bool = False,
                        quick: bool = False) -> ValidationResult:
    """
    Validate a single task/model combination.

    Args:
        task_name: Name of the task from TaskRegistry
        model_name: Name of the model from ModelRegistry
        verbose: Print detailed progress
        quick: Use quick settings (fewer epochs, smaller networks)

    Returns:
        ValidationResult with test outcome
    """
    try:
        # Create task
        task_cls = TaskRegistry.get(task_name)

        # Use smaller problem size for RBF-FD tasks in quick mode
        if quick and 'rbf-fd' in task_name:
            task = task_cls(n_interior=200, n_boundary=50)
        else:
            task = task_cls()

        # Load data
        data = task.data
        is_linear = task.is_linear() if hasattr(task, 'is_linear') else False
        n_points = data.N_ib

        if verbose:
            print(f"    Task loaded: {n_points} points, linear={is_linear}")

        # Create model with appropriate settings
        model_cls = ModelRegistry.get(model_name)

        # Quick mode adjustments
        if quick:
            if model_name in ['vanilla-pinn', 'dt-pinn']:
                model = model_cls(task, layers=2, nodes=20, epochs=50)
            elif model_name in ['dt-elm-pinn', 'dt-elm-pinn-cholesky', 'dt-elm-pinn-svd']:
                model = model_cls(task, hidden_sizes=[50])
            elif model_name == 'elm':
                model = model_cls(task, n_hidden=50)
            elif model_name == 'pielm':
                model = model_cls(task, n_hidden=50)
            else:
                model = model_cls(task)
        else:
            model = model_cls(task)

        # Train model
        start_time = time.perf_counter()
        result = model.train(verbose=verbose)
        train_time = time.perf_counter() - start_time

        # Get L2 error
        l2_error = result.l2_error

        return ValidationResult(
            task_name=task_name,
            model_name=model_name,
            success=True,
            l2_error=l2_error,
            train_time=train_time,
            is_linear=is_linear,
            n_points=n_points,
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        if verbose:
            traceback.print_exc()
        return ValidationResult(
            task_name=task_name,
            model_name=model_name,
            success=False,
            error_msg=error_msg,
        )


def run_validation(tasks: Optional[List[str]] = None,
                   models: Optional[List[str]] = None,
                   verbose: bool = False,
                   quick: bool = False) -> List[ValidationResult]:
    """
    Run validation across all specified tasks and models.

    Args:
        tasks: List of task names (None = all registered)
        models: List of model names (None = all registered)
        verbose: Print detailed progress
        quick: Use quick settings

    Returns:
        List of ValidationResults
    """
    # Get tasks and models
    all_tasks = tasks or TaskRegistry.list_tasks()
    all_models = models or ModelRegistry.list_models()

    print("=" * 80)
    print("DT-PINN Validation Suite")
    print("=" * 80)
    print(f"\nTasks ({len(all_tasks)}): {', '.join(all_tasks)}")
    print(f"Models ({len(all_models)}): {', '.join(all_models)}")
    print(f"Quick mode: {quick}")
    print(f"Total combinations: {len(all_tasks) * len(all_models)}")
    print("\n" + "-" * 80)

    results = []

    for task_name in all_tasks:
        print(f"\n[Task: {task_name}]")

        for model_name in all_models:
            print(f"  Testing {model_name}...", end=" ", flush=True)

            result = validate_task_model(
                task_name, model_name,
                verbose=verbose, quick=quick
            )
            results.append(result)

            if result.success:
                l2_str = f"L2={result.l2_error:.4e}" if result.l2_error else "L2=N/A"
                time_str = f"t={result.train_time:.2f}s" if result.train_time else "t=N/A"
                print(f"OK [{l2_str}, {time_str}]")
            else:
                print(f"FAILED [{result.error_msg}]")

    return results


def print_summary(results: List[ValidationResult]):
    """Print a summary of validation results."""
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Count successes and failures
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    print(f"\nTotal tests: {len(results)}")
    print(f"Passed: {len(successes)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\n" + "-" * 40)
        print("FAILURES:")
        print("-" * 40)
        for f in failures:
            print(f"  {f.task_name} + {f.model_name}: {f.error_msg}")

    # Summary table
    print("\n" + "-" * 80)
    print("L2 Error Summary (lower is better):")
    print("-" * 80)
    print(f"{'Task':<35} {'Model':<20} {'L2 Error':>12} {'Time':>10}")
    print("-" * 80)

    for r in successes:
        l2_str = f"{r.l2_error:.4e}" if r.l2_error else "N/A"
        time_str = f"{r.train_time:.2f}s" if r.train_time else "N/A"
        print(f"{r.task_name:<35} {r.model_name:<20} {l2_str:>12} {time_str:>10}")

    # Anomaly detection
    print("\n" + "-" * 80)
    print("ANOMALY CHECK:")
    print("-" * 80)

    anomalies = []
    for r in successes:
        if r.l2_error is not None:
            # Flag high L2 errors
            if r.l2_error > 1.0:
                anomalies.append(f"HIGH L2: {r.task_name} + {r.model_name}: L2={r.l2_error:.2e}")
            # Flag linear tasks with high errors
            if r.is_linear and r.l2_error > 0.1:
                anomalies.append(f"LINEAR HIGH: {r.task_name} + {r.model_name}: L2={r.l2_error:.2e}")

    if anomalies:
        for a in anomalies:
            print(f"  WARNING: {a}")
    else:
        print("  No anomalies detected.")

    print("\n" + "=" * 80)
    passed = len(failures) == 0
    print(f"OVERALL: {'PASSED' if passed else 'FAILED'}")
    print("=" * 80)

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate all tasks and models")
    parser.add_argument('--tasks', type=str, nargs='+', help='Specific tasks to test')
    parser.add_argument('--models', type=str, nargs='+', help='Specific models to test')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode (smaller problems)')
    parser.add_argument('--list', '-l', action='store_true', help='List available tasks and models')

    args = parser.parse_args()

    if args.list:
        print("Available Tasks:")
        for t in TaskRegistry.list_tasks():
            print(f"  - {t}")
        print("\nAvailable Models:")
        for m in ModelRegistry.list_models():
            print(f"  - {m}")
        return 0

    # Run validation
    results = run_validation(
        tasks=args.tasks,
        models=args.models,
        verbose=args.verbose,
        quick=args.quick,
    )

    # Print summary
    passed = print_summary(results)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
