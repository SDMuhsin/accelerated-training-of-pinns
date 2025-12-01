#!/usr/bin/env python
"""
train_pinn.py - Unified training script for PINN experiments.

Supports multiple tasks (PDE problems) and models (solver architectures).
Uses registries for extensibility.

Usage (from project root):
    python -m src.experiment_dt_elm_pinn.train_pinn --task nonlinear-poisson --model dt-elm-pinn
    python -m src.experiment_dt_elm_pinn.train_pinn --task nonlinear-poisson --model vanilla-pinn --epochs 500
    python -m src.experiment_dt_elm_pinn.train_pinn --list-tasks
    python -m src.experiment_dt_elm_pinn.train_pinn --list-models

Results are automatically saved to ./results/experiments.csv (thread-safe for parallel runs).
"""

import argparse
import sys
import os
import json
import time
import csv
import fcntl
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

# Determine project root and add paths for both direct and module execution
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = _THIS_FILE.parent  # experiment_dt_elm_pinn
_SRC_DIR = _PACKAGE_DIR.parent    # src
_PROJECT_ROOT = _SRC_DIR.parent   # dt-pinn

# Add package directory to path for relative imports
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from tasks import TaskRegistry, BaseTask
from models import ModelRegistry, BaseModel


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PINN models on PDE tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train DT-ELM-PINN on nonlinear Poisson
  python train_pinn.py --task nonlinear-poisson --model dt-elm-pinn

  # Train with custom architecture
  python train_pinn.py --task nonlinear-poisson --model dt-elm-pinn --hidden-sizes 150 50

  # Train vanilla PINN with L-BFGS
  python train_pinn.py --task nonlinear-poisson --model vanilla-pinn --optimizer lbfgs --epochs 500

  # Save results to file
  python train_pinn.py --task nonlinear-poisson --model dt-pinn --output results.json
        """
    )

    # Task and model selection
    parser.add_argument('--task', type=str, default='nonlinear-poisson',
                        help='Task name (PDE problem to solve)')
    parser.add_argument('--model', type=str, default='dt-elm-pinn',
                        help='Model name (solver architecture)')

    # Listing options
    parser.add_argument('--list-tasks', action='store_true',
                        help='List available tasks')
    parser.add_argument('--list-models', action='store_true',
                        help='List available models')

    # Task-specific arguments
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory for task')
    parser.add_argument('--file-name', type=str, default='2_2236',
                        help='Data file name/identifier')

    # Common model arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                        help='Print training progress')

    # ELM-specific arguments
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[100],
                        help='Hidden layer sizes for ELM models')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'sin', 'relu'],
                        help='Activation function')
    parser.add_argument('--max-iter', type=int, default=20,
                        help='Maximum Newton iterations (ELM models)')
    parser.add_argument('--tol', type=float, default=1e-8,
                        help='Convergence tolerance (ELM models)')
    parser.add_argument('--no-skip-connections', action='store_true',
                        help='Disable skip connections in multi-layer ELM')

    # PINN-specific arguments
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of hidden layers (PINN models)')
    parser.add_argument('--nodes', type=int, default=50,
                        help='Nodes per hidden layer (PINN models)')
    parser.add_argument('--optimizer', type=str, default='lbfgs',
                        choices=['lbfgs', 'adam'],
                        help='Optimizer (gradient-based models)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--pde-weight', type=float, default=1.0,
                        help='PDE loss weight (vanilla PINN)')
    parser.add_argument('--bc-weight', type=float, default=1.0,
                        help='BC loss weight (vanilla PINN)')

    # Hardware options
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')

    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results (JSON format)')
    parser.add_argument('--csv-output', type=str, default=None,
                        help='CSV file for results (default: ./results/experiments.csv)')
    parser.add_argument('--no-csv', action='store_true',
                        help='Disable automatic CSV output')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress non-essential output')

    return parser.parse_args()


def list_available(args):
    """List available tasks or models."""
    if args.list_tasks:
        print("\nAvailable tasks:")
        for name in TaskRegistry.list_tasks():
            print(f"  - {name}")
        return True

    if args.list_models:
        print("\nAvailable models:")
        for name in ModelRegistry.list_models():
            model_cls = ModelRegistry.get(name)
            defaults = model_cls.get_default_args()
            print(f"  - {name}")
            if defaults:
                print(f"    Defaults: {defaults}")
        return True

    return False


def build_task_kwargs(args) -> Dict[str, Any]:
    """Build keyword arguments for task construction."""
    kwargs = {
        'file_name': args.file_name,
    }
    if args.data_dir:
        kwargs['data_dir'] = args.data_dir
    return kwargs


def build_model_kwargs(args, model_name: str) -> Dict[str, Any]:
    """Build keyword arguments for model construction based on model type."""
    kwargs = {
        'seed': args.seed,
    }

    # ELM-based models
    if model_name in ['elm', 'dt-elm-pinn']:
        kwargs['hidden_sizes'] = args.hidden_sizes
        kwargs['activation'] = args.activation
        kwargs['max_iter'] = args.max_iter
        kwargs['tol'] = args.tol

        if model_name in ['dt-elm-pinn', 'dt-elm-pinn-cholesky', 'dt-elm-pinn-svd']:
            kwargs['use_skip_connections'] = not args.no_skip_connections

        if model_name == 'elm':
            kwargs['use_cuda'] = not args.no_cuda

    # Gradient-based PINN models
    if model_name in ['vanilla-pinn', 'dt-pinn']:
        kwargs['layers'] = args.layers
        kwargs['nodes'] = args.nodes
        kwargs['activation'] = args.activation
        kwargs['optimizer'] = args.optimizer
        kwargs['lr'] = args.lr
        kwargs['epochs'] = args.epochs
        kwargs['use_cuda'] = not args.no_cuda

        if model_name == 'vanilla-pinn':
            kwargs['pde_weight'] = args.pde_weight
            kwargs['bc_weight'] = args.bc_weight

    return kwargs


def format_results(result, task, model, args) -> Dict[str, Any]:
    """Format training results for output."""
    output = {
        'task': args.task,
        'model': args.model,
        'train_time': result.train_time,
        'l2_error': result.l2_error,
        'final_loss': result.final_loss,
        'n_iterations': result.n_iterations,
        'config': {
            'seed': args.seed,
        },
        'extra': result.extra,
    }

    # Add model-specific config
    if args.model in ['elm', 'dt-elm-pinn']:
        output['config']['hidden_sizes'] = args.hidden_sizes
        output['config']['activation'] = args.activation
        output['config']['max_iter'] = args.max_iter
    elif args.model in ['vanilla-pinn', 'dt-pinn']:
        output['config']['layers'] = args.layers
        output['config']['nodes'] = args.nodes
        output['config']['optimizer'] = args.optimizer
        output['config']['epochs'] = args.epochs

    return output


def write_csv_result_threadsafe(result, args, csv_path: Path):
    """
    Write experiment results to CSV with thread-safe file locking.

    Uses fcntl.flock for exclusive file access to allow multiple parallel
    instances to safely write to the same file.
    """
    # Build a flat row with all hyperparameters and metrics
    row = {
        # Identifiers
        'run_id': str(uuid.uuid4())[:8],
        'timestamp': datetime.now().isoformat(),

        # Task info
        'task': args.task,
        'file_name': args.file_name,

        # Model info
        'model': args.model,
        'seed': args.seed,

        # ELM-specific hyperparameters
        'hidden_sizes': str(args.hidden_sizes) if hasattr(args, 'hidden_sizes') else '',
        'activation': args.activation if hasattr(args, 'activation') else '',
        'max_iter': args.max_iter if hasattr(args, 'max_iter') else '',
        'tol': args.tol if hasattr(args, 'tol') else '',

        # PINN-specific hyperparameters
        'layers': args.layers if hasattr(args, 'layers') else '',
        'nodes': args.nodes if hasattr(args, 'nodes') else '',
        'optimizer': args.optimizer if hasattr(args, 'optimizer') else '',
        'lr': args.lr if hasattr(args, 'lr') else '',
        'epochs': args.epochs if hasattr(args, 'epochs') else '',
        'pde_weight': args.pde_weight if hasattr(args, 'pde_weight') else '',
        'bc_weight': args.bc_weight if hasattr(args, 'bc_weight') else '',

        # Results
        'l2_error': result.l2_error,
        'final_loss': result.final_loss,
        'train_time': result.train_time,
        'n_iterations': result.n_iterations,
    }

    # Ensure results directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Thread-safe write with file locking
    file_exists = csv_path.exists()

    # Open in append mode, create if doesn't exist
    with open(csv_path, 'a', newline='') as f:
        # Acquire exclusive lock (blocks until lock is available)
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            # Write header only if file is empty/new
            if not file_exists or csv_path.stat().st_size == 0:
                writer.writeheader()

            writer.writerow(row)
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def print_results(result, quiet: bool = False):
    """Print training results to console."""
    if quiet:
        # Minimal output
        if result.l2_error is not None:
            print(f"L2 error: {result.l2_error:.4e}")
        print(f"Time: {result.train_time:.3f}s")
        return

    print("\n" + "=" * 60)
    print("Training Results")
    print("=" * 60)
    print(f"  Training time:    {result.train_time:.4f} seconds")
    if result.l2_error is not None:
        print(f"  L2 error:         {result.l2_error:.6e}")
    if result.final_loss is not None:
        print(f"  Final loss:       {result.final_loss:.6e}")
    print(f"  Iterations:       {result.n_iterations}")

    if result.extra:
        print("\n  Extra info:")
        for key, value in result.extra.items():
            print(f"    {key}: {value}")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    # Handle listing requests
    if list_available(args):
        return 0

    # Validate task and model names
    try:
        task_cls = TaskRegistry.get(args.task)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-tasks to see available tasks")
        return 1

    try:
        model_cls = ModelRegistry.get(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        print("Use --list-models to see available models")
        return 1

    if not args.quiet:
        print(f"\nTask: {args.task}")
        print(f"Model: {args.model}")

    # Create task
    task_kwargs = build_task_kwargs(args)
    task = task_cls(**task_kwargs)

    if not args.quiet:
        print(f"\nLoading data...")
    task.load_data()

    if not args.quiet:
        data = task.data
        print(f"  Interior points: {data.N_interior}")
        print(f"  Boundary points: {data.N_boundary}")
        print(f"  Ghost points:    {data.N_ghost}")
        print(f"  Total points:    {data.N_total}")
        print(f"  Spatial dim:     {data.spatial_dim}")

    # Create model
    model_kwargs = build_model_kwargs(args, args.model)
    model = model_cls(task, **model_kwargs)

    if not args.quiet:
        print(f"\nModel configuration:")
        for key, value in model_kwargs.items():
            print(f"  {key}: {value}")

    # Train
    if not args.quiet:
        print(f"\nTraining...")

    result = model.train(verbose=args.verbose)

    # Print results
    print_results(result, quiet=args.quiet)

    # Save results to CSV (thread-safe)
    if not args.no_csv:
        if args.csv_output:
            csv_path = Path(args.csv_output)
        else:
            # Default: ./results/experiments.csv relative to project root
            csv_path = _PROJECT_ROOT / 'results' / 'experiments.csv'

        write_csv_result_threadsafe(result, args, csv_path)
        if not args.quiet:
            print(f"\nResults appended to: {csv_path}")

    # Save results if requested (JSON format)
    if args.output:
        output_data = format_results(result, task, model, args)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        if not args.quiet:
            print(f"JSON results saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
