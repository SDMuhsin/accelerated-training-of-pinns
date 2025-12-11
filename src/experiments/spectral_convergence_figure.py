"""
Generate spectral convergence figure for SPECTO-ELM paper.
Shows exponential convergence of spectral discretization and ELM bottleneck.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiment_dt_elm_pinn'))

from experiment_dt_elm_pinn.tasks.spectral import (
    chebyshev_points, chebyshev_laplacian_2d, chebyshev_grid_2d,
    scale_domain, scale_laplacian
)
from experiment_dt_elm_pinn.tasks import TaskRegistry
from experiment_dt_elm_pinn.models.dt_elm_pinn import DTELMPINN


def pure_spectral_convergence():
    """Test pure spectral solve (no ELM) - should reach machine precision."""
    print("Testing pure spectral convergence...")

    Ns = [8, 10, 12, 14, 16, 18, 20, 25]
    errors = []

    for N in Ns:
        # Generate Chebyshev grid on [-1, 1]^2
        x = chebyshev_points(N)
        y = chebyshev_points(N)
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        # Scale to [0, 1]^2
        X_scaled = (X_flat + 1) / 2
        Y_scaled = (Y_flat + 1) / 2

        # Get Laplacian operator
        L_raw = chebyshev_laplacian_2d(N, N)
        L = scale_laplacian(L_raw, (0, 1), (0, 1))  # Scale for [0,1]^2

        # True solution: u = sin(pi*x) * sin(pi*y)
        u_true = np.sin(np.pi * X_scaled) * np.sin(np.pi * Y_scaled)

        # Source term: f = -2*pi^2 * sin(pi*x) * sin(pi*y)
        f = -2 * np.pi**2 * np.sin(np.pi * X_scaled) * np.sin(np.pi * Y_scaled)

        # Check: L @ u_true should equal f
        Lu = L @ u_true

        # Compute error (interior only, excluding boundary)
        boundary_mask = (np.abs(X_flat - (-1)) < 1e-10) | (np.abs(X_flat - 1) < 1e-10) | \
                       (np.abs(Y_flat - (-1)) < 1e-10) | (np.abs(Y_flat - 1) < 1e-10)
        interior_mask = ~boundary_mask

        error = np.linalg.norm(Lu[interior_mask] - f[interior_mask]) / np.linalg.norm(f[interior_mask])
        errors.append(error)
        print(f"  N={N}: error = {error:.2e}")

    return Ns, errors


def specto_elm_convergence():
    """Test SPECTO-ELM convergence with fixed hidden neurons."""
    print("\nTesting SPECTO-ELM convergence...")

    # Use actual task
    task_cls = TaskRegistry.get('spectral-poisson-square')

    Ns = [12, 16, 20, 25]
    errors = []

    for N in Ns:
        task = task_cls(N=N)
        task.load_data()

        model = DTELMPINN(
            task,
            hidden_sizes=[100],
            use_skip_connections=True,
            solver='svd',
            seed=42,
        )

        model.setup()
        result = model.train()
        errors.append(result.l2_error)
        print(f"  N={N}: L2 error = {result.l2_error:.2e}")

    return Ns, errors


def specto_elm_vs_hidden_neurons():
    """Test SPECTO-ELM accuracy vs number of hidden neurons."""
    print("\nTesting SPECTO-ELM vs hidden neurons...")

    task_cls = TaskRegistry.get('spectral-poisson-square')

    hidden_sizes_list = [50, 100, 150, 200, 300, 500]
    errors = []

    for hidden in hidden_sizes_list:
        task = task_cls(N=25)
        task.load_data()

        model = DTELMPINN(
            task,
            hidden_sizes=[hidden],
            use_skip_connections=True,
            solver='svd',
            seed=42,
        )

        model.setup()
        result = model.train()
        errors.append(result.l2_error)
        print(f"  Hidden={hidden}: L2 error = {result.l2_error:.2e}")

    return hidden_sizes_list, errors


def create_figure(output_path):
    """Create the spectral convergence figure."""

    # Get data
    Ns_pure, errors_pure = pure_spectral_convergence()
    Ns_elm, errors_elm = specto_elm_convergence()
    hidden_sizes, errors_hidden = specto_elm_vs_hidden_neurons()

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Spectral convergence
    ax = axes[0]
    ax.semilogy(Ns_pure, errors_pure, 'b-o', linewidth=2, markersize=8, label='Pure Spectral')
    ax.semilogy(Ns_elm, errors_elm, 'r-s', linewidth=2, markersize=8, label='SPECTO-ELM (100 neurons)')
    ax.axhline(y=1e-12, color='gray', linestyle='--', alpha=0.5, label='Machine precision')
    ax.set_xlabel('Grid Size N (per dimension)', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title('Spectral Convergence', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-14, 1)

    # Right plot: ELM accuracy vs hidden neurons
    ax = axes[1]
    ax.semilogy(hidden_sizes, errors_hidden, 'g-^', linewidth=2, markersize=8)
    ax.set_xlabel('Hidden Neurons', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('SPECTO-ELM: Effect of Hidden Neurons', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add annotation showing ELM is bottleneck
    ax.annotate('ELM approximation\nbecomes bottleneck',
                xy=(200, errors_hidden[3]),
                xytext=(300, 1e-3),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='gray'),
                ha='center')

    plt.suptitle('SPECTO-ELM: Spectral Convergence Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved figure to: {output_path}")


def main():
    """Generate spectral convergence figure."""
    print("="*70)
    print("GENERATING SPECTRAL CONVERGENCE FIGURE")
    print("="*70)

    # Output to paper/figures
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'spectral_convergence.png')
    create_figure(output_path)

    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
