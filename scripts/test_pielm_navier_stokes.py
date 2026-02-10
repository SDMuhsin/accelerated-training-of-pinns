"""
Test and visualize PIELM_NavierStokes for lid-driven cavity flow.

This script:
1. Trains our PIELM model
2. Visualizes velocity and pressure fields
3. Computes and reports PDE residuals
4. Optionally compares with partner's PINN (if trained)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_dt_elm_pinn.models.pielm_navier_stokes import PIELM_NavierStokes


def run_experiment(mode='quick', solver='direct'):
    """
    Run PIELM_NavierStokes experiment.

    Args:
        mode: 'quick' for fast testing, 'full' for production-level
        solver: 'direct' (lstsq) or 'iterative' (lsqr)
    """
    print("=" * 70)
    print("PIELM for Navier-Stokes: Lid-Driven Cavity Flow")
    print("=" * 70)

    # Output directory
    output_dir = 'results/pielm_navier_stokes'
    os.makedirs(output_dir, exist_ok=True)

    # Model parameters based on mode
    if mode == 'quick':
        params = {
            'n_hidden': 200,
            'N_interior': 1000,
            'N_wall': 200,
            'N_lid': 200,
            'max_picard_iter': 50,
        }
        print("Mode: QUICK (small problem for testing)")
    elif mode == 'medium':
        params = {
            'n_hidden': 400,
            'N_interior': 3000,
            'N_wall': 400,
            'N_lid': 400,
            'max_picard_iter': 80,
        }
        print("Mode: MEDIUM (moderate problem size)")
    else:  # full
        params = {
            'n_hidden': 500,
            'N_interior': 6000,
            'N_wall': 800,
            'N_lid': 800,
            'max_picard_iter': 100,
        }
        print("Mode: FULL (matching partner's PINN)")

    print(f"Solver: {solver.upper()}")

    print(f"Parameters: n_hidden={params['n_hidden']}, "
          f"N_interior={params['N_interior']}")
    print()

    # Create model
    model = PIELM_NavierStokes(
        Re=1000.0,
        U_lid=1.0,
        Cs=0.1,
        n_hidden=params['n_hidden'],
        activation='tanh',
        max_picard_iter=params['max_picard_iter'],
        tol=1e-6,
        N_interior=params['N_interior'],
        N_wall=params['N_wall'],
        N_lid=params['N_lid'],
        bc_weight=10.0,
        verbose=True,
        seed=42,
    )
    model.solver = solver  # 'direct' or 'iterative'

    # Train
    print("Training PIELM...")
    results = model.train()

    print()
    print("-" * 40)
    print(f"Training time: {results['train_time']:.3f} seconds")
    print(f"Picard iterations: {results['n_iterations']}")
    print(f"Final rel. change: {results['final_residual']:.2e}")
    print(f"Converged: {results['converged']}")
    print("-" * 40)
    print()

    # Create visualization grid
    print("Generating visualizations...")
    nx, ny = 51, 51
    x_lin = np.linspace(0, 1, nx)
    y_lin = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_lin, y_lin)
    XY = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    # Predict on grid
    u, v, p = model.predict(XY)
    U = u.reshape(ny, nx)
    V = v.reshape(ny, nx)
    P = p.reshape(ny, nx)

    # Compute velocity magnitude
    speed = np.sqrt(U**2 + V**2)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # U-velocity contour
    ax = axes[0, 0]
    cf = ax.contourf(X, Y, U, levels=30, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='u')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('u-velocity')
    ax.set_aspect('equal')

    # V-velocity contour
    ax = axes[0, 1]
    cf = ax.contourf(X, Y, V, levels=30, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='v')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('v-velocity')
    ax.set_aspect('equal')

    # Pressure contour
    ax = axes[1, 0]
    cf = ax.contourf(X, Y, P, levels=30, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='p')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pressure')
    ax.set_aspect('equal')

    # Streamlines / velocity magnitude
    ax = axes[1, 1]
    cf = ax.contourf(X, Y, speed, levels=30, cmap='plasma')
    plt.colorbar(cf, ax=ax, label='|V|')
    # Add streamlines
    ax.streamplot(X, Y, U, V, color='white', density=1.5, linewidth=0.5, arrowsize=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Velocity magnitude with streamlines')
    ax.set_aspect('equal')

    plt.suptitle(f'PIELM Lid-Driven Cavity (Re=1000)\n'
                 f'Training: {results["train_time"]:.2f}s, {results["n_iterations"]} iterations',
                 fontsize=14)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, f'pielm_cavity_{mode}.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure: {fig_path}")
    plt.close()

    # Plot centerline profiles
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Vertical centerline (x=0.5): u vs y
    ax = axes[0]
    y_center = y_lin
    xy_vcenter = np.column_stack([np.full_like(y_center, 0.5), y_center])
    u_vcenter, _, _ = model.predict(xy_vcenter)
    ax.plot(u_vcenter, y_center, 'b-', linewidth=2, label='PIELM')
    ax.set_xlabel('u')
    ax.set_ylabel('y')
    ax.set_title('Vertical centerline (x=0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Horizontal centerline (y=0.5): v vs x
    ax = axes[1]
    x_center = x_lin
    xy_hcenter = np.column_stack([x_center, np.full_like(x_center, 0.5)])
    _, v_hcenter, _ = model.predict(xy_hcenter)
    ax.plot(x_center, v_hcenter, 'r-', linewidth=2, label='PIELM')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title('Horizontal centerline (y=0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Centerline velocity profiles', fontsize=14)
    plt.tight_layout()

    centerline_path = os.path.join(output_dir, f'pielm_centerlines_{mode}.png')
    plt.savefig(centerline_path, dpi=150, bbox_inches='tight')
    print(f"Saved centerline profiles: {centerline_path}")
    plt.close()

    # Compute PDE residuals on a test grid
    print("\nComputing PDE residuals...")
    n_test = 400
    xy_test = np.random.rand(n_test, 2)
    # Keep away from boundaries
    xy_test = 0.1 + 0.8 * xy_test

    residuals = model.compute_pde_residuals(xy_test)

    cont_rms = np.sqrt(np.mean(residuals['continuity']**2))
    momx_rms = np.sqrt(np.mean(residuals['momentum_x']**2))
    momy_rms = np.sqrt(np.mean(residuals['momentum_y']**2))

    print(f"Continuity RMS residual: {cont_rms:.6f}")
    print(f"Momentum-x RMS residual: {momx_rms:.6f}")
    print(f"Momentum-y RMS residual: {momy_rms:.6f}")

    # Check boundary conditions
    print("\nBoundary condition satisfaction:")
    # Lid
    xy_lid_test = np.column_stack([np.linspace(0.1, 0.9, 20), np.ones(20)])
    u_lid, v_lid, _ = model.predict(xy_lid_test)
    print(f"  Lid: max|u-1| = {np.max(np.abs(u_lid - 1.0)):.6f}, max|v| = {np.max(np.abs(v_lid)):.6f}")

    # Bottom wall
    xy_bot_test = np.column_stack([np.linspace(0.1, 0.9, 20), np.zeros(20)])
    u_bot, v_bot, _ = model.predict(xy_bot_test)
    print(f"  Bottom: max|u| = {np.max(np.abs(u_bot)):.6f}, max|v| = {np.max(np.abs(v_bot)):.6f}")

    # Left wall
    xy_left_test = np.column_stack([np.zeros(20), np.linspace(0.1, 0.9, 20)])
    u_left, v_left, _ = model.predict(xy_left_test)
    print(f"  Left: max|u| = {np.max(np.abs(u_left)):.6f}, max|v| = {np.max(np.abs(v_left)):.6f}")

    # Right wall
    xy_right_test = np.column_stack([np.ones(20), np.linspace(0.1, 0.9, 20)])
    u_right, v_right, _ = model.predict(xy_right_test)
    print(f"  Right: max|u| = {np.max(np.abs(u_right)):.6f}, max|v| = {np.max(np.abs(v_right)):.6f}")

    # Plot convergence history
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(model.residual_history)+1), model.residual_history, 'b-o', markersize=4)
    ax.set_xlabel('Picard iteration')
    ax.set_ylabel('Relative change')
    ax.set_title('Picard iteration convergence')
    ax.grid(True, alpha=0.3)

    conv_path = os.path.join(output_dir, f'pielm_convergence_{mode}.png')
    plt.savefig(conv_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved convergence plot: {conv_path}")
    plt.close()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Method: PIELM with Picard iteration + Smagorinsky turbulence")
    print(f"Training time: {results['train_time']:.3f} seconds")
    print(f"Speedup vs PINN (~50 min): {3000 / results['train_time']:.0f}x (estimated)")
    print(f"Picard iterations: {results['n_iterations']}")
    print(f"Converged: {results['converged']}")
    print(f"PDE residual (continuity): {cont_rms:.6f}")
    print(f"PDE residual (momentum): {max(momx_rms, momy_rms):.6f}")
    print("=" * 70)

    return model, results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test PIELM Navier-Stokes')
    parser.add_argument('--mode', choices=['quick', 'medium', 'full'], default='quick',
                       help='Problem size mode')
    parser.add_argument('--solver', choices=['direct', 'iterative'], default='direct',
                       help='Linear solver (direct=lstsq, iterative=lsqr)')
    args = parser.parse_args()

    run_experiment(mode=args.mode, solver=args.solver)
