#!/usr/bin/env python3
"""
End-to-End Comparison: Partner's PINN vs Our PIELM

This script provides an honest, apples-to-apples comparison of:
1. Training throughput (time)
2. Solution quality (PDE residuals, BC satisfaction)
3. Extrapolation to full 30K epoch performance

Run multiple PINN training durations and compare against PIELM.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.experiment_dt_elm_pinn.models.pielm_navier_stokes import PIELM_NavierStokes

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Physics parameters (matching partner's code exactly)
Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
rho = 1.0

# Problem size (matching partner's code)
N_interior = 6000
N_wall = 800
N_lid = 800

# Epoch counts for PINN experiments
EPOCH_COUNTS = [500, 1000, 2000, 4000, 8000]

# Evaluation grid
EVAL_NX, EVAL_NY = 41, 41

# Output
OUTPUT_DIR = 'results/end_to_end_comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("END-TO-END COMPARISON: Partner's PINN vs Our PIELM")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Device: {device}")
print(f"Problem: Lid-driven cavity, Re={Re}, Smagorinsky Cs={Cs}")
print(f"Collocation: {N_interior} interior, {N_wall} wall, {N_lid} lid points")
print(f"PINN epoch counts to test: {EPOCH_COUNTS}")
print("=" * 80)


# =============================================================================
# Helper functions (from partner's code)
# =============================================================================
def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


class PINN_Cavity(nn.Module):
    """Partner's exact architecture: 6 hidden layers, 64 units, tanh."""
    def __init__(self, in_dim=2, out_dim=3, hidden_layers=6, hidden_units=64):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_units))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def sample_points():
    """Sample collocation points (same as partner's code)."""
    np.random.seed(SEED)

    # Interior
    xy_int = np.random.rand(N_interior, 2)

    # Lid (y=1)
    x_lid = np.random.rand(N_lid, 1)
    y_lid = np.ones((N_lid, 1))
    xy_lid = np.hstack((x_lid, y_lid))

    # Walls (bottom, left, right)
    N_each = N_wall // 3
    xb, yb = np.random.rand(N_each, 1), np.zeros((N_each, 1))
    xl, yl = np.zeros((N_each, 1)), np.random.rand(N_each, 1)
    xr, yr = np.ones((N_each, 1)), np.random.rand(N_each, 1)
    xy_wall = np.vstack([np.hstack((xb, yb)), np.hstack((xl, yl)), np.hstack((xr, yr))])

    # Pressure anchor
    xy_p = np.array([[0.5, 0.5]])

    return xy_int, xy_lid, xy_wall, xy_p


def eddy_viscosity_torch(xy, u, v):
    """Smagorinsky eddy viscosity (partner's implementation)."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    d = torch.min(torch.min(x, 1.0 - x), torch.min(y, 1.0 - y))

    grad_u, grad_v = gradients(u, xy), gradients(v, xy)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2)
    S_mag = torch.sqrt(S_sq + 1e-12)

    nu_t = (Cs * d)**2 * S_mag
    return nu_laminar + nu_t, du_dx, du_dy, dv_dx, dv_dy


def pde_residuals_torch(model, xy):
    """Compute PDE residuals (partner's implementation)."""
    xy.requires_grad_(True)
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    nu_eff, du_dx, du_dy, dv_dx, dv_dy = eddy_viscosity_torch(xy, u, v)

    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy

    grad_p = gradients(p, xy)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # Full divergence form: ∇·(ν_eff ∇u)
    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy

    grad_qx_u, grad_qy_u = gradients(qx_u, xy), gradients(qy_u, xy)
    grad_qx_v, grad_qy_v = gradients(qx_v, xy), gradients(qy_v, xy)

    visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
    visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]

    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    return continuity, mom_u, mom_v


# =============================================================================
# Evaluation functions
# =============================================================================
def create_eval_grid():
    """Create evaluation grid."""
    x = np.linspace(0, 1, EVAL_NX)
    y = np.linspace(0, 1, EVAL_NY)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    return xy, X, Y


def evaluate_pinn(model, xy_eval):
    """Evaluate PINN on grid and compute metrics."""
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    pred = model(xy_t)
    u = pred[:, 0].detach().cpu().numpy()
    v = pred[:, 1].detach().cpu().numpy()
    p = pred[:, 2].detach().cpu().numpy()

    # Compute PDE residuals
    cont, mom_u, mom_v = pde_residuals_torch(model, xy_t)

    cont_np = cont.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    metrics = {
        'continuity_rms': float(np.sqrt(np.mean(cont_np**2))),
        'momentum_u_rms': float(np.sqrt(np.mean(mom_u_np**2))),
        'momentum_v_rms': float(np.sqrt(np.mean(mom_v_np**2))),
        'total_pde_rms': float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2))),
    }

    # BC satisfaction
    # Lid (y=1): u=1, v=0
    lid_mask = xy_eval[:, 1] > 0.99
    if lid_mask.sum() > 0:
        metrics['lid_u_error'] = float(np.sqrt(np.mean((u[lid_mask] - 1.0)**2)))
        metrics['lid_v_error'] = float(np.sqrt(np.mean(v[lid_mask]**2)))

    # Bottom wall (y=0): u=0, v=0
    bot_mask = xy_eval[:, 1] < 0.01
    if bot_mask.sum() > 0:
        metrics['bottom_u_error'] = float(np.sqrt(np.mean(u[bot_mask]**2)))
        metrics['bottom_v_error'] = float(np.sqrt(np.mean(v[bot_mask]**2)))

    return u, v, p, metrics


def evaluate_pielm(model, xy_eval):
    """Evaluate PIELM on grid and compute metrics."""
    u, v, p = model.predict(xy_eval)

    # PDE residuals via model's method
    residuals = model.compute_pde_residuals(xy_eval)

    metrics = {
        'continuity_rms': float(np.sqrt(np.mean(residuals['continuity']**2))),
        'momentum_u_rms': float(np.sqrt(np.mean(residuals['momentum_x']**2))),
        'momentum_v_rms': float(np.sqrt(np.mean(residuals['momentum_y']**2))),
        'total_pde_rms': float(np.sqrt(np.mean(
            residuals['continuity']**2 +
            residuals['momentum_x']**2 +
            residuals['momentum_y']**2
        ))),
    }

    # BC satisfaction
    lid_mask = xy_eval[:, 1] > 0.99
    if lid_mask.sum() > 0:
        metrics['lid_u_error'] = float(np.sqrt(np.mean((u[lid_mask] - 1.0)**2)))
        metrics['lid_v_error'] = float(np.sqrt(np.mean(v[lid_mask]**2)))

    bot_mask = xy_eval[:, 1] < 0.01
    if bot_mask.sum() > 0:
        metrics['bottom_u_error'] = float(np.sqrt(np.mean(u[bot_mask]**2)))
        metrics['bottom_v_error'] = float(np.sqrt(np.mean(v[bot_mask]**2)))

    return u, v, p, metrics


# =============================================================================
# Training functions
# =============================================================================
def train_pinn(n_epochs, verbose=True):
    """Train partner's PINN for n_epochs."""
    torch.manual_seed(SEED)

    xy_int_np, xy_lid_np, xy_wall_np, xy_p_np = sample_points()

    xy_int = torch.tensor(xy_int_np, dtype=torch.float32, device=device)
    xy_lid = torch.tensor(xy_lid_np, dtype=torch.float32, device=device)
    xy_wall = torch.tensor(xy_wall_np, dtype=torch.float32, device=device)
    xy_p = torch.tensor(xy_p_np, dtype=torch.float32, device=device)

    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    loss_history = []

    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # PDE residuals
        xy_int.requires_grad_(True)
        cont, mom_u, mom_v = pde_residuals_torch(model, xy_int)

        loss_cont = mse_loss(cont, torch.zeros_like(cont))
        loss_momu = mse_loss(mom_u, torch.zeros_like(mom_u))
        loss_momv = mse_loss(mom_v, torch.zeros_like(mom_v))
        loss_pde = loss_cont + loss_momu + loss_momv

        # Lid BC
        pred_lid = model(xy_lid)
        loss_lid = (mse_loss(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) +
                   mse_loss(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2])))

        # Wall BC
        pred_wall = model(xy_wall)
        loss_wall = (mse_loss(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) +
                    mse_loss(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2])))

        # Pressure anchor
        pred_p = model(xy_p)
        loss_p = mse_loss(pred_p[:, 2:3], torch.zeros_like(pred_p[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p

        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            print(f"    Epoch {epoch:5d}/{n_epochs}: loss={loss.item():.4e}, "
                  f"PDE={loss_pde.item():.4e}, BC={loss_lid.item() + loss_wall.item():.4e}")

    train_time = time.perf_counter() - start_time

    return model, train_time, loss_history


def train_pielm(verbose=True):
    """Train our PIELM."""
    np.random.seed(SEED)

    model = PIELM_NavierStokes(
        Re=Re,
        U_lid=U_lid,
        Cs=Cs,
        n_hidden=500,  # Comparable capacity to 6-layer 64-unit MLP
        activation='tanh',
        max_picard_iter=100,
        tol=1e-6,
        N_interior=N_interior,
        N_wall=N_wall,
        N_lid=N_lid,
        bc_weight=10.0,
        verbose=verbose,
        seed=SEED,
    )
    model.use_full_viscous = True  # Match partner's physics exactly
    model.relaxation = 0.7

    results = model.train()

    return model, results


# =============================================================================
# Main comparison
# =============================================================================
def run_comparison():
    """Run full comparison study."""

    xy_eval, X, Y = create_eval_grid()
    results = {
        'config': {
            'Re': Re,
            'U_lid': U_lid,
            'Cs': Cs,
            'N_interior': N_interior,
            'N_wall': N_wall,
            'N_lid': N_lid,
            'device': str(device),
            'seed': SEED,
            'eval_grid': (EVAL_NX, EVAL_NY),
        },
        'pinn_runs': [],
        'pielm_run': None,
    }

    # ==========================================================================
    # Run PIELM first (single run, no epochs)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PIELM Training")
    print("=" * 80)

    pielm_model, pielm_results = train_pielm(verbose=True)
    u_pielm, v_pielm, p_pielm, metrics_pielm = evaluate_pielm(pielm_model, xy_eval)

    results['pielm_run'] = {
        'train_time': pielm_results['train_time'],
        'n_iterations': pielm_results['n_iterations'],
        'converged': pielm_results['converged'],
        'final_residual': pielm_results['final_residual'],
        'metrics': metrics_pielm,
    }

    print(f"\nPIELM Results:")
    print(f"  Train time: {pielm_results['train_time']:.2f}s")
    print(f"  Picard iterations: {pielm_results['n_iterations']}")
    print(f"  Converged: {pielm_results['converged']}")
    print(f"  Total PDE RMS: {metrics_pielm['total_pde_rms']:.6f}")
    print(f"  Lid BC error (u): {metrics_pielm.get('lid_u_error', 'N/A')}")

    # ==========================================================================
    # Run PINN at multiple epoch counts
    # ==========================================================================
    for n_epochs in EPOCH_COUNTS:
        print("\n" + "=" * 80)
        print(f"PINN Training: {n_epochs} epochs")
        print("=" * 80)

        pinn_model, train_time, loss_history = train_pinn(n_epochs, verbose=True)
        u_pinn, v_pinn, p_pinn, metrics_pinn = evaluate_pinn(pinn_model, xy_eval)

        # Compute difference from PIELM (cross-comparison)
        u_diff = np.sqrt(np.mean((u_pinn - u_pielm)**2))
        v_diff = np.sqrt(np.mean((v_pinn - v_pielm)**2))

        run_data = {
            'n_epochs': n_epochs,
            'train_time': train_time,
            'time_per_epoch': train_time / n_epochs,
            'final_loss': loss_history[-1],
            'metrics': metrics_pinn,
            'diff_from_pielm': {
                'u_rms': float(u_diff),
                'v_rms': float(v_diff),
            }
        }
        results['pinn_runs'].append(run_data)

        print(f"\nPINN @ {n_epochs} epochs:")
        print(f"  Train time: {train_time:.2f}s ({train_time/n_epochs*1000:.2f}ms/epoch)")
        print(f"  Final loss: {loss_history[-1]:.6f}")
        print(f"  Total PDE RMS: {metrics_pinn['total_pde_rms']:.6f}")
        print(f"  Lid BC error (u): {metrics_pinn.get('lid_u_error', 'N/A')}")
        print(f"  Diff from PIELM - u: {u_diff:.6f}, v: {v_diff:.6f}")

    # ==========================================================================
    # Extrapolation to 30K epochs
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EXTRAPOLATION TO 30,000 EPOCHS")
    print("=" * 80)

    # Time extrapolation (linear with epoch count)
    avg_time_per_epoch = np.mean([r['time_per_epoch'] for r in results['pinn_runs']])
    extrapolated_time_30k = avg_time_per_epoch * 30000

    # Loss extrapolation (fit power law: loss = a * epoch^b)
    epochs = np.array([r['n_epochs'] for r in results['pinn_runs']])
    losses = np.array([r['final_loss'] for r in results['pinn_runs']])

    # Fit log-log regression: log(loss) = log(a) + b*log(epoch)
    log_epochs = np.log(epochs)
    log_losses = np.log(losses)
    b, log_a = np.polyfit(log_epochs, log_losses, 1)
    a = np.exp(log_a)

    extrapolated_loss_30k = a * (30000 ** b)

    # PDE residual extrapolation (similar power law)
    pde_rms = np.array([r['metrics']['total_pde_rms'] for r in results['pinn_runs']])
    log_pde = np.log(pde_rms + 1e-10)
    b_pde, log_a_pde = np.polyfit(log_epochs, log_pde, 1)
    a_pde = np.exp(log_a_pde)
    extrapolated_pde_30k = a_pde * (30000 ** b_pde)

    results['extrapolation'] = {
        'target_epochs': 30000,
        'avg_time_per_epoch_ms': avg_time_per_epoch * 1000,
        'extrapolated_time_s': extrapolated_time_30k,
        'extrapolated_time_min': extrapolated_time_30k / 60,
        'loss_fit': {'a': float(a), 'b': float(b)},
        'extrapolated_loss': float(extrapolated_loss_30k),
        'pde_fit': {'a': float(a_pde), 'b': float(b_pde)},
        'extrapolated_pde_rms': float(extrapolated_pde_30k),
    }

    print(f"Avg time per epoch: {avg_time_per_epoch*1000:.2f}ms")
    print(f"Extrapolated time for 30K epochs: {extrapolated_time_30k:.1f}s ({extrapolated_time_30k/60:.1f} min)")
    print(f"Loss power law fit: loss = {a:.4e} * epoch^{b:.4f}")
    print(f"Extrapolated loss at 30K: {extrapolated_loss_30k:.6f}")
    print(f"PDE RMS power law fit: pde = {a_pde:.4e} * epoch^{b_pde:.4f}")
    print(f"Extrapolated PDE RMS at 30K: {extrapolated_pde_30k:.6f}")

    # ==========================================================================
    # Summary comparison
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)

    pielm_time = results['pielm_run']['train_time']
    pielm_pde = results['pielm_run']['metrics']['total_pde_rms']

    print(f"\n{'Method':<30} {'Time (s)':<12} {'PDE RMS':<15} {'Speedup':<12}")
    print("-" * 70)
    print(f"{'PIELM':<30} {pielm_time:<12.2f} {pielm_pde:<15.6f} {'(baseline)':<12}")

    for run in results['pinn_runs']:
        speedup = run['train_time'] / pielm_time
        print(f"{'PINN @ ' + str(run['n_epochs']) + ' epochs':<30} "
              f"{run['train_time']:<12.2f} "
              f"{run['metrics']['total_pde_rms']:<15.6f} "
              f"{speedup:.2f}x slower")

    speedup_30k = extrapolated_time_30k / pielm_time
    print(f"{'PINN @ 30K epochs (extrap.)':<30} "
          f"{extrapolated_time_30k:<12.1f} "
          f"{extrapolated_pde_30k:<15.6f} "
          f"{speedup_30k:.1f}x slower")

    # Critical comparison: at what PINN epoch count does quality match PIELM?
    print("\n" + "-" * 70)
    print("QUALITY CROSSOVER ANALYSIS")
    print("-" * 70)

    # Find when PINN's PDE RMS matches or beats PIELM
    pielm_quality = pielm_pde
    pinn_better_epoch = None

    for run in results['pinn_runs']:
        if run['metrics']['total_pde_rms'] <= pielm_quality:
            pinn_better_epoch = run['n_epochs']
            pinn_better_time = run['train_time']
            break

    if pinn_better_epoch:
        print(f"PINN matches PIELM quality at ~{pinn_better_epoch} epochs ({pinn_better_time:.1f}s)")
        print(f"At crossover point: PIELM is {pinn_better_time/pielm_time:.1f}x faster")
    else:
        # Extrapolate to find crossover
        # Solve: a * epoch^b = pielm_quality
        crossover_epoch = (pielm_quality / a_pde) ** (1 / b_pde)
        crossover_time = crossover_epoch * avg_time_per_epoch
        print(f"PINN never matched PIELM quality in tested range")
        print(f"Extrapolated crossover: ~{crossover_epoch:.0f} epochs ({crossover_time:.1f}s)")
        if crossover_time > pielm_time:
            print(f"At crossover: PIELM would still be {crossover_time/pielm_time:.1f}x faster")
        else:
            print(f"At crossover: PINN would be {pielm_time/crossover_time:.1f}x faster")

    # ==========================================================================
    # Honest assessment
    # ==========================================================================
    print("\n" + "=" * 80)
    print("HONEST ASSESSMENT")
    print("=" * 80)

    # Check if PIELM quality is actually good
    print("\nPIELM Quality Check:")
    print(f"  PDE residual RMS: {pielm_pde:.6f}")
    print(f"  Continuity RMS: {results['pielm_run']['metrics']['continuity_rms']:.6f}")
    print(f"  Lid BC u-error: {results['pielm_run']['metrics'].get('lid_u_error', 'N/A')}")

    if pielm_pde > 0.1:
        print("  WARNING: PIELM PDE residual is HIGH (>0.1). Solution may be inaccurate.")
    elif pielm_pde > 0.01:
        print("  CAUTION: PIELM PDE residual is moderate (0.01-0.1).")
    else:
        print("  GOOD: PIELM PDE residual is low (<0.01).")

    print("\nKey Findings:")
    print(f"  1. PIELM training time: {pielm_time:.1f}s")
    print(f"  2. PINN @ 30K epochs (extrapolated): {extrapolated_time_30k/60:.1f} min")
    print(f"  3. Speedup: {speedup_30k:.0f}x")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    results = run_comparison()
