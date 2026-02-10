#!/usr/bin/env python3
"""
Pareto Frontier Analysis: Accuracy vs Training Time

Maps the tradeoff curve for both PINN and PIELM to enable fair comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.experiment_dt_elm_pinn.models.pielm_navier_stokes import PIELM_NavierStokes

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1

N_interior = 6000
N_wall = 800
N_lid = 800

# PINN epoch checkpoints (faster version)
PINN_EPOCHS = [500, 1000, 2000, 3000, 4000, 6000]

# PIELM configurations to test (key variants only)
PIELM_CONFIGS = [
    {'n_hidden': 300, 'max_iter': 50},
    {'n_hidden': 400, 'max_iter': 75},
    {'n_hidden': 500, 'max_iter': 100},
]

EVAL_NX, EVAL_NY = 41, 41
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PARETO FRONTIER: Accuracy vs Training Time")
print("=" * 70)
print(f"Device: {device}")


# =============================================================================
# Helper functions
# =============================================================================
def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


class PINN_Cavity(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, 64), nn.Tanh()]
        for _ in range(5):
            layers.extend([nn.Linear(64, 64), nn.Tanh()])
        layers.append(nn.Linear(64, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def sample_points():
    np.random.seed(SEED)
    xy_int = np.random.rand(N_interior, 2)
    x_lid = np.random.rand(N_lid, 1)
    y_lid = np.ones((N_lid, 1))
    xy_lid = np.hstack((x_lid, y_lid))
    N_each = N_wall // 3
    xb, yb = np.random.rand(N_each, 1), np.zeros((N_each, 1))
    xl, yl = np.zeros((N_each, 1)), np.random.rand(N_each, 1)
    xr, yr = np.ones((N_each, 1)), np.random.rand(N_each, 1)
    xy_wall = np.vstack([np.hstack((xb, yb)), np.hstack((xl, yl)), np.hstack((xr, yr))])
    xy_p = np.array([[0.5, 0.5]])
    return xy_int, xy_lid, xy_wall, xy_p


def eddy_viscosity_torch(xy, u, v):
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
    xy.requires_grad_(True)
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    nu_eff, du_dx, du_dy, dv_dx, dv_dy = eddy_viscosity_torch(xy, u, v)
    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    grad_p = gradients(p, xy)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
    grad_qx_u, grad_qy_u = gradients(qx_u, xy), gradients(qy_u, xy)
    grad_qx_v, grad_qy_v = gradients(qx_v, xy), gradients(qy_v, xy)
    visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
    visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    return continuity, mom_u, mom_v


def create_eval_grid():
    x = np.linspace(0, 1, EVAL_NX)
    y = np.linspace(0, 1, EVAL_NY)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    return xy


def compute_pinn_metrics(model, xy_eval):
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)
    model.eval()
    cont, mom_u, mom_v = pde_residuals_torch(model, xy_t)
    cont_np = cont.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))
    cont_rms = float(np.sqrt(np.mean(cont_np**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u_np**2 + mom_v_np**2)))

    # Get predictions for BC check
    with torch.no_grad():
        pred = model(torch.tensor(xy_eval, dtype=torch.float32, device=device))
        u = pred[:, 0].cpu().numpy()
        v = pred[:, 1].cpu().numpy()

    # Lid BC error
    lid_mask = xy_eval[:, 1] > 0.99
    lid_u_err = float(np.sqrt(np.mean((u[lid_mask] - 1.0)**2))) if lid_mask.sum() > 0 else 0
    lid_v_err = float(np.sqrt(np.mean(v[lid_mask]**2))) if lid_mask.sum() > 0 else 0

    return {
        'pde_rms': pde_rms,
        'continuity_rms': cont_rms,
        'momentum_rms': mom_rms,
        'lid_u_error': lid_u_err,
        'lid_v_error': lid_v_err,
    }


def compute_pielm_metrics(model, xy_eval):
    residuals = model.compute_pde_residuals(xy_eval)
    pde_rms = float(np.sqrt(np.mean(
        residuals['continuity']**2 +
        residuals['momentum_x']**2 +
        residuals['momentum_y']**2
    )))
    cont_rms = float(np.sqrt(np.mean(residuals['continuity']**2)))
    mom_rms = float(np.sqrt(np.mean(residuals['momentum_x']**2 + residuals['momentum_y']**2)))

    u, v, p = model.predict(xy_eval)
    lid_mask = xy_eval[:, 1] > 0.99
    lid_u_err = float(np.sqrt(np.mean((u[lid_mask] - 1.0)**2))) if lid_mask.sum() > 0 else 0
    lid_v_err = float(np.sqrt(np.mean(v[lid_mask]**2))) if lid_mask.sum() > 0 else 0

    return {
        'pde_rms': pde_rms,
        'continuity_rms': cont_rms,
        'momentum_rms': mom_rms,
        'lid_u_error': lid_u_err,
        'lid_v_error': lid_v_err,
    }


# =============================================================================
# Run PINN experiments with checkpointing
# =============================================================================
def train_pinn_with_checkpoints(epochs_list):
    """Train PINN once, evaluate at multiple checkpoints."""
    torch.manual_seed(SEED)

    xy_int_np, xy_lid_np, xy_wall_np, xy_p_np = sample_points()
    xy_int = torch.tensor(xy_int_np, dtype=torch.float32, device=device)
    xy_lid = torch.tensor(xy_lid_np, dtype=torch.float32, device=device)
    xy_wall = torch.tensor(xy_wall_np, dtype=torch.float32, device=device)
    xy_p = torch.tensor(xy_p_np, dtype=torch.float32, device=device)

    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()

    xy_eval = create_eval_grid()
    results = []
    max_epochs = max(epochs_list)
    checkpoints = sorted(epochs_list)
    next_checkpoint_idx = 0

    start_time = time.perf_counter()

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        xy_int.requires_grad_(True)
        cont, mom_u, mom_v = pde_residuals_torch(model, xy_int)
        loss_pde = (mse_loss(cont, torch.zeros_like(cont)) +
                   mse_loss(mom_u, torch.zeros_like(mom_u)) +
                   mse_loss(mom_v, torch.zeros_like(mom_v)))

        pred_lid = model(xy_lid)
        loss_lid = (mse_loss(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) +
                   mse_loss(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2])))

        pred_wall = model(xy_wall)
        loss_wall = (mse_loss(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) +
                    mse_loss(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2])))

        pred_p = model(xy_p)
        loss_p = mse_loss(pred_p[:, 2:3], torch.zeros_like(pred_p[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        # Check if we hit a checkpoint
        if next_checkpoint_idx < len(checkpoints) and (epoch + 1) == checkpoints[next_checkpoint_idx]:
            elapsed = time.perf_counter() - start_time
            metrics = compute_pinn_metrics(model, xy_eval)
            results.append({
                'epochs': epoch + 1,
                'time': elapsed,
                'loss': float(loss.item()),
                **metrics
            })
            print(f"  PINN @ {epoch+1:5d} epochs: time={elapsed:6.1f}s, PDE_RMS={metrics['pde_rms']:.5f}")
            next_checkpoint_idx += 1

    return results


# =============================================================================
# Run PIELM experiments
# =============================================================================
def run_pielm_experiments(configs):
    """Run PIELM with different configurations."""
    xy_eval = create_eval_grid()
    results = []

    for cfg in configs:
        np.random.seed(SEED)

        model = PIELM_NavierStokes(
            Re=Re, U_lid=U_lid, Cs=Cs,
            n_hidden=cfg['n_hidden'],
            activation='tanh',
            max_picard_iter=cfg['max_iter'],
            tol=1e-6,
            N_interior=N_interior,
            N_wall=N_wall,
            N_lid=N_lid,
            bc_weight=10.0,
            verbose=False,
            seed=SEED,
        )
        model.use_full_viscous = True
        model.relaxation = 0.7

        train_result = model.train()
        metrics = compute_pielm_metrics(model, xy_eval)

        results.append({
            'n_hidden': cfg['n_hidden'],
            'max_iter': cfg['max_iter'],
            'actual_iter': train_result['n_iterations'],
            'converged': train_result['converged'],
            'time': train_result['train_time'],
            **metrics
        })

        print(f"  PIELM h={cfg['n_hidden']:3d}, iter={train_result['n_iterations']:3d}: "
              f"time={train_result['train_time']:6.1f}s, PDE_RMS={metrics['pde_rms']:.5f}")

    return results


# =============================================================================
# Main
# =============================================================================
print("\n" + "-" * 70)
print("Running PINN experiments...")
print("-" * 70)
pinn_results = train_pinn_with_checkpoints(PINN_EPOCHS)

print("\n" + "-" * 70)
print("Running PIELM experiments...")
print("-" * 70)
pielm_results = run_pielm_experiments(PIELM_CONFIGS)

# =============================================================================
# Analysis
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\nPINN Results:")
print(f"{'Epochs':<8} {'Time (s)':<10} {'PDE RMS':<12} {'Cont RMS':<12} {'Mom RMS':<12}")
print("-" * 54)
for r in pinn_results:
    print(f"{r['epochs']:<8} {r['time']:<10.1f} {r['pde_rms']:<12.5f} {r['continuity_rms']:<12.5f} {r['momentum_rms']:<12.5f}")

print("\nPIELM Results:")
print(f"{'Hidden':<8} {'Iter':<6} {'Time (s)':<10} {'PDE RMS':<12} {'Cont RMS':<12} {'Mom RMS':<12}")
print("-" * 60)
for r in pielm_results:
    print(f"{r['n_hidden']:<8} {r['actual_iter']:<6} {r['time']:<10.1f} {r['pde_rms']:<12.5f} {r['continuity_rms']:<12.5f} {r['momentum_rms']:<12.5f}")

# Find Pareto-optimal points
print("\n" + "=" * 70)
print("PARETO ANALYSIS: Time vs PDE RMS")
print("=" * 70)

all_points = []
for r in pinn_results:
    all_points.append(('PINN', r['epochs'], r['time'], r['pde_rms']))
for r in pielm_results:
    all_points.append(('PIELM', r['n_hidden'], r['time'], r['pde_rms']))

# Sort by time
all_points.sort(key=lambda x: x[2])

print(f"\n{'Method':<20} {'Config':<10} {'Time (s)':<10} {'PDE RMS':<12} {'Pareto?':<8}")
print("-" * 60)

best_pde = float('inf')
for method, cfg, t, pde in all_points:
    is_pareto = pde < best_pde
    if is_pareto:
        best_pde = pde
    marker = "YES" if is_pareto else ""
    if method == 'PINN':
        print(f"{'PINN':<20} {str(cfg)+'ep':<10} {t:<10.1f} {pde:<12.5f} {marker:<8}")
    else:
        print(f"{'PIELM':<20} {'h='+str(cfg):<10} {t:<10.1f} {pde:<12.5f} {marker:<8}")

# Compute speedup at various quality levels
print("\n" + "=" * 70)
print("SPEEDUP ANALYSIS AT VARIOUS QUALITY LEVELS")
print("=" * 70)

# Find best PIELM
best_pielm = min(pielm_results, key=lambda x: x['pde_rms'])
pielm_time = best_pielm['time']
pielm_pde = best_pielm['pde_rms']

print(f"\nBest PIELM: h={best_pielm['n_hidden']}, time={pielm_time:.1f}s, PDE_RMS={pielm_pde:.5f}")

# Find PINN time to match PIELM quality
pinn_times = np.array([r['time'] for r in pinn_results])
pinn_pdes = np.array([r['pde_rms'] for r in pinn_results])

# Interpolate to find when PINN reaches PIELM quality
if pinn_pdes.min() < pielm_pde:
    # PINN can reach PIELM quality - find when
    idx = np.where(pinn_pdes < pielm_pde)[0][0]
    pinn_time_to_match = pinn_times[idx]
    print(f"PINN reaches PIELM quality ({pielm_pde:.4f}) at ~{pinn_time_to_match:.1f}s")
    print(f"Speedup (PIELM vs PINN-to-match): {pinn_time_to_match/pielm_time:.2f}x")
else:
    print(f"PINN never reaches PIELM quality in tested range")

# At PIELM's time, what's PINN quality?
idx = np.argmin(np.abs(pinn_times - pielm_time))
pinn_pde_at_pielm_time = pinn_pdes[idx]
print(f"\nAt PIELM time ({pielm_time:.1f}s):")
print(f"  PIELM PDE RMS: {pielm_pde:.5f}")
print(f"  PINN PDE RMS:  {pinn_pde_at_pielm_time:.5f}")
print(f"  PINN is {pielm_pde/pinn_pde_at_pielm_time:.2f}x more accurate")

# Extrapolate PINN to 30K
avg_time_per_epoch = np.mean([r['time']/r['epochs'] for r in pinn_results])
pinn_30k_time = avg_time_per_epoch * 30000
# Fit power law for quality
log_epochs = np.log([r['epochs'] for r in pinn_results])
log_pdes = np.log(pinn_pdes)
b, log_a = np.polyfit(log_epochs, log_pdes, 1)
pinn_30k_pde = np.exp(log_a) * (30000 ** b)

print(f"\nExtrapolated PINN @ 30K epochs:")
print(f"  Time: {pinn_30k_time:.1f}s ({pinn_30k_time/60:.1f} min)")
print(f"  PDE RMS: {pinn_30k_pde:.5f}")
print(f"  Speedup (PIELM vs PINN-30K): {pinn_30k_time/pielm_time:.1f}x")
print(f"  Quality ratio (PIELM/PINN-30K): {pielm_pde/pinn_30k_pde:.2f}x worse")

# Save results
output = {
    'pinn': pinn_results,
    'pielm': pielm_results,
    'analysis': {
        'best_pielm_time': pielm_time,
        'best_pielm_pde': pielm_pde,
        'pinn_30k_time_extrap': pinn_30k_time,
        'pinn_30k_pde_extrap': float(pinn_30k_pde),
        'speedup_vs_30k': pinn_30k_time / pielm_time,
        'quality_ratio_vs_30k': pielm_pde / pinn_30k_pde,
    }
}

os.makedirs('results/end_to_end_comparison', exist_ok=True)
with open('results/end_to_end_comparison/pareto_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to results/end_to_end_comparison/pareto_results.json")
