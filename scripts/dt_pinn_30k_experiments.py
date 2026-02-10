#!/usr/bin/env python3
"""
DT-PINN 30K Epoch Experiments

Full-scale comparison at production training length.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

N_grid = 50  # 50x50 = 2500 points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("DT-PINN 30K EPOCH EXPERIMENTS")
print("=" * 70)
print(f"Device: {device}")

# =============================================================================
# Build Infrastructure (same as before)
# =============================================================================
def chebyshev_points(N):
    i = np.arange(N)
    return np.cos(np.pi * i / (N - 1))

def chebyshev_diff_matrix(N):
    x = chebyshev_points(N)
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j)) / (x[i] - x[j])
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return D

def build_2d_operators(N):
    D1d = chebyshev_diff_matrix(N)
    D1d_scaled = D1d * 2.0
    I = np.eye(N)
    Dx = np.kron(I, D1d_scaled)
    Dy = np.kron(D1d_scaled, I)
    return Dx, Dy

def build_grid(N):
    x_ref = chebyshev_points(N)
    x = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x, x, indexing='xy')
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    return xy, x

print("\nBuilding discrete operators...")
Dx_np, Dy_np = build_2d_operators(N_grid)
xy_grid, x_1d = build_grid(N_grid)

eps = 1e-10
x_coords = xy_grid[:, 0]
y_coords = xy_grid[:, 1]
is_boundary = (x_coords < eps) | (x_coords > 1-eps) | (y_coords < eps) | (y_coords > 1-eps)
is_lid = (y_coords > 1-eps)
is_wall = is_boundary & ~is_lid
is_interior = ~is_boundary

interior_idx = np.where(is_interior)[0]
lid_idx = np.where(is_lid)[0]
wall_idx = np.where(is_wall)[0]

print(f"Grid: {N_grid}x{N_grid}, Interior: {len(interior_idx)}, Lid: {len(lid_idx)}, Wall: {len(wall_idx)}")

Dx_torch = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_torch = torch.tensor(Dy_np, dtype=torch.float32, device=device)
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_interior = xy_all[interior_idx]
xy_lid = xy_all[lid_idx]
xy_wall = xy_all[wall_idx]

x_t = xy_all[:, 0:1]
y_t = xy_all[:, 1:2]
d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))

# =============================================================================
# Network
# =============================================================================
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

# =============================================================================
# PDE Residual Functions
# =============================================================================
def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]

def pde_residuals_autodiff(model, xy):
    xy.requires_grad_(True)
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    x, y = xy[:, 0:1], xy[:, 1:2]
    d = torch.min(torch.min(x, 1.0 - x), torch.min(y, 1.0 - y))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d)**2 * S_mag
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

def pde_residuals_discrete(model, xy_all, Dx, Dy, d_wall, interior_idx):
    pred = model(xy_all)
    u_all = pred[:, 0:1]
    v_all = pred[:, 1:2]
    p_all = pred[:, 2:3]
    du_dx = Dx @ u_all
    du_dy = Dy @ u_all
    dv_dx = Dx @ v_all
    dv_dy = Dy @ v_all
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u_all * du_dx + v_all * du_dy
    v_conv = u_all * dv_dx + v_all * dv_dy
    dp_dx = Dx @ p_all
    dp_dy = Dy @ p_all
    visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
    visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    return continuity[interior_idx], mom_u[interior_idx], mom_v[interior_idx]

# =============================================================================
# Training Functions
# =============================================================================
def train_experiment(dt_epochs, auto_epochs, verbose=True, log_interval=5000):
    """Train with dt_epochs of DT-PINN followed by auto_epochs of autodiff."""
    total_epochs = dt_epochs + auto_epochs
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    loss_history = []
    epoch_times = []

    # Phase 1: DT-PINN
    if dt_epochs > 0 and verbose:
        print(f"  Phase 1: DT-PINN ({dt_epochs} epochs)")

    for epoch in range(dt_epochs):
        epoch_start = time.perf_counter()
        optimizer.zero_grad()

        cont, mom_u, mom_v = pde_residuals_discrete(
            model, xy_all, Dx_torch, Dy_torch, d_wall, interior_idx
        )
        loss_pde = mse(cont, torch.zeros_like(cont)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

        pred_lid = model(xy_lid)
        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                   mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))

        pred_wall = model(xy_wall)
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                    mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))

        xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        pred_center = model(xy_center)
        loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        epoch_times.append(time.perf_counter() - epoch_start)
        loss_history.append(loss.item())

        if verbose and (epoch + 1) % log_interval == 0:
            print(f"    Epoch {epoch+1}: loss = {loss.item():.6f}")

    # Phase 2: Autodiff
    if auto_epochs > 0:
        if verbose:
            print(f"  Phase 2: Autodiff ({auto_epochs} epochs)")

        xy_int = xy_interior.clone().detach().requires_grad_(True)

        for epoch in range(auto_epochs):
            epoch_start = time.perf_counter()
            optimizer.zero_grad()

            cont, mom_u, mom_v = pde_residuals_autodiff(model, xy_int)
            loss_pde = mse(cont, torch.zeros_like(cont)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))

            pred_lid = model(xy_lid)
            loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                       mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))

            pred_wall = model(xy_wall)
            loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                        mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))

            xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
            pred_center = model(xy_center)
            loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))

            loss = loss_pde + loss_lid + loss_wall + loss_p
            loss.backward()
            optimizer.step()

            epoch_times.append(time.perf_counter() - epoch_start)
            loss_history.append(loss.item())

            if verbose and (dt_epochs + epoch + 1) % log_interval == 0:
                print(f"    Epoch {dt_epochs + epoch + 1}: loss = {loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return model, total_time, loss_history, epoch_times

def evaluate_model(model):
    """Evaluate using autodiff on evaluation grid."""
    nx, ny = 41, 41
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    cont, mom_u, mom_v = pde_residuals_autodiff(model, xy_t)

    cont_np = cont.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))
    cont_rms = float(np.sqrt(np.mean(cont_np**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u_np**2 + mom_v_np**2)))

    return {
        'pde_rms': pde_rms,
        'continuity_rms': cont_rms,
        'momentum_rms': mom_rms,
    }

# =============================================================================
# Run Experiments
# =============================================================================
EXPERIMENTS = [
    {'name': 'Baseline (pure autodiff)', 'dt_epochs': 0, 'auto_epochs': 30000},
    {'name': 'Pure DT-PINN', 'dt_epochs': 30000, 'auto_epochs': 0},
    {'name': 'Hybrid 50-50', 'dt_epochs': 15000, 'auto_epochs': 15000},
    {'name': 'Hybrid 75-25', 'dt_epochs': 22500, 'auto_epochs': 7500},
    {'name': 'Hybrid 25-75', 'dt_epochs': 7500, 'auto_epochs': 22500},
    {'name': 'Hybrid 500+rest', 'dt_epochs': 500, 'auto_epochs': 29500},
]

results = []

print("\n" + "=" * 70)
print("RUNNING 30K EXPERIMENTS")
print("=" * 70)

for exp in EXPERIMENTS:
    print(f"\n{'='*70}")
    print(f"Experiment: {exp['name']}")
    print(f"DT-PINN: {exp['dt_epochs']}, Autodiff: {exp['auto_epochs']}")
    print("=" * 70)

    model, total_time, loss_history, epoch_times = train_experiment(
        exp['dt_epochs'], exp['auto_epochs'], verbose=True
    )
    metrics = evaluate_model(model)

    result = {
        'name': exp['name'],
        'dt_epochs': exp['dt_epochs'],
        'auto_epochs': exp['auto_epochs'],
        'total_time': total_time,
        'final_loss': loss_history[-1] if loss_history else None,
        **metrics,
    }
    results.append(result)

    print(f"\n  RESULT: Time={total_time:.1f}s ({total_time/60:.1f}min), PDE_RMS={metrics['pde_rms']:.5f}")

    # Save intermediate results
    os.makedirs('results/dt_pinn_30k', exist_ok=True)
    with open('results/dt_pinn_30k/results.json', 'w') as f:
        json.dump(results, f, indent=2)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("30K EXPERIMENT RESULTS SUMMARY")
print("=" * 70)

baseline = [r for r in results if r['name'] == 'Baseline (pure autodiff)'][0]

print(f"\n{'Config':<25} {'Time':<12} {'PDE RMS':<12} {'Speedup':<10} {'Acc Ratio':<10}")
print("-" * 70)
for r in results:
    speedup = baseline['total_time'] / r['total_time']
    acc_ratio = r['pde_rms'] / baseline['pde_rms']
    print(f"{r['name']:<25} {r['total_time']/60:.1f}min      {r['pde_rms']:<12.5f} {speedup:.2f}x      {acc_ratio:.2f}x")

print("\n" + "=" * 70)
print("Results saved to results/dt_pinn_30k/results.json")
