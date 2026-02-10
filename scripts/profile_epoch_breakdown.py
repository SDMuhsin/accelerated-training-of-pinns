#!/usr/bin/env python3
"""
Profile per-epoch time breakdown for autodiff vs DT-PINN.

Breaks each epoch into:
  1. Forward pass (model prediction)
  2. Derivative computation (autograd or matrix multiply)
  3. Loss computation (PDE residual assembly + MSE)
  4. Backward pass (loss.backward())
  5. Optimizer step

Runs 200 epochs (50 warmup + 150 measured) for each method.
Reports mean +/- std per component.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
N_WARMUP = 50
N_MEASURE = 150

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
N_grid = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# Infrastructure (copied from dt_pinn_30k_experiments.py)
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

Dx_torch = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_torch = torch.tensor(Dy_np, dtype=torch.float32, device=device)
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_interior = xy_all[interior_idx]
xy_lid = xy_all[lid_idx]
xy_wall = xy_all[wall_idx]
x_t = xy_all[:, 0:1]
y_t = xy_all[:, 1:2]
d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))

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

def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

# =============================================================================
# Profile Autodiff
# =============================================================================
def profile_autodiff():
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    xy_int = xy_interior.clone().detach().requires_grad_(True)

    timings = {
        'forward': [], 'derivatives': [], 'loss_assembly': [],
        'backward': [], 'optimizer': [], 'total': []
    }

    for epoch in range(N_WARMUP + N_MEASURE):
        sync()
        t_total_start = time.perf_counter()

        optimizer.zero_grad()

        # 1. Forward pass
        sync()
        t0 = time.perf_counter()
        xy_int_local = xy_int  # already requires_grad
        pred = model(xy_int_local)
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        sync()
        t_forward = time.perf_counter() - t0

        # 2. Derivative computation (autograd)
        sync()
        t0 = time.perf_counter()
        grad_u = gradients(u, xy_int_local)
        grad_v = gradients(v, xy_int_local)
        du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
        dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
        x_coord, y_coord = xy_int_local[:, 0:1], xy_int_local[:, 1:2]
        d = torch.min(torch.min(x_coord, 1.0 - x_coord), torch.min(y_coord, 1.0 - y_coord))
        Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
        S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
        nu_eff = nu_laminar + (Cs * d)**2 * S_mag
        grad_p = gradients(p, xy_int_local)
        dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
        qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
        qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
        grad_qx_u = gradients(qx_u, xy_int_local)
        grad_qy_u = gradients(qy_u, xy_int_local)
        grad_qx_v = gradients(qx_v, xy_int_local)
        grad_qy_v = gradients(qy_v, xy_int_local)
        sync()
        t_deriv = time.perf_counter() - t0

        # 3. Loss assembly
        sync()
        t0 = time.perf_counter()
        continuity = du_dx + dv_dy
        u_conv = u * du_dx + v * du_dy
        v_conv = u * dv_dx + v * dv_dy
        visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
        visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]
        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v

        loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
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
        sync()
        t_loss = time.perf_counter() - t0

        # 4. Backward
        sync()
        t0 = time.perf_counter()
        loss.backward()
        sync()
        t_backward = time.perf_counter() - t0

        # 5. Optimizer step
        sync()
        t0 = time.perf_counter()
        optimizer.step()
        sync()
        t_optim = time.perf_counter() - t0

        sync()
        t_total = time.perf_counter() - t_total_start

        if epoch >= N_WARMUP:
            timings['forward'].append(t_forward)
            timings['derivatives'].append(t_deriv)
            timings['loss_assembly'].append(t_loss)
            timings['backward'].append(t_backward)
            timings['optimizer'].append(t_optim)
            timings['total'].append(t_total)

    return timings


# =============================================================================
# Profile DT-PINN (discrete derivatives)
# =============================================================================
def profile_dtpinn():
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    timings = {
        'forward': [], 'derivatives': [], 'loss_assembly': [],
        'backward': [], 'optimizer': [], 'total': []
    }

    for epoch in range(N_WARMUP + N_MEASURE):
        sync()
        t_total_start = time.perf_counter()

        optimizer.zero_grad()

        # 1. Forward pass
        sync()
        t0 = time.perf_counter()
        pred = model(xy_all)
        u_all = pred[:, 0:1]
        v_all = pred[:, 1:2]
        p_all = pred[:, 2:3]
        sync()
        t_forward = time.perf_counter() - t0

        # 2. Derivative computation (matrix multiply)
        sync()
        t0 = time.perf_counter()
        du_dx = Dx_torch @ u_all
        du_dy = Dy_torch @ u_all
        dv_dx = Dx_torch @ v_all
        dv_dy = Dy_torch @ v_all
        Sxx, Syy = du_dx, dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
        nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag
        dp_dx = Dx_torch @ p_all
        dp_dy = Dy_torch @ p_all
        visc_u = Dx_torch @ (nu_eff * du_dx) + Dy_torch @ (nu_eff * du_dy)
        visc_v = Dx_torch @ (nu_eff * dv_dx) + Dy_torch @ (nu_eff * dv_dy)
        sync()
        t_deriv = time.perf_counter() - t0

        # 3. Loss assembly
        sync()
        t0 = time.perf_counter()
        continuity = du_dx + dv_dy
        u_conv = u_all * du_dx + v_all * du_dy
        v_conv = u_all * dv_dx + v_all * dv_dy
        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v

        cont_int = continuity[interior_idx]
        mom_u_int = mom_u[interior_idx]
        mom_v_int = mom_v[interior_idx]

        loss_pde = mse(cont_int, torch.zeros_like(cont_int)) + \
                   mse(mom_u_int, torch.zeros_like(mom_u_int)) + \
                   mse(mom_v_int, torch.zeros_like(mom_v_int))

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
        sync()
        t_loss = time.perf_counter() - t0

        # 4. Backward
        sync()
        t0 = time.perf_counter()
        loss.backward()
        sync()
        t_backward = time.perf_counter() - t0

        # 5. Optimizer step
        sync()
        t0 = time.perf_counter()
        optimizer.step()
        sync()
        t_optim = time.perf_counter() - t0

        sync()
        t_total = time.perf_counter() - t_total_start

        if epoch >= N_WARMUP:
            timings['forward'].append(t_forward)
            timings['derivatives'].append(t_deriv)
            timings['loss_assembly'].append(t_loss)
            timings['backward'].append(t_backward)
            timings['optimizer'].append(t_optim)
            timings['total'].append(t_total)

    return timings


# =============================================================================
# Run and report
# =============================================================================
print("\n" + "=" * 70)
print("PROFILING AUTODIFF (50 warmup + 150 measured epochs)")
print("=" * 70)
auto_timings = profile_autodiff()

print("\n" + "=" * 70)
print("PROFILING DT-PINN (50 warmup + 150 measured epochs)")
print("=" * 70)
dtpinn_timings = profile_dtpinn()

print("\n" + "=" * 70)
print("PER-EPOCH TIME BREAKDOWN (ms)")
print("=" * 70)

components = ['forward', 'derivatives', 'loss_assembly', 'backward', 'optimizer', 'total']

print(f"\n{'Component':<20} {'Autodiff (ms)':<25} {'DT-PINN (ms)':<25} {'Speedup':<10}")
print("-" * 80)

profile_results = {}
for comp in components:
    auto_arr = np.array(auto_timings[comp]) * 1000  # to ms
    dt_arr = np.array(dtpinn_timings[comp]) * 1000
    auto_mean, auto_std = auto_arr.mean(), auto_arr.std()
    dt_mean, dt_std = dt_arr.mean(), dt_arr.std()
    speedup = auto_mean / dt_mean if dt_mean > 0 else float('inf')
    print(f"{comp:<20} {auto_mean:>8.3f} +/- {auto_std:>6.3f}   {dt_mean:>8.3f} +/- {dt_std:>6.3f}   {speedup:>6.2f}x")
    profile_results[comp] = {
        'autodiff_mean_ms': float(auto_mean),
        'autodiff_std_ms': float(auto_std),
        'dtpinn_mean_ms': float(dt_mean),
        'dtpinn_std_ms': float(dt_std),
        'speedup': float(speedup),
    }

# Percentage breakdown
print(f"\n{'Component':<20} {'Autodiff %':<15} {'DT-PINN %':<15}")
print("-" * 50)
auto_total = np.array(auto_timings['total']).mean() * 1000
dt_total = np.array(dtpinn_timings['total']).mean() * 1000
for comp in components[:-1]:  # exclude total
    auto_pct = np.array(auto_timings[comp]).mean() * 1000 / auto_total * 100
    dt_pct = np.array(dtpinn_timings[comp]).mean() * 1000 / dt_total * 100
    print(f"{comp:<20} {auto_pct:>8.1f}%       {dt_pct:>8.1f}%")
    profile_results[comp]['autodiff_pct'] = float(auto_pct)
    profile_results[comp]['dtpinn_pct'] = float(dt_pct)

# Estimated 30K epoch times
print(f"\nEstimated 30K epoch wall clock:")
print(f"  Autodiff: {auto_total * 30000 / 1000 / 60:.1f} min")
print(f"  DT-PINN:  {dt_total * 30000 / 1000 / 60:.1f} min")
print(f"  Speedup:  {auto_total / dt_total:.2f}x")

# Save results
os.makedirs('results/profiling', exist_ok=True)
with open('results/profiling/epoch_breakdown.json', 'w') as f:
    json.dump(profile_results, f, indent=2)
print(f"\nResults saved to results/profiling/epoch_breakdown.json")
