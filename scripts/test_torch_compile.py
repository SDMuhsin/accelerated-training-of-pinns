#!/usr/bin/env python3
"""
Test torch.compile speedup on autodiff and DT-PINN derivative computation.
Quick test: 200 epochs (50 warmup + 150 measured).
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os

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

# Infrastructure
def chebyshev_points(N):
    i = np.arange(N)
    return np.cos(np.pi * i / (N - 1))

def chebyshev_diff_matrix(N):
    x = chebyshev_points(N)
    c = np.ones(N); c[0] = 2.0; c[-1] = 2.0
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j)) / (x[i] - x[j])
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return D

def build_2d_operators(N):
    D1d = chebyshev_diff_matrix(N) * 2.0
    I = np.eye(N)
    return np.kron(I, D1d), np.kron(D1d, I)

def build_grid(N):
    x_ref = chebyshev_points(N)
    x = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x, x, indexing='xy')
    return np.column_stack([xx.ravel(), yy.ravel()])

Dx_np, Dy_np = build_2d_operators(N_grid)
xy_grid = build_grid(N_grid)

eps = 1e-10
x_c, y_c = xy_grid[:, 0], xy_grid[:, 1]
is_boundary = (x_c < eps) | (x_c > 1-eps) | (y_c < eps) | (y_c > 1-eps)
is_lid = (y_c > 1-eps)
is_wall = is_boundary & ~is_lid
interior_idx = np.where(~is_boundary)[0]
lid_idx = np.where(is_lid)[0]
wall_idx = np.where(is_wall)[0]

Dx_torch = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_torch = torch.tensor(Dy_np, dtype=torch.float32, device=device)
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_interior = xy_all[interior_idx]
xy_lid = xy_all[lid_idx]
xy_wall = xy_all[wall_idx]
x_t, y_t = xy_all[:, 0:1], xy_all[:, 1:2]
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
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True, retain_graph=True)[0]

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

# =============================================================================
# Define train step functions for each method
# =============================================================================
def autodiff_step(model, optimizer, mse, xy_int):
    optimizer.zero_grad()
    pred = model(xy_int)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    grad_u = gradients(u, xy_int)
    grad_v = gradients(v, xy_int)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    x, y = xy_int[:, 0:1], xy_int[:, 1:2]
    d = torch.min(torch.min(x, 1.0 - x), torch.min(y, 1.0 - y))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d)**2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    grad_p = gradients(p, xy_int)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
    grad_qx_u, grad_qy_u = gradients(qx_u, xy_int), gradients(qy_u, xy_int)
    grad_qx_v, grad_qy_v = gradients(qx_v, xy_int), gradients(qy_v, xy_int)
    visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
    visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
               mse(mom_u, torch.zeros_like(mom_u)) + mse(mom_v, torch.zeros_like(mom_v))
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
    return loss.item()

def dtpinn_step(model, optimizer, mse):
    optimizer.zero_grad()
    pred = model(xy_all)
    u_all, v_all, p_all = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    du_dx = Dx_torch @ u_all
    du_dy = Dy_torch @ u_all
    dv_dx = Dx_torch @ v_all
    dv_dy = Dy_torch @ v_all
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u_all * du_dx + v_all * du_dy
    v_conv = u_all * dv_dx + v_all * dv_dy
    dp_dx = Dx_torch @ p_all
    dp_dy = Dy_torch @ p_all
    visc_u = Dx_torch @ (nu_eff * du_dx) + Dy_torch @ (nu_eff * du_dy)
    visc_v = Dx_torch @ (nu_eff * dv_dx) + Dy_torch @ (nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    loss_pde = mse(continuity[interior_idx], torch.zeros(len(interior_idx), 1, device=device)) + \
               mse(mom_u[interior_idx], torch.zeros(len(interior_idx), 1, device=device)) + \
               mse(mom_v[interior_idx], torch.zeros(len(interior_idx), 1, device=device))
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
    return loss.item()

def measure_method(name, step_fn, *args):
    times = []
    for epoch in range(N_WARMUP + N_MEASURE):
        sync()
        t0 = time.perf_counter()
        step_fn(*args)
        sync()
        t1 = time.perf_counter()
        if epoch >= N_WARMUP:
            times.append(t1 - t0)
    arr = np.array(times) * 1000
    return arr.mean(), arr.std()

# =============================================================================
# Test 1: Uncompiled baselines (sanity check)
# =============================================================================
print("=" * 70)
print("BASELINE (no torch.compile)")
print("=" * 70)

torch.manual_seed(SEED)
model_auto = PINN_Cavity().to(device)
opt_auto = torch.optim.Adam(model_auto.parameters(), lr=1e-3)
mse = nn.MSELoss()
xy_int = xy_interior.clone().detach().requires_grad_(True)

mean, std = measure_method("autodiff", autodiff_step, model_auto, opt_auto, mse, xy_int)
print(f"Autodiff:  {mean:.3f} +/- {std:.3f} ms/epoch")

torch.manual_seed(SEED)
model_dt = PINN_Cavity().to(device)
opt_dt = torch.optim.Adam(model_dt.parameters(), lr=1e-3)

mean_dt, std_dt = measure_method("dtpinn", dtpinn_step, model_dt, opt_dt, mse)
print(f"DT-PINN:   {mean_dt:.3f} +/- {std_dt:.3f} ms/epoch")

# =============================================================================
# Test 2: torch.compile on DT-PINN step
# =============================================================================
print("\n" + "=" * 70)
print("torch.compile on DT-PINN")
print("=" * 70)

torch.manual_seed(SEED)
model_dt2 = PINN_Cavity().to(device)
opt_dt2 = torch.optim.Adam(model_dt2.parameters(), lr=1e-3)

try:
    compiled_dt_step = torch.compile(dtpinn_step, mode="reduce-overhead")
    mean_c, std_c = measure_method("dtpinn_compiled", compiled_dt_step, model_dt2, opt_dt2, mse)
    print(f"DT-PINN compiled: {mean_c:.3f} +/- {std_c:.3f} ms/epoch")
    print(f"  vs uncompiled: {mean_dt/mean_c:.2f}x speedup")
except Exception as e:
    print(f"torch.compile failed for DT-PINN: {e}")

# =============================================================================
# Test 3: torch.compile on autodiff step
# =============================================================================
print("\n" + "=" * 70)
print("torch.compile on autodiff")
print("=" * 70)

torch.manual_seed(SEED)
model_auto2 = PINN_Cavity().to(device)
opt_auto2 = torch.optim.Adam(model_auto2.parameters(), lr=1e-3)
xy_int2 = xy_interior.clone().detach().requires_grad_(True)

try:
    compiled_auto_step = torch.compile(autodiff_step, mode="reduce-overhead")
    mean_ca, std_ca = measure_method("autodiff_compiled", compiled_auto_step, model_auto2, opt_auto2, mse, xy_int2)
    print(f"Autodiff compiled: {mean_ca:.3f} +/- {std_ca:.3f} ms/epoch")
    print(f"  vs uncompiled: {mean/mean_ca:.2f}x speedup")
except Exception as e:
    print(f"torch.compile failed for autodiff: {e}")

# =============================================================================
# Test 4: Forward-mode AD using torch.func.jvp
# =============================================================================
print("\n" + "=" * 70)
print("Forward-mode AD (torch.func.jvp) for derivatives only")
print("=" * 70)

torch.manual_seed(SEED)
model_jvp = PINN_Cavity().to(device)

# Measure just the derivative computation time
def measure_deriv_autodiff(model, xy, n_measure=150, n_warmup=50):
    times = []
    for i in range(n_warmup + n_measure):
        xy_local = xy.clone().detach().requires_grad_(True)
        pred = model(xy_local)
        u = pred[:, 0:1]
        sync()
        t0 = time.perf_counter()
        grad_u = gradients(u, xy_local)
        sync()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append(t1 - t0)
        # Prevent graph buildup
        del grad_u, pred, u
    return np.array(times) * 1000

def measure_deriv_jvp(model, xy, n_measure=150, n_warmup=50):
    """Use forward-mode AD: 2 JVP passes (one per input dim) instead of 3 VJP passes (one per output)."""
    from torch.func import jvp, vmap
    import functools

    times = []
    for i in range(n_warmup + n_measure):
        sync()
        t0 = time.perf_counter()

        # Tangent vectors for x and y directions
        tangent_x = torch.zeros_like(xy)
        tangent_x[:, 0] = 1.0
        tangent_y = torch.zeros_like(xy)
        tangent_y[:, 1] = 1.0

        # Forward pass + JVP for d/dx
        pred, dpred_dx = jvp(model, (xy,), (tangent_x,))
        # Forward pass + JVP for d/dy
        _, dpred_dy = jvp(model, (xy,), (tangent_y,))

        du_dx = dpred_dx[:, 0:1]
        du_dy = dpred_dy[:, 0:1]
        dv_dx = dpred_dx[:, 1:2]
        dv_dy = dpred_dy[:, 1:2]
        dp_dx = dpred_dx[:, 2:3]
        dp_dy = dpred_dy[:, 2:3]

        sync()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times.append(t1 - t0)
    return np.array(times) * 1000

xy_test = xy_interior.clone().detach()

auto_deriv_times = measure_deriv_autodiff(model_jvp, xy_test)
print(f"Autodiff derivatives (3 VJPs):  {auto_deriv_times.mean():.3f} +/- {auto_deriv_times.std():.3f} ms")

jvp_deriv_times = measure_deriv_jvp(model_jvp, xy_test)
print(f"JVP derivatives (2 JVPs):       {jvp_deriv_times.mean():.3f} +/- {jvp_deriv_times.std():.3f} ms")
print(f"  JVP vs autodiff: {auto_deriv_times.mean()/jvp_deriv_times.mean():.2f}x")

# But does JVP work with create_graph=True (needed for second derivatives in Smagorinsky)?
print("\nNote: JVP gives first derivatives. For Smagorinsky viscous terms")
print("(which need d/dx(nu_eff * du/dx)), we need second derivatives through nu_eff.")
print("This requires either nested JVP or mixing JVP+VJP.")

print("\nDone.")
