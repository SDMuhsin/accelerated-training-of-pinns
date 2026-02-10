#!/usr/bin/env python3
"""
Analyze gradient divergence between autodiff and DT-PINN over training.

Key question: Is DT-PINN's accuracy loss from random float32 noise or systematic bias?

Approach:
- Train both methods in lockstep for N epochs
- After each step, compare the parameter updates (gradients)
- Measure gradient cosine similarity and relative magnitude
- Track the parameter divergence over time
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os
import json
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
N_EPOCHS = 500

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
N_grid = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Infrastructure (minimal)
def chebyshev_points(N):
    return np.cos(np.pi * np.arange(N) / (N - 1))

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

D1d = chebyshev_diff_matrix(N_grid) * 2.0
I_mat = np.eye(N_grid)
Dx_np, Dy_np = np.kron(I_mat, D1d), np.kron(D1d, I_mat)
x_ref = chebyshev_points(N_grid)
x = 0.5 * (x_ref + 1.0)
xx, yy = np.meshgrid(x, x, indexing='xy')
xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

eps = 1e-10
xc, yc = xy_grid[:, 0], xy_grid[:, 1]
is_boundary = (xc < eps) | (xc > 1-eps) | (yc < eps) | (yc > 1-eps)
interior_idx = np.where(~is_boundary)[0]
lid_idx = np.where(yc > 1-eps)[0]
wall_idx = np.where(is_boundary & ~(yc > 1-eps))[0]

Dx_t = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_t = torch.tensor(Dy_np, dtype=torch.float32, device=device)
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_int = xy_all[interior_idx]
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

def compute_loss_autodiff(model, xy_int, xy_lid, xy_wall):
    xy_int_g = xy_int.clone().detach().requires_grad_(True)
    pred = model(xy_int_g)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    gu = gradients(u, xy_int_g); gv = gradients(v, xy_int_g)
    du_dx, du_dy = gu[:, 0:1], gu[:, 1:2]
    dv_dx, dv_dy = gv[:, 0:1], gv[:, 1:2]
    x_, y_ = xy_int_g[:, 0:1], xy_int_g[:, 1:2]
    d = torch.min(torch.min(x_, 1.0 - x_), torch.min(y_, 1.0 - y_))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5*(du_dy + dv_dx)
    S_mag = torch.sqrt(2.0*(Sxx**2 + Syy**2 + 2.0*Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs*d)**2 * S_mag
    cont = du_dx + dv_dy
    u_conv = u*du_dx + v*du_dy; v_conv = u*dv_dx + v*dv_dy
    gp = gradients(p, xy_int_g); dp_dx, dp_dy = gp[:, 0:1], gp[:, 1:2]
    qxu, qyu = nu_eff*du_dx, nu_eff*du_dy; qxv, qyv = nu_eff*dv_dx, nu_eff*dv_dy
    gqxu, gqyu = gradients(qxu, xy_int_g), gradients(qyu, xy_int_g)
    gqxv, gqyv = gradients(qxv, xy_int_g), gradients(qyv, xy_int_g)
    visc_u = gqxu[:, 0:1] + gqyu[:, 1:2]; visc_v = gqxv[:, 0:1] + gqyv[:, 1:2]
    mu = u_conv + dp_dx - visc_u; mv = v_conv + dp_dy - visc_v
    mse = nn.MSELoss()
    loss_pde = mse(cont, torch.zeros_like(cont)) + mse(mu, torch.zeros_like(mu)) + mse(mv, torch.zeros_like(mv))
    pred_lid = model(xy_lid)
    loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
    pred_wall = model(xy_wall)
    loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    pred_center = model(xy_center)
    loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))
    return loss_pde + loss_lid + loss_wall + loss_p

def compute_loss_dtpinn(model, xy_all, Dx, Dy, d_wall, interior_idx, xy_lid, xy_wall):
    pred = model(xy_all)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    du_dx, du_dy = Dx@u, Dy@u; dv_dx, dv_dy = Dx@v, Dy@v
    Sxx, Syy = du_dx, dv_dy; Sxy = 0.5*(du_dy + dv_dx)
    S_mag = torch.sqrt(2.0*(Sxx**2 + Syy**2 + 2.0*Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs*d_wall)**2 * S_mag
    cont = (du_dx + dv_dy)[interior_idx]
    u_conv = (u*du_dx + v*du_dy)[interior_idx]; v_conv = (u*dv_dx + v*dv_dy)[interior_idx]
    dp_dx = (Dx@p)[interior_idx]; dp_dy = (Dy@p)[interior_idx]
    visc_u = (Dx@(nu_eff*du_dx) + Dy@(nu_eff*du_dy))[interior_idx]
    visc_v = (Dx@(nu_eff*dv_dx) + Dy@(nu_eff*dv_dy))[interior_idx]
    mu = u_conv + dp_dx - visc_u; mv = v_conv + dp_dy - visc_v
    mse = nn.MSELoss()
    loss_pde = mse(cont, torch.zeros_like(cont)) + mse(mu, torch.zeros_like(mu)) + mse(mv, torch.zeros_like(mv))
    pred_lid = model(xy_lid)
    loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
    pred_wall = model(xy_wall)
    loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    pred_center = model(xy_center)
    loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))
    return loss_pde + loss_lid + loss_wall + loss_p

def get_flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

def get_flat_params(model):
    return torch.cat([p.flatten() for p in model.parameters()])

# Initialize IDENTICAL models
torch.manual_seed(SEED)
model_auto = PINN_Cavity().to(device)
model_dt = PINN_Cavity().to(device)
# Copy exact parameters
model_dt.load_state_dict(copy.deepcopy(model_auto.state_dict()))

# Use IDENTICAL Adam optimizer states
opt_auto = torch.optim.Adam(model_auto.parameters(), lr=1e-3)
opt_dt = torch.optim.Adam(model_dt.parameters(), lr=1e-3)

# Track metrics
metrics = {
    'grad_cosine_sim': [],
    'grad_rel_diff': [],
    'param_divergence': [],
    'loss_auto': [],
    'loss_dt': [],
}

print(f"Training {N_EPOCHS} epochs with both methods in lockstep...")
print(f"{'Epoch':<8} {'Cos Sim':<12} {'Grad RelDiff':<14} {'Param Div':<12} {'Loss Auto':<12} {'Loss DT':<12}")
print("-" * 70)

for epoch in range(N_EPOCHS):
    # Compute gradients for BOTH methods (same model state)
    opt_auto.zero_grad()
    loss_auto = compute_loss_autodiff(model_auto, xy_int, xy_lid, xy_wall)
    loss_auto.backward()
    grad_auto = get_flat_grad(model_auto).clone()

    opt_dt.zero_grad()
    loss_dt = compute_loss_dtpinn(model_dt, xy_all, Dx_t, Dy_t, d_wall, interior_idx, xy_lid, xy_wall)
    loss_dt.backward()
    grad_dt = get_flat_grad(model_dt).clone()

    # Gradient comparison metrics
    cos_sim = torch.nn.functional.cosine_similarity(grad_auto.unsqueeze(0), grad_dt.unsqueeze(0)).item()
    grad_diff = (grad_auto - grad_dt).norm().item()
    grad_norm = grad_auto.norm().item()
    rel_diff = grad_diff / (grad_norm + 1e-12)

    # Apply updates
    opt_auto.step()
    opt_dt.step()

    # Parameter divergence
    param_div = (get_flat_params(model_auto) - get_flat_params(model_dt)).norm().item()

    metrics['grad_cosine_sim'].append(cos_sim)
    metrics['grad_rel_diff'].append(rel_diff)
    metrics['param_divergence'].append(param_div)
    metrics['loss_auto'].append(loss_auto.item())
    metrics['loss_dt'].append(loss_dt.item())

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {cos_sim:<12.8f} {rel_diff:<14.6e} {param_div:<12.6e} {loss_auto.item():<12.6f} {loss_dt.item():<12.6f}")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
cos_arr = np.array(metrics['grad_cosine_sim'])
rel_arr = np.array(metrics['grad_rel_diff'])
div_arr = np.array(metrics['param_divergence'])

print(f"Gradient cosine similarity:")
print(f"  First 10:  mean={cos_arr[:10].mean():.10f}, min={cos_arr[:10].min():.10f}")
print(f"  Last 10:   mean={cos_arr[-10:].mean():.10f}, min={cos_arr[-10:].min():.10f}")
print(f"  Overall:   mean={cos_arr.mean():.10f}, min={cos_arr.min():.10f}")

print(f"\nGradient relative difference:")
print(f"  First 10:  mean={rel_arr[:10].mean():.6e}")
print(f"  Last 10:   mean={rel_arr[-10:].mean():.6e}")
print(f"  Overall:   mean={rel_arr.mean():.6e}")

print(f"\nParameter divergence (L2 norm):")
print(f"  Epoch 1:   {div_arr[0]:.6e}")
print(f"  Epoch 100: {div_arr[99]:.6e}")
print(f"  Epoch 500: {div_arr[-1]:.6e}")
print(f"  Growth rate (last/first): {div_arr[-1]/div_arr[0]:.1f}x")

# Is growth linear (systematic bias) or sqrt (random walk)?
if len(div_arr) > 1:
    # Fit: divergence ~ epoch^alpha
    epochs = np.arange(1, len(div_arr) + 1, dtype=float)
    # Use log-log regression
    log_e = np.log(epochs[9:])  # skip first few for stability
    log_d = np.log(div_arr[9:] + 1e-20)
    alpha = np.polyfit(log_e, log_d, 1)[0]
    print(f"  Divergence scaling: param_div ~ epoch^{alpha:.2f}")
    if alpha < 0.65:
        print(f"  Interpretation: sqrt growth → RANDOM noise accumulation")
    elif alpha > 0.85:
        print(f"  Interpretation: linear growth → SYSTEMATIC bias")
    else:
        print(f"  Interpretation: intermediate (between random and systematic)")

# Save
os.makedirs('results/gradient_analysis', exist_ok=True)
with open('results/gradient_analysis/divergence.json', 'w') as f:
    json.dump(metrics, f)
print(f"\nSaved to results/gradient_analysis/divergence.json")
