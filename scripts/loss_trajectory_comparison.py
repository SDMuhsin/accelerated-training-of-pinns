#!/usr/bin/env python3
"""
Compare loss trajectories: autodiff vs DT-PINN over 2000 epochs.
Quick comparison to see where/when DT-PINN diverges from autodiff.
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
N_EPOCHS = 2000

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
N_grid = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Infrastructure (compact)
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
I = np.eye(N_grid)
Dx_np, Dy_np = np.kron(I, D1d), np.kron(D1d, I)
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

def pde_residuals_autodiff(model, xy):
    xy.requires_grad_(True)
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    gu = gradients(u, xy); gv = gradients(v, xy)
    du_dx, du_dy = gu[:, 0:1], gu[:, 1:2]
    dv_dx, dv_dy = gv[:, 0:1], gv[:, 1:2]
    x, y = xy[:, 0:1], xy[:, 1:2]
    d = torch.min(torch.min(x, 1.0 - x), torch.min(y, 1.0 - y))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5*(du_dy + dv_dx)
    S_mag = torch.sqrt(2.0*(Sxx**2 + Syy**2 + 2.0*Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d)**2 * S_mag
    cont = du_dx + dv_dy
    u_conv = u*du_dx + v*du_dy; v_conv = u*dv_dx + v*dv_dy
    gp = gradients(p, xy); dp_dx, dp_dy = gp[:, 0:1], gp[:, 1:2]
    qxu, qyu = nu_eff*du_dx, nu_eff*du_dy
    qxv, qyv = nu_eff*dv_dx, nu_eff*dv_dy
    gqxu, gqyu = gradients(qxu, xy), gradients(qyu, xy)
    gqxv, gqyv = gradients(qxv, xy), gradients(qyv, xy)
    visc_u = gqxu[:, 0:1] + gqyu[:, 1:2]
    visc_v = gqxv[:, 0:1] + gqyv[:, 1:2]
    return cont, u_conv + dp_dx - visc_u, v_conv + dp_dy - visc_v

def pde_residuals_discrete(model, xy_all, Dx, Dy, d_wall, interior_idx):
    pred = model(xy_all)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    du_dx, du_dy = Dx@u, Dy@u
    dv_dx, dv_dy = Dx@v, Dy@v
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5*(du_dy + dv_dx)
    S_mag = torch.sqrt(2.0*(Sxx**2 + Syy**2 + 2.0*Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs*d_wall)**2 * S_mag
    cont = (du_dx + dv_dy)[interior_idx]
    u_conv = (u*du_dx + v*du_dy)[interior_idx]
    v_conv = (u*dv_dx + v*dv_dy)[interior_idx]
    dp_dx = (Dx@p)[interior_idx]; dp_dy = (Dy@p)[interior_idx]
    visc_u = (Dx@(nu_eff*du_dx) + Dy@(nu_eff*du_dy))[interior_idx]
    visc_v = (Dx@(nu_eff*dv_dx) + Dy@(nu_eff*dv_dy))[interior_idx]
    return cont, u_conv + dp_dx - visc_u, v_conv + dp_dy - visc_v

def evaluate_pde_rms(model):
    xg, yg = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 41), indexing='xy')
    xy_eval = np.column_stack([xg.ravel(), yg.ravel()])
    xy_e = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)
    model.eval()
    c, mu, mv = pde_residuals_autodiff(model, xy_e)
    c, mu, mv = c.detach().cpu().numpy(), mu.detach().cpu().numpy(), mv.detach().cpu().numpy()
    model.train()
    return float(np.sqrt(np.mean(c**2 + mu**2 + mv**2)))

def train_and_track(method, dt_epochs, auto_epochs):
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    xy_int_local = xy_int.clone().detach().requires_grad_(True)

    loss_history = []
    eval_points = [100, 200, 500, 1000, 1500, 2000]
    eval_results = {}

    total = dt_epochs + auto_epochs
    for epoch in range(total):
        optimizer.zero_grad()
        if epoch < dt_epochs:
            c, mu, mv = pde_residuals_discrete(model, xy_all, Dx_t, Dy_t, d_wall, interior_idx)
        else:
            c, mu, mv = pde_residuals_autodiff(model, xy_int_local)
        loss_pde = mse(c, torch.zeros_like(c)) + mse(mu, torch.zeros_like(mu)) + mse(mv, torch.zeros_like(mv))
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
        loss_history.append(loss.item())

        if (epoch + 1) in eval_points:
            pde_rms = evaluate_pde_rms(model)
            eval_results[epoch + 1] = pde_rms
            print(f"  [{method}] Epoch {epoch+1}: loss={loss.item():.6f}, PDE_RMS={pde_rms:.5f}")

    return loss_history, eval_results

# Run comparisons
print("=" * 70)
print(f"LOSS TRAJECTORY COMPARISON ({N_EPOCHS} epochs)")
print("=" * 70)

print("\n--- Pure Autodiff ---")
auto_loss, auto_eval = train_and_track("autodiff", 0, N_EPOCHS)

print("\n--- Pure DT-PINN ---")
dt_loss, dt_eval = train_and_track("dtpinn", N_EPOCHS, 0)

print("\n--- Hybrid 25-75 (500 DT + 1500 auto) ---")
hyb_loss, hyb_eval = train_and_track("hybrid", 500, 1500)

# Summary
print("\n" + "=" * 70)
print("SUMMARY: PDE RMS at evaluation points")
print("=" * 70)
print(f"{'Epoch':<10} {'Autodiff':<15} {'DT-PINN':<15} {'Hybrid':<15}")
print("-" * 55)
for ep in sorted(set(list(auto_eval.keys()) + list(dt_eval.keys()) + list(hyb_eval.keys()))):
    a = auto_eval.get(ep, float('nan'))
    d = dt_eval.get(ep, float('nan'))
    h = hyb_eval.get(ep, float('nan'))
    print(f"{ep:<10} {a:<15.5f} {d:<15.5f} {h:<15.5f}")

# Save
os.makedirs('results/trajectory', exist_ok=True)
with open('results/trajectory/2k_comparison.json', 'w') as f:
    json.dump({
        'autodiff': {'loss': auto_loss, 'eval': {str(k): v for k, v in auto_eval.items()}},
        'dtpinn': {'loss': dt_loss, 'eval': {str(k): v for k, v in dt_eval.items()}},
        'hybrid': {'loss': hyb_loss, 'eval': {str(k): v for k, v in hyb_eval.items()}},
    }, f, indent=2)
print("\nSaved to results/trajectory/2k_comparison.json")
