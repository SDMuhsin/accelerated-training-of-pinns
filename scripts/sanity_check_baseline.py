#!/usr/bin/env python3
"""
Sanity check: Run partner's exact PINN code and evaluate with our metric.
This verifies our baseline error is accurate.
"""

import numpy as np
import torch
import torch.nn as nn
import time

# =============================================================================
# EXACTLY from partner's code
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Re = 1000.0
U_lid = 1.0
rho = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
num_epochs = 30000
lr = 1e-3

N_interior = 6000
N_wall = 800
N_lid = 800
N_p_anchor = 1

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 70)
print("SANITY CHECK: Partner's PINN Code with Our Evaluation Metric")
print("=" * 70)
print(f"Device: {device}")
print(f"Config: Re={Re}, epochs={num_epochs}, interior={N_interior}")

def gradients(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]

class PINN_Cavity(nn.Module):
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

def sample_interior(N):
    x = np.random.rand(N, 1)
    y = np.random.rand(N, 1)
    return np.hstack((x, y))

def sample_lid(N):
    x = np.random.rand(N, 1)
    y = np.ones((N, 1))
    return np.hstack((x, y))

def sample_walls(N):
    N_each = N // 3
    xb = np.random.rand(N_each, 1)
    yb = np.zeros((N_each, 1))
    xl = np.zeros((N_each, 1))
    yl = np.random.rand(N_each, 1)
    xr = np.ones((N_each, 1))
    yr = np.random.rand(N_each, 1)
    return np.vstack((
        np.hstack((xb, yb)),
        np.hstack((xl, yl)),
        np.hstack((xr, yr)),
    ))

def sample_p_anchor():
    return np.array([[0.5, 0.5]])

def eddy_viscosity(xy, u, v):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    d_left = x
    d_right = 1.0 - x
    d_bottom = y
    d_top = 1.0 - y
    d = torch.min(torch.min(d_left, d_right), torch.min(d_bottom, d_top))

    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    du_dx = grad_u[:, 0:1]
    du_dy = grad_u[:, 1:2]
    dv_dx = grad_v[:, 0:1]
    dv_dy = grad_v[:, 1:2]

    Sxx = du_dx
    Syy = dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2)
    S_mag = torch.sqrt(S_sq + 1e-12)

    nu_t = (Cs * d)**2 * S_mag
    nu_eff = nu_laminar + nu_t
    return nu_eff, du_dx, du_dy, dv_dx, dv_dy

def pde_residuals(model, xy):
    xy.requires_grad_(True)
    pred = model(xy)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    nu_eff, du_dx, du_dy, dv_dx, dv_dy = eddy_viscosity(xy, u, v)

    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy

    grad_p = gradients(p, xy)
    dp_dx = grad_p[:, 0:1]
    dp_dy = grad_p[:, 1:2]

    qx_u = nu_eff * du_dx
    qy_u = nu_eff * du_dy
    qx_v = nu_eff * dv_dx
    qy_v = nu_eff * dv_dy

    grad_qx_u = gradients(qx_u, xy)
    grad_qy_u = gradients(qy_u, xy)
    grad_qx_v = gradients(qx_v, xy)
    grad_qy_v = gradients(qy_v, xy)

    dqx_u_dx = grad_qx_u[:, 0:1]
    dqy_u_dy = grad_qy_u[:, 1:2]
    dqx_v_dx = grad_qx_v[:, 0:1]
    dqy_v_dy = grad_qy_v[:, 1:2]

    visc_u = dqx_u_dx + dqy_u_dy
    visc_v = dqx_v_dx + dqy_v_dy

    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    return continuity, mom_u, mom_v, u, v, p

# =============================================================================
# Prepare training data (exactly as partner's code)
# =============================================================================
xy_int_np = sample_interior(N_interior)
xy_lid_np = sample_lid(N_lid)
xy_wall_np = sample_walls(N_wall)
xy_p_np = sample_p_anchor()

xy_int = torch.tensor(xy_int_np, dtype=torch.float32, device=device)
xy_lid = torch.tensor(xy_lid_np, dtype=torch.float32, device=device)
xy_wall = torch.tensor(xy_wall_np, dtype=torch.float32, device=device)
xy_p = torch.tensor(xy_p_np, dtype=torch.float32, device=device)

N_lid_eff = xy_lid.shape[0]
N_wall_eff = xy_wall.shape[0]
N_p_anchor_eff = xy_p.shape[0]

u_lid_target = torch.full((N_lid_eff, 1), U_lid, dtype=torch.float32, device=device)
zero_lid = torch.zeros((N_lid_eff, 1), dtype=torch.float32, device=device)
zero_wall = torch.zeros((N_wall_eff, 1), dtype=torch.float32, device=device)
zero_p = torch.zeros((N_p_anchor_eff, 1), dtype=torch.float32, device=device)

# =============================================================================
# Model, optimizer, loss
# =============================================================================
model = PINN_Cavity().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss()

# =============================================================================
# Training loop (exactly as partner's code)
# =============================================================================
print(f"\nTraining for {num_epochs} epochs...")
start_time = time.perf_counter()

for epoch in range(num_epochs):
    optimizer.zero_grad()

    cont, mom_u, mom_v, u_int, v_int, p_int = pde_residuals(model, xy_int)

    loss_cont = mse_loss(cont, torch.zeros_like(cont))
    loss_momu = mse_loss(mom_u, torch.zeros_like(mom_u))
    loss_momv = mse_loss(mom_v, torch.zeros_like(mom_v))

    pred_lid = model(xy_lid)
    u_lid_pred = pred_lid[:, 0:1]
    v_lid_pred = pred_lid[:, 1:2]
    loss_lid_u = mse_loss(u_lid_pred, u_lid_target)
    loss_lid_v = mse_loss(v_lid_pred, zero_lid)

    pred_wall = model(xy_wall)
    u_wall_pred = pred_wall[:, 0:1]
    v_wall_pred = pred_wall[:, 1:2]
    loss_wall_u = mse_loss(u_wall_pred, zero_wall)
    loss_wall_v = mse_loss(v_wall_pred, zero_wall)

    pred_p = model(xy_p)
    p_anchor_pred = pred_p[:, 2:3]
    loss_p_anchor = mse_loss(p_anchor_pred, zero_p)

    loss_pde = loss_cont + loss_momu + loss_momv
    loss_bc = loss_lid_u + loss_lid_v + loss_wall_u + loss_wall_v + loss_p_anchor
    loss = loss_pde + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 5000 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Loss: {loss.item():.3e} | "
            f"PDE: {loss_pde.item():.3e} | "
            f"BC: {loss_bc.item():.3e}"
        )

total_time = time.perf_counter() - start_time
print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Final training loss: {loss.item():.6f}")
print(f"Final PDE loss: {loss_pde.item():.6f}")
print(f"Final BC loss: {loss_bc.item():.6f}")

# =============================================================================
# OUR EVALUATION METRIC (same as in dt_pinn_30k_experiments.py)
# =============================================================================
print("\n" + "=" * 70)
print("EVALUATING WITH OUR METRIC (41x41 grid)")
print("=" * 70)

def evaluate_pde_rms(model):
    """Same evaluation as our 30K experiments."""
    nx, ny = 41, 41
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    cont, mom_u, mom_v, _, _, _ = pde_residuals(model, xy_t)

    cont_np = cont.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))
    cont_rms = float(np.sqrt(np.mean(cont_np**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u_np**2 + mom_v_np**2)))

    return pde_rms, cont_rms, mom_rms

pde_rms, cont_rms, mom_rms = evaluate_pde_rms(model)

print(f"\nPDE RMS Error (our metric): {pde_rms:.5f}")
print(f"  - Continuity RMS: {cont_rms:.5f}")
print(f"  - Momentum RMS: {mom_rms:.5f}")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"Partner's code after {num_epochs} epochs:")
print(f"  Training time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Final loss (their metric): {loss.item():.5f}")
print(f"  PDE RMS (our metric): {pde_rms:.5f}")
print()
print("Our baseline from 30K experiments:")
print("  Training time: 22.0 min")
print("  PDE RMS: 0.039")
print()
if abs(pde_rms - 0.039) < 0.01:
    print("✓ SANITY CHECK PASSED: Results are consistent")
else:
    print(f"⚠ DIFFERENCE DETECTED: {pde_rms:.5f} vs 0.039")
    print("  This may be due to different random seeds or slight implementation differences")
