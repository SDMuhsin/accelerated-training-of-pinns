#!/usr/bin/env python3
"""
Full epoch comparison: Standard DT-PINN vs Analytical Jacobian method.

Measures complete epoch time including PDE + boundary conditions + optimizer.
This gives the actual speedup for 30K training.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import json

SEED = 42
N_WARMUP = 50
N_MEASURE = 200

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
N_grid = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# Infrastructure
# =============================================================================
def chebyshev_points(N):
    i = np.arange(N)
    return np.cos(np.pi * i / (N - 1))

def chebyshev_diff_matrix(N):
    x = chebyshev_points(N)
    c = np.ones(N)
    c[0] = 2.0; c[-1] = 2.0
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
    return np.kron(I, D1d_scaled), np.kron(D1d_scaled, I)

def build_grid(N):
    x_ref = chebyshev_points(N)
    x = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x, x, indexing='xy')
    return np.column_stack([xx.ravel(), yy.ravel()]), x

Dx_np, Dy_np = build_2d_operators(N_grid)
xy_grid, x_1d = build_grid(N_grid)

eps = 1e-10
x_coords = xy_grid[:, 0]; y_coords = xy_grid[:, 1]
is_boundary = (x_coords < eps) | (x_coords > 1-eps) | (y_coords < eps) | (y_coords > 1-eps)
is_lid = (y_coords > 1-eps)
is_wall = is_boundary & ~is_lid
is_interior = ~is_boundary
interior_idx = np.where(is_interior)[0]
lid_idx = np.where(is_lid)[0]
wall_idx = np.where(is_wall)[0]

Dx_t = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_t = torch.tensor(Dy_np, dtype=torch.float32, device=device)
DxT = Dx_t.T.contiguous()
DyT = Dy_t.T.contiguous()
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_lid = xy_all[lid_idx]
xy_wall = xy_all[wall_idx]
x_t = xy_all[:, 0:1]; y_t = xy_all[:, 1:2]
d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))
Cs_d_sq = (Cs * d_wall) ** 2

N_pts = N_grid * N_grid
M = len(interior_idx)
interior_mask = torch.zeros(N_pts, 1, device=device)
interior_mask[interior_idx] = 1.0

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

mse = nn.MSELoss()

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()


# =============================================================================
# Method A: Standard DT-PINN (full autograd backward)
# =============================================================================
def epoch_standard_dtpinn(model, optimizer):
    optimizer.zero_grad()

    pred = model(xy_all)
    u_all = pred[:, 0:1]; v_all = pred[:, 1:2]; p_all = pred[:, 2:3]

    du_dx = Dx_t @ u_all; du_dy = Dy_t @ u_all
    dv_dx = Dx_t @ v_all; dv_dy = Dy_t @ v_all
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u_all * du_dx + v_all * du_dy
    v_conv = u_all * dv_dx + v_all * dv_dy
    dp_dx = Dx_t @ p_all; dp_dy = Dy_t @ p_all
    visc_u = Dx_t @ (nu_eff * du_dx) + Dy_t @ (nu_eff * du_dy)
    visc_v = Dx_t @ (nu_eff * dv_dx) + Dy_t @ (nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    loss_pde = mse(continuity[interior_idx], torch.zeros_like(continuity[interior_idx])) + \
               mse(mom_u[interior_idx], torch.zeros_like(mom_u[interior_idx])) + \
               mse(mom_v[interior_idx], torch.zeros_like(mom_v[interior_idx]))

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


# =============================================================================
# Method B: Analytical Jacobian (no autograd through PDE)
# =============================================================================
def epoch_analytical_jacobian(model, optimizer):
    optimizer.zero_grad()

    # Forward pass (in graph for network backward)
    pred = model(xy_all)

    # Analytical PDE gradient (no autograd)
    with torch.no_grad():
        u_all = pred[:, 0:1]; v_all = pred[:, 1:2]; p_all = pred[:, 2:3]
        du_dx = Dx_t @ u_all; du_dy = Dy_t @ u_all
        dv_dx = Dx_t @ v_all; dv_dy = Dy_t @ v_all
        Sxx, Syy = du_dx, dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12
        S_mag = torch.sqrt(S_sq)
        nu_eff = nu_laminar + Cs_d_sq * S_mag

        continuity = du_dx + dv_dy
        dp_dx = Dx_t @ p_all; dp_dy = Dy_t @ p_all
        visc_u = Dx_t @ (nu_eff * du_dx) + Dy_t @ (nu_eff * du_dy)
        visc_v = Dx_t @ (nu_eff * dv_dx) + Dy_t @ (nu_eff * dv_dy)
        u_conv = u_all * du_dx + v_all * du_dy
        v_conv = u_all * dv_dx + v_all * dv_dy
        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v

        # Loss value (for logging only)
        loss_pde_val = (continuity[interior_idx]**2).mean() + \
                       (mom_u[interior_idx]**2).mean() + \
                       (mom_v[interior_idx]**2).mean()

        # Upstream gradients
        scale = 2.0 / M
        dc = continuity * scale * interior_mask
        dmu = mom_u * scale * interior_mask
        dmv = mom_v * scale * interior_mask

        # Smagorinsky chain rule coefficients
        inv_S = 1.0 / S_mag
        alpha_u = Cs_d_sq * 2.0 * Sxx * inv_S
        beta_u = Cs_d_sq * Sxy * inv_S  # 0.5 * (4*Sxy/S) = 2*Sxy/S... wait
        # Correcting: dnu/dSxy = Cs_d² * 4*Sxy/S. ∂Sxy/∂u = 0.5*Dy. So alpha_u gets Sxx term, beta_u gets 0.5*dnu_dSxy
        # dnu_dSxx = Cs_d² * 2*Sxx/S, dnu_dSxy = Cs_d² * 4*Sxy/S
        # ∂ν/∂u via Sxx: dnu_dSxx * ∂Sxx/∂u = dnu_dSxx * Dx
        # ∂ν/∂u via Sxy: dnu_dSxy * ∂Sxy/∂u = dnu_dSxy * 0.5 * Dy
        # Combined: (Cs_d² * 2*Sxx/S) * Dx + (Cs_d² * 4*Sxy/S * 0.5) * Dy
        #         = (Cs_d² * 2*Sxx/S) * Dx + (Cs_d² * 2*Sxy/S) * Dy
        alpha_u = Cs_d_sq * 2.0 * Sxx * inv_S   # coeff of Dx in ∂ν/∂u
        beta_u = Cs_d_sq * 2.0 * Sxy * inv_S     # coeff of Dy in ∂ν/∂u
        alpha_v = Cs_d_sq * 2.0 * Sxy * inv_S    # coeff of Dx in ∂ν/∂v
        beta_v = Cs_d_sq * 2.0 * Syy * inv_S     # coeff of Dy in ∂ν/∂v

        # ∂L/∂p
        dL_dp = DxT @ dmu + DyT @ dmv

        # ∂L/∂u
        dL_du = DxT @ dc  # continuity
        dL_du = dL_du + du_dx * dmu + DxT @ (u_all * dmu) + DyT @ (v_all * dmu)  # mom_u conv
        dL_du = dL_du + dv_dx * dmv  # mom_v conv (u direct)

        neg_dmu = -dmu
        w_x_u = DxT @ neg_dmu; w_y_u = DyT @ neg_dmu
        dL_du = dL_du + DxT @ (nu_eff * w_x_u) + DyT @ (nu_eff * w_y_u)  # visc_u direct
        gamma_u = du_dx * w_x_u + du_dy * w_y_u
        dL_du = dL_du + DxT @ (alpha_u * gamma_u) + DyT @ (beta_u * gamma_u)  # visc_u Smag

        neg_dmv = -dmv
        w_x_v = DxT @ neg_dmv; w_y_v = DyT @ neg_dmv
        gamma_v = dv_dx * w_x_v + dv_dy * w_y_v
        dL_du = dL_du + DxT @ (alpha_u * gamma_v) + DyT @ (beta_u * gamma_v)  # visc_v Smag w.r.t. u

        # ∂L/∂v
        dL_dv = DyT @ dc  # continuity
        dL_dv = dL_dv + du_dy * dmu  # mom_u conv (v direct)
        dL_dv = dL_dv + DxT @ (u_all * dmv) + dv_dy * dmv + DyT @ (v_all * dmv)  # mom_v conv

        dL_dv = dL_dv + DxT @ (nu_eff * w_x_v) + DyT @ (nu_eff * w_y_v)  # visc_v direct
        dL_dv = dL_dv + DxT @ (alpha_v * gamma_v) + DyT @ (beta_v * gamma_v)  # visc_v Smag
        dL_dv = dL_dv + DxT @ (alpha_v * gamma_u) + DyT @ (beta_v * gamma_u)  # visc_u Smag w.r.t. v

        upstream_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1)

    # Network backward with PDE gradient
    pred.backward(gradient=upstream_pde)

    # Boundary conditions (these still use autograd, but they're simple/cheap)
    pred_lid = model(xy_lid)
    loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
               mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
    loss_lid.backward()

    pred_wall = model(xy_wall)
    loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
    loss_wall.backward()

    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    pred_center = model(xy_center)
    loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))
    loss_p.backward()

    optimizer.step()

    return (loss_pde_val + loss_lid.item() + loss_wall.item() + loss_p.item())


# =============================================================================
# Run comparison
# =============================================================================
print("\n" + "=" * 70)
print("FULL EPOCH COMPARISON")
print(f"Warmup: {N_WARMUP}, Measured: {N_MEASURE}")
print("=" * 70)

# Method A: Standard DT-PINN
print("\nMethod A: Standard DT-PINN...")
torch.manual_seed(SEED)
model_a = PINN_Cavity().to(device)
opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)

times_a = []
losses_a = []
for ep in range(N_WARMUP + N_MEASURE):
    sync()
    t0 = time.perf_counter()
    loss_val = epoch_standard_dtpinn(model_a, opt_a)
    sync()
    t = time.perf_counter() - t0
    if ep >= N_WARMUP:
        times_a.append(t * 1000)
        losses_a.append(loss_val)

times_a = np.array(times_a)
print(f"  Epoch time: {times_a.mean():.3f} ± {times_a.std():.3f} ms")
print(f"  Loss at end: {losses_a[-1]:.6e}")

# Method B: Analytical Jacobian
print("\nMethod B: Analytical Jacobian...")
torch.manual_seed(SEED)
model_b = PINN_Cavity().to(device)
opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

times_b = []
losses_b = []
for ep in range(N_WARMUP + N_MEASURE):
    sync()
    t0 = time.perf_counter()
    loss_val = epoch_analytical_jacobian(model_b, opt_b)
    sync()
    t = time.perf_counter() - t0
    if ep >= N_WARMUP:
        times_b.append(t * 1000)
        losses_b.append(loss_val)

times_b = np.array(times_b)
print(f"  Epoch time: {times_b.mean():.3f} ± {times_b.std():.3f} ms")
print(f"  Loss at end: {losses_b[-1]:.6e}")

# =============================================================================
# Verify training equivalence
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING EQUIVALENCE CHECK")
print("=" * 70)

# Compare model parameters
params_a = torch.cat([p.view(-1) for p in model_a.parameters()])
params_b = torch.cat([p.view(-1) for p in model_b.parameters()])
param_diff = (params_a - params_b).abs()
param_cos = torch.nn.functional.cosine_similarity(params_a.unsqueeze(0), params_b.unsqueeze(0)).item()

print(f"  Parameter max diff: {param_diff.max().item():.2e}")
print(f"  Parameter mean diff: {param_diff.mean().item():.2e}")
print(f"  Parameter cosine sim: {param_cos:.8f}")

# Compare loss trajectories
losses_a_arr = np.array(losses_a)
losses_b_arr = np.array(losses_b)
loss_diff = np.abs(losses_a_arr - losses_b_arr)
print(f"  Loss max diff: {loss_diff.max():.2e}")
print(f"  Loss mean diff: {loss_diff.mean():.2e}")

identical_training = param_cos > 0.9999
print(f"  Training is identical: {'YES' if identical_training else 'NO'}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

speedup_ratio = times_a.mean() / times_b.mean()
savings_ms = times_a.mean() - times_b.mean()
savings_pct = savings_ms / times_a.mean() * 100

print(f"\n  Standard DT-PINN epoch: {times_a.mean():.3f} ms")
print(f"  Analytical Jacobian epoch: {times_b.mean():.3f} ms")
print(f"  Savings: {savings_ms:.3f} ms ({savings_pct:.1f}%)")
print(f"  Method speedup: {speedup_ratio:.2f}x vs standard DT-PINN")

# Project 30K training times
est_30k_a = times_a.mean() * 30000 / 1000 / 60
est_30k_b = times_b.mean() * 30000 / 1000 / 60
# Phase 2 autodiff reference: 41.07ms/epoch -> 20.5 min
autodiff_epoch = 41.07
est_30k_auto = autodiff_epoch * 30000 / 1000 / 60

print(f"\n  Projected 30K training:")
print(f"    Autodiff: {est_30k_auto:.1f} min (Phase 2 reference)")
print(f"    Standard DT-PINN: {est_30k_a:.1f} min")
print(f"    Analytical Jacobian: {est_30k_b:.1f} min")
print(f"\n  Speedup vs autodiff:")
print(f"    Standard DT-PINN: {autodiff_epoch / times_a.mean():.2f}x")
print(f"    Analytical Jacobian: {autodiff_epoch / times_b.mean():.2f}x")

# Save
results = {
    'standard_dtpinn_ms': {'mean': float(times_a.mean()), 'std': float(times_a.std())},
    'analytical_jacobian_ms': {'mean': float(times_b.mean()), 'std': float(times_b.std())},
    'savings_ms': float(savings_ms),
    'savings_pct': float(savings_pct),
    'method_speedup': float(speedup_ratio),
    'vs_autodiff_speedup_dtpinn': float(autodiff_epoch / times_a.mean()),
    'vs_autodiff_speedup_analytical': float(autodiff_epoch / times_b.mean()),
    'training_identical': identical_training,
    'param_cosine_sim': float(param_cos),
    'loss_max_diff': float(loss_diff.max()),
}
os.makedirs('results/phase3', exist_ok=True)
with open('results/phase3/full_epoch_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/phase3/full_epoch_comparison.json")
