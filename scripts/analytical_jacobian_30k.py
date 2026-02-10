#!/usr/bin/env python3
"""
30K Epoch Training: Analytical Jacobian Method

Full-scale validation comparing:
1. Standard DT-PINN (separate forward passes, standard backward)
2. Batched DT-PINN (batched forward, standard backward)
3. Analytical Jacobian (batched forward, analytical backward)

Multi-seed (3 seeds). Evaluation on 51x51 uniform grid using autodiff.
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
SEEDS = [42, 43, 44]
N_EPOCHS = 30000
LOG_INTERVAL = 5000

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
N_grid = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("ANALYTICAL JACOBIAN 30K EPOCH VALIDATION")
print("=" * 70)
print(f"Device: {device}")
print(f"Seeds: {SEEDS}")
print(f"Epochs: {N_EPOCHS}")

# =============================================================================
# Infrastructure
# =============================================================================
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
        D[i, i] = -np.sum(D[i, :])
    return D

D1d = chebyshev_diff_matrix(N_grid) * 2.0
I_mat = np.eye(N_grid)
Dx_np = np.kron(I_mat, D1d); Dy_np = np.kron(D1d, I_mat)

x_ref = chebyshev_points(N_grid)
x_phys = 0.5 * (x_ref + 1.0)
xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

eps = 1e-10
xc, yc = xy_grid[:, 0], xy_grid[:, 1]
is_boundary = (xc < eps) | (xc > 1-eps) | (yc < eps) | (yc > 1-eps)
is_lid = (yc > 1-eps)
is_wall = is_boundary & ~is_lid
is_interior = ~is_boundary

interior_idx = np.where(is_interior)[0]
lid_idx = np.where(is_lid)[0]
wall_idx = np.where(is_wall)[0]

N_all = len(xy_grid)
N_lid = len(lid_idx); N_wall = len(wall_idx)
M = len(interior_idx)

Dx = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy = torch.tensor(Dy_np, dtype=torch.float32, device=device)
DxT = Dx.T.contiguous(); DyT = Dy.T.contiguous()
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_lid = xy_all[lid_idx]; xy_wall = xy_all[wall_idx]
x_t = xy_all[:, 0:1]; y_t = xy_all[:, 1:2]
d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))
Cs_d_sq = (Cs * d_wall) ** 2
interior_mask = torch.zeros(N_all, 1, device=device)
interior_mask[interior_idx] = 1.0

# Batched input
xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
xy_batched = torch.cat([xy_all, xy_lid, xy_wall, xy_center], dim=0)
off_lid = N_all; off_wall = N_all + N_lid; off_center = N_all + N_lid + N_wall


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


# =============================================================================
# Training methods
# =============================================================================
def train_standard_dtpinn(seed, verbose=True):
    """Standard DT-PINN (separate forward passes, single loss.backward)."""
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(xy_all)
        u_all, v_all, p_all = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        du_dx = Dx @ u_all; du_dy = Dy @ u_all
        dv_dx = Dx @ v_all; dv_dy = Dy @ v_all
        Sxx, Syy = du_dx, dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
        nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag
        continuity = du_dx + dv_dy
        u_conv = u_all * du_dx + v_all * du_dy
        v_conv = u_all * dv_dx + v_all * dv_dy
        dp_dx = Dx @ p_all; dp_dy = Dy @ p_all
        visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
        visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)
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
        pc = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        pred_center = model(pc)
        loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))
        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            print(f"    Epoch {epoch+1}: loss = {loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    return model, total_time


def train_analytical_jacobian(seed, verbose=True):
    """Analytical Jacobian method (batched forward, analytical backward)."""
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred_batch = model(xy_batched)

        with torch.no_grad():
            pred_pde = pred_batch[:N_all]
            pred_l = pred_batch[off_lid:off_wall]
            pred_w = pred_batch[off_wall:off_center]
            pred_c = pred_batch[off_center:]

            u = pred_pde[:, 0:1]; v = pred_pde[:, 1:2]; p = pred_pde[:, 2:3]
            du_dx = Dx @ u; du_dy = Dy @ u
            dv_dx = Dx @ v; dv_dy = Dy @ v
            Sxx, Syy = du_dx, dv_dy
            Sxy = 0.5 * (du_dy + dv_dx)
            S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12
            S_mag = torch.sqrt(S_sq); inv_S = 1.0 / S_mag
            nu_eff = nu_laminar + Cs_d_sq * S_mag

            continuity = du_dx + dv_dy
            dp_dx = Dx @ p; dp_dy = Dy @ p
            visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
            visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)
            mom_u = u * du_dx + v * du_dy + dp_dx - visc_u
            mom_v = u * dv_dx + v * dv_dy + dp_dy - visc_v

            scale = 2.0 / M
            dc = continuity * scale * interior_mask
            dmu = mom_u * scale * interior_mask
            dmv = mom_v * scale * interior_mask

            au = Cs_d_sq * 2.0 * Sxx * inv_S
            bu = Cs_d_sq * 2.0 * Sxy * inv_S
            av = bu; bv = Cs_d_sq * 2.0 * Syy * inv_S

            dL_dp = DxT @ dmu + DyT @ dmv
            dL_du = DxT @ dc + du_dx * dmu + DxT @ (u * dmu) + DyT @ (v * dmu) + dv_dx * dmv
            ndmu = -dmu; ndmv = -dmv
            wxu = DxT @ ndmu; wyu = DyT @ ndmu
            dL_du = dL_du + DxT @ (nu_eff * wxu) + DyT @ (nu_eff * wyu)
            gu = du_dx * wxu + du_dy * wyu
            dL_du = dL_du + DxT @ (au * gu) + DyT @ (bu * gu)
            wxv = DxT @ ndmv; wyv = DyT @ ndmv
            gv = dv_dx * wxv + dv_dy * wyv
            dL_du = dL_du + DxT @ (au * gv) + DyT @ (bu * gv)

            dL_dv = DyT @ dc + du_dy * dmu
            dL_dv = dL_dv + DxT @ (u * dmv) + dv_dy * dmv + DyT @ (v * dmv)
            dL_dv = dL_dv + DxT @ (nu_eff * wxv) + DyT @ (nu_eff * wyv)
            dL_dv = dL_dv + DxT @ (av * gv) + DyT @ (bv * gv)
            dL_dv = dL_dv + DxT @ (av * gu) + DyT @ (bv * gu)

            grad_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1)

            grad_lid = torch.zeros(N_lid, 3, device=device)
            grad_lid[:, 0:1] = 2.0 * (pred_l[:, 0:1] - 1.0) / N_lid
            grad_lid[:, 1:2] = 2.0 * pred_l[:, 1:2] / N_lid
            grad_wall = torch.zeros(N_wall, 3, device=device)
            grad_wall[:, 0:1] = 2.0 * pred_w[:, 0:1] / N_wall
            grad_wall[:, 1:2] = 2.0 * pred_w[:, 1:2] / N_wall
            grad_center = torch.zeros(1, 3, device=device)
            grad_center[:, 2:3] = 2.0 * pred_c[:, 2:3]

            upstream = torch.cat([grad_pde, grad_lid, grad_wall, grad_center], dim=0)

        pred_batch.backward(gradient=upstream)
        optimizer.step()

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            # Compute loss for logging
            with torch.no_grad():
                loss_val = (continuity[interior_idx]**2).mean() + \
                           (mom_u[interior_idx]**2).mean() + \
                           (mom_v[interior_idx]**2).mean()
            print(f"    Epoch {epoch+1}: pde_loss = {loss_val.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    return model, total_time


# =============================================================================
# Evaluation (autodiff on uniform grid — same as Phase 2)
# =============================================================================
def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True)[0]

def evaluate_model(model):
    """Evaluate PDE residual on 51x51 uniform grid using autodiff."""
    nx, ny = 51, 51
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    pred = model(xy_t)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = gradients(u, xy_t); grad_v = gradients(v, xy_t)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    x_coord, y_coord = xy_t[:, 0:1], xy_t[:, 1:2]
    d = torch.min(torch.min(x_coord, 1.0 - x_coord), torch.min(y_coord, 1.0 - y_coord))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d)**2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    grad_p = gradients(p, xy_t)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
    qx_u = nu_eff * du_dx; qy_u = nu_eff * du_dy
    qx_v = nu_eff * dv_dx; qy_v = nu_eff * dv_dy
    gqxu = gradients(qx_u, xy_t); gqyu = gradients(qy_u, xy_t)
    gqxv = gradients(qx_v, xy_t); gqyv = gradients(qy_v, xy_t)
    visc_u = gqxu[:, 0:1] + gqyu[:, 1:2]
    visc_v = gqxv[:, 0:1] + gqyv[:, 1:2]
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    cont_np = continuity.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))
    cont_rms = float(np.sqrt(np.mean(cont_np**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u_np**2 + mom_v_np**2)))

    model.train()
    return {'pde_rms': pde_rms, 'continuity_rms': cont_rms, 'momentum_rms': mom_rms}


# =============================================================================
# Run experiments
# =============================================================================
METHODS = [
    ('Standard DT-PINN', train_standard_dtpinn),
    ('Analytical Jacobian', train_analytical_jacobian),
]

all_results = {}

for method_name, train_fn in METHODS:
    print(f"\n{'='*70}")
    print(f"METHOD: {method_name}")
    print(f"{'='*70}")

    method_results = []
    for seed in SEEDS:
        print(f"\n  Seed {seed}:")
        model, total_time = train_fn(seed, verbose=True)
        metrics = evaluate_model(model)

        result = {
            'seed': seed,
            'total_time': total_time,
            'total_time_min': total_time / 60,
            **metrics,
        }
        method_results.append(result)
        print(f"  RESULT: Time={total_time:.1f}s ({total_time/60:.1f}min), PDE_RMS={metrics['pde_rms']:.5f}")

    all_results[method_name] = method_results

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("30K VALIDATION SUMMARY")
print("=" * 70)

# Phase 2 reference
autodiff_time_min = 22.4  # Phase 2 multi-seed mean
autodiff_rms = 0.060      # Phase 2 multi-seed mean

print(f"\n{'Method':<25} {'Time (min)':<18} {'PDE RMS':<18} {'Speedup':<10}")
print("-" * 71)

# Phase 2 baselines
print(f"{'Autodiff (Phase 2)':<25} {autodiff_time_min:<18.1f} {autodiff_rms:<18.3f} {'1.00x':<10}")

summary = {}
for method_name, results in all_results.items():
    times = [r['total_time_min'] for r in results]
    rms_vals = [r['pde_rms'] for r in results]
    t_mean, t_std = np.mean(times), np.std(times)
    r_mean, r_std = np.mean(rms_vals), np.std(rms_vals)
    speedup = autodiff_time_min / t_mean

    time_str = f"{t_mean:.1f} ± {t_std:.1f}"
    rms_str = f"{r_mean:.4f} ± {r_std:.4f}"
    print(f"{method_name:<25} {time_str:<18} {rms_str:<18} {speedup:.2f}x")

    summary[method_name] = {
        'time_mean_min': t_mean, 'time_std_min': t_std,
        'rms_mean': r_mean, 'rms_std': r_std,
        'speedup': speedup,
        'per_seed': results,
    }

# Check OBJECTIVE.md targets
print(f"\nOBJECTIVE.md TARGET EVALUATION:")
for method_name, s in summary.items():
    print(f"\n  {method_name}:")
    t_a = s['speedup'] >= 1.5 and s['rms_mean'] <= 0.046
    t_b = s['speedup'] >= 1.2 and s['rms_mean'] <= 0.028
    t_c = s['speedup'] >= 2.0 and s['rms_mean'] <= 0.055
    print(f"    Target A (≥1.5x, ≤0.046): {'MET' if t_a else 'NOT MET'} ({s['speedup']:.2f}x, {s['rms_mean']:.4f})")
    print(f"    Target B (≥1.2x, ≤0.028): {'MET' if t_b else 'NOT MET'} ({s['speedup']:.2f}x, {s['rms_mean']:.4f})")
    print(f"    Target C (≥2.0x, ≤0.055): {'MET' if t_c else 'NOT MET'} ({s['speedup']:.2f}x, {s['rms_mean']:.4f})")

# Save
os.makedirs('results/phase3', exist_ok=True)
with open('results/phase3/30k_validation.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nResults saved to results/phase3/30k_validation.json")
