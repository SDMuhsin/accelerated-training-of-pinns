#!/usr/bin/env python3
"""
Full epoch comparison v2: Standard DT-PINN vs Analytical Jacobian (batched).

Key optimization: Batch ALL forward passes into one call, compute ALL upstream
gradients analytically, and do a SINGLE backward through the network.
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
# Infrastructure (minimal)
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
I = np.eye(N_grid)
Dx_np = np.kron(I, D1d); Dy_np = np.kron(D1d, I)

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
N_lid = len(lid_idx)
N_wall = len(wall_idx)
N_int = len(interior_idx)
M = N_int  # number of interior points for MSE

Dx = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy = torch.tensor(Dy_np, dtype=torch.float32, device=device)
DxT = Dx.T.contiguous()
DyT = Dy.T.contiguous()
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_lid = xy_all[lid_idx]
xy_wall = xy_all[wall_idx]

x_t = xy_all[:, 0:1]; y_t = xy_all[:, 1:2]
d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))
Cs_d_sq = (Cs * d_wall) ** 2

interior_mask = torch.zeros(N_all, 1, device=device)
interior_mask[interior_idx] = 1.0

# Batched input: all grid points + lid + wall + center
xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
xy_batched = torch.cat([xy_all, xy_lid, xy_wall, xy_center], dim=0)
# Offsets for slicing batched predictions
off_lid = N_all
off_wall = N_all + N_lid
off_center = N_all + N_lid + N_wall
N_batched = off_center + 1

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
# Method A: Standard DT-PINN (reference)
# =============================================================================
def epoch_standard(model, optimizer):
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
    return loss.item()


# =============================================================================
# Method B: Analytical Jacobian (batched — single forward, single backward)
# =============================================================================
def epoch_analytical_batched(model, optimizer):
    optimizer.zero_grad()

    # Single forward pass through ALL points
    pred_batch = model(xy_batched)

    # Compute full upstream gradient analytically (no autograd through PDE or BC)
    with torch.no_grad():
        # Split predictions
        pred_pde = pred_batch[:N_all]
        pred_lid = pred_batch[off_lid:off_wall]
        pred_wall = pred_batch[off_wall:off_center]
        pred_center = pred_batch[off_center:]

        # --- PDE gradient (analytical) ---
        u = pred_pde[:, 0:1]; v = pred_pde[:, 1:2]; p = pred_pde[:, 2:3]

        du_dx = Dx @ u; du_dy = Dy @ u
        dv_dx = Dx @ v; dv_dy = Dy @ v
        Sxx, Syy = du_dx, dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12
        S_mag = torch.sqrt(S_sq)
        nu_eff = nu_laminar + Cs_d_sq * S_mag

        continuity = du_dx + dv_dy
        dp_dx = Dx @ p; dp_dy = Dy @ p
        visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
        visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)
        u_conv = u * du_dx + v * du_dy
        v_conv = u * dv_dx + v * dv_dy
        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v

        scale = 2.0 / M
        dc = continuity * scale * interior_mask
        dmu = mom_u * scale * interior_mask
        dmv = mom_v * scale * interior_mask

        inv_S = 1.0 / S_mag
        alpha_u = Cs_d_sq * 2.0 * Sxx * inv_S
        beta_u = Cs_d_sq * 2.0 * Sxy * inv_S
        alpha_v = Cs_d_sq * 2.0 * Sxy * inv_S
        beta_v = Cs_d_sq * 2.0 * Syy * inv_S

        # ∂L/∂p
        dL_dp = DxT @ dmu + DyT @ dmv

        # ∂L/∂u
        dL_du = DxT @ dc
        dL_du = dL_du + du_dx * dmu + DxT @ (u * dmu) + DyT @ (v * dmu)
        dL_du = dL_du + dv_dx * dmv

        neg_dmu = -dmu; neg_dmv = -dmv
        w_x_u = DxT @ neg_dmu; w_y_u = DyT @ neg_dmu
        dL_du = dL_du + DxT @ (nu_eff * w_x_u) + DyT @ (nu_eff * w_y_u)
        gamma_u = du_dx * w_x_u + du_dy * w_y_u
        dL_du = dL_du + DxT @ (alpha_u * gamma_u) + DyT @ (beta_u * gamma_u)

        w_x_v = DxT @ neg_dmv; w_y_v = DyT @ neg_dmv
        gamma_v = dv_dx * w_x_v + dv_dy * w_y_v
        dL_du = dL_du + DxT @ (alpha_u * gamma_v) + DyT @ (beta_u * gamma_v)

        # ∂L/∂v
        dL_dv = DyT @ dc
        dL_dv = dL_dv + du_dy * dmu
        dL_dv = dL_dv + DxT @ (u * dmv) + dv_dy * dmv + DyT @ (v * dmv)

        dL_dv = dL_dv + DxT @ (nu_eff * w_x_v) + DyT @ (nu_eff * w_y_v)
        dL_dv = dL_dv + DxT @ (alpha_v * gamma_v) + DyT @ (beta_v * gamma_v)
        dL_dv = dL_dv + DxT @ (alpha_v * gamma_u) + DyT @ (beta_v * gamma_u)

        grad_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1)  # (N_all, 3)

        # --- BC gradients (analytical — just MSE gradient) ---
        # Lid BC: loss = MSE(u_lid, 1) + MSE(v_lid, 0)
        #   = (1/N_lid)*sum((u-1)²) + (1/N_lid)*sum(v²)
        # ∂loss/∂u = 2*(u-1)/N_lid, ∂loss/∂v = 2*v/N_lid, ∂loss/∂p = 0
        grad_lid = torch.zeros(N_lid, 3, device=device)
        grad_lid[:, 0:1] = 2.0 * (pred_lid[:, 0:1] - 1.0) / N_lid
        grad_lid[:, 1:2] = 2.0 * pred_lid[:, 1:2] / N_lid

        # Wall BC: loss = MSE(u_wall, 0) + MSE(v_wall, 0)
        grad_wall = torch.zeros(N_wall, 3, device=device)
        grad_wall[:, 0:1] = 2.0 * pred_wall[:, 0:1] / N_wall
        grad_wall[:, 1:2] = 2.0 * pred_wall[:, 1:2] / N_wall

        # Pressure gauge: loss = MSE(p_center, 0) = p_center²
        grad_center = torch.zeros(1, 3, device=device)
        grad_center[:, 2:3] = 2.0 * pred_center[:, 2:3]  # MSE of single point: N=1, so 2*p

        # Combine into single upstream gradient
        upstream = torch.cat([grad_pde, grad_lid, grad_wall, grad_center], dim=0)

    # Single backward through network
    pred_batch.backward(gradient=upstream)
    optimizer.step()

    # Loss for logging (computed in no_grad above)
    with torch.no_grad():
        loss_val = (continuity[interior_idx]**2).mean() + \
                   (mom_u[interior_idx]**2).mean() + \
                   (mom_v[interior_idx]**2).mean() + \
                   ((pred_lid[:, 0:1] - 1.0)**2).mean() + \
                   (pred_lid[:, 1:2]**2).mean() + \
                   (pred_wall[:, 0:1]**2).mean() + \
                   (pred_wall[:, 1:2]**2).mean() + \
                   (pred_center[:, 2:3]**2).mean()
    return loss_val.item()


# =============================================================================
# Method C: Batched standard DT-PINN (single forward for fair comparison)
# =============================================================================
def epoch_standard_batched(model, optimizer):
    """Standard DT-PINN but with batched forward pass for fair comparison."""
    optimizer.zero_grad()

    # Single forward pass through ALL points
    pred_batch = model(xy_batched)

    # Split predictions
    pred_pde = pred_batch[:N_all]
    pred_lid = pred_batch[off_lid:off_wall]
    pred_wall = pred_batch[off_wall:off_center]
    pred_center = pred_batch[off_center:]

    u_all, v_all, p_all = pred_pde[:, 0:1], pred_pde[:, 1:2], pred_pde[:, 2:3]

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

    loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
               mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
    loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
    loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))

    loss = loss_pde + loss_lid + loss_wall + loss_p
    loss.backward()
    optimizer.step()
    return loss.item()


# =============================================================================
# Run comparisons
# =============================================================================
print("\n" + "=" * 70)
print("FULL EPOCH COMPARISON (BATCHED)")
print(f"Warmup: {N_WARMUP}, Measured: {N_MEASURE}")
print("=" * 70)

methods = [
    ("A: Standard DT-PINN (separate fwd)", epoch_standard),
    ("B: Standard DT-PINN (batched fwd)", epoch_standard_batched),
    ("C: Analytical Jacobian (batched)", epoch_analytical_batched),
]

results = {}

for name, epoch_fn in methods:
    print(f"\n{name}...")
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    times = []
    losses = []
    for ep in range(N_WARMUP + N_MEASURE):
        sync()
        t0 = time.perf_counter()
        loss_val = epoch_fn(model, opt)
        sync()
        t = (time.perf_counter() - t0) * 1000
        if ep >= N_WARMUP:
            times.append(t)
            losses.append(loss_val)

    times = np.array(times)
    print(f"  Epoch: {times.mean():.3f} ± {times.std():.3f} ms")
    print(f"  Final loss: {losses[-1]:.6e}")

    # Save final model params for comparison
    final_params = torch.cat([p.view(-1) for p in model.parameters()]).detach().clone()

    results[name] = {
        'times': times,
        'losses': losses,
        'params': final_params,
    }

# =============================================================================
# Comparisons
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Check training equivalence
print("\nTraining equivalence (parameter cosine similarity):")
keys = list(results.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        cos = torch.nn.functional.cosine_similarity(
            results[keys[i]]['params'].unsqueeze(0),
            results[keys[j]]['params'].unsqueeze(0)
        ).item()
        print(f"  {keys[i][:20]} vs {keys[j][:20]}: {cos:.8f}")

# Timing comparison
print(f"\n{'Method':<45} {'ms/epoch':<15} {'vs A':<10} {'vs autodiff':<12}")
print("-" * 82)
autodiff_ref = 41.07  # Phase 2 reference
base_time = results[keys[0]]['times'].mean()
for name, data in results.items():
    t = data['times'].mean()
    vs_a = base_time / t
    vs_auto = autodiff_ref / t
    print(f"{name:<45} {t:>8.3f} ms     {vs_a:>5.2f}x     {vs_auto:>5.2f}x")

# 30K projections
print(f"\nProjected 30K training times:")
for name, data in results.items():
    t_30k = data['times'].mean() * 30000 / 1000 / 60
    print(f"  {name}: {t_30k:.1f} min")
print(f"  Autodiff (Phase 2 ref): {autodiff_ref * 30000 / 1000 / 60:.1f} min")

# Per-component timing for analytical method
print("\n" + "=" * 70)
print("DETAILED TIMING: Analytical Jacobian Method")
print("=" * 70)

torch.manual_seed(SEED)
model_detail = PINN_Cavity().to(device)
opt_detail = torch.optim.Adam(model_detail.parameters(), lr=1e-3)

detail_times = {'forward': [], 'analytical': [], 'backward': [], 'optimizer': [], 'total': []}

for ep in range(N_WARMUP + N_MEASURE):
    sync(); t_total = time.perf_counter()

    opt_detail.zero_grad()

    # Forward
    sync(); t0 = time.perf_counter()
    pred_batch = model_detail(xy_batched)
    sync(); t_fwd = time.perf_counter() - t0

    # Analytical gradient
    sync(); t0 = time.perf_counter()
    with torch.no_grad():
        pred_pde = pred_batch[:N_all]
        pred_lid = pred_batch[off_lid:off_wall]
        pred_wall = pred_batch[off_wall:off_center]
        pred_center = pred_batch[off_center:]

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
        av = bu  # same as alpha_v
        bv = Cs_d_sq * 2.0 * Syy * inv_S

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
        grad_lid[:, 0:1] = 2.0 * (pred_lid[:, 0:1] - 1.0) / N_lid
        grad_lid[:, 1:2] = 2.0 * pred_lid[:, 1:2] / N_lid
        grad_wall = torch.zeros(N_wall, 3, device=device)
        grad_wall[:, 0:1] = 2.0 * pred_wall[:, 0:1] / N_wall
        grad_wall[:, 1:2] = 2.0 * pred_wall[:, 1:2] / N_wall
        grad_center = torch.zeros(1, 3, device=device)
        grad_center[:, 2:3] = 2.0 * pred_center[:, 2:3]

        upstream = torch.cat([grad_pde, grad_lid, grad_wall, grad_center], dim=0)
    sync(); t_anal = time.perf_counter() - t0

    # Backward
    sync(); t0 = time.perf_counter()
    pred_batch.backward(gradient=upstream)
    sync(); t_back = time.perf_counter() - t0

    # Optimizer
    sync(); t0 = time.perf_counter()
    opt_detail.step()
    sync(); t_opt = time.perf_counter() - t0

    sync(); t_tot = time.perf_counter() - t_total

    if ep >= N_WARMUP:
        detail_times['forward'].append(t_fwd * 1000)
        detail_times['analytical'].append(t_anal * 1000)
        detail_times['backward'].append(t_back * 1000)
        detail_times['optimizer'].append(t_opt * 1000)
        detail_times['total'].append(t_tot * 1000)

print(f"\n{'Component':<20} {'Time (ms)':<20} {'% of total':<12}")
print("-" * 52)
total_mean = np.mean(detail_times['total'])
for comp in ['forward', 'analytical', 'backward', 'optimizer', 'total']:
    arr = np.array(detail_times[comp])
    pct = arr.mean() / total_mean * 100
    print(f"{comp:<20} {arr.mean():>8.3f} ± {arr.std():.3f}    {pct:>5.1f}%")

# Save results
save_data = {
    'methods': {},
    'detail_timing': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                      for k, v in detail_times.items()},
}
for name, data in results.items():
    save_data['methods'][name] = {
        'mean_ms': float(data['times'].mean()),
        'std_ms': float(data['times'].std()),
        'final_loss': float(data['losses'][-1]),
        'vs_autodiff_speedup': float(autodiff_ref / data['times'].mean()),
    }

os.makedirs('results/phase3', exist_ok=True)
with open('results/phase3/full_epoch_comparison_v2.json', 'w') as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to results/phase3/full_epoch_comparison_v2.json")
