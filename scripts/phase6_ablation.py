#!/usr/bin/env python3
"""
Phase 6: Ablation and Robustness Study

Ablation variants (4):
  A. Standard DT-PINN:    separate forward + autograd backward (baseline)
  B. Batched forward only: batched forward + autograd backward
  C. Analytical backward only: separate forward + analytical backward
  D. Full method:          batched forward + analytical backward

Robustness (2 settings):
  Setting 1: N_grid=50 (default, matches Phase 5)
  Setting 2: N_grid=30 (smaller grid, different problem scale)

5 seeds per configuration. 30K epochs. Evaluation on 51x51 uniform grid with autodiff.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Configuration
# =============================================================================
SEEDS = [42, 43, 44, 45, 46]
N_EPOCHS = 30000
LOG_INTERVAL = 10000
GRID_SIZES = [50, 30]

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("PHASE 6: ABLATION AND ROBUSTNESS STUDY")
print("=" * 70)
print(f"Device: {device}")
print(f"Seeds: {SEEDS}")
print(f"Epochs: {N_EPOCHS}")
print(f"Grid sizes: {GRID_SIZES}")


# =============================================================================
# Infrastructure (parameterized by grid size)
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


def build_grid_data(N_grid):
    """Build all grid data for a given grid size."""
    D1d = chebyshev_diff_matrix(N_grid) * 2.0
    I_mat = np.eye(N_grid)
    Dx_np = np.kron(I_mat, D1d)
    Dy_np = np.kron(D1d, I_mat)

    x_ref = chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1-eps) | (yc < eps) | (yc > 1-eps)
    is_lid = (yc > 1-eps)
    is_wall = is_boundary & ~is_lid

    interior_idx = np.where(~is_boundary)[0]
    lid_idx = np.where(is_lid)[0]
    wall_idx = np.where(is_wall)[0]

    N_all = len(xy_grid)
    N_lid = len(lid_idx)
    N_wall = len(wall_idx)
    M = len(interior_idx)

    Dx = torch.tensor(Dx_np, dtype=torch.float32, device=device)
    Dy = torch.tensor(Dy_np, dtype=torch.float32, device=device)
    DxT = Dx.T.contiguous()
    DyT = Dy.T.contiguous()
    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_lid = xy_all[lid_idx]
    xy_wall = xy_all[wall_idx]

    x_t = xy_all[:, 0:1]
    y_t = xy_all[:, 1:2]
    d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))
    Cs_d_sq = (Cs * d_wall) ** 2

    interior_mask = torch.zeros(N_all, 1, device=device)
    interior_mask[interior_idx] = 1.0

    # Batched input
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    xy_batched = torch.cat([xy_all, xy_lid, xy_wall, xy_center], dim=0)
    off_lid = N_all
    off_wall = N_all + N_lid
    off_center = N_all + N_lid + N_wall

    return {
        'Dx': Dx, 'Dy': Dy, 'DxT': DxT, 'DyT': DyT,
        'xy_all': xy_all, 'xy_lid': xy_lid, 'xy_wall': xy_wall,
        'xy_batched': xy_batched,
        'interior_idx': interior_idx, 'lid_idx': lid_idx, 'wall_idx': wall_idx,
        'interior_mask': interior_mask, 'd_wall': d_wall, 'Cs_d_sq': Cs_d_sq,
        'N_all': N_all, 'N_lid': N_lid, 'N_wall': N_wall, 'M': M,
        'off_lid': off_lid, 'off_wall': off_wall, 'off_center': off_center,
        'N_grid': N_grid,
    }


mse = nn.MSELoss()


# =============================================================================
# Training variants
# =============================================================================

def compute_pde_terms(pred, g):
    """Compute PDE residual terms from prediction and grid data."""
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    du_dx = g['Dx'] @ u; du_dy = g['Dy'] @ u
    dv_dx = g['Dx'] @ v; dv_dy = g['Dy'] @ v
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + g['Cs_d_sq'] * S_mag
    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    dp_dx = g['Dx'] @ p; dp_dy = g['Dy'] @ p
    visc_u = g['Dx'] @ (nu_eff * du_dx) + g['Dy'] @ (nu_eff * du_dy)
    visc_v = g['Dx'] @ (nu_eff * dv_dx) + g['Dy'] @ (nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    return continuity, mom_u, mom_v, du_dx, du_dy, dv_dx, dv_dy, S_mag, nu_eff


def compute_analytical_grad(pred_det, g):
    """Compute ∂L/∂pred analytically for all loss terms."""
    u = pred_det[:g['N_all'], 0:1]
    v = pred_det[:g['N_all'], 1:2]
    p = pred_det[:g['N_all'], 2:3]

    du_dx = g['Dx'] @ u; du_dy = g['Dy'] @ u
    dv_dx = g['Dx'] @ v; dv_dy = g['Dy'] @ v
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12
    S_mag = torch.sqrt(S_sq)
    inv_S = 1.0 / S_mag
    nu_eff = nu_laminar + g['Cs_d_sq'] * S_mag

    continuity = du_dx + dv_dy
    dp_dx = g['Dx'] @ p; dp_dy = g['Dy'] @ p
    visc_u = g['Dx'] @ (nu_eff * du_dx) + g['Dy'] @ (nu_eff * du_dy)
    visc_v = g['Dx'] @ (nu_eff * dv_dx) + g['Dy'] @ (nu_eff * dv_dy)
    mom_u = u * du_dx + v * du_dy + dp_dx - visc_u
    mom_v = u * dv_dx + v * dv_dy + dp_dy - visc_v

    M = g['M']
    scale = 2.0 / M
    mask = g['interior_mask']
    dc = continuity * scale * mask
    dmu = mom_u * scale * mask
    dmv = mom_v * scale * mask

    au = g['Cs_d_sq'] * 2.0 * Sxx * inv_S
    bu = g['Cs_d_sq'] * 2.0 * Sxy * inv_S
    av = bu
    bv = g['Cs_d_sq'] * 2.0 * Syy * inv_S

    DxT, DyT = g['DxT'], g['DyT']

    # dL/dp
    dL_dp = DxT @ dmu + DyT @ dmv

    # dL/du
    dL_du = DxT @ dc
    dL_du = dL_du + du_dx * dmu + DxT @ (u * dmu) + DyT @ (v * dmu)
    dL_du = dL_du + dv_dx * dmv
    ndmu = -dmu; ndmv = -dmv
    wxu = DxT @ ndmu; wyu = DyT @ ndmu
    dL_du = dL_du + DxT @ (nu_eff * wxu) + DyT @ (nu_eff * wyu)
    gu = du_dx * wxu + du_dy * wyu
    dL_du = dL_du + DxT @ (au * gu) + DyT @ (bu * gu)
    wxv = DxT @ ndmv; wyv = DyT @ ndmv
    gv = dv_dx * wxv + dv_dy * wyv
    dL_du = dL_du + DxT @ (au * gv) + DyT @ (bu * gv)

    # dL/dv
    dL_dv = DyT @ dc
    dL_dv = dL_dv + du_dy * dmu
    dL_dv = dL_dv + DxT @ (u * dmv) + dv_dy * dmv + DyT @ (v * dmv)
    dL_dv = dL_dv + DxT @ (nu_eff * wxv) + DyT @ (nu_eff * wyv)
    dL_dv = dL_dv + DxT @ (av * gv) + DyT @ (bv * gv)
    dL_dv = dL_dv + DxT @ (av * gu) + DyT @ (bv * gu)

    grad_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1)
    return grad_pde


# --- Variant A: Standard DT-PINN (separate fwd, autograd bwd) ---
def train_A_standard(seed, g, verbose=True):
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(g['xy_all'])
        continuity, mom_u, mom_v, *_ = compute_pde_terms(pred, g)
        ii = g['interior_idx']
        loss_pde = mse(continuity[ii], torch.zeros_like(continuity[ii])) + \
                   mse(mom_u[ii], torch.zeros_like(mom_u[ii])) + \
                   mse(mom_v[ii], torch.zeros_like(mom_v[ii]))

        pred_lid = model(g['xy_lid'])
        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                   mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
        pred_wall = model(g['xy_wall'])
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                    mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
        pc = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        pred_c = model(pc)
        loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            print(f"      Epoch {epoch+1}: loss={loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, time.perf_counter() - start


# --- Variant B: Batched forward only (batched fwd, autograd bwd) ---
def train_B_batched_fwd(seed, g, verbose=True):
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred_batch = model(g['xy_batched'])

        pred_pde = pred_batch[:g['N_all']]
        pred_lid = pred_batch[g['off_lid']:g['off_wall']]
        pred_wall = pred_batch[g['off_wall']:g['off_center']]
        pred_c = pred_batch[g['off_center']:]

        continuity, mom_u, mom_v, *_ = compute_pde_terms(pred_pde, g)
        ii = g['interior_idx']
        loss_pde = mse(continuity[ii], torch.zeros_like(continuity[ii])) + \
                   mse(mom_u[ii], torch.zeros_like(mom_u[ii])) + \
                   mse(mom_v[ii], torch.zeros_like(mom_v[ii]))

        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                   mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                    mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
        loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            print(f"      Epoch {epoch+1}: loss={loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, time.perf_counter() - start


# --- Variant C: Analytical backward only (separate fwd, analytical bwd) ---
def train_C_analytical_bwd(seed, g, verbose=True):
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()

        # Separate forward passes (like standard)
        pred_pde = model(g['xy_all'])
        pred_lid = model(g['xy_lid'])
        pred_wall = model(g['xy_wall'])
        pc = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        pred_c = model(pc)

        with torch.no_grad():
            # Analytical PDE gradient
            grad_pde = compute_analytical_grad(pred_pde.detach(), g)

            # BC gradients
            N_lid, N_wall = g['N_lid'], g['N_wall']
            grad_lid = torch.zeros(N_lid, 3, device=device)
            grad_lid[:, 0:1] = 2.0 * (pred_lid.detach()[:, 0:1] - 1.0) / N_lid
            grad_lid[:, 1:2] = 2.0 * pred_lid.detach()[:, 1:2] / N_lid

            grad_wall = torch.zeros(N_wall, 3, device=device)
            grad_wall[:, 0:1] = 2.0 * pred_wall.detach()[:, 0:1] / N_wall
            grad_wall[:, 1:2] = 2.0 * pred_wall.detach()[:, 1:2] / N_wall

            grad_center = torch.zeros(1, 3, device=device)
            grad_center[:, 2:3] = 2.0 * pred_c.detach()[:, 2:3]

        # Backward through each prediction separately
        pred_pde.backward(gradient=grad_pde)
        pred_lid.backward(gradient=grad_lid)
        pred_wall.backward(gradient=grad_wall)
        pred_c.backward(gradient=grad_center)

        optimizer.step()

        if verbose and (epoch + 1) % LOG_INTERVAL == 0:
            with torch.no_grad():
                u = pred_pde[:, 0:1]; v = pred_pde[:, 1:2]; p = pred_pde[:, 2:3]
                du_dx = g['Dx'] @ u; du_dy = g['Dy'] @ u
                dv_dx = g['Dx'] @ v; dv_dy = g['Dy'] @ v
                Sxx, Syy = du_dx, dv_dy
                Sxy = 0.5 * (du_dy + dv_dx)
                S_mag = torch.sqrt(2.0*(Sxx**2+Syy**2+2.0*Sxy**2)+1e-12)
                nu_eff = nu_laminar + g['Cs_d_sq'] * S_mag
                cont = du_dx + dv_dy
                dp_dx = g['Dx'] @ p; dp_dy = g['Dy'] @ p
                visc_u = g['Dx']@(nu_eff*du_dx) + g['Dy']@(nu_eff*du_dy)
                visc_v = g['Dx']@(nu_eff*dv_dx) + g['Dy']@(nu_eff*dv_dy)
                mu = u*du_dx + v*du_dy + dp_dx - visc_u
                mv = u*dv_dx + v*dv_dy + dp_dy - visc_v
                ii = g['interior_idx']
                lv = (cont[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
            print(f"      Epoch {epoch+1}: pde_loss={lv.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, time.perf_counter() - start


# --- Variant D: Full method (batched fwd, analytical bwd) ---
def train_D_full(seed, g, verbose=True):
    torch.manual_seed(seed)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred_batch = model(g['xy_batched'])

        with torch.no_grad():
            pred_pde = pred_batch[:g['N_all']]
            pred_l = pred_batch[g['off_lid']:g['off_wall']]
            pred_w = pred_batch[g['off_wall']:g['off_center']]
            pred_c = pred_batch[g['off_center']:]

            grad_pde = compute_analytical_grad(pred_pde, g)

            N_lid, N_wall = g['N_lid'], g['N_wall']
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
            with torch.no_grad():
                u = pred_pde[:, 0:1]; v = pred_pde[:, 1:2]; p = pred_pde[:, 2:3]
                du_dx = g['Dx'] @ u; du_dy = g['Dy'] @ u
                dv_dx = g['Dx'] @ v; dv_dy = g['Dy'] @ v
                Sxx, Syy = du_dx, dv_dy
                Sxy = 0.5 * (du_dy + dv_dx)
                S_mag = torch.sqrt(2.0*(Sxx**2+Syy**2+2.0*Sxy**2)+1e-12)
                nu_eff = nu_laminar + g['Cs_d_sq'] * S_mag
                cont = du_dx + dv_dy
                dp_dx = g['Dx'] @ p; dp_dy = g['Dy'] @ p
                visc_u = g['Dx']@(nu_eff*du_dx) + g['Dy']@(nu_eff*du_dy)
                visc_v = g['Dx']@(nu_eff*dv_dx) + g['Dy']@(nu_eff*dv_dy)
                mu = u*du_dx + v*du_dy + dp_dx - visc_u
                mv = u*dv_dx + v*dv_dy + dp_dy - visc_v
                ii = g['interior_idx']
                lv = (cont[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
            print(f"      Epoch {epoch+1}: pde_loss={lv.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    return model, time.perf_counter() - start


# =============================================================================
# Evaluation
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
    ('A_Standard', train_A_standard),
    ('B_BatchedFwd', train_B_batched_fwd),
    ('C_AnalyticalBwd', train_C_analytical_bwd),
    ('D_Full', train_D_full),
]

all_results = {}

for N_grid in GRID_SIZES:
    setting_key = f"N{N_grid}"
    print(f"\n{'#'*70}")
    print(f"# SETTING: Grid size N={N_grid} ({N_grid*N_grid} points)")
    print(f"{'#'*70}")

    g = build_grid_data(N_grid)
    print(f"  Interior points: {g['M']}, Lid: {g['N_lid']}, Wall: {g['N_wall']}")
    print(f"  Total batched points: {len(g['xy_batched'])}")

    setting_results = {}

    for method_name, train_fn in METHODS:
        print(f"\n  {'='*60}")
        print(f"  METHOD: {method_name} (N={N_grid})")
        print(f"  {'='*60}")

        method_results = []
        for seed in SEEDS:
            print(f"\n    Seed {seed}:")
            model, total_time = train_fn(seed, g, verbose=True)
            metrics = evaluate_model(model)
            result = {
                'seed': seed,
                'total_time_s': total_time,
                'total_time_min': total_time / 60,
                **metrics,
            }
            method_results.append(result)
            print(f"    RESULT: Time={total_time:.1f}s ({total_time/60:.2f}min), PDE_RMS={metrics['pde_rms']:.5f}")
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        setting_results[method_name] = method_results

    all_results[setting_key] = setting_results


# =============================================================================
# Summary and analysis
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 6: ABLATION SUMMARY")
print("=" * 70)

# Reference autodiff time from Phase 2
autodiff_time_min = 22.4

summary = {}

for setting_key, setting_results in all_results.items():
    print(f"\n{'='*60}")
    print(f"Setting: {setting_key}")
    print(f"{'='*60}")
    print(f"\n{'Method':<22} {'Time (min)':<16} {'PDE RMS':<18} {'Speedup':<10}")
    print("-" * 66)

    setting_summary = {}
    for method_name, results in setting_results.items():
        times = [r['total_time_min'] for r in results]
        rms_vals = [r['pde_rms'] for r in results]
        t_mean, t_std = np.mean(times), np.std(times)
        r_mean, r_std = np.mean(rms_vals), np.std(rms_vals)
        speedup = autodiff_time_min / t_mean

        time_str = f"{t_mean:.2f} ± {t_std:.2f}"
        rms_str = f"{r_mean:.4f} ± {r_std:.4f}"
        print(f"{method_name:<22} {time_str:<16} {rms_str:<18} {speedup:.2f}x")

        setting_summary[method_name] = {
            'time_mean_min': float(t_mean),
            'time_std_min': float(t_std),
            'rms_mean': float(r_mean),
            'rms_std': float(r_std),
            'speedup_vs_autodiff': float(speedup),
            'per_seed': results,
        }

    summary[setting_key] = setting_summary

# Ablation analysis
print("\n" + "=" * 70)
print("ABLATION ANALYSIS")
print("=" * 70)

for setting_key, ss in summary.items():
    print(f"\n--- {setting_key} ---")

    if all(m in ss for m in ['A_Standard', 'B_BatchedFwd', 'C_AnalyticalBwd', 'D_Full']):
        t_A = ss['A_Standard']['time_mean_min']
        t_B = ss['B_BatchedFwd']['time_mean_min']
        t_C = ss['C_AnalyticalBwd']['time_mean_min']
        t_D = ss['D_Full']['time_mean_min']

        print(f"\nSpeed contributions:")
        print(f"  Batched forward alone:    {t_A/t_B:.2f}x faster than standard")
        print(f"  Analytical backward alone: {t_A/t_C:.2f}x faster than standard")
        print(f"  Both combined:            {t_A/t_D:.2f}x faster than standard")
        print(f"  Expected (multiplicative): {(t_A/t_B)*(t_A/t_C)/1:.2f}x")
        print(f"  Interaction effect:       {(t_A/t_D) / ((t_A/t_B)*(t_A/t_C)):.2f}x")

        r_A = ss['A_Standard']['rms_mean']
        r_B = ss['B_BatchedFwd']['rms_mean']
        r_C = ss['C_AnalyticalBwd']['rms_mean']
        r_D = ss['D_Full']['rms_mean']

        print(f"\nAccuracy:")
        print(f"  Standard:              {r_A:.4f} ± {ss['A_Standard']['rms_std']:.4f}")
        print(f"  + Batched fwd:         {r_B:.4f} ± {ss['B_BatchedFwd']['rms_std']:.4f}")
        print(f"  + Analytical bwd:      {r_C:.4f} ± {ss['C_AnalyticalBwd']['rms_std']:.4f}")
        print(f"  + Both:                {r_D:.4f} ± {ss['D_Full']['rms_std']:.4f}")

# Save results
os.makedirs('results/phase6', exist_ok=True)
output_path = 'results/phase6/ablation_results.json'
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# OBJECTIVE.md target check for full method
print("\n" + "=" * 70)
print("OBJECTIVE.md TARGET CHECK (Full Method, N50)")
print("=" * 70)

if 'N50' in summary and 'D_Full' in summary['N50']:
    s = summary['N50']['D_Full']
    sp = s['speedup_vs_autodiff']
    rm = s['rms_mean']
    t_a = sp >= 1.5 and rm <= 0.046
    t_b = sp >= 1.2 and rm <= 0.028
    t_c = sp >= 2.0 and rm <= 0.055
    print(f"  Target A (≥1.5x, ≤0.046): {'MET' if t_a else 'NOT MET'} ({sp:.2f}x, {rm:.4f})")
    print(f"  Target B (≥1.2x, ≤0.028): {'MET' if t_b else 'NOT MET'} ({sp:.2f}x, {rm:.4f})")
    print(f"  Target C (≥2.0x, ≤0.055): {'MET' if t_c else 'NOT MET'} ({sp:.2f}x, {rm:.4f})")
