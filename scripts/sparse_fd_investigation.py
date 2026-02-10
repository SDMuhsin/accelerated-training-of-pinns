#!/usr/bin/env python3
"""
Investigate sparse finite difference operators as an alternative to dense Chebyshev.

Chebyshev spectral: Dense 2500x2500 matrix, N^2 FLOPs per matvec.
Finite difference: Sparse matrix, ~5*N nonzeros per matvec (5-point stencil).

This script:
1. Builds FD operators on the Chebyshev grid and on a uniform grid
2. Compares derivative accuracy against autodiff
3. Benchmarks sparse vs dense matvec speed
4. Runs a short training comparison (1000 epochs)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.sparse
import time
import sys
import os
import json
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1
N_grid = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# Build Chebyshev operators (dense, reference)
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
    for i in range(N):
        D[i, i] = -np.sum(D[i, :])
    return D

D1d_cheb = chebyshev_diff_matrix(N_grid) * 2.0  # scaled for [0,1]
I = np.eye(N_grid)
Dx_cheb = np.kron(I, D1d_cheb)
Dy_cheb = np.kron(D1d_cheb, I)

# =============================================================================
# Build uniform-grid FD operators (2nd order central differences)
# =============================================================================
def build_fd_operators_uniform(N):
    """Build 2nd-order central FD operators on uniform [0,1] grid."""
    h = 1.0 / (N - 1)
    # 1D operator: central differences interior, one-sided at boundaries
    D1d = np.zeros((N, N))
    for i in range(1, N-1):
        D1d[i, i-1] = -1.0 / (2.0 * h)
        D1d[i, i+1] = 1.0 / (2.0 * h)
    # Forward difference at left boundary
    D1d[0, 0] = -3.0 / (2.0 * h)
    D1d[0, 1] = 4.0 / (2.0 * h)
    D1d[0, 2] = -1.0 / (2.0 * h)
    # Backward difference at right boundary
    D1d[-1, -2] = 1.0 / (2.0 * h)
    D1d[-1, -3] = -4.0 / (2.0 * h)
    D1d[-1, -1] = 3.0 / (2.0 * h)

    I = np.eye(N)
    Dx = np.kron(I, D1d)
    Dy = np.kron(D1d, I)
    return Dx, Dy

# Build FD on Chebyshev grid (non-uniform spacing)
def build_fd_operators_nonuniform(x_1d):
    """Build 2nd-order FD operators on non-uniform 1D grid, then Kronecker to 2D."""
    N = len(x_1d)
    D1d = np.zeros((N, N))

    for i in range(1, N-1):
        h_m = x_1d[i] - x_1d[i-1]
        h_p = x_1d[i+1] - x_1d[i]
        # 2nd order on non-uniform grid
        D1d[i, i-1] = -h_p / (h_m * (h_m + h_p))
        D1d[i, i] = (h_p - h_m) / (h_m * h_p)
        D1d[i, i+1] = h_m / (h_p * (h_m + h_p))

    # Boundaries: 2nd order one-sided
    h0 = x_1d[1] - x_1d[0]
    h1 = x_1d[2] - x_1d[1]
    D1d[0, 0] = -(2*h0 + h1) / (h0 * (h0 + h1))
    D1d[0, 1] = (h0 + h1) / (h0 * h1)
    D1d[0, 2] = -h0 / (h1 * (h0 + h1))

    hm1 = x_1d[-1] - x_1d[-2]
    hm2 = x_1d[-2] - x_1d[-3]
    D1d[-1, -1] = (2*hm1 + hm2) / (hm1 * (hm1 + hm2))
    D1d[-1, -2] = -(hm1 + hm2) / (hm1 * hm2)
    D1d[-1, -3] = hm1 / (hm2 * (hm1 + hm2))

    I = np.eye(N)
    Dx = np.kron(I, D1d)
    Dy = np.kron(D1d, I)
    return Dx, Dy

# Chebyshev 1D points mapped to [0,1]
x_cheb_1d = 0.5 * (chebyshev_points(N_grid) + 1.0)

Dx_fd_nonunif, Dy_fd_nonunif = build_fd_operators_nonuniform(x_cheb_1d)

# Uniform grid for comparison
x_unif_1d = np.linspace(0, 1, N_grid)
xx_unif, yy_unif = np.meshgrid(x_unif_1d, x_unif_1d, indexing='xy')
xy_unif = np.column_stack([xx_unif.ravel(), yy_unif.ravel()])
Dx_fd_unif, Dy_fd_unif = build_fd_operators_uniform(N_grid)

# =============================================================================
# Test 1: Derivative accuracy on known function
# =============================================================================
print("=" * 70)
print("TEST 1: Derivative accuracy on sin(2*pi*x)*cos(2*pi*y)")
print("=" * 70)

# Chebyshev grid
xx_c, yy_c = np.meshgrid(x_cheb_1d, x_cheb_1d, indexing='xy')
xy_cheb = np.column_stack([xx_c.ravel(), yy_c.ravel()])

f_cheb = np.sin(2*np.pi*xy_cheb[:, 0]) * np.cos(2*np.pi*xy_cheb[:, 1])
dfdx_exact_cheb = 2*np.pi*np.cos(2*np.pi*xy_cheb[:, 0]) * np.cos(2*np.pi*xy_cheb[:, 1])

dfdx_spectral = Dx_cheb @ f_cheb
dfdx_fd_nonunif = Dx_fd_nonunif @ f_cheb

f_unif = np.sin(2*np.pi*xy_unif[:, 0]) * np.cos(2*np.pi*xy_unif[:, 1])
dfdx_exact_unif = 2*np.pi*np.cos(2*np.pi*xy_unif[:, 0]) * np.cos(2*np.pi*xy_unif[:, 1])
dfdx_fd_unif = Dx_fd_unif @ f_unif

print(f"Chebyshev spectral (N={N_grid}):   max_err = {np.max(np.abs(dfdx_spectral - dfdx_exact_cheb)):.4e}")
print(f"FD on Cheb grid (N={N_grid}):      max_err = {np.max(np.abs(dfdx_fd_nonunif - dfdx_exact_cheb)):.4e}")
print(f"FD on uniform grid (N={N_grid}):   max_err = {np.max(np.abs(dfdx_fd_unif - dfdx_exact_unif)):.4e}")

# Sparsity
print(f"\nSparsity:")
print(f"  Chebyshev Dx: {np.count_nonzero(Dx_cheb)} / {Dx_cheb.size} = {np.count_nonzero(Dx_cheb)/Dx_cheb.size:.4f}")
print(f"  FD Dx (non-unif): {np.count_nonzero(Dx_fd_nonunif)} / {Dx_fd_nonunif.size} = {np.count_nonzero(Dx_fd_nonunif)/Dx_fd_nonunif.size:.6f}")
print(f"  FD Dx (uniform): {np.count_nonzero(Dx_fd_unif)} / {Dx_fd_unif.size} = {np.count_nonzero(Dx_fd_unif)/Dx_fd_unif.size:.6f}")

# =============================================================================
# Test 2: Matvec speed comparison (dense vs sparse)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Matvec speed (dense vs sparse)")
print("=" * 70)

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

# Dense Chebyshev
Dx_dense = torch.tensor(Dx_cheb, dtype=torch.float32, device=device)
vec = torch.randn(N_grid*N_grid, 1, device=device)

# Sparse FD (non-uniform)
Dx_fd_sp = sp.csr_matrix(Dx_fd_nonunif)
indices = torch.tensor(np.array([Dx_fd_sp.tocoo().row, Dx_fd_sp.tocoo().col]), dtype=torch.long)
values = torch.tensor(Dx_fd_sp.data, dtype=torch.float32)
Dx_sparse = torch.sparse_coo_tensor(indices, values, Dx_fd_sp.shape).to(device).coalesce()

# Benchmark dense
N_BENCH = 1000
for _ in range(100):  # warmup
    _ = Dx_dense @ vec
sync()
t0 = time.perf_counter()
for _ in range(N_BENCH):
    _ = Dx_dense @ vec
sync()
t_dense = (time.perf_counter() - t0) / N_BENCH * 1000

# Benchmark sparse
for _ in range(100):
    _ = torch.sparse.mm(Dx_sparse, vec)
sync()
t0 = time.perf_counter()
for _ in range(N_BENCH):
    _ = torch.sparse.mm(Dx_sparse, vec)
sync()
t_sparse = (time.perf_counter() - t0) / N_BENCH * 1000

print(f"Dense matvec (2500x2500 @ 2500x1):  {t_dense:.4f} ms")
print(f"Sparse matvec (nnz={Dx_fd_sp.nnz}):         {t_sparse:.4f} ms")
print(f"Speedup: {t_dense/t_sparse:.2f}x")

# Batch: 3 matvecs (for u,v,p)
vec3 = torch.randn(N_grid*N_grid, 3, device=device)
for _ in range(100):
    _ = Dx_dense @ vec3
sync()
t0 = time.perf_counter()
for _ in range(N_BENCH):
    _ = Dx_dense @ vec3
sync()
t_dense_batch = (time.perf_counter() - t0) / N_BENCH * 1000

for _ in range(100):
    _ = torch.sparse.mm(Dx_sparse, vec3)
sync()
t0 = time.perf_counter()
for _ in range(N_BENCH):
    _ = torch.sparse.mm(Dx_sparse, vec3)
sync()
t_sparse_batch = (time.perf_counter() - t0) / N_BENCH * 1000

print(f"\nBatched matvec (2500x2500 @ 2500x3):")
print(f"  Dense:  {t_dense_batch:.4f} ms")
print(f"  Sparse: {t_sparse_batch:.4f} ms")
print(f"  Speedup: {t_dense_batch/t_sparse_batch:.2f}x")

# =============================================================================
# Test 3: Full DT-PINN step comparison (dense Cheb vs dense FD on uniform grid)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Full training step comparison (1000 epochs)")
print("=" * 70)

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

def train_discrete(model, xy_all, Dx, Dy, d_wall, interior_idx, xy_lid, xy_wall, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    sync()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
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
        dp_dx, dp_dy = (Dx@p)[interior_idx], (Dy@p)[interior_idx]
        visc_u = (Dx@(nu_eff*du_dx) + Dy@(nu_eff*du_dy))[interior_idx]
        visc_v = (Dx@(nu_eff*dv_dx) + Dy@(nu_eff*dv_dy))[interior_idx]
        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v
        loss_pde = mse(cont, torch.zeros_like(cont)) + mse(mom_u, torch.zeros_like(mom_u)) + mse(mom_v, torch.zeros_like(mom_v))
        pred_lid = model(xy_lid)
        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
        pred_wall = model(xy_wall)
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
        xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        pred_center = model(xy_center)
        loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))
        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()
    sync()
    return time.perf_counter() - t0, loss.item()

# Setup Chebyshev grid
xy_cheb_t = torch.tensor(xy_cheb, dtype=torch.float32, device=device)
eps = 1e-10
is_bnd_c = (xy_cheb[:, 0] < eps) | (xy_cheb[:, 0] > 1-eps) | (xy_cheb[:, 1] < eps) | (xy_cheb[:, 1] > 1-eps)
int_idx_c = np.where(~is_bnd_c)[0]
lid_idx_c = np.where(xy_cheb[:, 1] > 1-eps)[0]
wall_idx_c = np.where(is_bnd_c & ~(xy_cheb[:, 1] > 1-eps))[0]
xy_lid_c = xy_cheb_t[lid_idx_c]
xy_wall_c = xy_cheb_t[wall_idx_c]
x_c, y_c = xy_cheb_t[:, 0:1], xy_cheb_t[:, 1:2]
d_wall_c = torch.min(torch.min(x_c, 1.0 - x_c), torch.min(y_c, 1.0 - y_c))

# Setup uniform grid
xy_unif_t = torch.tensor(xy_unif, dtype=torch.float32, device=device)
is_bnd_u = (xy_unif[:, 0] < eps) | (xy_unif[:, 0] > 1-eps) | (xy_unif[:, 1] < eps) | (xy_unif[:, 1] > 1-eps)
int_idx_u = np.where(~is_bnd_u)[0]
lid_idx_u = np.where(xy_unif[:, 1] > 1-eps)[0]
wall_idx_u = np.where(is_bnd_u & ~(xy_unif[:, 1] > 1-eps))[0]
xy_lid_u = xy_unif_t[lid_idx_u]
xy_wall_u = xy_unif_t[wall_idx_u]
x_u, y_u = xy_unif_t[:, 0:1], xy_unif_t[:, 1:2]
d_wall_u = torch.min(torch.min(x_u, 1.0 - x_u), torch.min(y_u, 1.0 - y_u))

# Tensors
Dx_cheb_t = torch.tensor(Dx_cheb, dtype=torch.float32, device=device)
Dy_cheb_t = torch.tensor(Dy_cheb, dtype=torch.float32, device=device)
Dx_fd_unif_t = torch.tensor(Dx_fd_unif, dtype=torch.float32, device=device)
Dy_fd_unif_t = torch.tensor(Dy_fd_unif, dtype=torch.float32, device=device)
Dx_fd_nonunif_t = torch.tensor(Dx_fd_nonunif, dtype=torch.float32, device=device)
Dy_fd_nonunif_t = torch.tensor(Dy_fd_nonunif, dtype=torch.float32, device=device)

N_TRAIN = 1000

# DT-PINN (Chebyshev spectral)
torch.manual_seed(SEED)
model1 = PINN_Cavity().to(device)
t1, loss1 = train_discrete(model1, xy_cheb_t, Dx_cheb_t, Dy_cheb_t, d_wall_c, int_idx_c, xy_lid_c, xy_wall_c, N_TRAIN)
print(f"Chebyshev spectral:  {t1:.1f}s ({t1*30:.0f}s est 30K), loss={loss1:.6f}")

# FD on uniform grid (dense)
torch.manual_seed(SEED)
model2 = PINN_Cavity().to(device)
t2, loss2 = train_discrete(model2, xy_unif_t, Dx_fd_unif_t, Dy_fd_unif_t, d_wall_u, int_idx_u, xy_lid_u, xy_wall_u, N_TRAIN)
print(f"FD uniform (dense):  {t2:.1f}s ({t2*30:.0f}s est 30K), loss={loss2:.6f}")

# FD on Chebyshev grid (dense)
torch.manual_seed(SEED)
model3 = PINN_Cavity().to(device)
t3, loss3 = train_discrete(model3, xy_cheb_t, Dx_fd_nonunif_t, Dy_fd_nonunif_t, d_wall_c, int_idx_c, xy_lid_c, xy_wall_c, N_TRAIN)
print(f"FD Cheb grid (dense):{t3:.1f}s ({t3*30:.0f}s est 30K), loss={loss3:.6f}")

# Evaluate PDE RMS for all 3
def evaluate_pde_rms(model):
    xg, yg = np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 41), indexing='xy')
    xy_eval = np.column_stack([xg.ravel(), yg.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)
    model.eval()
    pred = model(xy_t)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    gu = gradients(u, xy_t); gv = gradients(v, xy_t)
    du_dx, du_dy = gu[:, 0:1], gu[:, 1:2]
    dv_dx, dv_dy = gv[:, 0:1], gv[:, 1:2]
    x, y = xy_t[:, 0:1], xy_t[:, 1:2]
    d = torch.min(torch.min(x, 1.0 - x), torch.min(y, 1.0 - y))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5*(du_dy + dv_dx)
    S_mag = torch.sqrt(2.0*(Sxx**2 + Syy**2 + 2.0*Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d)**2 * S_mag
    cont = du_dx + dv_dy
    u_conv = u*du_dx + v*du_dy; v_conv = u*dv_dx + v*dv_dy
    gp = gradients(p, xy_t); dp_dx, dp_dy = gp[:, 0:1], gp[:, 1:2]
    qxu, qyu = nu_eff*du_dx, nu_eff*du_dy; qxv, qyv = nu_eff*dv_dx, nu_eff*dv_dy
    gqxu, gqyu = gradients(qxu, xy_t), gradients(qyu, xy_t)
    gqxv, gqyv = gradients(qxv, xy_t), gradients(qyv, xy_t)
    visc_u = gqxu[:, 0:1] + gqyu[:, 1:2]; visc_v = gqxv[:, 0:1] + gqyv[:, 1:2]
    mu = u_conv + dp_dx - visc_u; mv = v_conv + dp_dy - visc_v
    c_np = cont.detach().cpu().numpy(); mu_np = mu.detach().cpu().numpy(); mv_np = mv.detach().cpu().numpy()
    model.train()
    return float(np.sqrt(np.mean(c_np**2 + mu_np**2 + mv_np**2)))

rms1 = evaluate_pde_rms(model1)
rms2 = evaluate_pde_rms(model2)
rms3 = evaluate_pde_rms(model3)

print(f"\nPDE RMS after {N_TRAIN} epochs:")
print(f"  Chebyshev spectral:  {rms1:.5f}")
print(f"  FD uniform grid:     {rms2:.5f}")
print(f"  FD Chebyshev grid:   {rms3:.5f}")

# Save
os.makedirs('results/sparse_fd', exist_ok=True)
results = {
    'derivative_accuracy': {
        'chebyshev_spectral': float(np.max(np.abs(dfdx_spectral - dfdx_exact_cheb))),
        'fd_nonuniform': float(np.max(np.abs(dfdx_fd_nonunif - dfdx_exact_cheb))),
        'fd_uniform': float(np.max(np.abs(dfdx_fd_unif - dfdx_exact_unif))),
    },
    'matvec_speed_ms': {
        'dense': t_dense,
        'sparse': t_sparse,
        'dense_batch3': t_dense_batch,
        'sparse_batch3': t_sparse_batch,
    },
    'training_1k': {
        'chebyshev': {'time': t1, 'loss': loss1, 'pde_rms': rms1},
        'fd_uniform': {'time': t2, 'loss': loss2, 'pde_rms': rms2},
        'fd_cheb_grid': {'time': t3, 'loss': loss3, 'pde_rms': rms3},
    }
}
with open('results/sparse_fd/investigation.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/sparse_fd/investigation.json")
