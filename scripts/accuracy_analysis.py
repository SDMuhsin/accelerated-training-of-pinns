#!/usr/bin/env python3
"""
Analyze WHY DT-PINN is less accurate than autodiff.

Hypothesis space:
1. Spectral differentiation truncation error on 50x50 grid
2. Training on Chebyshev points but evaluating on uniform grid (aliasing)
3. Dense diff matrix introduces gradient pathologies (condition number)
4. Something about how the graph structure affects optimization

This script:
- Compares autodiff vs spectral derivatives on a KNOWN analytic function
- Measures differentiation error as a function of grid size
- Checks condition number of the diff matrices
- Compares the PDE residuals computed both ways on a trained model
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1

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

# =============================================================================
# Test 1: Spectral differentiation accuracy on known functions
# =============================================================================
print("=" * 70)
print("TEST 1: Spectral differentiation accuracy (1D)")
print("=" * 70)

for N in [10, 20, 30, 50, 70]:
    D = chebyshev_diff_matrix(N)
    D_scaled = D * 2.0  # scale from [-1,1] to [0,1]
    x_ref = chebyshev_points(N)
    x = 0.5 * (x_ref + 1.0)  # map to [0,1]

    # Test function: sin(2*pi*x) - known derivative: 2*pi*cos(2*pi*x)
    f = np.sin(2 * np.pi * x)
    df_exact = 2 * np.pi * np.cos(2 * np.pi * x)
    df_spectral = D_scaled @ f
    err1 = np.max(np.abs(df_spectral - df_exact))

    # Test function: sin(6*pi*x) - higher frequency
    f2 = np.sin(6 * np.pi * x)
    df2_exact = 6 * np.pi * np.cos(6 * np.pi * x)
    df2_spectral = D_scaled @ f2
    err2 = np.max(np.abs(df2_spectral - df2_exact))

    # Condition number
    cond = np.linalg.cond(D)

    print(f"N={N:3d}: sin(2pi*x) err={err1:.2e}, sin(6pi*x) err={err2:.2e}, cond(D)={cond:.2e}")

# =============================================================================
# Test 2: Compare autodiff vs spectral on a neural network
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Compare autodiff vs spectral derivatives on NN")
print("=" * 70)

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
    return xy

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

N = 50
Dx_np, Dy_np = build_2d_operators(N)
xy_grid = build_grid(N)

Dx_torch = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_torch = torch.tensor(Dy_np, dtype=torch.float32, device=device)
xy_t = torch.tensor(xy_grid, dtype=torch.float32, device=device, requires_grad=True)

# Random model (untrained - just to see derivative agreement)
torch.manual_seed(42)
model = PINN_Cavity().to(device)

# Autodiff derivatives
pred = model(xy_t)
u = pred[:, 0:1]
grad_u = gradients(u, xy_t)
du_dx_auto = grad_u[:, 0:1].detach()
du_dy_auto = grad_u[:, 1:2].detach()

# Spectral derivatives
with torch.no_grad():
    xy_nd = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    pred_nd = model(xy_nd)
    u_nd = pred_nd[:, 0:1]
    du_dx_spec = Dx_torch @ u_nd
    du_dy_spec = Dy_torch @ u_nd

# Compare
diff_x = (du_dx_auto - du_dx_spec).abs()
diff_y = (du_dy_auto - du_dy_spec).abs()
print(f"\nUntrained model (random weights):")
print(f"  du/dx: max_diff={diff_x.max().item():.6e}, mean_diff={diff_x.mean().item():.6e}")
print(f"  du/dy: max_diff={diff_y.max().item():.6e}, mean_diff={diff_y.mean().item():.6e}")
print(f"  du/dx range: [{du_dx_auto.min().item():.4f}, {du_dx_auto.max().item():.4f}]")
print(f"  Relative error du/dx: {(diff_x / (du_dx_auto.abs() + 1e-10)).mean().item():.6e}")

# =============================================================================
# Test 3: Spectral differentiation on DIFFERENT grid sizes
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Spectral derivative accuracy vs grid size (2D NN)")
print("=" * 70)

torch.manual_seed(42)
model = PINN_Cavity().to(device)
model.eval()

# Reference: autodiff on dense grid
xy_ref = torch.tensor(
    np.column_stack(np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100), indexing='xy')).reshape(-1, 2),
    dtype=torch.float32, device=device, requires_grad=True
)
pred_ref = model(xy_ref)
u_ref = pred_ref[:, 0:1]
grad_ref = gradients(u_ref, xy_ref)
du_dx_ref = grad_ref[:, 0:1].detach()

for N in [10, 20, 30, 50, 70]:
    Dx_n, Dy_n = build_2d_operators(N)
    xy_n = build_grid(N)
    Dx_t = torch.tensor(Dx_n, dtype=torch.float32, device=device)
    xy_tn = torch.tensor(xy_n, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_n = model(xy_tn)
        u_n = pred_n[:, 0:1]
        du_dx_n = Dx_t @ u_n

    # Also compute autodiff on Chebyshev points
    xy_tn_grad = xy_tn.clone().requires_grad_(True)
    pred_n_g = model(xy_tn_grad)
    u_n_g = pred_n_g[:, 0:1]
    grad_n = gradients(u_n_g, xy_tn_grad)
    du_dx_auto_n = grad_n[:, 0:1].detach()

    diff = (du_dx_n - du_dx_auto_n).abs()
    print(f"N={N:3d} ({N*N:5d} pts): spectral-autodiff max={diff.max().item():.4e}, mean={diff.mean().item():.4e}")

# =============================================================================
# Test 4: Condition numbers of 2D diff matrices
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Condition numbers of 2D operators")
print("=" * 70)

for N in [10, 20, 30, 50]:
    Dx_n, Dy_n = build_2d_operators(N)
    # Condition number of Dx
    cond_Dx = np.linalg.cond(Dx_n)
    # Singular values of Dx
    svd = np.linalg.svd(Dx_n, compute_uv=False)
    print(f"N={N:3d}: cond(Dx)={cond_Dx:.2e}, sv_max={svd[0]:.2e}, sv_min={svd[-1]:.2e}, rank={np.sum(svd > 1e-10)}/{N*N}")

# =============================================================================
# Test 5: Gradient flow comparison
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: Gradient norm comparison (single step)")
print("=" * 70)

N = 50
Dx_np5, Dy_np5 = build_2d_operators(N)
xy5 = build_grid(N)

eps = 1e-10
x_coords = xy5[:, 0]
y_coords = xy5[:, 1]
is_boundary = (x_coords < eps) | (x_coords > 1-eps) | (y_coords < eps) | (y_coords > 1-eps)
is_interior = ~is_boundary
interior_idx = np.where(is_interior)[0]

Dx5 = torch.tensor(Dx_np5, dtype=torch.float32, device=device)
Dy5 = torch.tensor(Dy_np5, dtype=torch.float32, device=device)
xy5_t = torch.tensor(xy5, dtype=torch.float32, device=device)
x_t = xy5_t[:, 0:1]
y_t = xy5_t[:, 1:2]
d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))

for method in ['autodiff', 'dtpinn']:
    torch.manual_seed(42)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    optimizer.zero_grad()

    if method == 'autodiff':
        xy_int = xy5_t[interior_idx].clone().detach().requires_grad_(True)
        pred = model(xy_int)
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        grad_u = gradients(u, xy_int)
        grad_v = gradients(v, xy_int)
        du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
        dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
        x_c, y_c = xy_int[:, 0:1], xy_int[:, 1:2]
        d = torch.min(torch.min(x_c, 1.0 - x_c), torch.min(y_c, 1.0 - y_c))
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
    else:
        pred = model(xy5_t)
        u_all, v_all, p_all = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        du_dx = Dx5 @ u_all
        du_dy = Dy5 @ u_all
        dv_dx = Dx5 @ v_all
        dv_dy = Dy5 @ v_all
        Sxx, Syy = du_dx, dv_dy
        Sxy = 0.5 * (du_dy + dv_dx)
        S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
        nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag
        continuity = (du_dx + dv_dy)[interior_idx]
        u_conv = (u_all * du_dx + v_all * du_dy)[interior_idx]
        v_conv = (u_all * dv_dx + v_all * dv_dy)[interior_idx]
        dp_dx = (Dx5 @ p_all)[interior_idx]
        dp_dy = (Dy5 @ p_all)[interior_idx]
        visc_u = (Dx5 @ (nu_eff * du_dx) + Dy5 @ (nu_eff * du_dy))[interior_idx]
        visc_v = (Dx5 @ (nu_eff * dv_dx) + Dy5 @ (nu_eff * dv_dy))[interior_idx]
        mom_u = u_conv + dp_dx - visc_u
        mom_v = v_conv + dp_dy - visc_v

    loss = mse(continuity, torch.zeros_like(continuity)) + \
           mse(mom_u, torch.zeros_like(mom_u)) + \
           mse(mom_v, torch.zeros_like(mom_v))
    loss.backward()

    # Collect gradient norms per layer
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    total_grad = sum(g**2 for _, g in grad_norms)**0.5
    print(f"\n{method}:")
    print(f"  Loss: {loss.item():.6e}")
    print(f"  Total grad norm: {total_grad:.6e}")
    for name, gn in grad_norms[:4]:
        print(f"    {name}: {gn:.6e}")
    for name, gn in grad_norms[-2:]:
        print(f"    {name}: {gn:.6e}")

print("\nDone.")
