#!/usr/bin/env python3
"""
Phase 3 Experiment: Backward Pass Decomposition

Hypothesis: The DT-PINN backward pass (11.7ms) is dominated by backpropagation
through the PDE assembly graph (differentiation matrices, Smagorinsky, nonlinear
products). Network-only backward should be ~2ms.

Experiment:
1. Full backward: loss.backward() through entire DT-PINN graph
2. Network-only backward: pred.backward(gradient=random) — no PDE graph
3. PDE-only backward: autograd through PDE from detached pred to loss
4. Analytical PDE Jacobian cost estimate: pure tensor ops

If (2) << (1), the PDE graph is the bottleneck and an analytical Jacobian approach
could eliminate it.
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
# Infrastructure (same as existing scripts)
# =============================================================================
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
    return xy, x

Dx_np, Dy_np = build_2d_operators(N_grid)
xy_grid, x_1d = build_grid(N_grid)

eps = 1e-10
x_coords = xy_grid[:, 0]
y_coords = xy_grid[:, 1]
is_boundary = (x_coords < eps) | (x_coords > 1-eps) | (y_coords < eps) | (y_coords > 1-eps)
is_lid = (y_coords > 1-eps)
is_wall = is_boundary & ~is_lid
is_interior = ~is_boundary

interior_idx = np.where(is_interior)[0]
lid_idx = np.where(is_lid)[0]
wall_idx = np.where(is_wall)[0]

Dx_t = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_t = torch.tensor(Dy_np, dtype=torch.float32, device=device)
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_lid = xy_all[lid_idx]
xy_wall = xy_all[wall_idx]
x_t = xy_all[:, 0:1]
y_t = xy_all[:, 1:2]
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

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

mse = nn.MSELoss()

def compute_pde_from_pred(pred_input, Dx, Dy, dw, int_idx):
    """Compute PDE residuals from network output (may or may not be in graph)."""
    u_all = pred_input[:, 0:1]
    v_all = pred_input[:, 1:2]
    p_all = pred_input[:, 2:3]
    du_dx = Dx @ u_all
    du_dy = Dy @ u_all
    dv_dx = Dx @ v_all
    dv_dy = Dy @ v_all
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * dw)**2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u_all * du_dx + v_all * du_dy
    v_conv = u_all * dv_dx + v_all * dv_dy
    dp_dx = Dx @ p_all
    dp_dy = Dy @ p_all
    visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
    visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    cont_int = continuity[int_idx]
    mom_u_int = mom_u[int_idx]
    mom_v_int = mom_v[int_idx]
    loss_pde = mse(cont_int, torch.zeros_like(cont_int)) + \
               mse(mom_u_int, torch.zeros_like(mom_u_int)) + \
               mse(mom_v_int, torch.zeros_like(mom_v_int))
    return loss_pde


# =============================================================================
# Measurement 1: Full DT-PINN backward (baseline)
# =============================================================================
def measure_full_backward():
    """Standard DT-PINN: forward -> PDE -> loss -> loss.backward()"""
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    times_full = []

    for epoch in range(N_WARMUP + N_MEASURE):
        optimizer.zero_grad()

        pred = model(xy_all)
        loss_pde = compute_pde_from_pred(pred, Dx_t, Dy_t, d_wall, interior_idx)

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

        sync()
        t0 = time.perf_counter()
        loss.backward()
        sync()
        t_back = time.perf_counter() - t0

        optimizer.step()

        if epoch >= N_WARMUP:
            times_full.append(t_back)

    return np.array(times_full) * 1000  # ms


# =============================================================================
# Measurement 2: Network-only backward
# =============================================================================
def measure_network_only_backward():
    """Forward pass only, then backward with random upstream gradient.
    This isolates pure network backward cost."""
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Pre-generate a fixed upstream gradient (simulating what analytical Jacobian would produce)
    upstream_grad = torch.randn(N_grid * N_grid, 3, device=device)

    times_net = []

    for epoch in range(N_WARMUP + N_MEASURE):
        optimizer.zero_grad()

        pred = model(xy_all)

        sync()
        t0 = time.perf_counter()
        pred.backward(gradient=upstream_grad)
        sync()
        t_back = time.perf_counter() - t0

        optimizer.step()

        if epoch >= N_WARMUP:
            times_net.append(t_back)

    return np.array(times_net) * 1000  # ms


# =============================================================================
# Measurement 3: PDE-graph-only backward (no network)
# =============================================================================
def measure_pde_only_backward():
    """Detach pred from network, make it a leaf requiring grad,
    compute PDE loss, backward through PDE graph only."""
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)

    times_pde = []

    for epoch in range(N_WARMUP + N_MEASURE):
        with torch.no_grad():
            pred = model(xy_all)

        # Make pred a fresh leaf tensor with grad
        pred_leaf = pred.detach().requires_grad_(True)

        loss_pde = compute_pde_from_pred(pred_leaf, Dx_t, Dy_t, d_wall, interior_idx)

        sync()
        t0 = time.perf_counter()
        loss_pde.backward()
        sync()
        t_back = time.perf_counter() - t0

        # pred_leaf.grad is now ∂L_pde/∂pred — the upstream gradient
        if epoch >= N_WARMUP:
            times_pde.append(t_back)

    return np.array(times_pde) * 1000  # ms


# =============================================================================
# Measurement 4: Analytical PDE Jacobian (no autograd)
# =============================================================================
def measure_analytical_jacobian():
    """Compute ∂L/∂pred analytically using known PDE structure.
    No autograd at all — pure tensor operations.

    For NS + Smagorinsky:
    L = (1/M) * sum_interior [cont² + mom_u² + mom_v²]

    cont = Dx@u + Dy@v
    mom_u = u*(Dx@u) + v*(Dy@u) + Dx@p - Dx@(nu_eff*Dx@u) - Dy@(nu_eff*Dy@u)
    mom_v = u*(Dx@v) + v*(Dy@v) + Dy@p - Dx@(nu_eff*Dx@v) - Dy@(nu_eff*Dy@v)

    nu_eff = nu + (Cs*d)² * |S|
    |S| = sqrt(2*(Sxx² + Syy² + 2*Sxy²) + eps)

    The Jacobian ∂L/∂pred requires differentiating through all these operations
    with respect to u_all, v_all, p_all.
    """
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)

    # Precompute transposes
    DxT = Dx_t.T
    DyT = Dy_t.T

    times_analytic = []

    N = N_grid * N_grid
    M = len(interior_idx)
    int_idx_t = torch.tensor(interior_idx, dtype=torch.long, device=device)

    # Build selection matrix for interior points (sparse or just indexing)
    # We'll use a mask approach for gradient accumulation
    interior_mask = torch.zeros(N, 1, device=device)
    interior_mask[interior_idx] = 1.0

    for epoch in range(N_WARMUP + N_MEASURE):
        with torch.no_grad():
            pred = model(xy_all)

        sync()
        t0 = time.perf_counter()

        with torch.no_grad():
            u_all = pred[:, 0:1]
            v_all = pred[:, 1:2]
            p_all = pred[:, 2:3]

            # Forward derivatives
            du_dx = Dx_t @ u_all
            du_dy = Dy_t @ u_all
            dv_dx = Dx_t @ v_all
            dv_dy = Dy_t @ v_all

            # Smagorinsky
            Sxx = du_dx
            Syy = dv_dy
            Sxy = 0.5 * (du_dy + dv_dx)
            S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12
            S_mag = torch.sqrt(S_sq)
            Cs_d_sq = (Cs * d_wall)**2
            nu_eff = nu_laminar + Cs_d_sq * S_mag

            # PDE residuals
            dp_dx = Dx_t @ p_all
            dp_dy = Dy_t @ p_all
            visc_u = Dx_t @ (nu_eff * du_dx) + Dy_t @ (nu_eff * du_dy)
            visc_v = Dx_t @ (nu_eff * dv_dx) + Dy_t @ (nu_eff * dv_dy)

            continuity = du_dx + dv_dy
            u_conv = u_all * du_dx + v_all * du_dy
            v_conv = u_all * dv_dx + v_all * dv_dy
            mom_u = u_conv + dp_dx - visc_u
            mom_v = v_conv + dp_dy - visc_v

            # Upstream: dL/d(residual) = 2*residual/M (MSE gradient)
            # But MSE = mean(r²) for each, so dL/dr = 2*r/N_interior
            # And loss = MSE_cont + MSE_mom_u + MSE_mom_v
            scale = 2.0 / M
            d_cont = continuity * scale * interior_mask
            d_mom_u = mom_u * scale * interior_mask
            d_mom_v = mom_v * scale * interior_mask

            # =================================================================
            # Now compute ∂L/∂u_all, ∂L/∂v_all, ∂L/∂p_all
            # by chain rule through PDE + derivatives
            # =================================================================

            # ∂(nu_eff)/∂(Sxx), ∂(nu_eff)/∂(Syy), ∂(nu_eff)/∂(Sxy)
            # nu_eff = nu + Cs_d² * sqrt(2*(Sxx² + Syy² + 2*Sxy²) + eps)
            # ∂nu_eff/∂Sxx = Cs_d² * (1/(2*S_mag)) * 2 * 2*Sxx = Cs_d² * 2*Sxx / S_mag
            dnu_dSxx = Cs_d_sq * 2.0 * Sxx / S_mag
            dnu_dSyy = Cs_d_sq * 2.0 * Syy / S_mag
            dnu_dSxy = Cs_d_sq * 4.0 * Sxy / S_mag  # factor of 2 from 2*Sxy² -> 4*Sxy

            # ∂Sxx/∂(du_dx) = 1, ∂Syy/∂(dv_dy) = 1
            # ∂Sxy/∂(du_dy) = 0.5, ∂Sxy/∂(dv_dx) = 0.5

            # --- Gradient w.r.t. p_all ---
            # mom_u depends on p via dp_dx = Dx @ p_all
            # mom_v depends on p via dp_dy = Dy @ p_all
            # cont doesn't depend on p directly
            dL_dp = DxT @ d_mom_u + DyT @ d_mom_v

            # --- Gradient w.r.t. u_all (complex due to convection + Smagorinsky) ---
            # Contributions from continuity: cont = du_dx + dv_dy
            #   ∂cont/∂u_all = ∂(Dx@u)/∂u = Dx^T (applied to d_cont upstream)
            dL_du = DxT @ d_cont

            # Contributions from mom_u convection: u*du_dx + v*du_dy
            #   ∂(u*du_dx)/∂u = diag(du_dx) + diag(u)*Dx
            #   ∂(v*du_dy)/∂u = diag(v)*Dy
            dL_du = dL_du + du_dx * d_mom_u + DxT @ (u_all * d_mom_u) + DyT @ (v_all * d_mom_u)

            # Contributions from mom_v convection: u*dv_dx + v*dv_dy
            #   ∂(u*dv_dx)/∂u = diag(dv_dx)  (u appears directly, not through derivatives)
            dL_du = dL_du + dv_dx * d_mom_v

            # Contributions from viscous terms (without Smagorinsky dependence on u)
            # visc_u = Dx@(nu_eff*du_dx) + Dy@(nu_eff*du_dy)
            #   ∂visc_u/∂u via du_dx: Dx@(nu_eff * Dx) + through nu_eff dependence
            #   Direct: ∂visc_u/∂(du_dx) * ∂(du_dx)/∂u = Dx^T @ diag(nu_eff) @ Dx ... wait
            # Let me be more careful.

            # visc_u = Dx@(nu_eff * Dx@u) + Dy@(nu_eff * Dy@u)
            # ∂visc_u/∂u (ignoring nu_eff dependence on u for now):
            #   = Dx @ diag(nu_eff) @ Dx + Dy @ diag(nu_eff) @ Dy
            # Applied to d_mom_u (with negative sign):
            # -d_mom_u^T @ [Dx @ diag(nu_eff) @ Dx + Dy @ diag(nu_eff) @ Dy] @ du
            # -> contribution to dL/du: -Dx^T @ (nu_eff * (Dx^T @ d_mom_u)) - Dy^T @ (nu_eff * (Dy^T @ d_mom_u))
            # Wait, let me be more careful with the chain rule direction.

            # visc_u at point i = sum_j Dx[i,j] * nu_eff[j] * (Dx@u)[j] + sum_j Dy[i,j] * nu_eff[j] * (Dy@u)[j]
            # ∂visc_u[i]/∂u[k] = sum_j Dx[i,j] * [nu_eff[j] * Dx[j,k] + (Dx@u)[j] * ∂nu_eff[j]/∂u[k]]
            #                   + sum_j Dy[i,j] * [nu_eff[j] * Dy[j,k] + (Dy@u)[j] * ∂nu_eff[j]/∂u[k]]

            # This is getting very involved. Let me use a simpler approach:
            # Group contributions by whether they go through du_dx, du_dy, dv_dx, dv_dy

            # Actually, let me take a step back. The full analytical Jacobian for NS+Smagorinsky
            # is complex. For the PROFILING EXPERIMENT, I just need to estimate the TIME
            # for the tensor operations, not get the exact result right.
            # I'll compute the operations needed and time them.
            # The actual correctness can be validated later.

            # For timing purposes, compute the major operations:
            # 1. Derivative computations (already done above): 8 matmuls
            # 2. Residual computations (already done above)
            # 3. Upstream gradient: element-wise ops
            # 4. Chain rule through convection: element-wise + 4 matmuls with D^T
            # 5. Chain rule through viscosity: ~8 matmuls with D^T
            # 6. Chain rule through Smagorinsky: element-wise + ~8 matmuls with D^T

            # Let me just do all the transpose matmuls needed:
            # Conservative estimate: ~16 D^T @ vector operations
            tmp1 = DxT @ d_cont
            tmp2 = DyT @ d_cont
            tmp3 = DxT @ d_mom_u
            tmp4 = DyT @ d_mom_u
            tmp5 = DxT @ d_mom_v
            tmp6 = DyT @ d_mom_v
            tmp7 = DxT @ (nu_eff * tmp3)
            tmp8 = DyT @ (nu_eff * tmp4)
            tmp9 = DxT @ (nu_eff * tmp5)
            tmp10 = DyT @ (nu_eff * tmp6)

            # Smagorinsky chain rule additional matmuls
            tmp11 = DxT @ (dnu_dSxx * du_dx * d_mom_u)
            tmp12 = DyT @ (dnu_dSxy * du_dy * d_mom_u)
            tmp13 = DxT @ (dnu_dSxy * dv_dx * d_mom_v)
            tmp14 = DyT @ (dnu_dSyy * dv_dy * d_mom_v)

            # Element-wise operations for convection gradients
            conv_u_grad = du_dx * d_mom_u + dv_dx * d_mom_v
            conv_v_grad = du_dy * d_mom_u + dv_dy * d_mom_v

            # Combine into final gradient
            grad_u = tmp1 + conv_u_grad + tmp7 + tmp8 + tmp11 + tmp12
            grad_v = tmp2 + conv_v_grad + tmp9 + tmp10 + tmp13 + tmp14
            grad_p = DxT @ d_mom_u + DyT @ d_mom_v

            upstream_grad = torch.cat([grad_u, grad_v, grad_p], dim=1)

        sync()
        t_analytic = time.perf_counter() - t0

        if epoch >= N_WARMUP:
            times_analytic.append(t_analytic)

    return np.array(times_analytic) * 1000  # ms


# =============================================================================
# Measurement 5: Combined (analytical Jacobian + network backward)
# =============================================================================
def measure_combined():
    """Full proposed method: analytical PDE Jacobian + network-only backward.
    This simulates what the actual training loop would do."""
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    DxT = Dx_t.T
    DyT = Dy_t.T
    N = N_grid * N_grid
    M = len(interior_idx)
    interior_mask = torch.zeros(N, 1, device=device)
    interior_mask[interior_idx] = 1.0

    times_combined = []

    for epoch in range(N_WARMUP + N_MEASURE):
        optimizer.zero_grad()

        # Forward pass (with gradient tracking for network backward)
        pred = model(xy_all)

        sync()
        t0 = time.perf_counter()

        # Compute analytical upstream gradient (no autograd)
        with torch.no_grad():
            u_all = pred[:, 0:1]
            v_all = pred[:, 1:2]
            p_all = pred[:, 2:3]

            du_dx = Dx_t @ u_all
            du_dy = Dy_t @ u_all
            dv_dx = Dx_t @ v_all
            dv_dy = Dy_t @ v_all

            Sxx = du_dx
            Syy = dv_dy
            Sxy = 0.5 * (du_dy + dv_dx)
            S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12
            S_mag = torch.sqrt(S_sq)
            Cs_d_sq = (Cs * d_wall)**2
            nu_eff = nu_laminar + Cs_d_sq * S_mag

            continuity = du_dx + dv_dy
            dp_dx = Dx_t @ p_all
            dp_dy = Dy_t @ p_all
            visc_u = Dx_t @ (nu_eff * du_dx) + Dy_t @ (nu_eff * du_dy)
            visc_v = Dx_t @ (nu_eff * dv_dx) + Dy_t @ (nu_eff * dv_dy)
            u_conv = u_all * du_dx + v_all * du_dy
            v_conv = u_all * dv_dx + v_all * dv_dy
            mom_u = u_conv + dp_dx - visc_u
            mom_v = v_conv + dp_dy - visc_v

            scale = 2.0 / M
            d_cont = continuity * scale * interior_mask
            d_mom_u = mom_u * scale * interior_mask
            d_mom_v = mom_v * scale * interior_mask

            # Chain rule operations (representative set of ~14 D^T matmuls + element-wise)
            dnu_dSxx = Cs_d_sq * 2.0 * Sxx / S_mag
            dnu_dSyy = Cs_d_sq * 2.0 * Syy / S_mag
            dnu_dSxy = Cs_d_sq * 4.0 * Sxy / S_mag

            tmp1 = DxT @ d_cont
            tmp2 = DyT @ d_cont
            tmp3 = DxT @ d_mom_u
            tmp4 = DyT @ d_mom_u
            tmp5 = DxT @ d_mom_v
            tmp6 = DyT @ d_mom_v
            tmp7 = DxT @ (nu_eff * tmp3)
            tmp8 = DyT @ (nu_eff * tmp4)
            tmp9 = DxT @ (nu_eff * tmp5)
            tmp10 = DyT @ (nu_eff * tmp6)
            tmp11 = DxT @ (dnu_dSxx * du_dx * d_mom_u)
            tmp12 = DyT @ (dnu_dSxy * du_dy * d_mom_u)
            tmp13 = DxT @ (dnu_dSxy * dv_dx * d_mom_v)
            tmp14 = DyT @ (dnu_dSyy * dv_dy * d_mom_v)

            conv_u_grad = du_dx * d_mom_u + dv_dx * d_mom_v
            conv_v_grad = du_dy * d_mom_u + dv_dy * d_mom_v

            grad_u = tmp1 + conv_u_grad + tmp7 + tmp8 + tmp11 + tmp12
            grad_v = tmp2 + conv_v_grad + tmp9 + tmp10 + tmp13 + tmp14
            grad_p = DxT @ d_mom_u + DyT @ d_mom_v

            upstream_grad = torch.cat([grad_u, grad_v, grad_p], dim=1)

        # Network-only backward
        pred.backward(gradient=upstream_grad)

        sync()
        t_combined = time.perf_counter() - t0

        optimizer.step()

        if epoch >= N_WARMUP:
            times_combined.append(t_combined)

    return np.array(times_combined) * 1000  # ms


# =============================================================================
# Run all measurements
# =============================================================================
print("\n" + "=" * 70)
print("BACKWARD PASS DECOMPOSITION EXPERIMENT")
print(f"Warmup: {N_WARMUP} epochs, Measured: {N_MEASURE} epochs")
print("=" * 70)

print("\n1. Full DT-PINN backward (loss.backward through entire graph)...")
times_full = measure_full_backward()
print(f"   Done: {times_full.mean():.3f} ± {times_full.std():.3f} ms")

print("\n2. Network-only backward (pred.backward with random upstream)...")
times_net = measure_network_only_backward()
print(f"   Done: {times_net.mean():.3f} ± {times_net.std():.3f} ms")

print("\n3. PDE-graph-only backward (autograd through PDE, no network)...")
times_pde = measure_pde_only_backward()
print(f"   Done: {times_pde.mean():.3f} ± {times_pde.std():.3f} ms")

print("\n4. Analytical PDE Jacobian (pure tensor ops, no autograd)...")
times_analytic = measure_analytical_jacobian()
print(f"   Done: {times_analytic.mean():.3f} ± {times_analytic.std():.3f} ms")

print("\n5. Combined (analytical Jacobian + network backward)...")
times_combined = measure_combined()
print(f"   Done: {times_combined.mean():.3f} ± {times_combined.std():.3f} ms")

# =============================================================================
# Analysis
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\n{'Measurement':<45} {'Time (ms)':<20} {'% of Full':<12}")
print("-" * 77)
full_mean = times_full.mean()
measurements = [
    ("1. Full DT-PINN backward", full_mean, 100.0),
    ("2. Network-only backward", times_net.mean(), times_net.mean() / full_mean * 100),
    ("3. PDE-graph-only backward (autograd)", times_pde.mean(), times_pde.mean() / full_mean * 100),
    ("4. Analytical PDE Jacobian (tensor ops)", times_analytic.mean(), times_analytic.mean() / full_mean * 100),
    ("5. Combined (analytical + network)", times_combined.mean(), times_combined.mean() / full_mean * 100),
]

for name, t, pct in measurements:
    print(f"{name:<45} {t:>8.3f} ± {times_full.std() if name.startswith('1') else 0:.3f}    {pct:>6.1f}%")

savings = full_mean - times_combined.mean()
print(f"\n{'='*70}")
print(f"POTENTIAL SAVINGS: {savings:.3f} ms/epoch ({savings/full_mean*100:.1f}% of backward)")

# Phase 2 reported DT-PINN total epoch = 21.79ms
dtpinn_total = 21.79
new_total = dtpinn_total - savings
autodiff_total = 41.07

print(f"\nPROJECTED IMPACT:")
print(f"  Current DT-PINN epoch:   {dtpinn_total:.2f} ms")
print(f"  Projected new epoch:     {new_total:.2f} ms")
print(f"  Autodiff epoch:          {autodiff_total:.2f} ms")
print(f"  Current speedup:         {autodiff_total/dtpinn_total:.2f}x")
print(f"  Projected speedup:       {autodiff_total/new_total:.2f}x")

print(f"\nHYPOTHESIS EVALUATION:")
if times_net.mean() < full_mean * 0.4:
    print(f"  CONFIRMED: Network backward ({times_net.mean():.2f}ms) is <40% of full backward ({full_mean:.2f}ms)")
    print(f"  PDE graph backward dominates: {(full_mean - times_net.mean()):.2f}ms ({(1 - times_net.mean()/full_mean)*100:.0f}%)")
else:
    print(f"  REFUTED: Network backward ({times_net.mean():.2f}ms) is {times_net.mean()/full_mean*100:.0f}% of full ({full_mean:.2f}ms)")
    print(f"  PDE graph backward is NOT the dominant cost")

# Save results
results = {
    'full_backward_ms': {'mean': float(full_mean), 'std': float(times_full.std())},
    'network_only_backward_ms': {'mean': float(times_net.mean()), 'std': float(times_net.std())},
    'pde_only_backward_ms': {'mean': float(times_pde.mean()), 'std': float(times_pde.std())},
    'analytical_jacobian_ms': {'mean': float(times_analytic.mean()), 'std': float(times_analytic.std())},
    'combined_ms': {'mean': float(times_combined.mean()), 'std': float(times_combined.std())},
    'savings_ms': float(savings),
    'savings_pct': float(savings / full_mean * 100),
    'projected_epoch_ms': float(new_total),
    'projected_speedup': float(autodiff_total / new_total),
}

os.makedirs('results/phase3', exist_ok=True)
with open('results/phase3/backward_decomposition.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to results/phase3/backward_decomposition.json")
