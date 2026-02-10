#!/usr/bin/env python3
"""
Analytical PDE Jacobian for Navier-Stokes + Smagorinsky with DT-PINN.

Computes ∂L_pde/∂pred analytically (no autograd through PDE) and validates
against autograd ground truth.

The PDE system:
  cont = Dx@u + Dy@v
  mom_u = u*(Dx@u) + v*(Dy@u) + Dx@p - Dx@(ν_eff*Dx@u) - Dy@(ν_eff*Dy@u)
  mom_v = u*(Dx@v) + v*(Dy@v) + Dy@p - Dx@(ν_eff*Dx@v) - Dy@(ν_eff*Dy@v)

  ν_eff = ν + (Cs*d)² * |S|
  |S| = sqrt(2*(Sxx² + Syy² + 2*Sxy²) + ε)
  Sxx = Dx@u, Syy = Dy@v, Sxy = 0.5*(Dy@u + Dx@v)

The Jacobian ∂L_pde/∂pred is computed via VJP (vector-Jacobian product) form,
where the upstream vector is the MSE gradient 2*residual/M.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import json

SEED = 42
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
is_interior = ~is_boundary

interior_idx = np.where(is_interior)[0]

Dx = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy = torch.tensor(Dy_np, dtype=torch.float32, device=device)
DxT = Dx.T.contiguous()
DyT = Dy.T.contiguous()
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
x_t = xy_all[:, 0:1]
y_t = xy_all[:, 1:2]
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

mse_fn = nn.MSELoss()


def autograd_pde_gradient(model, xy, Dx_mat, Dy_mat, dw, int_idx):
    """Compute ∂L_pde/∂pred using autograd (ground truth)."""
    pred = model(xy)
    pred_leaf = pred.detach().requires_grad_(True)

    u = pred_leaf[:, 0:1]
    v = pred_leaf[:, 1:2]
    p = pred_leaf[:, 2:3]

    du_dx = Dx_mat @ u
    du_dy = Dy_mat @ u
    dv_dx = Dx_mat @ v
    dv_dy = Dy_mat @ v

    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * dw)**2 * S_mag

    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    dp_dx = Dx_mat @ p
    dp_dy = Dy_mat @ p
    visc_u = Dx_mat @ (nu_eff * du_dx) + Dy_mat @ (nu_eff * du_dy)
    visc_v = Dx_mat @ (nu_eff * dv_dx) + Dy_mat @ (nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    cont_int = continuity[int_idx]
    mom_u_int = mom_u[int_idx]
    mom_v_int = mom_v[int_idx]

    loss_pde = mse_fn(cont_int, torch.zeros_like(cont_int)) + \
               mse_fn(mom_u_int, torch.zeros_like(mom_u_int)) + \
               mse_fn(mom_v_int, torch.zeros_like(mom_v_int))

    grad_pred = torch.autograd.grad(loss_pde, pred_leaf)[0]

    return grad_pred, pred.detach(), loss_pde.item()


def analytical_pde_gradient(pred_detached, Dx_mat, Dy_mat, DxT_mat, DyT_mat,
                             dw, Cs_d_sq_val, int_idx, int_mask, M_int):
    """
    Compute ∂L_pde/∂pred analytically using VJP through PDE structure.

    All operations are pure tensor ops (no autograd).

    Derivation:
    L = MSE(cont[int],0) + MSE(mom_u[int],0) + MSE(mom_v[int],0)

    VJP: ∂L/∂pred_k = Σ_r (∂r/∂pred_k)^T @ (2/M * r * mask)
    where r ∈ {cont, mom_u, mom_v}

    For each output u, v, p, we compute the VJP through all PDE terms.
    """
    u = pred_detached[:, 0:1]
    v = pred_detached[:, 1:2]
    p = pred_detached[:, 2:3]

    # Forward: compute derivatives
    du_dx = Dx_mat @ u
    du_dy = Dy_mat @ u
    dv_dx = Dx_mat @ v
    dv_dy = Dy_mat @ v

    # Smagorinsky quantities
    Sxx = du_dx
    Syy = dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12
    S_mag = torch.sqrt(S_sq)
    nu_eff = nu_laminar + Cs_d_sq_val * S_mag

    # PDE residuals
    continuity = du_dx + dv_dy
    dp_dx = Dx_mat @ p
    dp_dy = Dy_mat @ p
    visc_u = Dx_mat @ (nu_eff * du_dx) + Dy_mat @ (nu_eff * du_dy)
    visc_v = Dx_mat @ (nu_eff * dv_dx) + Dy_mat @ (nu_eff * dv_dy)
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    # Upstream gradients (MSE gradient, masked to interior)
    scale = 2.0 / M_int
    dc = continuity * scale * int_mask       # δ_cont
    dmu = mom_u * scale * int_mask           # δ_mom_u
    dmv = mom_v * scale * int_mask           # δ_mom_v

    # ∂ν_eff/∂Sxx, ∂ν_eff/∂Syy, ∂ν_eff/∂Sxy (for Smagorinsky chain rule)
    # ν_eff = ν + Cs_d² * sqrt(2*(Sxx² + Syy² + 2*Sxy²) + ε)
    # ∂ν_eff/∂Sxx = Cs_d² * 2*Sxx / |S|
    # ∂ν_eff/∂Syy = Cs_d² * 2*Syy / |S|
    # ∂ν_eff/∂Sxy = Cs_d² * 4*Sxy / |S|
    inv_S = 1.0 / S_mag
    dnu_dSxx = Cs_d_sq_val * 2.0 * Sxx * inv_S
    dnu_dSyy = Cs_d_sq_val * 2.0 * Syy * inv_S
    dnu_dSxy = Cs_d_sq_val * 4.0 * Sxy * inv_S

    # ∂ν_eff/∂u = dnu_dSxx * Dx + 0.5*dnu_dSxy * Dy  (via ∂Sxx/∂u = Dx, ∂Sxy/∂u = 0.5*Dy)
    # ∂ν_eff/∂v = dnu_dSyy * Dy + 0.5*dnu_dSxy * Dx  (via ∂Syy/∂v = Dy, ∂Sxy/∂v = 0.5*Dx)

    # Define convenience: α, β for ∂ν_eff/∂u = diag(α)@Dx + diag(β)@Dy
    alpha_u = dnu_dSxx               # coefficient of Dx in ∂ν_eff/∂u
    beta_u = 0.5 * dnu_dSxy          # coefficient of Dy in ∂ν_eff/∂u
    alpha_v = 0.5 * dnu_dSxy         # coefficient of Dx in ∂ν_eff/∂v
    beta_v = dnu_dSyy                # coefficient of Dy in ∂ν_eff/∂v

    # =====================================================================
    # ∂L/∂p: Only mom_u and mom_v depend on p through pressure gradient
    # mom_u has dp_dx = Dx@p, mom_v has dp_dy = Dy@p
    # VJP: DxT @ dmu + DyT @ dmv
    # =====================================================================
    dL_dp = DxT_mat @ dmu + DyT_mat @ dmv

    # =====================================================================
    # ∂L/∂u: Contributions from continuity, convection, viscosity
    # =====================================================================

    # (A) Continuity: cont = Dx@u + Dy@v → ∂cont/∂u = Dx → VJP: DxT @ dc
    dL_du = DxT_mat @ dc

    # (B) mom_u convection: u*(Dx@u) + v*(Dy@u)
    #   ∂/∂u of u*(Dx@u): J = diag(Dx@u) + diag(u)@Dx → VJP: (Dx@u)⊙dmu + DxT@(u⊙dmu)
    #   ∂/∂u of v*(Dy@u): J = diag(v)@Dy → VJP: DyT@(v⊙dmu)
    dL_du = dL_du + du_dx * dmu + DxT_mat @ (u * dmu) + DyT_mat @ (v * dmu)

    # (C) mom_v convection: u*(Dx@v) + v*(Dy@v)
    #   ∂/∂u of u*(Dx@v): J = diag(Dx@v) → VJP: (Dx@v)⊙dmv
    #   (v*(Dy@v) doesn't depend on u)
    dL_du = dL_du + dv_dx * dmv

    # (D) Viscous terms for mom_u: -Dx@(ν_eff*Dx@u) - Dy@(ν_eff*Dy@u)
    #   The negative sign is already in the mom_u residual.
    #   For VJP through -visc_u w.r.t. u, the upstream is -dmu (from chain rule).
    #
    #   visc_u = Dx@(ν_eff * Dx@u) + Dy@(ν_eff * Dy@u)
    #   ∂visc_u/∂u involves both direct (Dx,Dy through ν_eff-weighted) and
    #   indirect (ν_eff depends on u through strain rates).
    #
    #   VJP: w_x = DxT @ δ, w_y = DyT @ δ  where δ = -dmu
    #   Direct part: DxT@[(ν_eff)⊙w_x] + DyT@[(ν_eff)⊙w_y]
    #   Smagorinsky part (∂ν_eff/∂u contributes):
    #     Need: (du_dx ⊙ w_x + du_dy ⊙ w_y) — weighted by dν/du chain
    neg_dmu = -dmu
    w_x_u = DxT_mat @ neg_dmu
    w_y_u = DyT_mat @ neg_dmu

    # Direct viscous VJP (ν_eff is constant w.r.t. u — the "linear" part)
    dL_du = dL_du + DxT_mat @ (nu_eff * w_x_u) + DyT_mat @ (nu_eff * w_y_u)

    # Smagorinsky chain rule for visc_u w.r.t. u:
    # The ∂ν_eff/∂u contribution to visc_u:
    # Each point j contributes: (Dx@u)[j]*w_x[j] + (Dy@u)[j]*w_y[j] to the
    # "sensitivity of visc_u to ν_eff at point j", which then chains through
    # ∂ν_eff[j]/∂u[k] = alpha_u[j]*Dx[j,k] + beta_u[j]*Dy[j,k]
    gamma_u = du_dx * w_x_u + du_dy * w_y_u   # how much visc_u cares about ν_eff through u
    dL_du = dL_du + DxT_mat @ (alpha_u * gamma_u) + DyT_mat @ (beta_u * gamma_u)

    # (E) Viscous terms for mom_v: -Dx@(ν_eff*Dx@v) - Dy@(ν_eff*Dy@v)
    #   ν_eff depends on u through strain rates, but Dx@v, Dy@v don't.
    #   VJP of visc_v w.r.t. u: only through ν_eff.
    neg_dmv = -dmv
    w_x_v = DxT_mat @ neg_dmv
    w_y_v = DyT_mat @ neg_dmv

    gamma_v = dv_dx * w_x_v + dv_dy * w_y_v
    dL_du = dL_du + DxT_mat @ (alpha_u * gamma_v) + DyT_mat @ (beta_u * gamma_v)

    # =====================================================================
    # ∂L/∂v: Contributions from continuity, convection, viscosity
    # =====================================================================

    # (A) Continuity: cont = Dx@u + Dy@v → ∂cont/∂v = Dy → VJP: DyT @ dc
    dL_dv = DyT_mat @ dc

    # (B) mom_u convection: u*(Dx@u) + v*(Dy@u)
    #   ∂/∂v of v*(Dy@u): J = diag(Dy@u) → VJP: (Dy@u)⊙dmu
    dL_dv = dL_dv + du_dy * dmu

    # (C) mom_v convection: u*(Dx@v) + v*(Dy@v)
    #   ∂/∂v of u*(Dx@v): J = diag(u)@Dx → VJP: DxT@(u⊙dmv)
    #   ∂/∂v of v*(Dy@v): J = diag(Dy@v) + diag(v)@Dy → VJP: (Dy@v)⊙dmv + DyT@(v⊙dmv)
    dL_dv = dL_dv + DxT_mat @ (u * dmv) + dv_dy * dmv + DyT_mat @ (v * dmv)

    # (D) Viscous terms for mom_v: -Dx@(ν_eff*Dx@v) - Dy@(ν_eff*Dy@v)
    #   Direct part (ν_eff treated as constant w.r.t. v is WRONG — ν_eff depends on v through Syy, Sxy)
    #   But for the Dx@v, Dy@v terms treated linearly in v:
    dL_dv = dL_dv + DxT_mat @ (nu_eff * w_x_v) + DyT_mat @ (nu_eff * w_y_v)

    # Smagorinsky chain rule for visc_v w.r.t. v:
    # gamma_v already computed = (Dx@v)⊙w_x_v + (Dy@v)⊙w_y_v
    dL_dv = dL_dv + DxT_mat @ (alpha_v * gamma_v) + DyT_mat @ (beta_v * gamma_v)

    # (E) Viscous terms for mom_u w.r.t. v (ν_eff depends on v through Syy, Sxy)
    # gamma_u already computed = (Dx@u)⊙w_x_u + (Dy@u)⊙w_y_u
    dL_dv = dL_dv + DxT_mat @ (alpha_v * gamma_u) + DyT_mat @ (beta_v * gamma_u)

    # Combine
    grad_pred = torch.cat([dL_du, dL_dv, dL_dp], dim=1)
    return grad_pred


# =============================================================================
# Validation
# =============================================================================
print("\n" + "=" * 70)
print("ANALYTICAL PDE JACOBIAN VALIDATION")
print("=" * 70)

torch.manual_seed(SEED)
model = PINN_Cavity().to(device)

# Test at multiple training states
states = [
    ("Untrained (epoch 0)", 0),
    ("After 10 epochs", 10),
    ("After 100 epochs", 100),
    ("After 500 epochs", 500),
]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

errors = []

for state_name, target_epoch in states:
    # Train to target epoch
    current_epoch = 0
    if target_epoch > 0:
        for ep in range(target_epoch - (0 if state_name == states[0][0] else states[states.index((state_name, target_epoch)) - 1][1])):
            optimizer.zero_grad()
            pred = model(xy_all)
            u_all = pred[:, 0:1]
            v_all = pred[:, 1:2]
            p_all = pred[:, 2:3]
            du_dx_t = Dx @ u_all
            du_dy_t = Dy @ u_all
            dv_dx_t = Dx @ v_all
            dv_dy_t = Dy @ v_all
            Sxx_t = du_dx_t
            Syy_t = dv_dy_t
            Sxy_t = 0.5 * (du_dy_t + dv_dx_t)
            S_mag_t = torch.sqrt(2.0 * (Sxx_t**2 + Syy_t**2 + 2.0 * Sxy_t**2) + 1e-12)
            nu_eff_t = nu_laminar + (Cs * d_wall)**2 * S_mag_t
            cont_t = du_dx_t + dv_dy_t
            u_conv_t = u_all * du_dx_t + v_all * du_dy_t
            v_conv_t = u_all * dv_dx_t + v_all * dv_dy_t
            dp_dx_t = Dx @ p_all
            dp_dy_t = Dy @ p_all
            visc_u_t = Dx @ (nu_eff_t * du_dx_t) + Dy @ (nu_eff_t * du_dy_t)
            visc_v_t = Dx @ (nu_eff_t * dv_dx_t) + Dy @ (nu_eff_t * dv_dy_t)
            mom_u_t = u_conv_t + dp_dx_t - visc_u_t
            mom_v_t = v_conv_t + dp_dy_t - visc_v_t
            loss = mse_fn(cont_t[interior_idx], torch.zeros_like(cont_t[interior_idx])) + \
                   mse_fn(mom_u_t[interior_idx], torch.zeros_like(mom_u_t[interior_idx])) + \
                   mse_fn(mom_v_t[interior_idx], torch.zeros_like(mom_v_t[interior_idx]))
            loss.backward()
            optimizer.step()

    # Compute autograd gradient
    grad_auto, pred_det, loss_val = autograd_pde_gradient(
        model, xy_all, Dx, Dy, d_wall, interior_idx
    )

    # Compute analytical gradient
    grad_analytic = analytical_pde_gradient(
        pred_det, Dx, Dy, DxT, DyT, d_wall, Cs_d_sq, interior_idx, interior_mask, M
    )

    # Compare
    diff = (grad_auto - grad_analytic).abs()
    rel_err = diff / (grad_auto.abs() + 1e-10)

    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()

    # Per-component errors
    for i, name in enumerate(['u', 'v', 'p']):
        comp_diff = diff[:, i]
        comp_auto = grad_auto[:, i].abs()
        comp_rel = comp_diff / (comp_auto + 1e-10)
        print(f"  {name}: max_abs={comp_diff.max().item():.2e}, mean_abs={comp_diff.mean().item():.2e}, "
              f"max_rel={comp_rel.max().item():.2e}, mean_rel={comp_rel.mean().item():.2e}, "
              f"auto_norm={grad_auto[:, i].norm().item():.4e}")

    cosine_sim = torch.nn.functional.cosine_similarity(
        grad_auto.flatten().unsqueeze(0),
        grad_analytic.flatten().unsqueeze(0)
    ).item()

    error_record = {
        'state': state_name,
        'loss': loss_val,
        'max_abs_err': max_abs_err,
        'mean_abs_err': mean_abs_err,
        'max_rel_err': max_rel_err,
        'mean_rel_err': mean_rel_err,
        'cosine_similarity': cosine_sim,
        'auto_norm': grad_auto.norm().item(),
        'analytic_norm': grad_analytic.norm().item(),
    }
    errors.append(error_record)

    print(f"\n{state_name} (loss={loss_val:.6f}):")
    print(f"  Max absolute error: {max_abs_err:.2e}")
    print(f"  Mean absolute error: {mean_abs_err:.2e}")
    print(f"  Max relative error: {max_rel_err:.2e}")
    print(f"  Mean relative error: {mean_rel_err:.2e}")
    print(f"  Cosine similarity: {cosine_sim:.8f}")
    print(f"  Autograd norm: {grad_auto.norm().item():.4e}")
    print(f"  Analytical norm: {grad_analytic.norm().item():.4e}")
    print(f"  Norm ratio: {grad_analytic.norm().item() / (grad_auto.norm().item() + 1e-10):.6f}")

    match = cosine_sim > 0.9999
    print(f"  MATCH: {'YES' if match else 'NO'}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

all_match = all(e['cosine_similarity'] > 0.9999 for e in errors)
print(f"\nAll states match (cosine > 0.9999): {'YES' if all_match else 'NO'}")

for e in errors:
    status = "PASS" if e['cosine_similarity'] > 0.9999 else "FAIL"
    print(f"  [{status}] {e['state']}: cosine={e['cosine_similarity']:.8f}, "
          f"max_rel={e['max_rel_err']:.2e}")

# Save results
os.makedirs('results/phase3', exist_ok=True)
with open('results/phase3/jacobian_validation.json', 'w') as f:
    json.dump(errors, f, indent=2)
print(f"\nResults saved to results/phase3/jacobian_validation.json")
