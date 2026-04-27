"""
Experiment 1: Derivative Mismatch Diagnostic

Tests the hypothesis that spectral D-matrix derivatives differ significantly
from autograd derivatives for the trained DT-PINN neural network.

Loads the trained DT-PINN model. At the Chebyshev grid points, computes PDE
residuals TWO ways:
  (a) Spectral: u_x = Dx @ NN_values (what DT-PINN trained on)
  (b) Autograd: u_x = d/dx NN(x) (what evaluation uses / "true" NN derivative)

If (a) is small but (b) is large, the derivative mismatch is the root cause.
Also computes raw derivative differences to quantify the gap.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sage_partner_ns import (
    FNN_NS, build_3d_grid, compute_pde_ns_3d, NU, V0,
    X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX,
)

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# =============================================================================
# 1. Load trained DT-PINN model
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Derivative Mismatch Test")
print("=" * 70)

model = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
ckpt = torch.load('results/sage_partner/model_ns_dtpinn.pt', map_location=device, weights_only=True)
model.load_state_dict(ckpt)
model.eval()
print(f"Loaded DT-PINN model: {sum(p.numel() for p in model.parameters())} params")


# =============================================================================
# 2. Build Chebyshev grid (same as training: Nx=55, Ny=15, Nt=30)
# =============================================================================
print("\nBuilding Chebyshev grid (Nx=55, Ny=15, Nt=30)...")
g = build_3d_grid(55, 15, 30, device)
N_all = g['N_all']
ii = g['interior_idx']
print(f"Total points: {N_all}, Interior points: {len(ii)}")


# =============================================================================
# 3a. Spectral derivatives: Dx @ NN_values
# =============================================================================
print("\n--- Computing SPECTRAL derivatives (Dx @ NN_values) ---")

with torch.no_grad():
    pred = model(g['xyt_all'])  # (N_all, 3)
    u_sp = pred[:, 0:1]
    v_sp = pred[:, 1:2]
    p_sp = pred[:, 2:3]

    # First derivatives via D matrices
    u_x_sp = torch.sparse.mm(g['Dx'], u_sp)
    u_y_sp = torch.sparse.mm(g['Dy'], u_sp)
    u_t_sp = torch.sparse.mm(g['Dt'], u_sp)
    v_x_sp = torch.sparse.mm(g['Dx'], v_sp)
    v_y_sp = torch.sparse.mm(g['Dy'], v_sp)
    v_t_sp = torch.sparse.mm(g['Dt'], v_sp)
    p_x_sp = torch.sparse.mm(g['Dx'], p_sp)
    p_y_sp = torch.sparse.mm(g['Dy'], p_sp)

    # Second derivatives via D² matrices
    u_xx_sp = torch.sparse.mm(g['Dxx'], u_sp)
    u_yy_sp = torch.sparse.mm(g['Dyy'], u_sp)
    v_xx_sp = torch.sparse.mm(g['Dxx'], v_sp)
    v_yy_sp = torch.sparse.mm(g['Dyy'], v_sp)

    # PDE residuals (spectral)
    cont_sp = u_x_sp + v_y_sp
    mom_u_sp = u_t_sp + u_sp * u_x_sp + v_sp * u_y_sp + p_x_sp - NU * (u_xx_sp + u_yy_sp)
    mom_v_sp = v_t_sp + u_sp * v_x_sp + v_sp * v_y_sp + p_y_sp - NU * (v_xx_sp + v_yy_sp)


# =============================================================================
# 3b. Autograd derivatives: d/dx NN(x)
# =============================================================================
print("--- Computing AUTOGRAD derivatives (d/dx NN(x)) ---")

xyt = g['xyt_all'].detach().requires_grad_(True)
pred_ag = model(xyt)
u_ag, v_ag, p_ag = pred_ag[:, 0:1], pred_ag[:, 1:2], pred_ag[:, 2:3]

# First derivatives
grad_u = torch.autograd.grad(u_ag.sum(), xyt, create_graph=True)[0]
grad_v = torch.autograd.grad(v_ag.sum(), xyt, create_graph=True)[0]
grad_p = torch.autograd.grad(p_ag.sum(), xyt, create_graph=True)[0]

u_x_ag, u_y_ag, u_t_ag = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
v_x_ag, v_y_ag, v_t_ag = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
p_x_ag, p_y_ag = grad_p[:, 0:1], grad_p[:, 1:2]

# Second derivatives
grad_u_x = torch.autograd.grad(u_x_ag.sum(), xyt, create_graph=False, retain_graph=True)[0]
grad_u_y = torch.autograd.grad(u_y_ag.sum(), xyt, create_graph=False, retain_graph=True)[0]
grad_v_x = torch.autograd.grad(v_x_ag.sum(), xyt, create_graph=False, retain_graph=True)[0]
grad_v_y = torch.autograd.grad(v_y_ag.sum(), xyt, create_graph=False, retain_graph=True)[0]

u_xx_ag = grad_u_x[:, 0:1]
u_yy_ag = grad_u_y[:, 1:2]
v_xx_ag = grad_v_x[:, 0:1]
v_yy_ag = grad_v_y[:, 1:2]

# PDE residuals (autograd)
cont_ag = u_x_ag + v_y_ag
mom_u_ag = u_t_ag + u_ag * u_x_ag + v_ag * u_y_ag + p_x_ag - NU * (u_xx_ag + u_yy_ag)
mom_v_ag = v_t_ag + u_ag * v_x_ag + v_ag * v_y_ag + p_y_ag - NU * (v_xx_ag + v_yy_ag)

# Detach for comparison
cont_ag = cont_ag.detach()
mom_u_ag = mom_u_ag.detach()
mom_v_ag = mom_v_ag.detach()

u_x_ag = u_x_ag.detach(); u_y_ag = u_y_ag.detach(); u_t_ag = u_t_ag.detach()
v_x_ag = v_x_ag.detach(); v_y_ag = v_y_ag.detach(); v_t_ag = v_t_ag.detach()
p_x_ag = p_x_ag.detach(); p_y_ag = p_y_ag.detach()
u_xx_ag = u_xx_ag.detach(); u_yy_ag = u_yy_ag.detach()
v_xx_ag = v_xx_ag.detach(); v_yy_ag = v_yy_ag.detach()


# =============================================================================
# 4. Compare PDE residuals: spectral vs autograd
# =============================================================================
print("\n" + "=" * 70)
print("PDE RESIDUALS: Spectral vs Autograd at Chebyshev points")
print("=" * 70)

# Interior points only (what training loss uses)
print("\n--- At INTERIOR points only (what training optimizes) ---")
for name, sp, ag in [
    ("Continuity", cont_sp, cont_ag),
    ("Momentum-u", mom_u_sp, mom_u_ag),
    ("Momentum-v", mom_v_sp, mom_v_ag),
]:
    sp_int = sp[ii].cpu().numpy().flatten()
    ag_int = ag[ii].cpu().numpy().flatten()
    sp_rms = np.sqrt(np.mean(sp_int ** 2))
    ag_rms = np.sqrt(np.mean(ag_int ** 2))
    print(f"  {name:12s}:  Spectral RMS = {sp_rms:.6f}  |  Autograd RMS = {ag_rms:.6f}  |  Ratio = {ag_rms/max(sp_rms,1e-15):.2f}x")

# Combined PDE RMS
sp_all = np.concatenate([cont_sp[ii].cpu().numpy().flatten(),
                          mom_u_sp[ii].cpu().numpy().flatten(),
                          mom_v_sp[ii].cpu().numpy().flatten()])
ag_all = np.concatenate([cont_ag[ii].cpu().numpy().flatten(),
                          mom_u_ag[ii].cpu().numpy().flatten(),
                          mom_v_ag[ii].cpu().numpy().flatten()])
print(f"\n  Combined PDE:  Spectral RMS = {np.sqrt(np.mean(sp_all**2)):.6f}  |  Autograd RMS = {np.sqrt(np.mean(ag_all**2)):.6f}")

# All points (including boundary)
print("\n--- At ALL points (including boundary) ---")
for name, sp, ag in [
    ("Continuity", cont_sp, cont_ag),
    ("Momentum-u", mom_u_sp, mom_u_ag),
    ("Momentum-v", mom_v_sp, mom_v_ag),
]:
    sp_np = sp.cpu().numpy().flatten()
    ag_np = ag.cpu().numpy().flatten()
    sp_rms = np.sqrt(np.mean(sp_np ** 2))
    ag_rms = np.sqrt(np.mean(ag_np ** 2))
    print(f"  {name:12s}:  Spectral RMS = {sp_rms:.6f}  |  Autograd RMS = {ag_rms:.6f}  |  Ratio = {ag_rms/max(sp_rms,1e-15):.2f}x")


# =============================================================================
# 5. Raw derivative differences: |Dx @ NN - d/dx NN|
# =============================================================================
print("\n" + "=" * 70)
print("RAW DERIVATIVE DIFFERENCES: |Spectral - Autograd|")
print("=" * 70)

print("\n--- First derivatives (all points) ---")
for name, sp_d, ag_d in [
    ("u_x", u_x_sp, u_x_ag), ("u_y", u_y_sp, u_y_ag), ("u_t", u_t_sp, u_t_ag),
    ("v_x", v_x_sp, v_x_ag), ("v_y", v_y_sp, v_y_ag), ("v_t", v_t_sp, v_t_ag),
    ("p_x", p_x_sp, p_x_ag), ("p_y", p_y_sp, p_y_ag),
]:
    diff = (sp_d - ag_d).abs().cpu().numpy().flatten()
    ag_mag = ag_d.abs().cpu().numpy().flatten()
    print(f"  {name:4s}:  mean|diff| = {diff.mean():.6f}  max|diff| = {diff.max():.6f}  "
          f"mean|ag| = {ag_mag.mean():.6f}  relative_err = {diff.mean()/max(ag_mag.mean(), 1e-15):.4f}")

print("\n--- Second derivatives (all points) ---")
for name, sp_d, ag_d in [
    ("u_xx", u_xx_sp, u_xx_ag), ("u_yy", u_yy_sp, u_yy_ag),
    ("v_xx", v_xx_sp, v_xx_ag), ("v_yy", v_yy_sp, v_yy_ag),
]:
    diff = (sp_d - ag_d).abs().cpu().numpy().flatten()
    ag_mag = ag_d.abs().cpu().numpy().flatten()
    print(f"  {name:4s}:  mean|diff| = {diff.mean():.6f}  max|diff| = {diff.max():.6f}  "
          f"mean|ag| = {ag_mag.mean():.6f}  relative_err = {diff.mean()/max(ag_mag.mean(), 1e-15):.4f}")

print("\n--- First derivatives (interior only) ---")
for name, sp_d, ag_d in [
    ("u_x", u_x_sp, u_x_ag), ("u_y", u_y_sp, u_y_ag), ("u_t", u_t_sp, u_t_ag),
    ("v_x", v_x_sp, v_x_ag), ("v_y", v_y_sp, v_y_ag), ("v_t", v_t_sp, v_t_ag),
    ("p_x", p_x_sp, p_x_ag), ("p_y", p_y_sp, p_y_ag),
]:
    diff = (sp_d[ii] - ag_d[ii]).abs().cpu().numpy().flatten()
    ag_mag = ag_d[ii].abs().cpu().numpy().flatten()
    print(f"  {name:4s}:  mean|diff| = {diff.mean():.6f}  max|diff| = {diff.max():.6f}  "
          f"mean|ag| = {ag_mag.mean():.6f}  relative_err = {diff.mean()/max(ag_mag.mean(), 1e-15):.4f}")

print("\n--- Second derivatives (interior only) ---")
for name, sp_d, ag_d in [
    ("u_xx", u_xx_sp, u_xx_ag), ("u_yy", u_yy_sp, u_yy_ag),
    ("v_xx", v_xx_sp, v_xx_ag), ("v_yy", v_yy_sp, v_yy_ag),
]:
    diff = (sp_d[ii] - ag_d[ii]).abs().cpu().numpy().flatten()
    ag_mag = ag_d[ii].abs().cpu().numpy().flatten()
    print(f"  {name:4s}:  mean|diff| = {diff.mean():.6f}  max|diff| = {diff.max():.6f}  "
          f"mean|ag| = {ag_mag.mean():.6f}  relative_err = {diff.mean()/max(ag_mag.mean(), 1e-15):.4f}")


# =============================================================================
# 6. Spatial analysis: where are the worst mismatches?
# =============================================================================
print("\n" + "=" * 70)
print("SPATIAL ANALYSIS: Where are the worst derivative mismatches?")
print("=" * 70)

xyt_np = g['xyt_all'].cpu().numpy()

# Combined first-derivative difference (all 8 components)
all_first_diffs = torch.cat([
    (u_x_sp - u_x_ag).abs(),
    (u_y_sp - u_y_ag).abs(),
    (u_t_sp - u_t_ag).abs(),
    (v_x_sp - v_x_ag).abs(),
    (v_y_sp - v_y_ag).abs(),
    (v_t_sp - v_t_ag).abs(),
    (p_x_sp - p_x_ag).abs(),
    (p_y_sp - p_y_ag).abs(),
], dim=1)  # (N_all, 8)
mean_diff_per_point = all_first_diffs.mean(dim=1).cpu().numpy()  # (N_all,)

# Top 20 worst points
top_k = 20
worst_idx = np.argsort(mean_diff_per_point)[-top_k:][::-1]
print(f"\nTop {top_k} points with largest mean first-derivative mismatch:")
print(f"  {'Rank':>4s}  {'x':>6s}  {'y':>6s}  {'t':>6s}  {'mean|diff|':>10s}  {'boundary?':>9s}")
for rank, idx in enumerate(worst_idx):
    x, y, t = xyt_np[idx]
    is_bdy = idx not in set(ii)
    print(f"  {rank+1:4d}  {x:6.3f}  {y:6.3f}  {t:6.3f}  {mean_diff_per_point[idx]:10.6f}  {'YES' if is_bdy else 'no'}")

# Distribution by region
print("\n--- Mean derivative mismatch by region ---")
xc = xyt_np[:, 0]
regions = [
    ("x < 0.05 (near inlet)", xc < 0.05),
    ("0.05 <= x < 0.2", (xc >= 0.05) & (xc < 0.2)),
    ("0.2 <= x < 1.0 (mid)", (xc >= 0.2) & (xc < 1.0)),
    ("1.0 <= x < 1.8 (mid)", (xc >= 1.0) & (xc < 1.8)),
    ("x >= 1.8 (near outlet)", xc >= 1.8),
]
for label, mask in regions:
    if mask.sum() > 0:
        print(f"  {label:30s}: mean|diff| = {mean_diff_per_point[mask].mean():.6f}  "
              f"(n={mask.sum()})")

# PDE residual mismatch by region
print("\n--- PDE residual (autograd) RMS by region ---")
pde_res_ag = np.sqrt(cont_ag.cpu().numpy().flatten()**2 +
                      mom_u_ag.cpu().numpy().flatten()**2 +
                      mom_v_ag.cpu().numpy().flatten()**2)
pde_res_sp = np.sqrt(cont_sp.cpu().numpy().flatten()**2 +
                      mom_u_sp.cpu().numpy().flatten()**2 +
                      mom_v_sp.cpu().numpy().flatten()**2)
for label, mask in regions:
    if mask.sum() > 0:
        ag_rms = np.sqrt(np.mean(pde_res_ag[mask]**2))
        sp_rms = np.sqrt(np.mean(pde_res_sp[mask]**2))
        print(f"  {label:30s}: Spectral PDE RMS = {sp_rms:.6f}  |  Autograd PDE RMS = {ag_rms:.6f}  |  Ratio = {ag_rms/max(sp_rms,1e-15):.2f}x")


# =============================================================================
# 7. CRITICAL TEST: What DT-PINN "thinks" its training loss is vs reality
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING LOSS vs REALITY")
print("=" * 70)

# Spectral PDE loss (what DT-PINN optimized)
loss_cont_sp = np.mean(cont_sp[ii].cpu().numpy()**2)
loss_mu_sp = np.mean(mom_u_sp[ii].cpu().numpy()**2)
loss_mv_sp = np.mean(mom_v_sp[ii].cpu().numpy()**2)
loss_pde_sp = loss_cont_sp + loss_mu_sp + loss_mv_sp

# Autograd PDE loss (ground truth)
loss_cont_ag = np.mean(cont_ag[ii].cpu().numpy()**2)
loss_mu_ag = np.mean(mom_u_ag[ii].cpu().numpy()**2)
loss_mv_ag = np.mean(mom_v_ag[ii].cpu().numpy()**2)
loss_pde_ag = loss_cont_ag + loss_mu_ag + loss_mv_ag

print(f"\nPDE loss at interior Chebyshev points:")
print(f"  Spectral (what DT-PINN trained on): {loss_pde_sp:.6e}")
print(f"    continuity: {loss_cont_sp:.6e}  mom_u: {loss_mu_sp:.6e}  mom_v: {loss_mv_sp:.6e}")
print(f"  Autograd (ground truth):            {loss_pde_ag:.6e}")
print(f"    continuity: {loss_cont_ag:.6e}  mom_u: {loss_mu_ag:.6e}  mom_v: {loss_mv_ag:.6e}")
print(f"  Ratio (autograd/spectral):          {loss_pde_ag/max(loss_pde_sp, 1e-30):.2f}x")

# Also compute PDE RMS (as in evaluation)
pde_rms_sp = np.sqrt(loss_pde_sp)  # This is not quite right for multi-component...
pde_rms_sp = np.sqrt(np.mean(
    cont_sp[ii].cpu().numpy().flatten()**2 +
    mom_u_sp[ii].cpu().numpy().flatten()**2 +
    mom_v_sp[ii].cpu().numpy().flatten()**2))
pde_rms_ag = np.sqrt(np.mean(
    cont_ag[ii].cpu().numpy().flatten()**2 +
    mom_u_ag[ii].cpu().numpy().flatten()**2 +
    mom_v_ag[ii].cpu().numpy().flatten()**2))

print(f"\n  Combined PDE RMS at interior Chebyshev points:")
print(f"    Spectral: {pde_rms_sp:.6f}")
print(f"    Autograd: {pde_rms_ag:.6f}")
print(f"    Ratio:    {pde_rms_ag/max(pde_rms_sp, 1e-15):.2f}x")


# =============================================================================
# 8. For reference: DeepXDE PDE RMS on same Chebyshev points
# =============================================================================
print("\n" + "=" * 70)
print("REFERENCE: DeepXDE on eval grid (from previous run)")
print("=" * 70)
print("  DeepXDE PDE RMS (161x81x20 uniform grid): 0.55")
print("  DT-PINN PDE RMS (161x81x20 uniform grid): 18.05  (autograd eval)")
print(f"  DT-PINN PDE RMS (Chebyshev interior, autograd): {pde_rms_ag:.2f}")
print(f"  DT-PINN PDE RMS (Chebyshev interior, spectral): {pde_rms_sp:.2f}")


# =============================================================================
# 9. Summary and conclusion
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

ratio_pde = loss_pde_ag / max(loss_pde_sp, 1e-30)
if ratio_pde > 5:
    print(f"\n  DERIVATIVE MISMATCH CONFIRMED as major contributor.")
    print(f"  The autograd PDE loss is {ratio_pde:.1f}x larger than spectral PDE loss.")
    print(f"  DT-PINN's spectral derivatives DO NOT match the actual network derivatives.")
    print(f"  The model learned to satisfy the PDE in spectral-derivative space,")
    print(f"  but this does NOT transfer to real (autograd) derivative space.")
elif ratio_pde > 1.5:
    print(f"\n  DERIVATIVE MISMATCH is a contributor but not the sole cause.")
    print(f"  The autograd PDE loss is {ratio_pde:.1f}x larger than spectral PDE loss.")
    print(f"  There is a gap, but other factors also contribute.")
else:
    print(f"\n  DERIVATIVE MISMATCH is NOT a significant contributor.")
    print(f"  The autograd PDE loss is only {ratio_pde:.1f}x the spectral PDE loss.")
    print(f"  The spectral and autograd derivatives agree well.")
    print(f"  Look at other hypotheses (L-BFGS config, point distribution).")

print("\nDone.")
