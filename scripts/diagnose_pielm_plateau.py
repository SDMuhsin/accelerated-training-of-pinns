#!/usr/bin/env python3
"""
Diagnose why PIELM accuracy plateaus at ~0.093 PDE RMS.

Hypotheses to test:
1. Picard linearization error - does iteration converge? what's the residual?
2. ELM representational capacity - can ELM fit a known good solution?
3. Least-squares conditioning - is the matrix ill-conditioned?
4. Spatial error distribution - where is the error largest?
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.experiment_dt_elm_pinn.models.pielm_navier_stokes import PIELM_NavierStokes

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

Re = 1000.0
U_lid = 1.0
Cs = 0.1
N_interior = 6000
N_wall = 800
N_lid = 800

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('results/diagnosis', exist_ok=True)

print("=" * 70)
print("DIAGNOSING PIELM ACCURACY PLATEAU")
print("=" * 70)

# =============================================================================
# Test 1: Picard Convergence Analysis
# =============================================================================
print("\n" + "=" * 70)
print("TEST 1: Picard Iteration Convergence")
print("=" * 70)

model = PIELM_NavierStokes(
    Re=Re, U_lid=U_lid, Cs=Cs,
    n_hidden=500,
    N_interior=N_interior, N_wall=N_wall, N_lid=N_lid,
    max_picard_iter=100, tol=1e-8,
    verbose=True, seed=SEED,
)
model.use_full_viscous = True
results = model.train()

print(f"\nConvergence history (last 20 iterations):")
for i, r in enumerate(model.residual_history[-20:]):
    print(f"  Iter {len(model.residual_history)-20+i+1}: rel_change = {r:.6e}")

# Check if residual is oscillating or stuck
residuals = np.array(model.residual_history)
if len(residuals) > 10:
    late_residuals = residuals[-10:]
    oscillation = np.std(late_residuals) / np.mean(late_residuals)
    print(f"\nLate-stage oscillation (std/mean of last 10): {oscillation:.4f}")
    if oscillation > 0.5:
        print("  -> Picard iteration is OSCILLATING (unstable)")
    elif np.mean(late_residuals) > 1e-5:
        print("  -> Picard iteration is STUCK (not converging)")
    else:
        print("  -> Picard iteration converged well")

# Plot convergence
plt.figure(figsize=(10, 5))
plt.semilogy(range(1, len(residuals)+1), residuals, 'b-o', markersize=3)
plt.xlabel('Picard Iteration')
plt.ylabel('Relative Change')
plt.title('Picard Iteration Convergence')
plt.grid(True, alpha=0.3)
plt.savefig('results/diagnosis/picard_convergence.png', dpi=150)
plt.close()
print("Saved: results/diagnosis/picard_convergence.png")

# =============================================================================
# Test 2: Spatial Error Distribution
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Spatial Error Distribution")
print("=" * 70)

nx, ny = 51, 51
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
xy_grid = np.column_stack([X.ravel(), Y.ravel()])

# Get PIELM predictions and residuals
u, v, p = model.predict(xy_grid)
residuals_dict = model.compute_pde_residuals(xy_grid)

U = u.reshape(ny, nx)
V = v.reshape(ny, nx)
cont = residuals_dict['continuity'].reshape(ny, nx)
mom_x = residuals_dict['momentum_x'].reshape(ny, nx)
mom_y = residuals_dict['momentum_y'].reshape(ny, nx)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Velocity fields
ax = axes[0, 0]
cf = ax.contourf(X, Y, U, levels=30, cmap='RdBu_r')
plt.colorbar(cf, ax=ax)
ax.set_title('u-velocity')
ax.set_aspect('equal')

ax = axes[0, 1]
cf = ax.contourf(X, Y, V, levels=30, cmap='RdBu_r')
plt.colorbar(cf, ax=ax)
ax.set_title('v-velocity')
ax.set_aspect('equal')

ax = axes[0, 2]
speed = np.sqrt(U**2 + V**2)
cf = ax.contourf(X, Y, speed, levels=30, cmap='plasma')
plt.colorbar(cf, ax=ax)
ax.streamplot(X, Y, U, V, color='white', density=1.5, linewidth=0.5)
ax.set_title('Velocity magnitude + streamlines')
ax.set_aspect('equal')

# Residual fields
ax = axes[1, 0]
cf = ax.contourf(X, Y, np.abs(cont), levels=30, cmap='hot')
plt.colorbar(cf, ax=ax)
ax.set_title(f'|Continuity residual| (RMS={np.sqrt(np.mean(cont**2)):.4f})')
ax.set_aspect('equal')

ax = axes[1, 1]
cf = ax.contourf(X, Y, np.abs(mom_x), levels=30, cmap='hot')
plt.colorbar(cf, ax=ax)
ax.set_title(f'|Momentum-x residual| (RMS={np.sqrt(np.mean(mom_x**2)):.4f})')
ax.set_aspect('equal')

ax = axes[1, 2]
cf = ax.contourf(X, Y, np.abs(mom_y), levels=30, cmap='hot')
plt.colorbar(cf, ax=ax)
ax.set_title(f'|Momentum-y residual| (RMS={np.sqrt(np.mean(mom_y**2)):.4f})')
ax.set_aspect('equal')

plt.suptitle('PIELM Solution and Residual Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('results/diagnosis/spatial_distribution.png', dpi=150)
plt.close()
print("Saved: results/diagnosis/spatial_distribution.png")

# Find where error is largest
total_residual = np.sqrt(cont**2 + mom_x**2 + mom_y**2)
max_idx = np.unravel_index(np.argmax(total_residual), total_residual.shape)
print(f"Maximum residual location: x={X[max_idx]:.3f}, y={Y[max_idx]:.3f}")
print(f"Maximum residual value: {total_residual[max_idx]:.4f}")

# =============================================================================
# Test 3: ELM Representational Capacity (Fitting Test)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: ELM Representational Capacity")
print("=" * 70)
print("Testing if ELM can fit a simple target function (not PDE, just regression)")

# Create a simple target that mimics cavity flow structure
def target_u(xy):
    x, y = xy[:, 0], xy[:, 1]
    # Simplified cavity flow: lid at y=1, walls elsewhere
    return y * (1 - 4*(x - 0.5)**2)  # Parabolic profile

def target_v(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Sample points
xy_train = np.random.rand(2000, 2)
u_target = target_u(xy_train)
v_target = target_v(xy_train)

# Build ELM features
np.random.seed(SEED)
n_hidden = 500
W = np.random.uniform(-1, 1, (2, n_hidden))
b = np.random.uniform(-1, 1, n_hidden)
z = xy_train @ W + b
H = np.tanh(z)

# Fit via least squares
beta_u, _, _, _ = np.linalg.lstsq(H, u_target, rcond=None)
beta_v, _, _, _ = np.linalg.lstsq(H, v_target, rcond=None)

# Evaluate
u_pred = H @ beta_u
v_pred = H @ beta_v

u_fit_error = np.sqrt(np.mean((u_pred - u_target)**2))
v_fit_error = np.sqrt(np.mean((v_pred - v_target)**2))

print(f"ELM fitting error (simple target):")
print(f"  u: {u_fit_error:.6f}")
print(f"  v: {v_fit_error:.6f}")

if u_fit_error < 0.01 and v_fit_error < 0.01:
    print("  -> ELM CAN fit smooth functions well")
    print("  -> Problem is NOT representational capacity")
else:
    print("  -> ELM struggles to fit even simple functions")
    print("  -> May need more neurons")

# =============================================================================
# Test 4: Least-Squares Conditioning
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Least-Squares Matrix Conditioning")
print("=" * 70)

# Build a small version of the PIELM matrix to check conditioning
np.random.seed(SEED)
n_hidden_small = 100
N_int_small = 500

W_small = np.random.uniform(-1, 1, (2, n_hidden_small))
b_small = np.random.uniform(-1, 1, n_hidden_small)

xy_small = np.random.rand(N_int_small, 2)
z = xy_small @ W_small + b_small
H = np.tanh(z)

# Compute derivatives
tanh_prime = 1 - np.tanh(z)**2
tanh_pp = -2 * np.tanh(z) * tanh_prime

dH_dx = tanh_prime * W_small[0, :]
dH_dy = tanh_prime * W_small[1, :]
LapH = tanh_pp * (W_small[0, :]**2 + W_small[1, :]**2)

# Simple momentum equation matrix: u·∂u/∂x - ν·∇²u
# Using zero velocity (first iteration)
nu = 0.001  # laminar viscosity
A_mom = -nu * LapH  # Simplified momentum

# Stack equations
A = np.vstack([
    dH_dx + dH_dy,  # Continuity
    A_mom,           # Simplified momentum
])

# Condition number
try:
    cond = np.linalg.cond(A)
    print(f"Condition number of A: {cond:.2e}")
    if cond > 1e10:
        print("  -> Matrix is ILL-CONDITIONED (cond > 1e10)")
        print("  -> Least-squares solve may be inaccurate")
    elif cond > 1e6:
        print("  -> Matrix is moderately conditioned")
    else:
        print("  -> Matrix is well-conditioned")
except:
    print("Could not compute condition number")

# SVD analysis
U_svd, S, Vt = np.linalg.svd(A, full_matrices=False)
print(f"Singular value range: {S.max():.4e} to {S.min():.4e}")
print(f"Ratio (max/min): {S.max()/S.min():.2e}")

# =============================================================================
# Test 5: Linearization Error Analysis
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: Picard Linearization Error")
print("=" * 70)

# The Picard linearization replaces:
#   u·∂u/∂x + v·∂u/∂y  (nonlinear)
# with:
#   u^k·∂u^{k+1}/∂x + v^k·∂u^{k+1}/∂y  (linearized)

# At convergence, these should be equal. Let's check.
u_final, v_final, p_final, grads = model.predict_with_gradients(xy_grid)

# Nonlinear term
nonlinear_u = u_final * grads['du_dx'] + v_final * grads['du_dy']
nonlinear_v = u_final * grads['dv_dx'] + v_final * grads['dv_dy']

print(f"Convective term magnitudes:")
print(f"  |u·∇u| RMS: {np.sqrt(np.mean(nonlinear_u**2)):.4f}")
print(f"  |u·∇v| RMS: {np.sqrt(np.mean(nonlinear_v**2)):.4f}")

# Compare to viscous term
nu_eff = model._compute_eddy_viscosity(xy_grid, grads['du_dx'], grads['du_dy'],
                                        grads['dv_dx'], grads['dv_dy'])
LapH_grid = model._compute_laplacian_features(xy_grid)
Lap_u = LapH_grid @ model.beta_u
Lap_v = LapH_grid @ model.beta_v

viscous_u = nu_eff * Lap_u
viscous_v = nu_eff * Lap_v

print(f"Viscous term magnitudes:")
print(f"  |ν·∇²u| RMS: {np.sqrt(np.mean(viscous_u**2)):.4f}")
print(f"  |ν·∇²v| RMS: {np.sqrt(np.mean(viscous_v**2)):.4f}")

ratio = np.sqrt(np.mean(nonlinear_u**2)) / np.sqrt(np.mean(viscous_u**2))
print(f"\nConvective/Viscous ratio: {ratio:.2f}")
if ratio > 10:
    print("  -> Convection-dominated (Re=1000 is high)")
    print("  -> Picard linearization may struggle")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)

print("""
1. PICARD CONVERGENCE: Check picard_convergence.png
   - If oscillating: linearization is unstable
   - If stuck at high residual: fundamental accuracy limit

2. SPATIAL DISTRIBUTION: Check spatial_distribution.png
   - Where is error concentrated?
   - Near walls? In vortex core? Near lid corners?

3. ELM CAPACITY: Can fit smooth functions well
   - Not a representational bottleneck

4. MATRIX CONDITIONING: Check condition number
   - If ill-conditioned: numerical errors in solve

5. LINEARIZATION ERROR: Convective vs Viscous
   - At Re=1000, convection >> viscosity
   - Picard linearization error may be significant
""")

print("\nDiagnostic plots saved to results/diagnosis/")
