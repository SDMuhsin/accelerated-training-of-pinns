#!/usr/bin/env python3
"""
DT-PINN vs Vanilla PINN Comparison for Navier-Stokes

Compares:
1. Vanilla PINN: Uses autodiff to compute all derivatives
2. DT-PINN: Uses precomputed discrete matrices for derivatives

Both use the same:
- Network architecture (6 layers, 64 neurons)
- Collocation points
- Loss function
- Optimizer (Adam)
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re
Cs = 0.1

# Use structured grid for DT-PINN (required for discrete operators)
# We'll use same number of points as original but on a grid
N_grid = 50  # 50x50 = 2500 interior points
N_bc_per_edge = 50  # boundary points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("DT-PINN vs Vanilla PINN for Navier-Stokes")
print("=" * 70)
print(f"Device: {device}")
print(f"Grid: {N_grid}x{N_grid} = {N_grid**2} total points")

# =============================================================================
# Build Discrete Operators (Chebyshev Spectral)
# =============================================================================
def chebyshev_points(N):
    """Chebyshev-Gauss-Lobatto points on [-1, 1]."""
    i = np.arange(N)
    return np.cos(np.pi * i / (N - 1))

def chebyshev_diff_matrix(N):
    """First derivative matrix on Chebyshev points."""
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
    """
    Build 2D derivative operators on [0,1]^2 domain.

    Returns Dx, Dy as dense matrices operating on flattened grid.
    Points ordered as: for j in range(N): for i in range(N): point (x[i], y[j])
    """
    D1d = chebyshev_diff_matrix(N)

    # Scale for [0,1] domain (chain rule: d/dx_phys = 2 * d/dx_ref)
    D1d_scaled = D1d * 2.0

    I = np.eye(N)

    # Dx = I_y ⊗ D_x (derivative in x, for each y)
    Dx = np.kron(I, D1d_scaled)

    # Dy = D_y ⊗ I_x (derivative in y, for each x)
    Dy = np.kron(D1d_scaled, I)

    return Dx, Dy

def build_grid(N):
    """Build 2D Chebyshev grid on [0,1]^2."""
    x_ref = chebyshev_points(N)
    # Map from [-1,1] to [0,1]
    x = 0.5 * (x_ref + 1.0)

    # Use indexing='xy' so x varies fastest (required for Kronecker product structure)
    xx, yy = np.meshgrid(x, x, indexing='xy')
    xy = np.column_stack([xx.ravel(), yy.ravel()])

    return xy, x

print("\nBuilding discrete operators...")
t0 = time.perf_counter()
Dx_np, Dy_np = build_2d_operators(N_grid)
xy_grid, x_1d = build_grid(N_grid)
t_build = time.perf_counter() - t0
print(f"Operator build time: {t_build:.2f}s")
print(f"Grid shape: {xy_grid.shape}")

# Identify interior vs boundary points
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

N_interior = len(interior_idx)
N_lid = len(lid_idx)
N_wall = len(wall_idx)
N_total = len(xy_grid)

print(f"Interior: {N_interior}, Lid: {N_lid}, Wall: {N_wall}, Total: {N_total}")

# Convert operators to torch
Dx_torch = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_torch = torch.tensor(Dy_np, dtype=torch.float32, device=device)

# Grid as torch tensors
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
xy_interior = xy_all[interior_idx]
xy_lid = xy_all[lid_idx]
xy_wall = xy_all[wall_idx]

# Precompute distance to wall for Smagorinsky
x_t = xy_all[:, 0:1]
y_t = xy_all[:, 1:2]
d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))

# =============================================================================
# Network Architecture (same for both methods)
# =============================================================================
class PINN_Cavity(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, 64), nn.Tanh()]
        for _ in range(5):
            layers.extend([nn.Linear(64, 64), nn.Tanh()])
        layers.append(nn.Linear(64, 3))  # u, v, p
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =============================================================================
# Autodiff PINN (Baseline)
# =============================================================================
def gradients(y, x):
    """Compute gradient via autodiff."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]

def pde_residuals_autodiff(model, xy):
    """Compute PDE residuals using autodiff."""
    xy.requires_grad_(True)
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    # Velocity gradients (autodiff)
    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    # Smagorinsky eddy viscosity
    x, y = xy[:, 0:1], xy[:, 1:2]
    d = torch.min(torch.min(x, 1.0 - x), torch.min(y, 1.0 - y))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d)**2 * S_mag

    # Continuity
    continuity = du_dx + dv_dy

    # Convection
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy

    # Pressure gradient
    grad_p = gradients(p, xy)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # Viscous term: ∇·(ν_eff·∇u)
    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
    grad_qx_u, grad_qy_u = gradients(qx_u, xy), gradients(qy_u, xy)
    grad_qx_v, grad_qy_v = gradients(qx_v, xy), gradients(qy_v, xy)
    visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
    visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]

    # Momentum
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    return continuity, mom_u, mom_v

# =============================================================================
# DT-PINN (Discrete Operators)
# =============================================================================
def pde_residuals_discrete(model, xy_all, Dx, Dy, d_wall, interior_idx):
    """
    Compute PDE residuals using discrete operators.

    Key difference: derivatives computed via matrix multiplication, not autodiff.
    """
    # Forward pass (still need gradients for network params, but not spatial derivatives)
    pred = model(xy_all)
    u_all = pred[:, 0:1]
    v_all = pred[:, 1:2]
    p_all = pred[:, 2:3]

    # Velocity gradients via matrix multiply
    du_dx = Dx @ u_all
    du_dy = Dy @ u_all
    dv_dx = Dx @ v_all
    dv_dy = Dy @ v_all

    # Smagorinsky eddy viscosity
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag

    # Continuity
    continuity = du_dx + dv_dy

    # Convection
    u_conv = u_all * du_dx + v_all * du_dy
    v_conv = u_all * dv_dx + v_all * dv_dy

    # Pressure gradient
    dp_dx = Dx @ p_all
    dp_dy = Dy @ p_all

    # Viscous term: ∇·(ν_eff·∇u)
    # = d/dx(ν_eff·du/dx) + d/dy(ν_eff·du/dy)
    visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
    visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)

    # Momentum
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    # Return only interior residuals
    return continuity[interior_idx], mom_u[interior_idx], mom_v[interior_idx]

# =============================================================================
# Training Functions
# =============================================================================
def train_pinn_autodiff(epochs, verbose=True):
    """Train PINN using autodiff for derivatives."""
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    xy_int = xy_interior.clone().detach().requires_grad_(True)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # PDE loss (interior)
        cont, mom_u, mom_v = pde_residuals_autodiff(model, xy_int)
        loss_pde = mse(cont, torch.zeros_like(cont)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

        # BC loss (lid)
        pred_lid = model(xy_lid)
        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                   mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))

        # BC loss (walls)
        pred_wall = model(xy_wall)
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                    mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))

        # Pressure anchor
        xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        pred_center = model(xy_center)
        loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 500 == 0:
            print(f"  Autodiff epoch {epoch+1}: loss = {loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return model, elapsed

def train_pinn_discrete(epochs, verbose=True):
    """Train PINN using discrete operators for derivatives."""
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # PDE loss (interior) - using discrete operators
        cont, mom_u, mom_v = pde_residuals_discrete(
            model, xy_all, Dx_torch, Dy_torch, d_wall, interior_idx
        )
        loss_pde = mse(cont, torch.zeros_like(cont)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

        # BC loss (lid)
        pred_lid = model(xy_lid)
        loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                   mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))

        # BC loss (walls)
        pred_wall = model(xy_wall)
        loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                    mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))

        # Pressure anchor
        xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
        pred_center = model(xy_center)
        loss_p = mse(pred_center[:, 2:3], torch.zeros_like(pred_center[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 500 == 0:
            print(f"  Discrete epoch {epoch+1}: loss = {loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return model, elapsed

# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(model, method_name):
    """Evaluate model on a fine grid using autodiff (ground truth residuals)."""
    # Create evaluation grid
    nx, ny = 41, 41
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    cont, mom_u, mom_v = pde_residuals_autodiff(model, xy_t)

    cont_np = cont.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))
    cont_rms = float(np.sqrt(np.mean(cont_np**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u_np**2 + mom_v_np**2)))

    return {
        'method': method_name,
        'pde_rms': pde_rms,
        'continuity_rms': cont_rms,
        'momentum_rms': mom_rms,
    }

# =============================================================================
# Hybrid Training (DT-PINN then Autodiff)
# =============================================================================
def train_hybrid(discrete_epochs, autodiff_epochs, verbose=True):
    """
    Hybrid training: DT-PINN for early epochs, then autodiff for fine-tuning.

    This combines:
    - DT-PINN's speed advantage in early training
    - Autodiff's stability for fine-tuning
    """
    torch.manual_seed(SEED)
    model = PINN_Cavity().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    # Phase 1: DT-PINN (fast)
    if verbose:
        print(f"  Phase 1: DT-PINN ({discrete_epochs} epochs)")

    for epoch in range(discrete_epochs):
        optimizer.zero_grad()

        cont, mom_u, mom_v = pde_residuals_discrete(
            model, xy_all, Dx_torch, Dy_torch, d_wall, interior_idx
        )
        loss_pde = mse(cont, torch.zeros_like(cont)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    phase1_time = time.perf_counter() - start

    if verbose:
        print(f"    Phase 1 done: {phase1_time:.1f}s, loss = {loss.item():.6f}")

    # Phase 2: Autodiff (stable fine-tuning)
    if autodiff_epochs > 0:
        if verbose:
            print(f"  Phase 2: Autodiff ({autodiff_epochs} epochs)")

        xy_int = xy_interior.clone().detach().requires_grad_(True)

        for epoch in range(autodiff_epochs):
            optimizer.zero_grad()

            cont, mom_u, mom_v = pde_residuals_autodiff(model, xy_int)
            loss_pde = mse(cont, torch.zeros_like(cont)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))

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

        if verbose:
            print(f"    Phase 2 done: loss = {loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return model, total_time, phase1_time

# =============================================================================
# Run Experiments
# =============================================================================
EPOCH_CHECKPOINTS = [500, 1000, 1500, 2000, 3000]

print("\n" + "=" * 70)
print("RUNNING EXPERIMENTS")
print("=" * 70)

results = []

for epochs in EPOCH_CHECKPOINTS:
    print(f"\n--- {epochs} epochs ---")

    # Autodiff PINN
    print(f"\nAutodiff PINN ({epochs} epochs):")
    model_auto, time_auto = train_pinn_autodiff(epochs, verbose=True)
    metrics_auto = evaluate_model(model_auto, 'autodiff')
    metrics_auto['epochs'] = epochs
    metrics_auto['time'] = time_auto
    results.append(metrics_auto)
    print(f"  Time: {time_auto:.1f}s, PDE RMS: {metrics_auto['pde_rms']:.5f}")

    # Discrete PINN
    print(f"\nDiscrete PINN ({epochs} epochs):")
    model_disc, time_disc = train_pinn_discrete(epochs, verbose=True)
    metrics_disc = evaluate_model(model_disc, 'discrete')
    metrics_disc['epochs'] = epochs
    metrics_disc['time'] = time_disc
    results.append(metrics_disc)
    print(f"  Time: {time_disc:.1f}s, PDE RMS: {metrics_disc['pde_rms']:.5f}")

    # Speedup
    speedup = time_auto / time_disc
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Accuracy ratio: {metrics_disc['pde_rms'] / metrics_auto['pde_rms']:.2f}x")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\n{'Method':<12} {'Epochs':<8} {'Time (s)':<10} {'PDE RMS':<12} {'Cont RMS':<12} {'Mom RMS':<12}")
print("-" * 70)
for r in results:
    print(f"{r['method']:<12} {r['epochs']:<8} {r['time']:<10.1f} {r['pde_rms']:<12.5f} {r['continuity_rms']:<12.5f} {r['momentum_rms']:<12.5f}")

print("\n" + "=" * 70)
print("SPEEDUP ANALYSIS")
print("=" * 70)

for epochs in EPOCH_CHECKPOINTS:
    auto_r = [r for r in results if r['method'] == 'autodiff' and r['epochs'] == epochs][0]
    disc_r = [r for r in results if r['method'] == 'discrete' and r['epochs'] == epochs][0]

    speedup = auto_r['time'] / disc_r['time']
    accuracy_ratio = disc_r['pde_rms'] / auto_r['pde_rms']

    print(f"\nAt {epochs} epochs:")
    print(f"  Autodiff: {auto_r['time']:.1f}s, PDE RMS = {auto_r['pde_rms']:.5f}")
    print(f"  Discrete: {disc_r['time']:.1f}s, PDE RMS = {disc_r['pde_rms']:.5f}")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Accuracy ratio (discrete/autodiff): {accuracy_ratio:.2f}x")

# =============================================================================
# HYBRID EXPERIMENTS
# =============================================================================
print("\n" + "=" * 70)
print("HYBRID TRAINING EXPERIMENTS")
print("=" * 70)
print("Testing DT-PINN for early epochs + Autodiff for fine-tuning")

HYBRID_CONFIGS = [
    # (discrete_epochs, autodiff_epochs)
    (500, 500),    # 1000 total, 50% DT-PINN
    (1000, 1000),  # 2000 total, 50% DT-PINN
    (500, 1500),   # 2000 total, 25% DT-PINN
    (1000, 2000),  # 3000 total, 33% DT-PINN
]

hybrid_results = []

for disc_ep, auto_ep in HYBRID_CONFIGS:
    total_ep = disc_ep + auto_ep
    print(f"\n--- Hybrid: {disc_ep} DT-PINN + {auto_ep} Autodiff = {total_ep} total ---")

    model_hybrid, time_hybrid, phase1_time = train_hybrid(disc_ep, auto_ep, verbose=True)
    metrics_hybrid = evaluate_model(model_hybrid, f'hybrid_{disc_ep}+{auto_ep}')

    # Compare to pure autodiff at same total epochs
    auto_result = [r for r in results if r['method'] == 'autodiff' and r['epochs'] == total_ep]
    if auto_result:
        auto_time = auto_result[0]['time']
        auto_pde = auto_result[0]['pde_rms']
        speedup = auto_time / time_hybrid
        accuracy_ratio = metrics_hybrid['pde_rms'] / auto_pde
        print(f"\n  Hybrid: {time_hybrid:.1f}s, PDE RMS = {metrics_hybrid['pde_rms']:.5f}")
        print(f"  Pure Autodiff @ {total_ep}ep: {auto_time:.1f}s, PDE RMS = {auto_pde:.5f}")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Accuracy ratio: {accuracy_ratio:.2f}x")

    hybrid_results.append({
        'discrete_epochs': disc_ep,
        'autodiff_epochs': auto_ep,
        'total_epochs': total_ep,
        'time': time_hybrid,
        'phase1_time': phase1_time,
        **metrics_hybrid,
    })

# Summary table
print("\n" + "=" * 70)
print("HYBRID RESULTS SUMMARY")
print("=" * 70)
print(f"{'Config':<20} {'Time (s)':<10} {'PDE RMS':<12} {'vs Pure Auto':<15}")
print("-" * 60)
for r in hybrid_results:
    config = f"{r['discrete_epochs']}+{r['autodiff_epochs']}"
    total_ep = r['total_epochs']
    auto_result = [x for x in results if x['method'] == 'autodiff' and x['epochs'] == total_ep]
    if auto_result:
        speedup = auto_result[0]['time'] / r['time']
        acc_ratio = r['pde_rms'] / auto_result[0]['pde_rms']
        print(f"{config:<20} {r['time']:<10.1f} {r['pde_rms']:<12.5f} {speedup:.2f}x speed, {acc_ratio:.2f}x acc")

# Save results
import json
os.makedirs('results/dt_pinn_comparison', exist_ok=True)
output = {
    'pure_methods': results,
    'hybrid': hybrid_results,
}
with open('results/dt_pinn_comparison/navier_stokes_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nResults saved to results/dt_pinn_comparison/navier_stokes_results.json")
