#!/usr/bin/env python3
"""
DT-PINN for Battery Cooling Channel Geometry

Extends DT-PINN to handle irregular geometry defined by an image mask.
Uses overset grid approach: Chebyshev grid over bounding box with masked interior.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Physics parameters (from partner's lid-driven cavity)
Re = 1000.0
U_inlet = 1.0  # Normalized inlet velocity
nu_laminar = U_inlet / Re
Cs = 0.1  # Smagorinsky constant

# Training parameters
NUM_EPOCHS = 5000  # Full experiment
LR = 1e-3

# Image and geometry
IMAGE_PATH = "from_partner_team/SourceCode/Pipe6.png"
WIDTH_MM = 2000.0
HEIGHT_MM = 1000.0
INLET_XY_MM = (10.0, 50.0)
OUTLET_XY_MM = (1900.0, 900.0)

# Grid resolution (Chebyshev grid over bounding box)
N_GRID_X = 80  # More points in x (longer dimension)
N_GRID_Y = 50  # Fewer points in y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("DT-PINN FOR BATTERY COOLING CHANNEL")
print("=" * 70)
print(f"Device: {device}")
print(f"Image: {IMAGE_PATH}")
print(f"Physical domain: {WIDTH_MM} x {HEIGHT_MM} mm")

# =============================================================================
# Load Image and Create Domain Mask
# =============================================================================
def load_domain_from_image(image_path, threshold=0.8):
    """Load image and create domain mask (dark pixels = inside)."""
    img = Image.open(image_path).convert("L")
    gray = np.asarray(img, dtype=np.float64) / 255.0
    inside = gray < threshold  # dark = inside, white = outside

    nrows, ncols = inside.shape
    print(f"Image size: {ncols} x {nrows} pixels")
    print(f"Inside domain: {inside.sum()} pixels ({100*inside.sum()/(nrows*ncols):.1f}%)")

    return inside, nrows, ncols

def find_boundary_pixels(inside):
    """Find pixels that are on the boundary of the domain."""
    nrows, ncols = inside.shape
    boundary = np.zeros_like(inside)

    # A pixel is a boundary if it's inside AND has at least one outside neighbor
    for i in range(nrows):
        for j in range(ncols):
            if inside[i, j]:
                # Check 4-connected neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= nrows or nj < 0 or nj >= ncols:
                        boundary[i, j] = True
                        break
                    if not inside[ni, nj]:
                        boundary[i, j] = True
                        break

    print(f"Boundary pixels: {boundary.sum()}")
    return boundary

print("\nLoading domain geometry...")
inside_mask, nrows, ncols = load_domain_from_image(IMAGE_PATH)
boundary_mask = find_boundary_pixels(inside_mask)
interior_mask = inside_mask & ~boundary_mask

# =============================================================================
# Create Chebyshev Grid Over Bounding Box
# =============================================================================
def chebyshev_points(N):
    """Chebyshev points on [-1, 1]."""
    i = np.arange(N)
    return np.cos(np.pi * i / (N - 1))

def chebyshev_diff_matrix(N):
    """1D Chebyshev differentiation matrix."""
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

def build_2d_operators(Nx, Ny, Lx, Ly):
    """Build 2D differentiation operators for rectangular domain [0, Lx] x [0, Ly]."""
    D1d_x = chebyshev_diff_matrix(Nx) * (2.0 / Lx)  # Scale for [0, Lx]
    D1d_y = chebyshev_diff_matrix(Ny) * (2.0 / Ly)  # Scale for [0, Ly]

    Ix = np.eye(Nx)
    Iy = np.eye(Ny)

    # Kronecker products: D_x operates on rows, D_y operates on columns
    Dx = np.kron(Iy, D1d_x)  # Shape: (Nx*Ny, Nx*Ny)
    Dy = np.kron(D1d_y, Ix)  # Shape: (Nx*Ny, Nx*Ny)

    return Dx, Dy

def build_grid(Nx, Ny, Lx, Ly):
    """Build 2D Chebyshev grid on [0, Lx] x [0, Ly]."""
    x_ref = chebyshev_points(Nx)
    y_ref = chebyshev_points(Ny)

    # Map from [-1, 1] to [0, Lx] and [0, Ly]
    x = Lx * 0.5 * (x_ref + 1.0)
    y = Ly * 0.5 * (y_ref + 1.0)

    # Create 2D grid (x varies faster)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    xy = np.column_stack([xx.ravel(), yy.ravel()])

    return xy, x, y

print(f"\nBuilding Chebyshev grid ({N_GRID_X} x {N_GRID_Y})...")

# Normalize domain to [0, 1] x [0, 1] for numerical stability
Lx, Ly = 1.0, 1.0  # Normalized domain
Dx_np, Dy_np = build_2d_operators(N_GRID_X, N_GRID_Y, Lx, Ly)
xy_grid, x_1d, y_1d = build_grid(N_GRID_X, N_GRID_Y, Lx, Ly)

print(f"Grid points: {xy_grid.shape[0]}")

# =============================================================================
# Map Grid Points to Image Domain
# =============================================================================
def grid_point_in_domain(xy, inside_mask, ncols, nrows):
    """Check if a grid point (normalized [0,1] x [0,1]) is inside the domain."""
    x, y = xy
    # Map to pixel coordinates
    px = int(x * (ncols - 1))
    py = int((1 - y) * (nrows - 1))  # Flip y (image y is top-to-bottom)

    px = min(max(px, 0), ncols - 1)
    py = min(max(py, 0), nrows - 1)

    return inside_mask[py, px]

def grid_point_on_boundary(xy, boundary_mask, ncols, nrows):
    """Check if a grid point is on the domain boundary."""
    x, y = xy
    px = int(x * (ncols - 1))
    py = int((1 - y) * (nrows - 1))

    px = min(max(px, 0), ncols - 1)
    py = min(max(py, 0), nrows - 1)

    return boundary_mask[py, px]

def mm_to_normalized(xy_mm, width_mm, height_mm):
    """Convert mm coordinates to normalized [0, 1] x [0, 1]."""
    return (xy_mm[0] / width_mm, xy_mm[1] / height_mm)

# Classify grid points
print("\nClassifying grid points...")
inside_idx = []
boundary_idx = []
interior_idx = []
inlet_idx = []
outlet_idx = []

# Use x-position to identify inlet (left) and outlet (right) boundary regions
# Inlet: leftmost 5% of x-range among boundary points
# Outlet: rightmost 5% of x-range among boundary points
inlet_x_threshold = 0.08  # Left 8% of domain
outlet_x_threshold = 0.92  # Right 8% of domain

print(f"Inlet region: x < {inlet_x_threshold}")
print(f"Outlet region: x > {outlet_x_threshold}")

for i, (x, y) in enumerate(xy_grid):
    if grid_point_in_domain((x, y), inside_mask, ncols, nrows):
        inside_idx.append(i)

        # Check if on boundary
        is_on_boundary = grid_point_on_boundary((x, y), boundary_mask, ncols, nrows)

        # Classify boundary points as inlet, outlet, or wall
        if is_on_boundary:
            if x < inlet_x_threshold:
                inlet_idx.append(i)
            elif x > outlet_x_threshold:
                outlet_idx.append(i)
            else:
                boundary_idx.append(i)
        else:
            interior_idx.append(i)

inside_idx = np.array(inside_idx)
boundary_idx = np.array(boundary_idx)
interior_idx = np.array(interior_idx)
inlet_idx = np.array(inlet_idx)
outlet_idx = np.array(outlet_idx)

print(f"Inside domain: {len(inside_idx)} points")
print(f"  Interior: {len(interior_idx)} points")
print(f"  Boundary (walls): {len(boundary_idx)} points")
print(f"  Inlet region: {len(inlet_idx)} points")
print(f"  Outlet region: {len(outlet_idx)} points")

if len(interior_idx) < 100:
    print("\nWARNING: Very few interior points. Consider increasing grid resolution.")

# =============================================================================
# Convert to PyTorch
# =============================================================================
Dx_torch = torch.tensor(Dx_np, dtype=torch.float32, device=device)
Dy_torch = torch.tensor(Dy_np, dtype=torch.float32, device=device)
xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)

xy_inside = xy_all[inside_idx] if len(inside_idx) > 0 else None
xy_interior = xy_all[interior_idx] if len(interior_idx) > 0 else None
xy_boundary = xy_all[boundary_idx] if len(boundary_idx) > 0 else None
xy_inlet = xy_all[inlet_idx] if len(inlet_idx) > 0 else None
xy_outlet = xy_all[outlet_idx] if len(outlet_idx) > 0 else None

# Compute wall distance for all grid points
def compute_wall_distance(xy, boundary_points):
    """Compute distance to nearest boundary point."""
    if boundary_points is None or len(boundary_points) == 0:
        # Fallback: distance to domain edges
        x, y = xy[:, 0:1], xy[:, 1:2]
        return torch.min(torch.min(x, 1.0 - x), torch.min(y, 1.0 - y))

    # Compute distance to nearest boundary point
    xy_expanded = xy.unsqueeze(1)  # (N, 1, 2)
    boundary_expanded = boundary_points.unsqueeze(0)  # (1, M, 2)
    distances = torch.norm(xy_expanded - boundary_expanded, dim=2)  # (N, M)
    min_dist = distances.min(dim=1, keepdim=True)[0]  # (N, 1)
    return min_dist

print("\nComputing wall distances...")
d_wall_all = compute_wall_distance(xy_all, xy_boundary)
d_wall_inside = d_wall_all[inside_idx] if len(inside_idx) > 0 else None

# =============================================================================
# Network
# =============================================================================
class PINN_Channel(nn.Module):
    """PINN for channel flow: outputs (u, v, p)."""
    def __init__(self, hidden_layers=6, hidden_units=64):
        super().__init__()
        layers = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.Tanh()])
        layers.append(nn.Linear(hidden_units, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =============================================================================
# PDE Residual Functions
# =============================================================================
def gradients(y, x):
    """Compute gradient of y with respect to x using autodiff."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]

def pde_residuals_autodiff(model, xy, d_wall):
    """Compute PDE residuals using autodiff."""
    xy.requires_grad_(True)
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    # Smagorinsky turbulence model
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag

    # Continuity
    continuity = du_dx + dv_dy

    # Convection
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy

    # Pressure gradient
    grad_p = gradients(p, xy)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # Viscous terms (with variable viscosity)
    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
    grad_qx_u, grad_qy_u = gradients(qx_u, xy), gradients(qy_u, xy)
    grad_qx_v, grad_qy_v = gradients(qx_v, xy), gradients(qy_v, xy)
    visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
    visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]

    # Momentum residuals
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    return continuity, mom_u, mom_v

def pde_residuals_discrete(model, xy_all, Dx, Dy, d_wall, inside_idx, interior_idx):
    """Compute PDE residuals using discrete Chebyshev differentiation."""
    pred = model(xy_all)
    u_all = pred[:, 0:1]
    v_all = pred[:, 1:2]
    p_all = pred[:, 2:3]

    # Compute derivatives on full grid
    du_dx = Dx @ u_all
    du_dy = Dy @ u_all
    dv_dx = Dx @ v_all
    dv_dy = Dy @ v_all

    # Smagorinsky model
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

    # Viscous terms (variable viscosity)
    visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
    visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)

    # Momentum residuals
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    # Return only interior points (where PDE should hold)
    return continuity[interior_idx], mom_u[interior_idx], mom_v[interior_idx]

# =============================================================================
# Training Function
# =============================================================================
def train_experiment(dt_epochs, auto_epochs, verbose=True, log_interval=500):
    """Train with DT-PINN followed by autodiff."""
    total_epochs = dt_epochs + auto_epochs

    torch.manual_seed(SEED)
    model = PINN_Channel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()

    loss_history = []

    # Targets for BCs
    zero_boundary = torch.zeros((len(boundary_idx), 1), device=device) if len(boundary_idx) > 0 else None
    u_inlet_target = torch.ones((len(inlet_idx), 1), device=device) * U_inlet if len(inlet_idx) > 0 else None
    zero_inlet_v = torch.zeros((len(inlet_idx), 1), device=device) if len(inlet_idx) > 0 else None
    zero_outlet_p = torch.zeros((len(outlet_idx), 1), device=device) if len(outlet_idx) > 0 else None

    # Phase 1: DT-PINN
    if dt_epochs > 0 and verbose:
        print(f"  Phase 1: DT-PINN ({dt_epochs} epochs)")

    for epoch in range(dt_epochs):
        optimizer.zero_grad()

        # PDE loss at interior
        if len(interior_idx) > 0:
            cont, mom_u, mom_v = pde_residuals_discrete(
                model, xy_all, Dx_torch, Dy_torch, d_wall_all, inside_idx, interior_idx
            )
            loss_pde = mse(cont, torch.zeros_like(cont)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))
        else:
            loss_pde = torch.tensor(0.0, device=device)

        # Wall BCs: no-slip (u=0, v=0)
        if len(boundary_idx) > 0:
            pred_wall = model(xy_boundary)
            loss_wall = mse(pred_wall[:, 0:1], zero_boundary) + \
                        mse(pred_wall[:, 1:2], zero_boundary)
        else:
            loss_wall = torch.tensor(0.0, device=device)

        # Inlet BC: u=U_inlet, v=0
        if len(inlet_idx) > 0:
            pred_inlet = model(xy_inlet)
            loss_inlet = mse(pred_inlet[:, 0:1], u_inlet_target) + \
                         mse(pred_inlet[:, 1:2], zero_inlet_v)
        else:
            loss_inlet = torch.tensor(0.0, device=device)

        # Outlet BC: p=0 (pressure reference)
        if len(outlet_idx) > 0:
            pred_outlet = model(xy_outlet)
            loss_outlet = mse(pred_outlet[:, 2:3], zero_outlet_p)
        else:
            loss_outlet = torch.tensor(0.0, device=device)

        loss = loss_pde + loss_wall + loss_inlet + loss_outlet
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and (epoch + 1) % log_interval == 0:
            print(f"    Epoch {epoch+1}: loss = {loss.item():.6f} "
                  f"(PDE={loss_pde.item():.6f}, Wall={loss_wall.item():.6f}, "
                  f"In={loss_inlet.item():.6f}, Out={loss_outlet.item():.6f})")

    # Phase 2: Autodiff
    if auto_epochs > 0:
        if verbose:
            print(f"  Phase 2: Autodiff ({auto_epochs} epochs)")

        xy_int = xy_interior.clone().detach().requires_grad_(True) if xy_interior is not None else None

        # Compute d_wall for interior points only
        d_wall_int = None
        if xy_int is not None:
            d_wall_int = compute_wall_distance(xy_int.detach(), xy_boundary)

        for epoch in range(auto_epochs):
            optimizer.zero_grad()

            # PDE loss
            if xy_int is not None:
                cont, mom_u, mom_v = pde_residuals_autodiff(model, xy_int, d_wall_int)
                loss_pde = mse(cont, torch.zeros_like(cont)) + \
                           mse(mom_u, torch.zeros_like(mom_u)) + \
                           mse(mom_v, torch.zeros_like(mom_v))
            else:
                loss_pde = torch.tensor(0.0, device=device)

            # Wall BCs
            if len(boundary_idx) > 0:
                pred_wall = model(xy_boundary)
                loss_wall = mse(pred_wall[:, 0:1], zero_boundary) + \
                            mse(pred_wall[:, 1:2], zero_boundary)
            else:
                loss_wall = torch.tensor(0.0, device=device)

            # Inlet BC
            if len(inlet_idx) > 0:
                pred_inlet = model(xy_inlet)
                loss_inlet = mse(pred_inlet[:, 0:1], u_inlet_target) + \
                             mse(pred_inlet[:, 1:2], zero_inlet_v)
            else:
                loss_inlet = torch.tensor(0.0, device=device)

            # Outlet BC
            if len(outlet_idx) > 0:
                pred_outlet = model(xy_outlet)
                loss_outlet = mse(pred_outlet[:, 2:3], zero_outlet_p)
            else:
                loss_outlet = torch.tensor(0.0, device=device)

            loss = loss_pde + loss_wall + loss_inlet + loss_outlet
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if verbose and (dt_epochs + epoch + 1) % log_interval == 0:
                print(f"    Epoch {dt_epochs + epoch + 1}: loss = {loss.item():.6f}")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return model, total_time, loss_history

def evaluate_model(model):
    """Evaluate PDE residuals on the domain."""
    model.eval()

    # Evaluate at inside points
    if xy_inside is None or len(inside_idx) == 0:
        return {'pde_rms': float('nan')}

    # Use autodiff for evaluation (ground truth)
    xy_eval = xy_inside.clone().detach().requires_grad_(True)
    d_wall_eval = compute_wall_distance(xy_eval.detach(), xy_boundary)

    cont, mom_u, mom_v = pde_residuals_autodiff(model, xy_eval, d_wall_eval)

    cont_np = cont.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))

    return {'pde_rms': pde_rms}

def visualize_solution(model, save_path=None):
    """Visualize the solution on the domain."""
    model.eval()

    # Create a dense evaluation grid
    nx, ny = 200, 100
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_dense = np.column_stack([X.ravel(), Y.ravel()])

    # Filter to domain points
    in_domain = np.array([grid_point_in_domain((xi, yi), inside_mask, ncols, nrows)
                          for xi, yi in xy_dense])

    xy_eval = torch.tensor(xy_dense[in_domain], dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = model(xy_eval)
        u = pred[:, 0].cpu().numpy()
        v = pred[:, 1].cpu().numpy()
        p = pred[:, 2].cpu().numpy()

    # Create full arrays with NaN outside domain
    U_full = np.full(len(xy_dense), np.nan)
    V_full = np.full(len(xy_dense), np.nan)
    P_full = np.full(len(xy_dense), np.nan)

    U_full[in_domain] = u
    V_full[in_domain] = v
    P_full[in_domain] = p

    U = U_full.reshape(ny, nx)
    V = V_full.reshape(ny, nx)
    P = P_full.reshape(ny, nx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(U, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
    axes[0].set_title('u-velocity')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(V, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
    axes[1].set_title('v-velocity')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(P, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
    axes[2].set_title('Pressure')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    plt.close()

# =============================================================================
# Run Experiments
# =============================================================================
if __name__ == "__main__":
    os.makedirs('results/dt_pinn_battery', exist_ok=True)

    EXPERIMENTS = [
        {'name': 'Baseline (autodiff)', 'dt_epochs': 0, 'auto_epochs': NUM_EPOCHS},
        {'name': 'Pure DT-PINN', 'dt_epochs': NUM_EPOCHS, 'auto_epochs': 0},
        {'name': 'Hybrid 25-75', 'dt_epochs': int(NUM_EPOCHS * 0.25), 'auto_epochs': int(NUM_EPOCHS * 0.75)},
    ]

    results = []

    print("\n" + "=" * 70)
    print(f"RUNNING BATTERY CHANNEL EXPERIMENTS ({NUM_EPOCHS} epochs)")
    print("=" * 70)

    for exp in EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"Experiment: {exp['name']}")
        print(f"DT-PINN: {exp['dt_epochs']}, Autodiff: {exp['auto_epochs']}")
        print("=" * 70)

        model, total_time, loss_history = train_experiment(
            exp['dt_epochs'], exp['auto_epochs'], verbose=True
        )
        metrics = evaluate_model(model)

        result = {
            'name': exp['name'],
            'dt_epochs': exp['dt_epochs'],
            'auto_epochs': exp['auto_epochs'],
            'total_time': total_time,
            'final_loss': loss_history[-1] if loss_history else None,
            **metrics,
        }
        results.append(result)

        print(f"\n  RESULT: Time={total_time:.1f}s, PDE_RMS={metrics['pde_rms']:.5f}")

        # Visualize
        viz_path = f"results/dt_pinn_battery/{exp['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        visualize_solution(model, viz_path)

        # Save results
        with open('results/dt_pinn_battery/results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("BATTERY CHANNEL EXPERIMENT RESULTS")
    print("=" * 70)

    baseline = [r for r in results if 'Baseline' in r['name']][0]

    print(f"\n{'Config':<25} {'Time':<12} {'PDE RMS':<12} {'Speedup':<10}")
    print("-" * 60)
    for r in results:
        speedup = baseline['total_time'] / r['total_time']
        print(f"{r['name']:<25} {r['total_time']:.1f}s       {r['pde_rms']:.5f}      {speedup:.2f}x")

    print("\nResults saved to results/dt_pinn_battery/")
