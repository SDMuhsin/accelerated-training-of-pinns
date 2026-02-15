#!/usr/bin/env python3
"""
Unified Benchmark Script: Lid-Driven Cavity NS+Smagorinsky

Single CLI entry point for all PINN training methods. Runs ONE training+evaluation
pass and appends results as a row to a CSV file.

Methods:
  autodiff   - Plain autograd PINN (no spectral matrices)
  dtpinn     - Standard DT-PINN (Chebyshev spectral matrices)
  analytical - Analytical Jacobian (our method)
  ropinn     - RoPINN (region-optimized, autograd-based)
  pielm      - PIELM (extreme learning machine)

Technique modifiers:
  none       - No modification
  amp        - Mixed precision (torch.amp)
  compile    - torch.compile on model
  cuda-graph - CUDA graph capture (only works with analytical)

Usage:
  python -u src/lid_benchmark.py --method analytical --seed 42
  python -u src/lid_benchmark.py --method autodiff --seed 42 --epochs 30000
  python -u src/lid_benchmark.py --method pielm --seed 42
  # Shell sweep:
  for seed in 42 43 44 45 46; do
    python -u src/lid_benchmark.py --method analytical --seed $seed
  done
"""

import argparse
import csv
import fcntl
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Physics constants
# =============================================================================
Re = 1000.0
U_lid = 1.0
nu_laminar = U_lid / Re  # 0.001
Cs = 0.1

# Kovasznay flow constants
Re_kov = 40.0
nu_kov = 1.0 / Re_kov  # 0.025
lambda_kov = Re_kov / 2.0 - math.sqrt(Re_kov**2 / 4.0 + 4.0 * math.pi**2)

# Elasticity constants (Lamé parameters)
lam_e = 1.0   # Lamé first parameter λ
mu_e = 0.5    # Shear modulus μ
Q_e = 4.0     # Load parameter for manufactured solution

LOG_INTERVAL = 5000

mse = nn.MSELoss()

# Lazy-initialized generated backward functions from symbolic VJP engine
_generated_dense_backward = None
_generated_sparse_backward = None

def _get_generated_backward(sparse=False):
    """Lazily generate and cache backward functions from symbolic VJP engine."""
    global _generated_dense_backward, _generated_sparse_backward
    if sparse:
        if _generated_sparse_backward is None:
            from src.symbolic_vjp import TracedVar, trace_pde_forward, emit_backward
            tape = []
            outputs, inputs = trace_pde_forward(compute_pde_terms_sparse, None, tape, sparse=True)
            _, _generated_sparse_backward = emit_backward(
                tape, list(outputs), ['dc', 'dmu', 'dmv'], inputs, sparse=True)
        return _generated_sparse_backward
    else:
        if _generated_dense_backward is None:
            from src.symbolic_vjp import TracedVar, trace_pde_forward, emit_backward
            tape = []
            outputs, inputs = trace_pde_forward(compute_pde_terms, None, tape, sparse=False)
            _, _generated_dense_backward = emit_backward(
                tape, list(outputs), ['dc', 'dmu', 'dmv'], inputs, sparse=False)
        return _generated_dense_backward


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified benchmark for PINN methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--problem", default="cavity",
                        choices=["cavity", "kovasznay", "elasticity"],
                        help="PDE problem to solve")
    parser.add_argument("--method", required=True,
                        choices=["autodiff", "dtpinn", "analytical", "ropinn", "pielm", "sk-pinn", "sage"],
                        help="Training method")
    parser.add_argument("--model", default="mlp", choices=["mlp", "tsa-pinn", "pirate-net"],
                        help="Network architecture (ignored for pielm)")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "lbfgs"],
                        help="Optimizer (ignored for pielm)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=30000,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="Grid N (NxN points). Default: 50 for Chebyshev methods, 200 for sk-pinn")
    parser.add_argument("--technique", default="none",
                        choices=["none", "amp", "compile", "cuda-graph"],
                        help="Optional technique modifier")
    parser.add_argument("--output-csv", default="results/lid_benchmark_results.csv",
                        help="Path to CSV output file")
    parser.add_argument("--tag", default="",
                        help="Optional string tag for this run")
    parser.add_argument("--track", action="store_true",
                        help="Enable per-epoch tracking for ablation studies")
    parser.add_argument("--track-interval", type=int, default=100,
                        help="Epoch interval for tracking evaluations")
    return parser.parse_args()


# =============================================================================
# Network architecture (shared across all gradient-based methods)
# =============================================================================
class PINN_Cavity(nn.Module):
    """6-layer/64-unit tanh MLP. Output: (u, v, p). 21,827 params."""
    def __init__(self, output_dim=3):
        super().__init__()
        layers = [nn.Linear(2, 64), nn.Tanh()]
        for _ in range(5):
            layers.extend([nn.Linear(64, 64), nn.Tanh()])
        layers.append(nn.Linear(64, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_model(model_name, output_dim=3):
    """Factory function: create a model by name."""
    if model_name == "mlp":
        return PINN_Cavity(output_dim=output_dim)
    elif model_name == "tsa-pinn":
        from src.experiment_dt_elm_pinn.models.tsa_pinn import TSA_PINN_Cavity
        return TSA_PINN_Cavity(initial_freq=1.0, output_dim=output_dim)
    elif model_name == "pirate-net":
        from src.experiment_dt_elm_pinn.models.pirate_net import PirateNet_Cavity
        return PirateNet_Cavity(output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def model_reg_loss(model):
    """Get regularization loss from model (0 if not supported)."""
    if hasattr(model, 'regularization_loss'):
        return model.regularization_loss()
    return 0.0


# =============================================================================
# Chebyshev grid + spectral matrices (for dtpinn, analytical)
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


def build_grid_data(N_grid, device):
    """Build Chebyshev grid + spectral differentiation matrices."""
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
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    is_lid = (yc > 1 - eps)
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

    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
    xy_batched = torch.cat([xy_all, xy_lid, xy_wall, xy_center], dim=0)
    off_lid = N_all
    off_wall = N_all + N_lid
    off_center = N_all + N_lid + N_wall

    Dx_h = Dx.half()
    Dy_h = Dy.half()
    DxT_h = DxT.half()
    DyT_h = DyT.half()

    return {
        'Dx': Dx, 'Dy': Dy, 'DxT': DxT, 'DyT': DyT,
        'Dx_h': Dx_h, 'Dy_h': Dy_h, 'DxT_h': DxT_h, 'DyT_h': DyT_h,
        'xy_all': xy_all, 'xy_lid': xy_lid, 'xy_wall': xy_wall,
        'xy_batched': xy_batched,
        'interior_idx': interior_idx, 'lid_idx': lid_idx, 'wall_idx': wall_idx,
        'interior_mask': interior_mask, 'd_wall': d_wall, 'Cs_d_sq': Cs_d_sq,
        'N_all': N_all, 'N_lid': N_lid, 'N_wall': N_wall, 'M': M,
        'off_lid': off_lid, 'off_wall': off_wall, 'off_center': off_center,
        'N_grid': N_grid,
    }


def build_collocation_points(N_grid, device):
    """Build Chebyshev collocation points for autograd methods (no spectral matrices)."""
    x_ref = chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    is_lid = (yc > 1 - eps)
    is_wall = is_boundary & ~is_lid

    interior_idx = np.where(~is_boundary)[0]
    lid_idx = np.where(is_lid)[0]
    wall_idx = np.where(is_wall)[0]

    return {
        'xy_interior': torch.tensor(xy_grid[interior_idx], dtype=torch.float32, device=device),
        'xy_lid': torch.tensor(xy_grid[lid_idx], dtype=torch.float32, device=device),
        'xy_wall': torch.tensor(xy_grid[wall_idx], dtype=torch.float32, device=device),
        'N_interior': len(interior_idx),
        'N_lid': len(lid_idx),
        'N_wall': len(wall_idx),
        'N_grid': N_grid,
    }


# =============================================================================
# SK-PINN: RKPM smoothing-kernel differentiation matrices (uniform grid)
# =============================================================================
def _sk_find_neighborhoods(coords, radius):
    """KD-tree neighbor search returning padded tensors."""
    from scipy.spatial import cKDTree
    kdtree = cKDTree(coords)
    neighborhoods = []
    distances = []
    distance_vectors = []
    for point in coords:
        indices = kdtree.query_ball_point(point, radius)
        nbr_coords = coords[indices]
        dist = np.linalg.norm(nbr_coords - point, axis=1)
        dist_vec = nbr_coords - point
        neighborhoods.append(torch.tensor(indices, dtype=torch.long))
        distances.append(torch.tensor(dist, dtype=torch.float64))
        distance_vectors.append(torch.tensor(dist_vec, dtype=torch.float64))
    neighborhoods = torch.nn.utils.rnn.pad_sequence(neighborhoods, batch_first=True, padding_value=-1)
    distances = torch.nn.utils.rnn.pad_sequence(distances, batch_first=True, padding_value=-1)
    distance_vectors = torch.nn.utils.rnn.pad_sequence(distance_vectors, batch_first=True, padding_value=0)
    return neighborhoods, distances, distance_vectors


def _sk_sph_kernel(distances, h):
    """Cubic spline SPH kernel (2D)."""
    q = distances / h
    result = torch.zeros_like(distances, dtype=torch.float64)
    within_range = (0 <= q) & (q <= 2)
    q_in = q[within_range]
    result[within_range] = (15.0 / (7.0 * np.pi * h ** 2)) * (
        (2.0/3.0 - q_in ** 2 + 0.5 * q_in ** 3) * (q_in <= 1) +
        (1.0/6.0 * (2.0 - q_in) ** 3) * ((1 < q_in) & (q_in <= 2))
    )
    return result


def _sk_compute_C(distance_vectors, kernel, dxdy, order):
    """RKPM correction coefficients (reproducing kernel particle method)."""
    moment_terms = [torch.ones(kernel.shape, dtype=torch.float64)]
    terms_num = sum(range(1, order + 2))
    for i in range(1, order + 1):
        for j in range(i + 1):
            term = (distance_vectors[:, :, 0:1] ** (i - j)) * (distance_vectors[:, :, 1:2] ** j) / (dxdy ** (i / 2))
            moment_terms.append(term)
    moment_vector = torch.cat(moment_terms, dim=2)
    # H matrix maps moment vector to derivative operators
    H = torch.tensor([
        [0, 1.0 / dxdy ** 0.5, 0, 0, 0, 0],
        [0, 0, 1.0 / dxdy ** 0.5, 0, 0, 0],
        [0, 0, 0, 2.0 / dxdy, 0, 0],
        [0, 0, 0, 0, 1.0 / dxdy, 0],
        [0, 0, 0, 0, 0, 2.0 / dxdy],
    ], dtype=torch.float64)
    H0 = torch.nn.functional.pad(H, (0, terms_num - H.shape[1]), value=0)
    matrix = torch.matmul(moment_vector.unsqueeze(3), moment_vector.unsqueeze(2)) * kernel.unsqueeze(-1)
    matrix_sum = torch.sum(matrix, dim=1)
    matrix_inverse = torch.inverse(matrix_sum)
    C = torch.matmul(
        torch.matmul(moment_vector, matrix_inverse),
        H0.t().view(1, 1, terms_num, -1)
    ).squeeze(0)
    return C


def build_sk_data(N_grid, device):
    """Build SK-PINN differentiation matrices on a uniform grid.

    Uses RKPM (Reproducing Kernel Particle Method) with SPH cubic spline
    kernels to construct Dx, Dy matrices on a uniform N×N grid over [0,1]².
    Returns the same dict format as build_grid_data() so it can be used
    as a drop-in replacement for train_dtpinn() / train_analytical().
    """
    dx = 1.0 / (N_grid - 1)
    h = dx * 1.4
    radius = 2.0 * h
    dxdy = dx * dx

    # Uniform grid
    x = np.linspace(0, 1, N_grid)
    xx, yy = np.meshgrid(x, x, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])
    N_all = len(xy_grid)

    print(f"  SK-PINN: building RKPM matrices for {N_grid}x{N_grid} uniform grid...")
    print(f"  dx={dx:.6f}, h={h:.6f}, radius={radius:.6f}")

    # Neighbor search + kernel + RKPM correction
    neighborhoods, distances, distance_vectors = _sk_find_neighborhoods(xy_grid, radius)
    kernel = _sk_sph_kernel(distances, h)
    C = _sk_compute_C(distance_vectors, kernel.unsqueeze(-1), dxdy, order=2)
    # C shape: [N_all, max_neighbors, 5]
    # C[:,:,0] = du/dx weights, C[:,:,1] = du/dy weights

    # Assemble sparse Dx, Dy matrices (COO format)
    kernel_np = kernel.numpy()
    C_np = C.numpy()
    nb_np = neighborhoods.numpy()

    rows, cols, dx_vals, dy_vals = [], [], [], []
    for i in range(N_all):
        for j_idx in range(nb_np.shape[1]):
            j = nb_np[i, j_idx]
            if j == -1:
                break
            w = kernel_np[i, j_idx]
            dx_val = C_np[i, j_idx, 0] * w
            dy_val = C_np[i, j_idx, 1] * w
            if dx_val != 0 or dy_val != 0:
                rows.append(i)
                cols.append(j)
                dx_vals.append(dx_val)
                dy_vals.append(dy_val)

    nnz = len(rows)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    Dx = torch.sparse_coo_tensor(
        indices, torch.tensor(dx_vals, dtype=torch.float32), (N_all, N_all)
    ).to(device).coalesce()
    Dy = torch.sparse_coo_tensor(
        indices, torch.tensor(dy_vals, dtype=torch.float32), (N_all, N_all)
    ).to(device).coalesce()

    print(f"  SK-PINN: sparse Dx/Dy nnz={nnz}/{N_all*N_all} "
          f"({100*nnz/(N_all*N_all):.2f}% dense)")

    # Boundary classification (same logic as build_grid_data)
    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    is_lid = (yc > 1 - eps)
    is_wall = is_boundary & ~is_lid

    interior_idx = np.where(~is_boundary)[0]
    lid_idx = np.where(is_lid)[0]
    wall_idx = np.where(is_wall)[0]

    N_lid = len(lid_idx)
    N_wall = len(wall_idx)
    M = len(interior_idx)

    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_lid = xy_all[lid_idx]
    xy_wall = xy_all[wall_idx]

    x_t = xy_all[:, 0:1]
    y_t = xy_all[:, 1:2]
    d_wall = torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t))
    Cs_d_sq = (Cs * d_wall) ** 2

    interior_mask = torch.zeros(N_all, 1, device=device)
    interior_mask[interior_idx] = 1.0

    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    print(f"  SK-PINN: N_all={N_all}, interior={M}, lid={N_lid}, wall={N_wall}")

    return {
        'Dx': Dx, 'Dy': Dy,
        'sparse': True,
        'xy_all': xy_all, 'xy_lid': xy_lid, 'xy_wall': xy_wall,
        'xy_center': xy_center,
        'interior_idx': interior_idx, 'lid_idx': lid_idx, 'wall_idx': wall_idx,
        'interior_mask': interior_mask, 'd_wall': d_wall, 'Cs_d_sq': Cs_d_sq,
        'N_all': N_all, 'N_lid': N_lid, 'N_wall': N_wall, 'M': M,
        'N_grid': N_grid,
    }


# =============================================================================
# Autograd helpers
# =============================================================================
def gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True)[0]


def pde_residuals_autodiff(model, xy):
    """Full NS+Smagorinsky PDE residuals via autograd. xy must have requires_grad=True."""
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    x_coord, y_coord = xy[:, 0:1], xy[:, 1:2]
    d_wall = torch.min(torch.min(x_coord, 1.0 - x_coord),
                       torch.min(y_coord, 1.0 - y_coord))
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + (Cs * d_wall)**2 * S_mag

    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    grad_p = gradients(p, xy)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
    grad_qx_u = gradients(qx_u, xy)
    grad_qy_u = gradients(qy_u, xy)
    grad_qx_v = gradients(qx_v, xy)
    grad_qy_v = gradients(qy_v, xy)
    visc_u = grad_qx_u[:, 0:1] + grad_qy_u[:, 1:2]
    visc_v = grad_qx_v[:, 0:1] + grad_qy_v[:, 1:2]

    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    return continuity, mom_u, mom_v


# =============================================================================
# Spectral PDE computation
# =============================================================================
def compute_pde_terms(pred, g):
    """PDE residuals via spectral differentiation matrices."""
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
    return continuity, mom_u, mom_v


def compute_pde_terms_sparse(pred, g):
    """PDE residuals via sparse RKPM differentiation matrices."""
    Dx, Dy = g['Dx'], g['Dy']
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    du_dx = torch.sparse.mm(Dx, u); du_dy = torch.sparse.mm(Dy, u)
    dv_dx = torch.sparse.mm(Dx, v); dv_dy = torch.sparse.mm(Dy, v)
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_laminar + g['Cs_d_sq'] * S_mag
    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    dp_dx = torch.sparse.mm(Dx, p); dp_dy = torch.sparse.mm(Dy, p)
    visc_u = torch.sparse.mm(Dx, nu_eff * du_dx) + torch.sparse.mm(Dy, nu_eff * du_dy)
    visc_v = torch.sparse.mm(Dx, nu_eff * dv_dx) + torch.sparse.mm(Dy, nu_eff * dv_dy)
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    return continuity, mom_u, mom_v


# =============================================================================
# Kovasznay flow — exact solution, PDE forward, grid builders
# =============================================================================
def kovasznay_exact(x, y):
    """Exact Kovasznay solution. x, y are tensors.

    Returns (u, v, p) each with shape matching input.
    """
    lam = lambda_kov
    u = 1.0 - torch.exp(lam * x) * torch.cos(2.0 * math.pi * y)
    v = (lam / (2.0 * math.pi)) * torch.exp(lam * x) * torch.sin(2.0 * math.pi * y)
    p = 0.5 * (1.0 - torch.exp(2.0 * lam * x))
    return u, v, p


def compute_pde_kovasznay(pred, g):
    """Kovasznay PDE residuals via spectral differentiation matrices.

    Standard incompressible NS with constant viscosity nu_kov = 1/Re_kov.
    No Smagorinsky model.
    """
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    du_dx = g['Dx'] @ u;  du_dy = g['Dy'] @ u
    dv_dx = g['Dx'] @ v;  dv_dy = g['Dy'] @ v
    dp_dx = g['Dx'] @ p;  dp_dy = g['Dy'] @ p

    # Second derivatives for viscous term (constant viscosity)
    d2u_dx2 = g['Dx'] @ du_dx;  d2u_dy2 = g['Dy'] @ du_dy
    d2v_dx2 = g['Dx'] @ dv_dx;  d2v_dy2 = g['Dy'] @ dv_dy

    # Continuity: du/dx + dv/dy = 0
    continuity = du_dx + dv_dy

    # Momentum-u: u*du/dx + v*du/dy + dp/dx - nu*(d2u/dx2 + d2u/dy2) = 0
    mom_u = u * du_dx + v * du_dy + dp_dx - nu_kov * (d2u_dx2 + d2u_dy2)

    # Momentum-v: u*dv/dx + v*dv/dy + dp/dy - nu*(d2v/dx2 + d2v/dy2) = 0
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu_kov * (d2v_dx2 + d2v_dy2)

    return continuity, mom_u, mom_v


def build_grid_data_kovasznay(N_grid, device):
    """Build Chebyshev grid + spectral differentiation matrices for Kovasznay flow.

    Domain: [-0.5, 1.0] x [-0.5, 1.5] (Lx=1.5, Ly=2.0).
    Non-square domain requires separate Dx/Dy scaling.
    All boundaries: Dirichlet BCs from exact solution.
    """
    Lx, Ly = 1.5, 2.0
    x0, y0 = -0.5, -0.5

    D1d = chebyshev_diff_matrix(N_grid)
    # Scale for non-square domain: D_phys = D_ref * (2/L)
    Dx_1d = D1d * (2.0 / Lx)
    Dy_1d = D1d * (2.0 / Ly)

    I_mat = np.eye(N_grid)
    Dx_np = np.kron(I_mat, Dx_1d)
    Dy_np = np.kron(Dy_1d, I_mat)

    x_ref = chebyshev_points(N_grid)
    x_phys = x0 + Lx * 0.5 * (x_ref + 1.0)  # map [-1,1] -> [x0, x0+Lx]
    y_phys = y0 + Ly * 0.5 * (x_ref + 1.0)   # map [-1,1] -> [y0, y0+Ly]
    xx, yy = np.meshgrid(x_phys, y_phys, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = ((xc < x0 + eps) | (xc > x0 + Lx - eps) |
                   (yc < y0 + eps) | (yc > y0 + Ly - eps))

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    N_all = len(xy_grid)
    N_bc = len(bc_idx)
    M = len(interior_idx)

    Dx = torch.tensor(Dx_np, dtype=torch.float32, device=device)
    Dy = torch.tensor(Dy_np, dtype=torch.float32, device=device)
    DxT = Dx.T.contiguous()
    DyT = Dy.T.contiguous()
    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_bc = xy_all[bc_idx]

    interior_mask = torch.zeros(N_all, 1, device=device)
    interior_mask[interior_idx] = 1.0

    # Exact BC values
    x_bc = xy_bc[:, 0:1]
    y_bc = xy_bc[:, 1:2]
    u_ex, v_ex, p_ex = kovasznay_exact(x_bc, y_bc)
    bc_target = torch.cat([u_ex, v_ex, p_ex], dim=1)  # (N_bc, 3)

    # Pressure reference at domain center
    x_center = torch.tensor([[x0 + Lx / 2, y0 + Ly / 2]], dtype=torch.float32, device=device)
    u_ctr, v_ctr, p_ctr = kovasznay_exact(x_center[:, 0:1], x_center[:, 1:2])
    p_center_exact = p_ctr.item()

    # Batched input: all grid + BC points + center
    xy_batched = torch.cat([xy_all, xy_bc, x_center], dim=0)
    off_bc = N_all
    off_center = N_all + N_bc

    return {
        'Dx': Dx, 'Dy': Dy, 'DxT': DxT, 'DyT': DyT,
        'xy_all': xy_all, 'xy_bc': xy_bc, 'xy_batched': xy_batched,
        'interior_idx': interior_idx, 'bc_idx': bc_idx,
        'interior_mask': interior_mask,
        'bc_target': bc_target, 'p_center_exact': p_center_exact,
        'N_all': N_all, 'N_bc': N_bc, 'M': M,
        'off_bc': off_bc, 'off_center': off_center,
        'N_grid': N_grid, 'Lx': Lx, 'Ly': Ly,
    }


def build_collocation_points_kovasznay(N_grid, device):
    """Build Chebyshev collocation points for Kovasznay autograd method."""
    Lx, Ly = 1.5, 2.0
    x0, y0 = -0.5, -0.5

    x_ref = chebyshev_points(N_grid)
    x_phys = x0 + Lx * 0.5 * (x_ref + 1.0)
    y_phys = y0 + Ly * 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, y_phys, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = ((xc < x0 + eps) | (xc > x0 + Lx - eps) |
                   (yc < y0 + eps) | (yc > y0 + Ly - eps))

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    xy_int = torch.tensor(xy_grid[interior_idx], dtype=torch.float32, device=device)
    xy_bc = torch.tensor(xy_grid[bc_idx], dtype=torch.float32, device=device)

    # Exact BC values
    u_ex, v_ex, p_ex = kovasznay_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([u_ex, v_ex, p_ex], dim=1)

    # Center point
    x_center = torch.tensor([[-0.5 + Lx / 2, -0.5 + Ly / 2]], dtype=torch.float32, device=device)
    _, _, p_ctr = kovasznay_exact(x_center[:, 0:1], x_center[:, 1:2])
    p_center_exact = p_ctr.item()

    return {
        'xy_interior': xy_int,
        'xy_bc': xy_bc,
        'bc_target': bc_target,
        'xy_center': x_center,
        'p_center_exact': p_center_exact,
        'N_interior': len(interior_idx),
        'N_bc': len(bc_idx),
        'N_grid': N_grid,
    }


def pde_residuals_kovasznay_autodiff(model, xy):
    """Kovasznay PDE residuals via autograd. xy must have requires_grad=True."""
    pred = model(xy)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = gradients(u, xy)
    grad_v = gradients(v, xy)
    grad_p = gradients(p, xy)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # Second derivatives
    grad_du_dx = gradients(du_dx, xy)
    grad_du_dy = gradients(du_dy, xy)
    grad_dv_dx = gradients(dv_dx, xy)
    grad_dv_dy = gradients(dv_dy, xy)
    d2u_dx2 = grad_du_dx[:, 0:1]
    d2u_dy2 = grad_du_dy[:, 1:2]
    d2v_dx2 = grad_dv_dx[:, 0:1]
    d2v_dy2 = grad_dv_dy[:, 1:2]

    continuity = du_dx + dv_dy
    mom_u = u * du_dx + v * du_dy + dp_dx - nu_kov * (d2u_dx2 + d2u_dy2)
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu_kov * (d2v_dx2 + d2v_dy2)

    return continuity, mom_u, mom_v


# Lazy-initialized generated backward for Kovasznay
_generated_kovasznay_backward = None

def _get_generated_backward_kovasznay():
    """Lazily generate and cache backward function for Kovasznay PDE."""
    global _generated_kovasznay_backward
    if _generated_kovasznay_backward is None:
        from src.symbolic_vjp import trace_pde_forward, emit_backward
        tape = []
        outputs, inputs = trace_pde_forward(
            compute_pde_kovasznay, None, tape, sparse=False,
            constants=['Dx', 'Dy'])
        _, _generated_kovasznay_backward = emit_backward(
            tape, list(outputs), ['dc', 'dmu', 'dmv'], inputs, sparse=False,
            func_name='generated_kovasznay_grad')
    return _generated_kovasznay_backward


# =============================================================================
# Analytical Jacobian backward
# =============================================================================
def compute_analytical_grad(pred_det, g):
    """Compute dL/dpred analytically for PDE loss terms."""
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

    dL_dp = DxT @ dmu + DyT @ dmv

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

    dL_dv = DyT @ dc
    dL_dv = dL_dv + du_dy * dmu
    dL_dv = dL_dv + DxT @ (u * dmv) + dv_dy * dmv + DyT @ (v * dmv)
    dL_dv = dL_dv + DxT @ (nu_eff * wxv) + DyT @ (nu_eff * wyv)
    dL_dv = dL_dv + DxT @ (av * gv) + DyT @ (bv * gv)
    dL_dv = dL_dv + DxT @ (av * gu) + DyT @ (bv * gu)

    grad_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1)
    return grad_pde


def compute_analytical_grad_amp(pred_det, g):
    """Analytical gradient in float16 for AMP."""
    u = pred_det[:g['N_all'], 0:1]
    v = pred_det[:g['N_all'], 1:2]
    p = pred_det[:g['N_all'], 2:3]

    Dx, Dy = g['Dx_h'], g['Dy_h']
    DxT, DyT = g['DxT_h'], g['DyT_h']

    u_h = u.half(); v_h = v.half(); p_h = p.half()

    du_dx = Dx @ u_h; du_dy = Dy @ u_h
    dv_dx = Dx @ v_h; dv_dy = Dy @ v_h
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_sq = 2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-4
    S_mag = torch.sqrt(S_sq)
    inv_S = 1.0 / S_mag
    Cs_d_sq_h = g['Cs_d_sq'].half()
    nu_eff = nu_laminar + Cs_d_sq_h * S_mag

    continuity = du_dx + dv_dy
    dp_dx = Dx @ p_h; dp_dy = Dy @ p_h
    visc_u = Dx @ (nu_eff * du_dx) + Dy @ (nu_eff * du_dy)
    visc_v = Dx @ (nu_eff * dv_dx) + Dy @ (nu_eff * dv_dy)
    mom_u = u_h * du_dx + v_h * du_dy + dp_dx - visc_u
    mom_v = u_h * dv_dx + v_h * dv_dy + dp_dy - visc_v

    M = g['M']
    scale = 2.0 / M
    mask_h = g['interior_mask'].half()
    dc = continuity * scale * mask_h
    dmu = mom_u * scale * mask_h
    dmv = mom_v * scale * mask_h

    au = Cs_d_sq_h * 2.0 * Sxx * inv_S
    bu = Cs_d_sq_h * 2.0 * Sxy * inv_S
    av = bu
    bv = Cs_d_sq_h * 2.0 * Syy * inv_S

    dL_dp = DxT @ dmu + DyT @ dmv

    dL_du = DxT @ dc
    dL_du = dL_du + du_dx * dmu + DxT @ (u_h * dmu) + DyT @ (v_h * dmu)
    dL_du = dL_du + dv_dx * dmv
    ndmu = -dmu; ndmv = -dmv
    wxu = DxT @ ndmu; wyu = DyT @ ndmu
    dL_du = dL_du + DxT @ (nu_eff * wxu) + DyT @ (nu_eff * wyu)
    gu = du_dx * wxu + du_dy * wyu
    dL_du = dL_du + DxT @ (au * gu) + DyT @ (bu * gu)
    wxv = DxT @ ndmv; wyv = DyT @ ndmv
    gv = dv_dx * wxv + dv_dy * wyv
    dL_du = dL_du + DxT @ (au * gv) + DyT @ (bu * gv)

    dL_dv = DyT @ dc
    dL_dv = dL_dv + du_dy * dmu
    dL_dv = dL_dv + DxT @ (u_h * dmv) + dv_dy * dmv + DyT @ (v_h * dmv)
    dL_dv = dL_dv + DxT @ (nu_eff * wxv) + DyT @ (nu_eff * wyv)
    dL_dv = dL_dv + DxT @ (av * gv) + DyT @ (bv * gv)
    dL_dv = dL_dv + DxT @ (av * gu) + DyT @ (bv * gu)

    grad_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1).float()
    return grad_pde


# =============================================================================
# RoPINN gradient variance
# =============================================================================
ROPINN_INITIAL_REGION = 1e-4
ROPINN_SAMPLE_NUM = 1
ROPINN_PAST_ITERATIONS = 10
ROPINN_REGION_MAX = 0.01


def compute_gradient_variance(gradient_list):
    """Normalized gradient variance for trust region calibration."""
    if len(gradient_list) < 2:
        return 1.0
    gradient_array = np.array(gradient_list)
    std_grad = np.std(gradient_array, axis=0)
    mean_abs_grad = np.mean(np.abs(gradient_array), axis=0) + 1e-6
    variance = float((std_grad / mean_abs_grad).mean())
    if variance == 0:
        variance = 1.0
    return variance


# =============================================================================
# Evaluation (autodiff on 51x51 uniform grid -- same for ALL methods)
# =============================================================================
def evaluate_model(model, device):
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


def evaluate_pielm(pielm_model):
    """Evaluate PIELM model on 51x51 uniform grid (numpy-based, matches autodiff eval physics)."""
    nx, ny = 51, 51
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])

    u, v, p, grads = pielm_model.predict_with_gradients(xy_eval)
    du_dx = grads['du_dx']; du_dy = grads['du_dy']
    dv_dx = grads['dv_dx']; dv_dy = grads['dv_dy']
    dp_dx = grads['dp_dx']; dp_dy = grads['dp_dy']

    # Smagorinsky
    d_wall = pielm_model._compute_wall_distance(xy_eval)
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = np.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = pielm_model.nu_laminar + (pielm_model.Cs * d_wall)**2 * S_mag

    # Continuity
    continuity = du_dx + dv_dy

    # Momentum (use Laplacian for viscous term — same as PIELM's own evaluation)
    LapH = pielm_model._compute_laplacian_features(xy_eval)
    Lap_u = LapH @ pielm_model.beta_u
    Lap_v = LapH @ pielm_model.beta_v

    mom_u = u * du_dx + v * du_dy + dp_dx - nu_eff * Lap_u
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu_eff * Lap_v

    pde_rms = float(np.sqrt(np.mean(continuity**2 + mom_u**2 + mom_v**2)))
    cont_rms = float(np.sqrt(np.mean(continuity**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u**2 + mom_v**2)))

    return {'pde_rms': pde_rms, 'continuity_rms': cont_rms, 'momentum_rms': mom_rms}


# =============================================================================
# Training tracker for ablation studies
# =============================================================================

TRACKING_COLUMNS = [
    'problem', 'method', 'model', 'optimizer', 'lr', 'seed', 'grid_size',
    'technique', 'tag', 'epoch', 'train_loss',
    'pde_rms', 'continuity_rms', 'momentum_rms',
    'u_rms_error', 'v_rms_error', 'p_rms_error',
]


class TrainingTracker:
    """Periodically evaluates model during training and writes per-epoch stats to CSV."""

    def __init__(self, args, device):
        self.interval = args.track_interval
        self.is_kovasznay = (args.problem == "kovasznay")
        self.is_elasticity = (args.problem == "elasticity")
        self.device = device
        tag_part = f"_{args.tag}" if args.tag else ""
        self.csv_path = (f"results/tracking_{args.problem}_{args.method}"
                         f"_{args.model}_s{args.seed}{tag_part}.csv")
        self.metadata = {
            'problem': args.problem,
            'method': args.method,
            'model': args.model if args.method != "pielm" else "pielm",
            'optimizer': args.optimizer,
            'lr': args.lr,
            'seed': args.seed,
            'grid_size': args.grid_size,
            'technique': args.technique,
            'tag': args.tag,
        }
        self._initialized = False

    def _init_csv(self):
        """Write CSV header on first call."""
        os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRACKING_COLUMNS)
            writer.writeheader()
        self._initialized = True

    def step(self, epoch, train_loss, model):
        """Call every epoch. At self.interval boundaries, evaluate and log."""
        if (epoch + 1) % self.interval != 0:
            return
        if not self._initialized:
            self._init_csv()

        # Unwrap compiled model if needed
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        if self.is_elasticity:
            metrics = evaluate_elasticity(base_model, self.device)
        elif self.is_kovasznay:
            metrics = evaluate_kovasznay(base_model, self.device)
        else:
            metrics = evaluate_model(base_model, self.device)

        row = dict(self.metadata)
        row['epoch'] = epoch + 1
        row['train_loss'] = round(train_loss, 8) if not math.isnan(train_loss) else 'NaN'
        row['pde_rms'] = round(metrics['pde_rms'], 6)
        row['continuity_rms'] = round(metrics['continuity_rms'], 6)
        row['momentum_rms'] = round(metrics['momentum_rms'], 6)
        row['u_rms_error'] = round(metrics['u_rms_error'], 6) if 'u_rms_error' in metrics and not math.isnan(metrics.get('u_rms_error', float('nan'))) else ''
        row['v_rms_error'] = round(metrics['v_rms_error'], 6) if 'v_rms_error' in metrics and not math.isnan(metrics.get('v_rms_error', float('nan'))) else ''
        row['p_rms_error'] = round(metrics['p_rms_error'], 6) if 'p_rms_error' in metrics and not math.isnan(metrics.get('p_rms_error', float('nan'))) else ''

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRACKING_COLUMNS)
            writer.writerow(row)


# =============================================================================
# TRAINING METHODS
# =============================================================================

# --- Method: autodiff ---
def train_autodiff(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Plain autodiff PINN: autograd derivatives, separate forward passes."""
    coll = build_collocation_points(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        model = torch.compile(model, mode='reduce-overhead')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xy_int = coll['xy_interior'].clone().requires_grad_(True)
    xy_lid = coll['xy_lid']
    xy_wall = coll['xy_wall']
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    scaler = torch.amp.GradScaler('cuda') if technique == "amp" else None
    use_amp = technique == "amp"

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                continuity, mom_u, mom_v = pde_residuals_autodiff(model, xy_int)
                loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                           mse(mom_u, torch.zeros_like(mom_u)) + \
                           mse(mom_v, torch.zeros_like(mom_v))
                pred_lid = model(xy_lid)
                loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                           mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
                pred_wall = model(xy_wall)
                loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                            mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
                pred_c = model(xy_center)
                loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
                loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            continuity, mom_u, mom_v = pde_residuals_autodiff(model, xy_int)
            loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))
            pred_lid = model(xy_lid)
            loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                       mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
            pred_wall = model(xy_wall)
            loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                        mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
            pred_c = model(xy_center)
            loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
            loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    # Unwrap compiled model for evaluation
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    return base_model, train_time, final_loss


# --- Method: dtpinn ---
def train_dtpinn(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", grid_data=None, tracker=None):
    """Standard DT-PINN: spectral matrices, separate forward passes, autograd backward."""
    g = grid_data or build_grid_data(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = torch.amp.GradScaler('cuda') if technique == "amp" else None
    use_amp = technique == "amp"

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred = compiled_model(g['xy_all'])
                continuity, mom_u, mom_v = compute_pde_terms(pred, g)
                ii = g['interior_idx']
                loss_pde = mse(continuity[ii], torch.zeros_like(continuity[ii])) + \
                           mse(mom_u[ii], torch.zeros_like(mom_u[ii])) + \
                           mse(mom_v[ii], torch.zeros_like(mom_v[ii]))
                pred_lid = compiled_model(g['xy_lid'])
                loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                           mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
                pred_wall = compiled_model(g['xy_wall'])
                loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                            mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
                pc = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
                pred_c = compiled_model(pc)
                loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
                loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = compiled_model(g['xy_all'])
            continuity, mom_u, mom_v = compute_pde_terms(pred, g)
            ii = g['interior_idx']
            loss_pde = mse(continuity[ii], torch.zeros_like(continuity[ii])) + \
                       mse(mom_u[ii], torch.zeros_like(mom_u[ii])) + \
                       mse(mom_v[ii], torch.zeros_like(mom_v[ii]))
            pred_lid = compiled_model(g['xy_lid'])
            loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                       mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
            pred_wall = compiled_model(g['xy_wall'])
            loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                        mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
            pc = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)
            pred_c = compiled_model(pc)
            loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
            loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: analytical ---
def train_analytical(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", grid_data=None, tracker=None):
    """Analytical Jacobian: batched forward, analytical backward."""
    g = grid_data or build_grid_data(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    use_amp = technique == "amp"
    use_cuda_graph = technique == "cuda-graph"

    if use_cuda_graph:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      capturable=True, foreach=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    # ---- CUDA Graph path ----
    if use_cuda_graph:
        return _train_analytical_cuda_graph(model, optimizer, g, n_epochs, device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                pred_batch = compiled_model(g['xy_batched'])
            with torch.no_grad():
                pred_pde = pred_batch[:g['N_all']]
                pred_l = pred_batch[g['off_lid']:g['off_wall']]
                pred_w = pred_batch[g['off_wall']:g['off_center']]
                pred_c = pred_batch[g['off_center']:]
                grad_pde = compute_analytical_grad_amp(pred_pde, g)
                N_lid, N_wall = g['N_lid'], g['N_wall']
                grad_lid = torch.zeros(N_lid, 3, device=device)
                grad_lid[:, 0:1] = 2.0 * (pred_l[:, 0:1].float() - 1.0) / N_lid
                grad_lid[:, 1:2] = 2.0 * pred_l[:, 1:2].float() / N_lid
                grad_wall = torch.zeros(N_wall, 3, device=device)
                grad_wall[:, 0:1] = 2.0 * pred_w[:, 0:1].float() / N_wall
                grad_wall[:, 1:2] = 2.0 * pred_w[:, 1:2].float() / N_wall
                grad_center = torch.zeros(1, 3, device=device)
                grad_center[:, 2:3] = 2.0 * pred_c[:, 2:3].float()
                upstream = torch.cat([grad_pde, grad_lid, grad_wall, grad_center], dim=0)
            pred_batch.backward(gradient=upstream)
            reg = model_reg_loss(model)
            if isinstance(reg, torch.Tensor):
                reg.backward()
            optimizer.step()
        else:
            pred_batch = compiled_model(g['xy_batched'])
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
            reg = model_reg_loss(model)
            if isinstance(reg, torch.Tensor):
                reg.backward()
            optimizer.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track:
            with torch.no_grad():
                u = pred_pde[:, 0:1]; v = pred_pde[:, 1:2]; p = pred_pde[:, 2:3]
                du_dx = g['Dx'] @ u; du_dy = g['Dy'] @ u
                dv_dx = g['Dx'] @ v; dv_dy = g['Dy'] @ v
                Sxx, Syy = du_dx, dv_dy
                Sxy = 0.5 * (du_dy + dv_dx)
                S_mag = torch.sqrt(2.0*(Sxx**2+Syy**2+2.0*Sxy**2)+1e-12)
                nu_eff_val = nu_laminar + g['Cs_d_sq'] * S_mag
                cont = du_dx + dv_dy
                dp_dx = g['Dx'] @ p; dp_dy = g['Dy'] @ p
                visc_u = g['Dx']@(nu_eff_val*du_dx) + g['Dy']@(nu_eff_val*du_dy)
                visc_v = g['Dx']@(nu_eff_val*dv_dx) + g['Dy']@(nu_eff_val*dv_dy)
                mu = u*du_dx + v*du_dy + dp_dx - visc_u
                mv = u*dv_dx + v*dv_dy + dp_dy - visc_v
                ii = g['interior_idx']
                lv = (cont[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                final_loss = lv.item()
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: pde_loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: sage (Symbolic Analytical Gradient Engine) ---
def train_sage(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """SAGE: auto-generated backward via symbolic VJP engine."""
    g = build_grid_data(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    # Pre-generate the backward function (one-time cost)
    generated_backward = _get_generated_backward(sparse=False)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_batch = compiled_model(g['xy_batched'])
        with torch.no_grad():
            pred_pde = pred_batch[:g['N_all']]
            pred_l = pred_batch[g['off_lid']:g['off_wall']]
            pred_w = pred_batch[g['off_wall']:g['off_center']]
            pred_c = pred_batch[g['off_center']:]
            grad_pde = generated_backward(pred_pde, g)
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
        reg = model_reg_loss(model)
        if isinstance(reg, torch.Tensor):
            reg.backward()
        optimizer.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track:
            with torch.no_grad():
                u = pred_pde[:, 0:1]; v = pred_pde[:, 1:2]; p = pred_pde[:, 2:3]
                du_dx = g['Dx'] @ u; du_dy = g['Dy'] @ u
                dv_dx = g['Dx'] @ v; dv_dy = g['Dy'] @ v
                Sxx, Syy = du_dx, dv_dy
                Sxy = 0.5 * (du_dy + dv_dx)
                S_mag = torch.sqrt(2.0*(Sxx**2+Syy**2+2.0*Sxy**2)+1e-12)
                nu_eff_val = nu_laminar + g['Cs_d_sq'] * S_mag
                cont = du_dx + dv_dy
                dp_dx = g['Dx'] @ p; dp_dy = g['Dy'] @ p
                visc_u = g['Dx']@(nu_eff_val*du_dx) + g['Dy']@(nu_eff_val*du_dy)
                visc_v = g['Dx']@(nu_eff_val*dv_dx) + g['Dy']@(nu_eff_val*dv_dy)
                mu = u*du_dx + v*du_dy + dp_dx - visc_u
                mv = u*dv_dx + v*dv_dy + dp_dy - visc_v
                ii = g['interior_idx']
                lv = (cont[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                final_loss = lv.item()
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: pde_loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


def _train_analytical_cuda_graph(model, optimizer, g, n_epochs, device):
    """Analytical Jacobian with CUDA graph capture."""
    # Warmup (must match graph structure exactly)
    for _ in range(11):
        optimizer.zero_grad(set_to_none=True)
        pred_batch = model(g['xy_batched'])

        pred_pde = pred_batch[:g['N_all']]
        pred_l = pred_batch[g['off_lid']:g['off_wall']]
        pred_w = pred_batch[g['off_wall']:g['off_center']]
        pred_c = pred_batch[g['off_center']:]

        u = pred_pde[:, 0:1].detach()
        v = pred_pde[:, 1:2].detach()
        p = pred_pde[:, 2:3].detach()

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

        M_val = g['M']
        scale = 2.0 / M_val
        mask = g['interior_mask']
        dc = continuity * scale * mask
        dmu = mom_u * scale * mask
        dmv = mom_v * scale * mask

        au = g['Cs_d_sq'] * 2.0 * Sxx * inv_S
        bu = g['Cs_d_sq'] * 2.0 * Sxy * inv_S
        av = bu
        bv = g['Cs_d_sq'] * 2.0 * Syy * inv_S

        DxT, DyT = g['DxT'], g['DyT']

        dL_dp = DxT @ dmu + DyT @ dmv
        dL_du = DxT @ dc + du_dx * dmu + DxT @ (u * dmu) + DyT @ (v * dmu) + dv_dx * dmv
        ndmu = -dmu; ndmv = -dmv
        wxu = DxT @ ndmu; wyu = DyT @ ndmu
        dL_du = dL_du + DxT @ (nu_eff * wxu) + DyT @ (nu_eff * wyu)
        gu_v = du_dx * wxu + du_dy * wyu
        dL_du = dL_du + DxT @ (au * gu_v) + DyT @ (bu * gu_v)
        wxv = DxT @ ndmv; wyv = DyT @ ndmv
        gv_v = dv_dx * wxv + dv_dy * wyv
        dL_du = dL_du + DxT @ (au * gv_v) + DyT @ (bu * gv_v)

        dL_dv = DyT @ dc + du_dy * dmu
        dL_dv = dL_dv + DxT @ (u * dmv) + dv_dy * dmv + DyT @ (v * dmv)
        dL_dv = dL_dv + DxT @ (nu_eff * wxv) + DyT @ (nu_eff * wyv)
        dL_dv = dL_dv + DxT @ (av * gv_v) + DyT @ (bv * gv_v)
        dL_dv = dL_dv + DxT @ (av * gu_v) + DyT @ (bv * gu_v)

        grad_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1)

        N_lid, N_wall = g['N_lid'], g['N_wall']
        grad_lid = pred_l.detach().clone()
        grad_lid[:, 0:1] = 2.0 * (pred_l[:, 0:1].detach() - 1.0) / N_lid
        grad_lid[:, 1:2] = 2.0 * pred_l[:, 1:2].detach() / N_lid
        grad_lid[:, 2:3] = 0.0

        grad_wall = pred_w.detach().clone()
        grad_wall[:, 0:1] = 2.0 * pred_w[:, 0:1].detach() / N_wall
        grad_wall[:, 1:2] = 2.0 * pred_w[:, 1:2].detach() / N_wall
        grad_wall[:, 2:3] = 0.0

        grad_center = pred_c.detach().clone()
        grad_center[:, 0:1] = 0.0
        grad_center[:, 1:2] = 0.0
        grad_center[:, 2:3] = 2.0 * pred_c[:, 2:3].detach()

        upstream = torch.cat([grad_pde, grad_lid, grad_wall, grad_center], dim=0)
        pred_batch.backward(gradient=upstream)
        optimizer.step()

    torch.cuda.synchronize()

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.graph(graph):
        pred_batch = model(g['xy_batched'])

        pred_pde = pred_batch[:g['N_all']]
        pred_l = pred_batch[g['off_lid']:g['off_wall']]
        pred_w = pred_batch[g['off_wall']:g['off_center']]
        pred_c = pred_batch[g['off_center']:]

        u = pred_pde[:, 0:1].detach()
        v = pred_pde[:, 1:2].detach()
        p = pred_pde[:, 2:3].detach()

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

        M_val = g['M']
        scale = 2.0 / M_val
        mask = g['interior_mask']
        dc = continuity * scale * mask
        dmu = mom_u * scale * mask
        dmv = mom_v * scale * mask

        au = g['Cs_d_sq'] * 2.0 * Sxx * inv_S
        bu = g['Cs_d_sq'] * 2.0 * Sxy * inv_S
        av = bu
        bv = g['Cs_d_sq'] * 2.0 * Syy * inv_S

        DxT, DyT = g['DxT'], g['DyT']

        dL_dp = DxT @ dmu + DyT @ dmv
        dL_du = DxT @ dc + du_dx * dmu + DxT @ (u * dmu) + DyT @ (v * dmu) + dv_dx * dmv
        ndmu = -dmu; ndmv = -dmv
        wxu = DxT @ ndmu; wyu = DyT @ ndmu
        dL_du = dL_du + DxT @ (nu_eff * wxu) + DyT @ (nu_eff * wyu)
        gu_v = du_dx * wxu + du_dy * wyu
        dL_du = dL_du + DxT @ (au * gu_v) + DyT @ (bu * gu_v)
        wxv = DxT @ ndmv; wyv = DyT @ ndmv
        gv_v = dv_dx * wxv + dv_dy * wyv
        dL_du = dL_du + DxT @ (au * gv_v) + DyT @ (bu * gv_v)

        dL_dv = DyT @ dc + du_dy * dmu
        dL_dv = dL_dv + DxT @ (u * dmv) + dv_dy * dmv + DyT @ (v * dmv)
        dL_dv = dL_dv + DxT @ (nu_eff * wxv) + DyT @ (nu_eff * wyv)
        dL_dv = dL_dv + DxT @ (av * gv_v) + DyT @ (bv * gv_v)
        dL_dv = dL_dv + DxT @ (av * gu_v) + DyT @ (bv * gu_v)

        grad_pde = torch.cat([dL_du, dL_dv, dL_dp], dim=1)

        N_lid, N_wall = g['N_lid'], g['N_wall']
        grad_lid = pred_l.detach().clone()
        grad_lid[:, 0:1] = 2.0 * (pred_l[:, 0:1].detach() - 1.0) / N_lid
        grad_lid[:, 1:2] = 2.0 * pred_l[:, 1:2].detach() / N_lid
        grad_lid[:, 2:3] = 0.0

        grad_wall = pred_w.detach().clone()
        grad_wall[:, 0:1] = 2.0 * pred_w[:, 0:1].detach() / N_wall
        grad_wall[:, 1:2] = 2.0 * pred_w[:, 1:2].detach() / N_wall
        grad_wall[:, 2:3] = 0.0

        grad_center = pred_c.detach().clone()
        grad_center[:, 0:1] = 0.0
        grad_center[:, 1:2] = 0.0
        grad_center[:, 2:3] = 2.0 * pred_c[:, 2:3].detach()

        upstream = torch.cat([grad_pde, grad_lid, grad_wall, grad_center], dim=0)
        pred_batch.backward(gradient=upstream)
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    for epoch in range(n_epochs):
        graph.replay()
        if (epoch + 1) % LOG_INTERVAL == 0:
            torch.cuda.synchronize()
            print(f"  Epoch {epoch+1}")

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, float('nan')  # No loss tracking in CUDA graph mode


# --- Method: ropinn ---
def train_ropinn(seed, device, n_epochs, lr, optimizer_type, technique, grid_size, model_name="mlp", tracker=None):
    """RoPINN: region-optimized PINN with trust region calibration."""
    coll = build_collocation_points(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        model = torch.compile(model, mode='reduce-overhead')

    xy_int_base = coll['xy_interior']
    xy_lid = coll['xy_lid']
    xy_wall = coll['xy_wall']
    xy_center = torch.tensor([[0.5, 0.5]], dtype=torch.float32, device=device)

    gradient_list = []
    gradient_variance = 1.0
    final_loss = float('nan')

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
        gradient_list_overall = []
        gradient_list_temp = []

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(n_epochs):
            current_region = np.clip(
                ROPINN_INITIAL_REGION / gradient_variance,
                a_min=0, a_max=ROPINN_REGION_MAX
            )

            def closure():
                optimizer.zero_grad()
                perturbation = torch.rand_like(xy_int_base) * current_region
                xy_perturbed = torch.clamp(xy_int_base + perturbation, 0.0, 1.0)
                xy_perturbed = xy_perturbed.detach().requires_grad_(True)

                continuity, mom_u, mom_v = pde_residuals_autodiff(model, xy_perturbed)
                loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                           mse(mom_u, torch.zeros_like(mom_u)) + \
                           mse(mom_v, torch.zeros_like(mom_v))

                pred_lid = model(xy_lid)
                loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                           mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
                pred_wall = model(xy_wall)
                loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                            mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
                pred_c = model(xy_center)
                loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))

                loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
                loss.backward()

                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1))
                flat_grad = torch.cat(grads).cpu().numpy()
                gradient_list_temp.append(flat_grad)

                return loss

            loss = optimizer.step(closure)
            final_loss = loss.item() if isinstance(loss, torch.Tensor) else loss

            if gradient_list_temp:
                avg_gradient = np.mean(np.array(gradient_list_temp), axis=0)
                gradient_list_overall.append(avg_gradient)
                gradient_list_overall = gradient_list_overall[-ROPINN_PAST_ITERATIONS:]
                gradient_variance = compute_gradient_variance(gradient_list_overall)
                gradient_list_temp.clear()

            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}, "
                      f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")
            if tracker is not None:
                tracker.step(epoch, final_loss, model)

    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            current_region = np.clip(
                ROPINN_INITIAL_REGION / gradient_variance,
                a_min=0, a_max=ROPINN_REGION_MAX
            )

            perturbation = torch.rand_like(xy_int_base) * current_region
            xy_perturbed = torch.clamp(xy_int_base + perturbation, 0.0, 1.0)
            xy_perturbed = xy_perturbed.detach().requires_grad_(True)

            continuity, mom_u, mom_v = pde_residuals_autodiff(model, xy_perturbed)
            loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))

            pred_lid = model(xy_lid)
            loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                       mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
            pred_wall = model(xy_wall)
            loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                        mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
            pred_c = model(xy_center)
            loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))

            loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

            # Gradient tracking for trust region calibration
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            flat_grad = torch.cat(grads).cpu().numpy()
            gradient_list.append(flat_grad)
            gradient_list = gradient_list[-ROPINN_PAST_ITERATIONS:]
            gradient_variance = compute_gradient_variance(gradient_list)

            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}, "
                      f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")
            if tracker is not None:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    return base_model, train_time, final_loss


# --- Method: pielm ---
def train_pielm(seed, grid_size):
    """PIELM: Physics-Informed Extreme Learning Machine."""
    from src.experiment_dt_elm_pinn.models.pielm_navier_stokes import PIELM_NavierStokes

    pielm = PIELM_NavierStokes(
        Re=Re, U_lid=U_lid, Cs=Cs,
        n_hidden=500, activation='tanh',
        max_picard_iter=50, tol=1e-6,
        seed=seed,
        N_interior=6000, N_wall=800, N_lid=800,
        bc_weight=10.0, verbose=True,
    )

    results = pielm.train()
    return pielm, results['train_time'], results.get('final_residual', float('nan'))


# =============================================================================
# CSV output with file locking
# =============================================================================
CSV_COLUMNS = [
    'timestamp', 'problem', 'method', 'model', 'optimizer', 'lr', 'epochs', 'seed', 'grid_size',
    'technique', 'tag',
    'train_time_s', 'train_time_min', 'peak_gpu_memory_mb', 'gpu_memory_reserved_mb',
    'ms_per_epoch', 'n_params',
    'pde_rms', 'continuity_rms', 'momentum_rms', 'final_loss',
    'status', 'device', 'gpu_name', 'pytorch_version',
]


def append_csv_row(csv_path, row_dict):
    """Append a single row to CSV with file locking for concurrent safety."""
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists or os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerow(row_dict)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- Method: sk-pinn ---
def train_sk_pinn(seed, device, n_epochs, lr, grid_size, model_name, grid_data, tracker=None):
    """SK-PINN: sparse RKPM differentiation matrices, autograd backward.

    Uses weight decay to prevent the model from learning high-frequency features
    that exceed the RKPM operator's resolution (O(h^2) algebraic convergence).
    """
    g = grid_data

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred = model(g['xy_all'])
        continuity, mom_u, mom_v = compute_pde_terms_sparse(pred, g)
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

        pred_c = model(g['xy_center'])
        loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))

        loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# =============================================================================
# Kovasznay training methods
# =============================================================================

# --- Method: sage (Kovasznay) ---
def train_sage_kovasznay(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """SAGE: auto-generated backward for Kovasznay flow."""
    g = build_grid_data_kovasznay(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    generated_backward = _get_generated_backward_kovasznay()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_batch = compiled_model(g['xy_batched'])
        with torch.no_grad():
            pred_pde = pred_batch[:g['N_all']]
            pred_bc = pred_batch[g['off_bc']:g['off_center']]
            pred_c = pred_batch[g['off_center']:]

            grad_pde = generated_backward(pred_pde, g)

            N_bc = g['N_bc']
            grad_bc = 2.0 * (pred_bc - g['bc_target']) / N_bc

            grad_center = torch.zeros(1, 3, device=device)
            grad_center[:, 2:3] = 2.0 * (pred_c[:, 2:3] - g['p_center_exact'])

            upstream = torch.cat([grad_pde, grad_bc, grad_center], dim=0)
        pred_batch.backward(gradient=upstream)
        reg = model_reg_loss(model)
        if isinstance(reg, torch.Tensor):
            reg.backward()
        optimizer.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track:
            with torch.no_grad():
                c, mu, mv = compute_pde_kovasznay(pred_pde, g)
                ii = g['interior_idx']
                lv = (c[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                final_loss = lv.item()
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: pde_loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: dtpinn (Kovasznay) ---
def train_dtpinn_kovasznay(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Standard DT-PINN for Kovasznay flow: spectral matrices, autograd backward."""
    g = build_grid_data_kovasznay(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred = compiled_model(g['xy_all'])
        continuity, mom_u, mom_v = compute_pde_kovasznay(pred, g)
        ii = g['interior_idx']
        loss_pde = mse(continuity[ii], torch.zeros_like(continuity[ii])) + \
                   mse(mom_u[ii], torch.zeros_like(mom_u[ii])) + \
                   mse(mom_v[ii], torch.zeros_like(mom_v[ii]))

        pred_bc = compiled_model(g['xy_bc'])
        loss_bc = mse(pred_bc, g['bc_target'])

        xy_center = g['xy_batched'][g['off_center']:]
        pred_c = compiled_model(xy_center)
        p_target = torch.tensor([[g['p_center_exact']]], dtype=torch.float32, device=device)
        loss_p = mse(pred_c[:, 2:3], p_target)

        loss = loss_pde + loss_bc + loss_p + model_reg_loss(model)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: autodiff (Kovasznay) ---
def train_autodiff_kovasznay(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Plain autodiff PINN for Kovasznay flow."""
    coll = build_collocation_points_kovasznay(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        model = torch.compile(model, mode='reduce-overhead')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xy_int = coll['xy_interior'].clone().requires_grad_(True)
    xy_bc = coll['xy_bc']
    bc_target = coll['bc_target']
    xy_center = coll['xy_center']
    p_center_exact = coll['p_center_exact']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        continuity, mom_u, mom_v = pde_residuals_kovasznay_autodiff(model, xy_int)
        loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

        pred_bc = model(xy_bc)
        loss_bc = mse(pred_bc, bc_target)

        pred_c = model(xy_center)
        p_target = torch.tensor([[p_center_exact]], dtype=torch.float32, device=device)
        loss_p = mse(pred_c[:, 2:3], p_target)

        loss = loss_pde + loss_bc + loss_p + model_reg_loss(model)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    return base_model, train_time, final_loss


# --- Method: ropinn (Kovasznay) ---
def train_ropinn_kovasznay(seed, device, n_epochs, lr, optimizer_type, technique, grid_size, model_name="mlp", tracker=None):
    """RoPINN: region-optimized PINN with trust region calibration for Kovasznay flow."""
    coll = build_collocation_points_kovasznay(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        model = torch.compile(model, mode='reduce-overhead')

    xy_int_base = coll['xy_interior']
    xy_bc = coll['xy_bc']
    bc_target = coll['bc_target']
    xy_center = coll['xy_center']
    p_center_exact = coll['p_center_exact']

    # Kovasznay domain bounds for clamping
    x_lo, x_hi = -0.5, 1.0
    y_lo, y_hi = -0.5, 1.5

    gradient_list = []
    gradient_variance = 1.0
    final_loss = float('nan')

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
        gradient_list_overall = []
        gradient_list_temp = []

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(n_epochs):
            current_region = np.clip(
                ROPINN_INITIAL_REGION / gradient_variance,
                a_min=0, a_max=ROPINN_REGION_MAX
            )

            def closure():
                optimizer.zero_grad()
                perturbation = torch.rand_like(xy_int_base) * current_region
                xy_perturbed = xy_int_base + perturbation
                # Clamp to Kovasznay domain
                xy_perturbed[:, 0].clamp_(x_lo, x_hi)
                xy_perturbed[:, 1].clamp_(y_lo, y_hi)
                xy_perturbed = xy_perturbed.detach().requires_grad_(True)

                continuity, mom_u, mom_v = pde_residuals_kovasznay_autodiff(model, xy_perturbed)
                loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                           mse(mom_u, torch.zeros_like(mom_u)) + \
                           mse(mom_v, torch.zeros_like(mom_v))

                pred_bc = model(xy_bc)
                loss_bc = mse(pred_bc, bc_target)

                pred_c = model(xy_center)
                p_target = torch.tensor([[p_center_exact]], dtype=torch.float32, device=device)
                loss_p = mse(pred_c[:, 2:3], p_target)

                loss = loss_pde + loss_bc + loss_p + model_reg_loss(model)
                loss.backward()

                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1))
                flat_grad = torch.cat(grads).cpu().numpy()
                gradient_list_temp.append(flat_grad)

                return loss

            loss = optimizer.step(closure)
            final_loss = loss.item() if isinstance(loss, torch.Tensor) else loss

            if gradient_list_temp:
                avg_gradient = np.mean(np.array(gradient_list_temp), axis=0)
                gradient_list_overall.append(avg_gradient)
                gradient_list_overall = gradient_list_overall[-ROPINN_PAST_ITERATIONS:]
                gradient_variance = compute_gradient_variance(gradient_list_overall)
                gradient_list_temp.clear()

            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}, "
                      f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")
            if tracker is not None:
                tracker.step(epoch, final_loss, model)

    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            current_region = np.clip(
                ROPINN_INITIAL_REGION / gradient_variance,
                a_min=0, a_max=ROPINN_REGION_MAX
            )

            perturbation = torch.rand_like(xy_int_base) * current_region
            xy_perturbed = xy_int_base + perturbation
            # Clamp to Kovasznay domain
            xy_perturbed[:, 0].clamp_(x_lo, x_hi)
            xy_perturbed[:, 1].clamp_(y_lo, y_hi)
            xy_perturbed = xy_perturbed.detach().requires_grad_(True)

            continuity, mom_u, mom_v = pde_residuals_kovasznay_autodiff(model, xy_perturbed)
            loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))

            pred_bc = model(xy_bc)
            loss_bc = mse(pred_bc, bc_target)

            pred_c = model(xy_center)
            p_target = torch.tensor([[p_center_exact]], dtype=torch.float32, device=device)
            loss_p = mse(pred_c[:, 2:3], p_target)

            loss = loss_pde + loss_bc + loss_p + model_reg_loss(model)
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

            # Gradient tracking for trust region calibration
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            flat_grad = torch.cat(grads).cpu().numpy()
            gradient_list.append(flat_grad)
            gradient_list = gradient_list[-ROPINN_PAST_ITERATIONS:]
            gradient_variance = compute_gradient_variance(gradient_list)

            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}, "
                      f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")
            if tracker is not None:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    return base_model, train_time, final_loss


# --- SK-PINN support for Kovasznay ---
def build_sk_data_kovasznay(N_grid, device):
    """Build SK-PINN RKPM differentiation matrices for Kovasznay flow.

    Uniform grid on [-0.5, 1.0] x [-0.5, 1.5] (non-square: Lx=1.5, Ly=2.0).
    No Smagorinsky model — no Cs_d_sq or d_wall.
    """
    Lx, Ly = 1.5, 2.0
    x0, y0 = -0.5, -0.5

    dx = Lx / (N_grid - 1)
    dy = Ly / (N_grid - 1)
    h = min(dx, dy) * 1.4
    radius = 2.0 * h
    dxdy = dx * dy

    # Uniform grid on non-square domain
    x = np.linspace(x0, x0 + Lx, N_grid)
    y = np.linspace(y0, y0 + Ly, N_grid)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])
    N_all = len(xy_grid)

    print(f"  SK-PINN Kovasznay: building RKPM matrices for {N_grid}x{N_grid} uniform grid...")
    print(f"  Lx={Lx}, Ly={Ly}, dx={dx:.6f}, dy={dy:.6f}, h={h:.6f}, radius={radius:.6f}")

    # Neighbor search + kernel + RKPM correction
    neighborhoods, distances, distance_vectors = _sk_find_neighborhoods(xy_grid, radius)
    kernel = _sk_sph_kernel(distances, h)
    C = _sk_compute_C(distance_vectors, kernel.unsqueeze(-1), dxdy, order=2)

    # Assemble sparse Dx, Dy matrices (COO format)
    kernel_np = kernel.numpy()
    C_np = C.numpy()
    nb_np = neighborhoods.numpy()

    rows, cols, dx_vals, dy_vals = [], [], [], []
    for i in range(N_all):
        for j_idx in range(nb_np.shape[1]):
            j = nb_np[i, j_idx]
            if j == -1:
                break
            w = kernel_np[i, j_idx]
            dx_val = C_np[i, j_idx, 0] * w
            dy_val = C_np[i, j_idx, 1] * w
            if dx_val != 0 or dy_val != 0:
                rows.append(i)
                cols.append(j)
                dx_vals.append(dx_val)
                dy_vals.append(dy_val)

    nnz = len(rows)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    Dx = torch.sparse_coo_tensor(
        indices, torch.tensor(dx_vals, dtype=torch.float32), (N_all, N_all)
    ).to(device).coalesce()
    Dy = torch.sparse_coo_tensor(
        indices, torch.tensor(dy_vals, dtype=torch.float32), (N_all, N_all)
    ).to(device).coalesce()

    print(f"  SK-PINN Kovasznay: sparse Dx/Dy nnz={nnz}/{N_all*N_all} "
          f"({100*nnz/(N_all*N_all):.2f}% dense)")

    # Boundary classification
    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = ((xc < x0 + eps) | (xc > x0 + Lx - eps) |
                   (yc < y0 + eps) | (yc > y0 + Ly - eps))

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    N_bc = len(bc_idx)
    M = len(interior_idx)

    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_bc = xy_all[bc_idx]

    # Exact BC values from Kovasznay solution
    u_ex, v_ex, p_ex = kovasznay_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([u_ex, v_ex, p_ex], dim=1)

    # Pressure reference at domain center
    xy_center = torch.tensor([[x0 + Lx / 2, y0 + Ly / 2]], dtype=torch.float32, device=device)
    _, _, p_ctr = kovasznay_exact(xy_center[:, 0:1], xy_center[:, 1:2])
    p_center_exact = p_ctr.item()

    interior_mask = torch.zeros(N_all, 1, device=device)
    interior_mask[interior_idx] = 1.0

    print(f"  SK-PINN Kovasznay: N_all={N_all}, interior={M}, bc={N_bc}")

    return {
        'Dx': Dx, 'Dy': Dy,
        'sparse': True,
        'xy_all': xy_all, 'xy_bc': xy_bc, 'bc_target': bc_target,
        'xy_center': xy_center, 'p_center_exact': p_center_exact,
        'interior_idx': interior_idx, 'bc_idx': bc_idx,
        'interior_mask': interior_mask,
        'N_all': N_all, 'N_bc': N_bc, 'M': M, 'N_grid': N_grid,
    }


def compute_pde_kovasznay_sparse(pred, g):
    """Kovasznay PDE residuals via sparse RKPM differentiation matrices.

    Constant viscosity NS (no Smagorinsky).
    """
    Dx, Dy = g['Dx'], g['Dy']
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    du_dx = torch.sparse.mm(Dx, u);  du_dy = torch.sparse.mm(Dy, u)
    dv_dx = torch.sparse.mm(Dx, v);  dv_dy = torch.sparse.mm(Dy, v)
    dp_dx = torch.sparse.mm(Dx, p);  dp_dy = torch.sparse.mm(Dy, p)

    # Second derivatives for viscous term
    d2u_dx2 = torch.sparse.mm(Dx, du_dx);  d2u_dy2 = torch.sparse.mm(Dy, du_dy)
    d2v_dx2 = torch.sparse.mm(Dx, dv_dx);  d2v_dy2 = torch.sparse.mm(Dy, dv_dy)

    # Continuity
    continuity = du_dx + dv_dy

    # Momentum (constant viscosity)
    mom_u = u * du_dx + v * du_dy + dp_dx - nu_kov * (d2u_dx2 + d2u_dy2)
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu_kov * (d2v_dx2 + d2v_dy2)

    return continuity, mom_u, mom_v


def train_sk_pinn_kovasznay(seed, device, n_epochs, lr, grid_size, model_name, grid_data, tracker=None):
    """SK-PINN for Kovasznay flow: sparse RKPM matrices, autograd backward.

    Uses weight decay to prevent model from learning beyond RKPM resolution.
    """
    g = grid_data

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred = model(g['xy_all'])
        continuity, mom_u, mom_v = compute_pde_kovasznay_sparse(pred, g)
        ii = g['interior_idx']
        loss_pde = mse(continuity[ii], torch.zeros_like(continuity[ii])) + \
                   mse(mom_u[ii], torch.zeros_like(mom_u[ii])) + \
                   mse(mom_v[ii], torch.zeros_like(mom_v[ii]))

        pred_bc = model(g['xy_bc'])
        loss_bc = mse(pred_bc, g['bc_target'])

        pred_c = model(g['xy_center'])
        p_target = torch.tensor([[g['p_center_exact']]], dtype=torch.float32, device=device)
        loss_p = mse(pred_c[:, 2:3], p_target)

        loss = loss_pde + loss_bc + loss_p + model_reg_loss(model)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# =============================================================================
# Kovasznay evaluation
# =============================================================================
def evaluate_kovasznay(model, device):
    """Evaluate Kovasznay flow on 51x51 uniform grid: PDE residuals + solution error."""
    Lx, Ly = 1.5, 2.0
    x0, y0 = -0.5, -0.5
    nx, ny = 51, 51
    x = np.linspace(x0, x0 + Lx, nx)
    y = np.linspace(y0, y0 + Ly, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    pred = model(xy_t)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    # PDE residuals via autograd
    grad_u = gradients(u, xy_t); grad_v = gradients(v, xy_t); grad_p = gradients(p, xy_t)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    grad_du_dx = gradients(du_dx, xy_t); grad_du_dy = gradients(du_dy, xy_t)
    grad_dv_dx = gradients(dv_dx, xy_t); grad_dv_dy = gradients(dv_dy, xy_t)
    d2u_dx2 = grad_du_dx[:, 0:1]; d2u_dy2 = grad_du_dy[:, 1:2]
    d2v_dx2 = grad_dv_dx[:, 0:1]; d2v_dy2 = grad_dv_dy[:, 1:2]

    continuity = du_dx + dv_dy
    mom_u = u * du_dx + v * du_dy + dp_dx - nu_kov * (d2u_dx2 + d2u_dy2)
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu_kov * (d2v_dx2 + d2v_dy2)

    cont_np = continuity.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(cont_np**2 + mom_u_np**2 + mom_v_np**2)))
    cont_rms = float(np.sqrt(np.mean(cont_np**2)))
    mom_rms = float(np.sqrt(np.mean(mom_u_np**2 + mom_v_np**2)))

    # Solution error vs exact
    x_coords = xy_t[:, 0:1].detach()
    y_coords = xy_t[:, 1:2].detach()
    u_ex, v_ex, p_ex = kovasznay_exact(x_coords, y_coords)

    u_err = (u.detach() - u_ex).cpu().numpy().flatten()
    v_err = (v.detach() - v_ex).cpu().numpy().flatten()
    p_err = (p.detach() - p_ex).cpu().numpy().flatten()
    u_rms_err = float(np.sqrt(np.mean(u_err**2)))
    v_rms_err = float(np.sqrt(np.mean(v_err**2)))
    p_rms_err = float(np.sqrt(np.mean(p_err**2)))

    model.train()
    return {
        'pde_rms': pde_rms, 'continuity_rms': cont_rms, 'momentum_rms': mom_rms,
        'u_rms_error': u_rms_err, 'v_rms_error': v_rms_err, 'p_rms_error': p_rms_err,
    }


# =============================================================================
# 2D Linear Elasticity (Navier-Cauchy, displacement formulation)
# =============================================================================
# Manufactured exact solution on [0,1]² (DeepXDE benchmark):
#   ux(x,y) = cos(2πx)·sin(πy)
#   uy(x,y) = sin(πx)·Q·y⁴/4
#
# Navier-Cauchy equations:
#   (λ+2μ)·∂²ux/∂x² + μ·∂²ux/∂y² + (λ+μ)·∂²uy/∂x∂y + fx = 0
#   μ·∂²uy/∂x² + (λ+2μ)·∂²uy/∂y² + (λ+μ)·∂²ux/∂x∂y + fy = 0

def elasticity_exact(x, y):
    """Exact manufactured solution for 2D linear elasticity.

    Args:
        x, y: tensors of coordinates on [0,1]²

    Returns:
        (ux, uy) displacement tensors
    """
    ux = torch.cos(2.0 * math.pi * x) * torch.sin(math.pi * y)
    uy = torch.sin(math.pi * x) * Q_e * y ** 4 / 4.0
    return ux, uy


def elasticity_body_forces(x, y):
    """Analytically derived body forces for the manufactured solution.

    fx = -[(λ+2μ)·ux_xx + μ·ux_yy + (λ+μ)·uy_xy]
    fy = -[μ·uy_xx + (λ+2μ)·uy_yy + (λ+μ)·ux_xy]
    """
    pi = math.pi
    # Second derivatives of ux = cos(2πx)·sin(πy)
    ux_xx = -(2 * pi) ** 2 * torch.cos(2 * pi * x) * torch.sin(pi * y)
    ux_yy = -(pi ** 2) * torch.cos(2 * pi * x) * torch.sin(pi * y)
    # Cross derivative of ux
    ux_xy = -2 * pi ** 2 * torch.sin(2 * pi * x) * torch.cos(pi * y)

    # Second derivatives of uy = sin(πx)·Q·y⁴/4
    uy_xx = -(pi ** 2) * torch.sin(pi * x) * Q_e * y ** 4 / 4.0
    uy_yy = torch.sin(pi * x) * Q_e * 3.0 * y ** 2
    # Cross derivative of uy
    uy_xy = pi * torch.cos(pi * x) * Q_e * y ** 3

    fx = -((lam_e + 2 * mu_e) * ux_xx + mu_e * ux_yy + (lam_e + mu_e) * uy_xy)
    fy = -(mu_e * uy_xx + (lam_e + 2 * mu_e) * uy_yy + (lam_e + mu_e) * ux_xy)
    return fx, fy


def compute_pde_elasticity(pred, g):
    """Elasticity PDE residuals via spectral differentiation matrices.

    Uses precomputed D² matrices (Dxx, Dyy, Dxy) for single-matmul second
    derivatives, avoiding chained Dx@Dx@u which amplifies float32 error.
    Returns (eq_x, eq_y) — 2 residual terms.
    """
    ux, uy = pred[:, 0:1], pred[:, 1:2]

    # Second derivatives via precomputed D² (single matmul, not chained)
    d2ux_dx2 = g['Dxx'] @ ux
    d2ux_dy2 = g['Dyy'] @ ux
    d2uy_dx2 = g['Dxx'] @ uy
    d2uy_dy2 = g['Dyy'] @ uy

    # Cross derivatives via precomputed Dxy = Dy @ Dx
    d2uy_dxdy = g['Dxy'] @ uy
    d2ux_dxdy = g['Dxy'] @ ux

    # Navier-Cauchy equations + body forces
    eq_x = ((lam_e + 2 * mu_e) * d2ux_dx2 + mu_e * d2ux_dy2
            + (lam_e + mu_e) * d2uy_dxdy + g['fx'])
    eq_y = (mu_e * d2uy_dx2 + (lam_e + 2 * mu_e) * d2uy_dy2
            + (lam_e + mu_e) * d2ux_dxdy + g['fy'])

    return eq_x, eq_y


def compute_pde_elasticity_sparse(pred, g):
    """Elasticity PDE residuals via sparse RKPM differentiation matrices."""
    Dx, Dy = g['Dx'], g['Dy']
    ux, uy = pred[:, 0:1], pred[:, 1:2]

    dux_dx = torch.sparse.mm(Dx, ux)
    dux_dy = torch.sparse.mm(Dy, ux)
    duy_dx = torch.sparse.mm(Dx, uy)
    duy_dy = torch.sparse.mm(Dy, uy)

    d2ux_dx2 = torch.sparse.mm(Dx, dux_dx)
    d2ux_dy2 = torch.sparse.mm(Dy, dux_dy)
    d2uy_dx2 = torch.sparse.mm(Dx, duy_dx)
    d2uy_dy2 = torch.sparse.mm(Dy, duy_dy)

    d2uy_dxdy = torch.sparse.mm(Dy, duy_dx)
    d2ux_dxdy = torch.sparse.mm(Dy, dux_dx)

    eq_x = ((lam_e + 2 * mu_e) * d2ux_dx2 + mu_e * d2ux_dy2
            + (lam_e + mu_e) * d2uy_dxdy + g['fx'])
    eq_y = (mu_e * d2uy_dx2 + (lam_e + 2 * mu_e) * d2uy_dy2
            + (lam_e + mu_e) * d2ux_dxdy + g['fy'])

    return eq_x, eq_y


def pde_residuals_elasticity_autodiff(model, xy):
    """Elasticity PDE residuals via autograd. xy must have requires_grad=True."""
    pred = model(xy)
    ux, uy = pred[:, 0:1], pred[:, 1:2]

    grad_ux = gradients(ux, xy)
    grad_uy = gradients(uy, xy)
    dux_dx, dux_dy = grad_ux[:, 0:1], grad_ux[:, 1:2]
    duy_dx, duy_dy = grad_uy[:, 0:1], grad_uy[:, 1:2]

    # Second derivatives
    grad_dux_dx = gradients(dux_dx, xy)
    grad_dux_dy = gradients(dux_dy, xy)
    grad_duy_dx = gradients(duy_dx, xy)
    grad_duy_dy = gradients(duy_dy, xy)
    d2ux_dx2 = grad_dux_dx[:, 0:1]
    d2ux_dy2 = grad_dux_dy[:, 1:2]
    d2uy_dx2 = grad_duy_dx[:, 0:1]
    d2uy_dy2 = grad_duy_dy[:, 1:2]

    # Cross derivatives
    d2uy_dxdy = grad_duy_dx[:, 1:2]
    d2ux_dxdy = grad_dux_dx[:, 1:2]

    # Body forces at collocation points
    x_coord, y_coord = xy[:, 0:1], xy[:, 1:2]
    fx, fy = elasticity_body_forces(x_coord, y_coord)

    eq_x = ((lam_e + 2 * mu_e) * d2ux_dx2 + mu_e * d2ux_dy2
            + (lam_e + mu_e) * d2uy_dxdy + fx)
    eq_y = (mu_e * d2uy_dx2 + (lam_e + 2 * mu_e) * d2uy_dy2
            + (lam_e + mu_e) * d2ux_dxdy + fy)

    return eq_x, eq_y


def build_grid_data_elasticity(N_grid, device):
    """Build Chebyshev grid + spectral differentiation matrices for elasticity.

    Domain: [0,1]² (unit square). Lx=Ly=1.0.
    Precomputes D², D_xy second-derivative matrices in float64 for accuracy.
    """
    D1d = chebyshev_diff_matrix(N_grid) * 2.0  # scale for [0,1]
    I_mat = np.eye(N_grid)
    Dx_np = np.kron(I_mat, D1d)
    Dy_np = np.kron(D1d, I_mat)

    # Precompute second-derivative matrices in float64 for numerical accuracy.
    # Chained float32 matmuls (Dx @ Dx @ u) amplify rounding error ~900x vs
    # single matmuls with precomputed D² matrices.
    Dxx_np = Dx_np @ Dx_np   # d²/dx²
    Dyy_np = Dy_np @ Dy_np   # d²/dy²
    Dxy_np = Dy_np @ Dx_np   # d²/dxdy (= Dy applied to d/dx)

    x_ref = chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    N_all = len(xy_grid)
    N_bc = len(bc_idx)
    M = len(interior_idx)

    Dx = torch.tensor(Dx_np, dtype=torch.float32, device=device)
    Dy = torch.tensor(Dy_np, dtype=torch.float32, device=device)
    DxT = Dx.T.contiguous()
    DyT = Dy.T.contiguous()

    Dxx = torch.tensor(Dxx_np, dtype=torch.float32, device=device)
    Dyy = torch.tensor(Dyy_np, dtype=torch.float32, device=device)
    Dxy = torch.tensor(Dxy_np, dtype=torch.float32, device=device)
    DxxT = Dxx.T.contiguous()
    DyyT = Dyy.T.contiguous()
    DxyT = Dxy.T.contiguous()

    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_bc = xy_all[bc_idx]

    interior_mask = torch.zeros(N_all, 1, device=device)
    interior_mask[interior_idx] = 1.0

    # Precompute body forces at all grid points (constant, stored in grid_data)
    fx_all, fy_all = elasticity_body_forces(xy_all[:, 0:1], xy_all[:, 1:2])

    # Exact BC values (2 displacements)
    ux_ex, uy_ex = elasticity_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([ux_ex, uy_ex], dim=1)  # (N_bc, 2)

    # Batched input: all grid + BC points
    xy_batched = torch.cat([xy_all, xy_bc], dim=0)
    off_bc = N_all

    return {
        'Dx': Dx, 'Dy': Dy, 'DxT': DxT, 'DyT': DyT,
        'Dxx': Dxx, 'Dyy': Dyy, 'Dxy': Dxy,
        'DxxT': DxxT, 'DyyT': DyyT, 'DxyT': DxyT,
        'xy_all': xy_all, 'xy_bc': xy_bc, 'xy_batched': xy_batched,
        'interior_idx': interior_idx, 'bc_idx': bc_idx,
        'interior_mask': interior_mask,
        'fx': fx_all, 'fy': fy_all,
        'bc_target': bc_target,
        'N_all': N_all, 'N_bc': N_bc, 'M': M,
        'off_bc': off_bc,
        'N_grid': N_grid,
    }


def build_collocation_points_elasticity(N_grid, device):
    """Build Chebyshev collocation points for elasticity autograd methods."""
    x_ref = chebyshev_points(N_grid)
    x_phys = 0.5 * (x_ref + 1.0)
    xx, yy = np.meshgrid(x_phys, x_phys, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    xy_int = torch.tensor(xy_grid[interior_idx], dtype=torch.float32, device=device)
    xy_bc = torch.tensor(xy_grid[bc_idx], dtype=torch.float32, device=device)

    # Exact BC values (2 displacements)
    ux_ex, uy_ex = elasticity_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([ux_ex, uy_ex], dim=1)  # (N_bc, 2)

    return {
        'xy_interior': xy_int,
        'xy_bc': xy_bc,
        'bc_target': bc_target,
        'N_interior': len(interior_idx),
        'N_bc': len(bc_idx),
        'N_grid': N_grid,
    }


def build_sk_data_elasticity(N_grid, device):
    """Build SK-PINN RKPM differentiation matrices for elasticity.

    Uniform grid on [0,1]² (unit square).
    """
    Lx, Ly = 1.0, 1.0
    x0, y0 = 0.0, 0.0

    dx = Lx / (N_grid - 1)
    dy = Ly / (N_grid - 1)
    h = min(dx, dy) * 1.4
    radius = 2.0 * h
    dxdy = dx * dy

    x = np.linspace(x0, x0 + Lx, N_grid)
    y = np.linspace(y0, y0 + Ly, N_grid)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])
    N_all = len(xy_grid)

    print(f"  SK-PINN Elasticity: building RKPM matrices for {N_grid}x{N_grid} uniform grid...")
    print(f"  dx={dx:.6f}, dy={dy:.6f}, h={h:.6f}, radius={radius:.6f}")

    neighborhoods, distances, distance_vectors = _sk_find_neighborhoods(xy_grid, radius)
    kernel = _sk_sph_kernel(distances, h)
    C = _sk_compute_C(distance_vectors, kernel.unsqueeze(-1), dxdy, order=2)

    kernel_np = kernel.numpy()
    C_np = C.numpy()
    nb_np = neighborhoods.numpy()

    rows, cols, dx_vals, dy_vals = [], [], [], []
    for i in range(N_all):
        for j_idx in range(nb_np.shape[1]):
            j = nb_np[i, j_idx]
            if j == -1:
                break
            w = kernel_np[i, j_idx]
            dx_val = C_np[i, j_idx, 0] * w
            dy_val = C_np[i, j_idx, 1] * w
            if dx_val != 0 or dy_val != 0:
                rows.append(i)
                cols.append(j)
                dx_vals.append(dx_val)
                dy_vals.append(dy_val)

    nnz = len(rows)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    Dx = torch.sparse_coo_tensor(
        indices, torch.tensor(dx_vals, dtype=torch.float32), (N_all, N_all)
    ).to(device).coalesce()
    Dy = torch.sparse_coo_tensor(
        indices, torch.tensor(dy_vals, dtype=torch.float32), (N_all, N_all)
    ).to(device).coalesce()

    print(f"  SK-PINN Elasticity: sparse Dx/Dy nnz={nnz}/{N_all*N_all} "
          f"({100*nnz/(N_all*N_all):.2f}% dense)")

    # Boundary classification
    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    N_bc = len(bc_idx)
    M = len(interior_idx)

    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_bc = xy_all[bc_idx]

    # Body forces at all grid points
    fx_all, fy_all = elasticity_body_forces(xy_all[:, 0:1], xy_all[:, 1:2])

    # Exact BC values
    ux_ex, uy_ex = elasticity_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([ux_ex, uy_ex], dim=1)

    interior_mask = torch.zeros(N_all, 1, device=device)
    interior_mask[interior_idx] = 1.0

    print(f"  SK-PINN Elasticity: N_all={N_all}, interior={M}, bc={N_bc}")

    return {
        'Dx': Dx, 'Dy': Dy,
        'sparse': True,
        'xy_all': xy_all, 'xy_bc': xy_bc, 'bc_target': bc_target,
        'fx': fx_all, 'fy': fy_all,
        'interior_idx': interior_idx, 'bc_idx': bc_idx,
        'interior_mask': interior_mask,
        'N_all': N_all, 'N_bc': N_bc, 'M': M, 'N_grid': N_grid,
    }


# Lazy-initialized generated backward for Elasticity
_generated_elasticity_backward = None

def _get_generated_backward_elasticity():
    """Lazily generate and cache backward function for Elasticity PDE."""
    global _generated_elasticity_backward
    if _generated_elasticity_backward is None:
        from src.symbolic_vjp import trace_pde_forward, emit_backward
        tape = []
        outputs, inputs = trace_pde_forward(
            compute_pde_elasticity, None, tape, sparse=False,
            constants=['Dxx', 'Dyy', 'Dxy', 'fx', 'fy'],
            input_names=['ux', 'uy'])
        _, _generated_elasticity_backward = emit_backward(
            tape, list(outputs), ['deq_x', 'deq_y'], inputs, sparse=False,
            func_name='generated_elasticity_grad',
            input_names=['ux', 'uy'])
    return _generated_elasticity_backward


# --- Method: sage (Elasticity) ---
def train_sage_elasticity(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """SAGE: auto-generated backward for elasticity."""
    g = build_grid_data_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    generated_backward = _get_generated_backward_elasticity()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred_batch = compiled_model(g['xy_batched'])
        with torch.no_grad():
            pred_pde = pred_batch[:g['N_all']]
            pred_bc = pred_batch[g['off_bc']:g['off_bc'] + g['N_bc']]

            grad_pde = generated_backward(pred_pde, g)

            N_bc = g['N_bc']
            n_out = pred_bc.shape[1]  # output_dim (2 for elasticity)
            grad_bc = 2.0 * (pred_bc - g['bc_target']) / (N_bc * n_out)

            upstream = torch.cat([grad_pde, grad_bc], dim=0)
        pred_batch.backward(gradient=upstream)
        reg = model_reg_loss(model)
        if isinstance(reg, torch.Tensor):
            reg.backward()
        optimizer.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track:
            with torch.no_grad():
                eq_x, eq_y = compute_pde_elasticity(pred_pde, g)
                ii = g['interior_idx']
                lv = (eq_x[ii]**2).mean() + (eq_y[ii]**2).mean()
                final_loss = lv.item()
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: pde_loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: dtpinn (Elasticity) ---
def train_dtpinn_elasticity(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Standard DT-PINN for elasticity: spectral matrices, autograd backward."""
    g = build_grid_data_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred = compiled_model(g['xy_all'])
        eq_x, eq_y = compute_pde_elasticity(pred, g)
        ii = g['interior_idx']
        loss_pde = mse(eq_x[ii], torch.zeros_like(eq_x[ii])) + \
                   mse(eq_y[ii], torch.zeros_like(eq_y[ii]))

        pred_bc = compiled_model(g['xy_bc'])
        loss_bc = mse(pred_bc, g['bc_target'])

        loss = loss_pde + loss_bc + model_reg_loss(model)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: autodiff (Elasticity) ---
def train_autodiff_elasticity(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Plain autodiff PINN for elasticity."""
    coll = build_collocation_points_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)

    if technique == "compile":
        model = torch.compile(model, mode='reduce-overhead')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    xy_int = coll['xy_interior'].clone().requires_grad_(True)
    xy_bc = coll['xy_bc']
    bc_target = coll['bc_target']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        eq_x, eq_y = pde_residuals_elasticity_autodiff(model, xy_int)
        loss_pde = mse(eq_x, torch.zeros_like(eq_x)) + \
                   mse(eq_y, torch.zeros_like(eq_y))

        pred_bc = model(xy_bc)
        loss_bc = mse(pred_bc, bc_target)

        loss = loss_pde + loss_bc + model_reg_loss(model)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    return base_model, train_time, final_loss


# --- Method: ropinn (Elasticity) ---
def train_ropinn_elasticity(seed, device, n_epochs, lr, optimizer_type, technique, grid_size, model_name="mlp", tracker=None):
    """RoPINN: region-optimized PINN with trust region calibration for elasticity."""
    coll = build_collocation_points_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)

    if technique == "compile":
        model = torch.compile(model, mode='reduce-overhead')

    xy_int_base = coll['xy_interior']
    xy_bc = coll['xy_bc']
    bc_target = coll['bc_target']

    # Elasticity domain bounds for clamping [0,1]²
    x_lo, x_hi = 0.0, 1.0
    y_lo, y_hi = 0.0, 1.0

    gradient_list = []
    gradient_variance = 1.0
    final_loss = float('nan')

    if optimizer_type == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
        gradient_list_overall = []
        gradient_list_temp = []

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(n_epochs):
            current_region = np.clip(
                ROPINN_INITIAL_REGION / gradient_variance,
                a_min=0, a_max=ROPINN_REGION_MAX
            )

            def closure():
                optimizer.zero_grad()
                perturbation = torch.rand_like(xy_int_base) * current_region
                xy_perturbed = xy_int_base + perturbation
                xy_perturbed[:, 0].clamp_(x_lo, x_hi)
                xy_perturbed[:, 1].clamp_(y_lo, y_hi)
                xy_perturbed = xy_perturbed.detach().requires_grad_(True)

                eq_x, eq_y = pde_residuals_elasticity_autodiff(model, xy_perturbed)
                loss_pde = mse(eq_x, torch.zeros_like(eq_x)) + \
                           mse(eq_y, torch.zeros_like(eq_y))

                pred_bc = model(xy_bc)
                loss_bc = mse(pred_bc, bc_target)

                loss = loss_pde + loss_bc + model_reg_loss(model)
                loss.backward()

                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1))
                flat_grad = torch.cat(grads).cpu().numpy()
                gradient_list_temp.append(flat_grad)

                return loss

            loss = optimizer.step(closure)
            final_loss = loss.item() if isinstance(loss, torch.Tensor) else loss

            if gradient_list_temp:
                avg_gradient = np.mean(np.array(gradient_list_temp), axis=0)
                gradient_list_overall.append(avg_gradient)
                gradient_list_overall = gradient_list_overall[-ROPINN_PAST_ITERATIONS:]
                gradient_variance = compute_gradient_variance(gradient_list_overall)
                gradient_list_temp.clear()

            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}, "
                      f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")
            if tracker is not None:
                tracker.step(epoch, final_loss, model)

    else:  # adam
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            current_region = np.clip(
                ROPINN_INITIAL_REGION / gradient_variance,
                a_min=0, a_max=ROPINN_REGION_MAX
            )

            perturbation = torch.rand_like(xy_int_base) * current_region
            xy_perturbed = xy_int_base + perturbation
            xy_perturbed[:, 0].clamp_(x_lo, x_hi)
            xy_perturbed[:, 1].clamp_(y_lo, y_hi)
            xy_perturbed = xy_perturbed.detach().requires_grad_(True)

            eq_x, eq_y = pde_residuals_elasticity_autodiff(model, xy_perturbed)
            loss_pde = mse(eq_x, torch.zeros_like(eq_x)) + \
                       mse(eq_y, torch.zeros_like(eq_y))

            pred_bc = model(xy_bc)
            loss_bc = mse(pred_bc, bc_target)

            loss = loss_pde + loss_bc + model_reg_loss(model)
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

            # Gradient tracking for trust region calibration
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
            flat_grad = torch.cat(grads).cpu().numpy()
            gradient_list.append(flat_grad)
            gradient_list = gradient_list[-ROPINN_PAST_ITERATIONS:]
            gradient_variance = compute_gradient_variance(gradient_list)

            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}, "
                      f"region={current_region:.2e}, grad_var={gradient_variance:.4f}")
            if tracker is not None:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    return base_model, train_time, final_loss


# --- Method: sk-pinn (Elasticity) ---
def train_sk_pinn_elasticity(seed, device, n_epochs, lr, grid_size, model_name, grid_data, tracker=None):
    """SK-PINN for elasticity: sparse RKPM matrices, autograd backward."""
    g = grid_data

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        pred = model(g['xy_all'])
        eq_x, eq_y = compute_pde_elasticity_sparse(pred, g)
        ii = g['interior_idx']
        loss_pde = mse(eq_x[ii], torch.zeros_like(eq_x[ii])) + \
                   mse(eq_y[ii], torch.zeros_like(eq_y[ii]))

        pred_bc = model(g['xy_bc'])
        loss_bc = mse(pred_bc, g['bc_target'])

        loss = loss_pde + loss_bc + model_reg_loss(model)
        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        if (epoch + 1) % LOG_INTERVAL == 0:
            print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
        if tracker is not None:
            tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# =============================================================================
# Elasticity evaluation
# =============================================================================
def evaluate_elasticity(model, device):
    """Evaluate elasticity on 51x51 uniform grid: PDE residuals + solution error."""
    nx, ny = 51, 51
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()
    pred = model(xy_t)
    ux, uy = pred[:, 0:1], pred[:, 1:2]

    # PDE residuals via autograd
    grad_ux = gradients(ux, xy_t)
    grad_uy = gradients(uy, xy_t)
    dux_dx, dux_dy = grad_ux[:, 0:1], grad_ux[:, 1:2]
    duy_dx, duy_dy = grad_uy[:, 0:1], grad_uy[:, 1:2]

    grad_dux_dx = gradients(dux_dx, xy_t)
    grad_dux_dy = gradients(dux_dy, xy_t)
    grad_duy_dx = gradients(duy_dx, xy_t)
    grad_duy_dy = gradients(duy_dy, xy_t)
    d2ux_dx2 = grad_dux_dx[:, 0:1]
    d2ux_dy2 = grad_dux_dy[:, 1:2]
    d2uy_dx2 = grad_duy_dx[:, 0:1]
    d2uy_dy2 = grad_duy_dy[:, 1:2]

    d2uy_dxdy = grad_duy_dx[:, 1:2]
    d2ux_dxdy = grad_dux_dx[:, 1:2]

    # Body forces at eval points
    x_coord, y_coord = xy_t[:, 0:1].detach(), xy_t[:, 1:2].detach()
    fx, fy = elasticity_body_forces(x_coord, y_coord)

    eq_x = ((lam_e + 2 * mu_e) * d2ux_dx2 + mu_e * d2ux_dy2
            + (lam_e + mu_e) * d2uy_dxdy + fx)
    eq_y = (mu_e * d2uy_dx2 + (lam_e + 2 * mu_e) * d2uy_dy2
            + (lam_e + mu_e) * d2ux_dxdy + fy)

    eq_x_np = eq_x.detach().cpu().numpy().flatten()
    eq_y_np = eq_y.detach().cpu().numpy().flatten()

    pde_rms = float(np.sqrt(np.mean(eq_x_np**2 + eq_y_np**2)))
    eq_x_rms = float(np.sqrt(np.mean(eq_x_np**2)))
    eq_y_rms = float(np.sqrt(np.mean(eq_y_np**2)))

    # Solution error vs exact
    ux_ex, uy_ex = elasticity_exact(x_coord, y_coord)

    ux_err = (ux.detach() - ux_ex).cpu().numpy().flatten()
    uy_err = (uy.detach() - uy_ex).cpu().numpy().flatten()
    ux_rms_err = float(np.sqrt(np.mean(ux_err**2)))
    uy_rms_err = float(np.sqrt(np.mean(uy_err**2)))

    model.train()
    return {
        'pde_rms': pde_rms, 'continuity_rms': eq_x_rms, 'momentum_rms': eq_y_rms,
        'u_rms_error': ux_rms_err, 'v_rms_error': uy_rms_err, 'p_rms_error': float('nan'),
    }


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_kovasznay = (args.problem == "kovasznay")
    is_elasticity = (args.problem == "elasticity")

    # Validate problem + method combinations
    if is_kovasznay and args.method not in ("sage", "dtpinn", "autodiff", "ropinn", "sk-pinn"):
        print(f"ERROR: Kovasznay problem only supports sage, dtpinn, autodiff, ropinn, sk-pinn methods, "
              f"not '{args.method}'")
        sys.exit(1)
    if is_elasticity and args.method not in ("sage", "dtpinn", "autodiff", "ropinn", "sk-pinn"):
        print(f"ERROR: Elasticity problem only supports sage, dtpinn, autodiff, ropinn, sk-pinn methods, "
              f"not '{args.method}'")
        sys.exit(1)

    # Per-method default grid sizes (problem-aware)
    if args.grid_size is None:
        if is_kovasznay:
            if args.method == 'sk-pinn':
                args.grid_size = 150
            else:
                args.grid_size = 30
        elif is_elasticity:
            if args.method == 'sk-pinn':
                args.grid_size = 100
            else:
                args.grid_size = 30
        else:
            method_defaults = {
                'autodiff': 50, 'dtpinn': 50, 'analytical': 50,
                'ropinn': 50, 'pielm': 50, 'sk-pinn': 200, 'sage': 50,
            }
            args.grid_size = method_defaults[args.method]

    problem_labels = {
        'kovasznay': "Kovasznay Flow (Re=40)",
        'elasticity': "2D Linear Elasticity (Navier-Cauchy)",
        'cavity': "Lid-Driven Cavity NS+Smagorinsky",
    }
    problem_label = problem_labels[args.problem]
    print("=" * 70)
    print(f"UNIFIED BENCHMARK: {problem_label}")
    print("=" * 70)
    print(f"Problem:   {args.problem}")
    print(f"Method:    {args.method}")
    print(f"Model:     {args.model}")
    print(f"Optimizer: {args.optimizer}")
    print(f"LR:        {args.lr}")
    print(f"Epochs:    {args.epochs}")
    print(f"Seed:      {args.seed}")
    print(f"Grid:      {args.grid_size}x{args.grid_size}")
    print(f"Technique: {args.technique}")
    print(f"Device:    {device}")
    if device.type == 'cuda':
        print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"PyTorch:   {torch.__version__}")
    print(f"Tag:       {args.tag or '(none)'}")
    print(f"Output:    {args.output_csv}")
    if args.track:
        print(f"Tracking:  every {args.track_interval} epochs")
    print("=" * 70)

    # Validate technique + method combinations
    if args.technique == "cuda-graph" and args.method != "analytical":
        print(f"ERROR: cuda-graph technique only works with 'analytical' method, "
              f"not '{args.method}'")
        sys.exit(1)

    if args.model != "mlp" and args.method == "pielm":
        print(f"ERROR: --model is not compatible with 'pielm' method (PIELM has its own architecture)")
        sys.exit(1)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create tracker if --track enabled
    tracker = TrainingTracker(args, device) if args.track else None
    if tracker:
        print(f"Tracking to: {tracker.csv_path}")

    # Track status
    status = "OK"
    model = None
    train_time = 0.0
    final_loss = float('nan')
    n_params = 0
    metrics = {'pde_rms': float('nan'), 'continuity_rms': float('nan'), 'momentum_rms': float('nan')}

    try:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # ---- Train ----
        if is_elasticity:
            # Elasticity problem dispatch
            if args.method == "sage":
                model, train_time, final_loss = train_sage_elasticity(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "dtpinn":
                model, train_time, final_loss = train_dtpinn_elasticity(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "autodiff":
                model, train_time, final_loss = train_autodiff_elasticity(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "ropinn":
                model, train_time, final_loss = train_ropinn_elasticity(
                    args.seed, device, args.epochs, args.lr, args.optimizer, args.technique,
                    args.grid_size, args.model, tracker=tracker)
            elif args.method == "sk-pinn":
                g = build_sk_data_elasticity(args.grid_size, device)
                model, train_time, final_loss = train_sk_pinn_elasticity(
                    args.seed, device, args.epochs, args.lr, args.grid_size,
                    args.model, g, tracker=tracker)
            n_params = sum(p.numel() for p in model.parameters())
        elif is_kovasznay:
            # Kovasznay problem dispatch
            if args.method == "sage":
                model, train_time, final_loss = train_sage_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "dtpinn":
                model, train_time, final_loss = train_dtpinn_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "autodiff":
                model, train_time, final_loss = train_autodiff_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "ropinn":
                model, train_time, final_loss = train_ropinn_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.optimizer, args.technique,
                    args.grid_size, args.model, tracker=tracker)
            elif args.method == "sk-pinn":
                g = build_sk_data_kovasznay(args.grid_size, device)
                model, train_time, final_loss = train_sk_pinn_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.grid_size,
                    args.model, g, tracker=tracker)
            n_params = sum(p.numel() for p in model.parameters())
        else:
            # Cavity problem dispatch (original)
            if args.method == "autodiff":
                model, train_time, final_loss = train_autodiff(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "dtpinn":
                model, train_time, final_loss = train_dtpinn(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "analytical":
                model, train_time, final_loss = train_analytical(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "ropinn":
                model, train_time, final_loss = train_ropinn(
                    args.seed, device, args.epochs, args.lr, args.optimizer, args.technique,
                    args.grid_size, args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "pielm":
                model, train_time, final_loss = train_pielm(args.seed, args.grid_size)
                n_params = model.n_hidden * 3  # output weights only (hidden frozen)

            elif args.method == "sk-pinn":
                g = build_sk_data(args.grid_size, device)
                model, train_time, final_loss = train_sk_pinn(
                    args.seed, device, args.epochs, args.lr, args.grid_size,
                    args.model, g, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "sage":
                model, train_time, final_loss = train_sage(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

        # ---- Evaluate ----
        print("\nEvaluating on 51x51 uniform grid...")
        if is_elasticity:
            metrics = evaluate_elasticity(model, device)
        elif is_kovasznay:
            metrics = evaluate_kovasznay(model, device)
        elif args.method == "pielm":
            metrics = evaluate_pielm(model)
        else:
            metrics = evaluate_model(model, device)

        # Check for NaN
        if math.isnan(metrics['pde_rms']):
            status = "DIVERGED"

    except Exception as e:
        status = "FAILED"
        print(f"\nERROR: {e}")
        traceback.print_exc()

    # ---- Collect memory stats ----
    peak_gpu_mb = 0.0
    reserved_gpu_mb = 0.0
    if device.type == 'cuda':
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        reserved_gpu_mb = torch.cuda.memory_reserved() / (1024 * 1024)

    ms_per_epoch = (train_time / args.epochs * 1000) if args.epochs > 0 else 0.0

    # ---- Print results ----
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Status:          {status}")
    print(f"Train time:      {train_time:.2f}s ({train_time/60:.2f} min)")
    print(f"ms/epoch:        {ms_per_epoch:.2f}")
    print(f"Peak GPU mem:    {peak_gpu_mb:.1f} MB")
    print(f"Parameters:      {n_params}")
    print(f"PDE RMS:         {metrics['pde_rms']:.6f}")
    print(f"Continuity RMS:  {metrics['continuity_rms']:.6f}")
    print(f"Momentum RMS:    {metrics['momentum_rms']:.6f}")
    if (is_kovasznay or is_elasticity) and 'u_rms_error' in metrics:
        print(f"u RMS error:     {metrics['u_rms_error']:.6f}")
        print(f"v RMS error:     {metrics['v_rms_error']:.6f}")
        print(f"p RMS error:     {metrics['p_rms_error']:.6f}")
    print(f"Final loss:      {final_loss}")
    print("=" * 70)

    # ---- Write CSV row ----
    row = {
        'timestamp': datetime.now().isoformat(),
        'problem': args.problem,
        'method': args.method,
        'model': args.model if args.method != "pielm" else "pielm",
        'optimizer': args.optimizer if args.method != "pielm" else "direct",
        'lr': args.lr if args.method != "pielm" else 0,
        'epochs': args.epochs,
        'seed': args.seed,
        'grid_size': args.grid_size,
        'technique': args.technique,
        'tag': args.tag,
        'train_time_s': round(train_time, 3),
        'train_time_min': round(train_time / 60, 4),
        'peak_gpu_memory_mb': round(peak_gpu_mb, 1),
        'gpu_memory_reserved_mb': round(reserved_gpu_mb, 1),
        'ms_per_epoch': round(ms_per_epoch, 2),
        'n_params': n_params,
        'pde_rms': round(metrics['pde_rms'], 6) if not math.isnan(metrics['pde_rms']) else 'NaN',
        'continuity_rms': round(metrics['continuity_rms'], 6) if not math.isnan(metrics['continuity_rms']) else 'NaN',
        'momentum_rms': round(metrics['momentum_rms'], 6) if not math.isnan(metrics['momentum_rms']) else 'NaN',
        'final_loss': round(final_loss, 6) if not math.isnan(final_loss) else 'NaN',
        'status': status,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
        'pytorch_version': torch.__version__,
    }

    append_csv_row(args.output_csv, row)
    print(f"\nResults appended to {args.output_csv}")
    if tracker:
        print(f"Tracking data saved to {tracker.csv_path}")


if __name__ == "__main__":
    main()
