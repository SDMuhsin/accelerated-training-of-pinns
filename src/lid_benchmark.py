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

LOG_INTERVAL = 5000

mse = nn.MSELoss()


# =============================================================================
# Argument parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified benchmark for lid-driven cavity NS+Smagorinsky PINN methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", required=True,
                        choices=["autodiff", "dtpinn", "analytical", "ropinn", "pielm", "sk-pinn"],
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
    return parser.parse_args()


# =============================================================================
# Network architecture (shared across all gradient-based methods)
# =============================================================================
class PINN_Cavity(nn.Module):
    """6-layer/64-unit tanh MLP. Output: (u, v, p). 21,827 params."""
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, 64), nn.Tanh()]
        for _ in range(5):
            layers.extend([nn.Linear(64, 64), nn.Tanh()])
        layers.append(nn.Linear(64, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_model(model_name):
    """Factory function: create a model by name."""
    if model_name == "mlp":
        return PINN_Cavity()
    elif model_name == "tsa-pinn":
        from src.experiment_dt_elm_pinn.models.tsa_pinn import TSA_PINN_Cavity
        return TSA_PINN_Cavity(initial_freq=1.0)
    elif model_name == "pirate-net":
        from src.experiment_dt_elm_pinn.models.pirate_net import PirateNet_Cavity
        return PirateNet_Cavity()
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
# TRAINING METHODS
# =============================================================================

# --- Method: autodiff ---
def train_autodiff(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp"):
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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start

    # Unwrap compiled model for evaluation
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    return base_model, train_time, final_loss


# --- Method: dtpinn ---
def train_dtpinn(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", grid_data=None):
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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: analytical ---
def train_analytical(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", grid_data=None):
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

        if (epoch + 1) % LOG_INTERVAL == 0:
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
            print(f"  Epoch {epoch+1}: pde_loss={final_loss:.6f}")

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
def train_ropinn(seed, device, n_epochs, lr, optimizer_type, technique, grid_size, model_name="mlp"):
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
    'timestamp', 'method', 'model', 'optimizer', 'lr', 'epochs', 'seed', 'grid_size',
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
def train_sk_pinn(seed, device, n_epochs, lr, grid_size, model_name, grid_data):
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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Per-method default grid sizes
    if args.grid_size is None:
        method_defaults = {
            'autodiff': 50, 'dtpinn': 50, 'analytical': 50,
            'ropinn': 50, 'pielm': 50, 'sk-pinn': 200,
        }
        args.grid_size = method_defaults[args.method]

    print("=" * 70)
    print("UNIFIED BENCHMARK: Lid-Driven Cavity NS+Smagorinsky")
    print("=" * 70)
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
        if args.method == "autodiff":
            model, train_time, final_loss = train_autodiff(
                args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                args.model)
            n_params = sum(p.numel() for p in model.parameters())

        elif args.method == "dtpinn":
            model, train_time, final_loss = train_dtpinn(
                args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                args.model)
            n_params = sum(p.numel() for p in model.parameters())

        elif args.method == "analytical":
            model, train_time, final_loss = train_analytical(
                args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                args.model)
            n_params = sum(p.numel() for p in model.parameters())

        elif args.method == "ropinn":
            model, train_time, final_loss = train_ropinn(
                args.seed, device, args.epochs, args.lr, args.optimizer, args.technique,
                args.grid_size, args.model)
            n_params = sum(p.numel() for p in model.parameters())

        elif args.method == "pielm":
            model, train_time, final_loss = train_pielm(args.seed, args.grid_size)
            n_params = model.n_hidden * 3  # output weights only (hidden frozen)

        elif args.method == "sk-pinn":
            g = build_sk_data(args.grid_size, device)
            model, train_time, final_loss = train_sk_pinn(
                args.seed, device, args.epochs, args.lr, args.grid_size,
                args.model, g)
            n_params = sum(p.numel() for p in model.parameters())

        # ---- Evaluate ----
        print("\nEvaluating on 51x51 uniform grid...")
        if args.method == "pielm":
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
    print(f"Final loss:      {final_loss}")
    print("=" * 70)

    # ---- Write CSV row ----
    row = {
        'timestamp': datetime.now().isoformat(),
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


if __name__ == "__main__":
    main()
