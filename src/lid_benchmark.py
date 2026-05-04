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
import copy
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

# SK-PINN model-specific weight decay: RKPM's O(h^2) accuracy limits the useful
# model complexity. More expressive architectures need stronger regularization to
# prevent overfitting the discretization (learning features that satisfy the RKPM-
# discretized PDE but not the true PDE).  PirateNet's multiplicative gating is
# especially prone to this; its momentum residual (evaluated by autograd) can
# degrade 5.5x while training loss still decreases with weight_decay=1e-4.
_SK_PINN_WD = {
    'mlp':        1e-4,   # baseline — mild overfitting (2.2x) is acceptable
    'tsa-pinn':   5e-4,   # trainable sinusoidal frequencies add capacity
    'pirate-net': 1e-3,   # adaptive gating amplifies RKPM exploitation
}

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
                        choices=["autodiff", "dtpinn", "chebyshev-pinn", "analytical", "ropinn", "pielm", "sk-pinn", "sage", "jaxpinn", "sage-jax", "bfsa", "sdccg", "slrm", "slrm-jax", "stencil-adjoint", "can-pinn-faithful"],
                        help="Training method. 'dtpinn' is the paper-faithful "
                             "Sharma & Shankar 2022 RBF-FD + fp64 + L-BFGS variant. "
                             "'chebyshev-pinn' is the prior Chebyshev-spectral baseline. "
                             "'can-pinn-faithful' is the Chiu et al. 2022 CMAME coupled "
                             "automatic-numerical differentiation method (cavity only in Phase 2).")
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
    # ---- Faithful DT-PINN (Sharma & Shankar 2022) controls ----
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp64"],
                        help="Compute precision. dtpinn defaults to fp64 if --dtype is "
                             "left at fp32; pass fp32 explicitly to force fp32 for ablation.")
    parser.add_argument("--rbf-fd-order", type=int, default=4, choices=[2, 3, 4, 5],
                        help="RBF-FD polynomial order p (only used by --method dtpinn)")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="Approximate number of scattered nodes (Ni+Nb) for "
                             "--method dtpinn. Default: ~grid_size² to keep the "
                             "node count comparable to the Chebyshev baseline.")
    parser.add_argument("--match-protocol", action="store_true",
                        help="(SK-PINN, DT-PINN) drop method-specific protocol "
                             "knobs and run on the Chebyshev-paired uniform grid "
                             "(N=50 cavity / N=30 kovasznay / N=30 elasticity) so the "
                             "row is apples-to-apples with the rest of the "
                             "matched-protocol baselines (autodiff, sage, "
                             "chebyshev-pinn, can-pinn-faithful). For sk-pinn this "
                             "drops weight decay and the elasticity cosine scheduler. "
                             "For dtpinn this skips the paper-faithful four-flag "
                             "override (Adam-only, fp32, no L-BFGS, no auto-restart) "
                             "while keeping the RBF-FD operators (the method's "
                             "distinguishing feature). Tag the run explicitly "
                             "(e.g., --tag=sk_pinn_matched_<date> or "
                             "--tag=dtpinn_matched_<date>) to keep results separable "
                             "from the historical/faithful rows.")
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


# =============================================================================
# Compact-stencil (FD) adjoint operators for stencil-adjoint method
# =============================================================================
def _fornberg_weights(z, x, m):
    """Fornberg 1988 recurrence: FD weights for derivatives up to order m
    at location z using nodes x[0..n].  Returns array of shape (n+1, m+1);
    column k gives the k-th-derivative weights.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x) - 1
    c = np.zeros((n + 1, m + 1))
    c1 = 1.0
    c4 = x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def _fd_stencil_1d(x_nodes, order=1, half_bandwidth=3):
    """Banded 1D differentiation matrix on (possibly non-uniform) nodes.

    Row i approximates d^order/dx^order at x_nodes[i] using the window of
    2*half_bandwidth+1 nearest nodes (clipped to the grid boundaries via
    one-sided stencils).  Returns (N, N) float64 dense ndarray whose
    nonzero pattern is banded + one-sided near edges.
    """
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    N = len(x_nodes)
    w = 2 * half_bandwidth + 1
    w = min(w, N)
    D = np.zeros((N, N))
    for i in range(N):
        lo = max(0, i - half_bandwidth)
        hi = lo + w
        if hi > N:
            hi = N
            lo = hi - w
        idx = np.arange(lo, hi)
        weights = _fornberg_weights(x_nodes[i], x_nodes[idx], order)
        D[i, idx] = weights[:, order]
    return D


def _inject_stencil_adjoint(g, problem, half_bandwidth=3):
    """Mutate grid-data g in place: replace transpose-operator entries
    (DxT, DyT, and for elasticity DxxT, DyyT, DxyT) with compact-stencil
    approximations.  Forward operators Dx, Dy, ... are left untouched so
    the observed residual (and loss value) remains exact.

    The stencil is derived from a local polynomial interpolant on the
    same Chebyshev node set; the approximation error on each operator is
    O(h^p) where p = 2*half_bandwidth.
    """
    N = g['N_grid']
    device = g['xy_all'].device

    if problem in ('cavity', 'elasticity'):
        x_ref = chebyshev_points(N)
        x_phys = 0.5 * (x_ref + 1.0)  # physical nodes on [0,1]
        x_nodes = y_nodes = x_phys
    elif problem == 'kovasznay':
        Lx, Ly = 1.5, 2.0
        x0, y0 = -0.5, -0.5
        x_ref = chebyshev_points(N)
        x_nodes = x0 + Lx * 0.5 * (x_ref + 1.0)
        y_nodes = y0 + Ly * 0.5 * (x_ref + 1.0)
    else:
        raise ValueError(f"unknown problem for stencil injector: {problem}")

    # Fornberg weights on physical nodes are already in physical units; no
    # extra domain-length scaling needed.
    Dx1d = _fd_stencil_1d(x_nodes, order=1, half_bandwidth=half_bandwidth)
    Dy1d = _fd_stencil_1d(y_nodes, order=1, half_bandwidth=half_bandwidth)

    I_mat = np.eye(N)
    Dx_sten = np.kron(I_mat, Dx1d).astype(np.float32)
    Dy_sten = np.kron(Dy1d, I_mat).astype(np.float32)

    g['DxT'] = torch.tensor(Dx_sten.T, dtype=torch.float32, device=device).contiguous()
    g['DyT'] = torch.tensor(Dy_sten.T, dtype=torch.float32, device=device).contiguous()

    if problem == 'elasticity':
        Dxx_sten = (Dx_sten @ Dx_sten).astype(np.float32)
        Dyy_sten = (Dy_sten @ Dy_sten).astype(np.float32)
        Dxy_sten = (Dy_sten @ Dx_sten).astype(np.float32)
        g['DxxT'] = torch.tensor(Dxx_sten.T, dtype=torch.float32, device=device).contiguous()
        g['DyyT'] = torch.tensor(Dyy_sten.T, dtype=torch.float32, device=device).contiguous()
        g['DxyT'] = torch.tensor(Dxy_sten.T, dtype=torch.float32, device=device).contiguous()

    g['_stencil_hbw'] = half_bandwidth
    return g


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
        'nu_lam': float(nu_laminar),  # Program-B F1 parametric threading
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
    """PDE residuals via spectral differentiation matrices.

    Laminar viscosity is read from g['nu_lam'] if present (Program-B parametric
    family F1), otherwise falls back to the module-level nu_laminar for legacy
    callers that expect the fixed Re=1000 cavity.
    """
    nu_lam = g['nu_lam'] if 'nu_lam' in g else nu_laminar
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    du_dx = g['Dx'] @ u; du_dy = g['Dy'] @ u
    dv_dx = g['Dx'] @ v; dv_dy = g['Dy'] @ v
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_lam + g['Cs_d_sq'] * S_mag
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
    nu_lam = g['nu_lam'] if 'nu_lam' in g else nu_laminar
    Dx, Dy = g['Dx'], g['Dy']
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    du_dx = torch.sparse.mm(Dx, u); du_dy = torch.sparse.mm(Dy, u)
    dv_dx = torch.sparse.mm(Dx, v); dv_dy = torch.sparse.mm(Dy, v)
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx**2 + Syy**2 + 2.0 * Sxy**2) + 1e-12)
    nu_eff = nu_lam + g['Cs_d_sq'] * S_mag
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
# CAN-PINN faithful (Chiu et al. 2022, CMAME 395, 114909) — independent
# PyTorch reimplementation of the coupled automatic-numerical differentiation
# scheme. NO upstream code copied; algorithm reconstructed from the paper and
# the spec at llmdocs/trackers/can_pinn_replication_2026-04-29.md.
#
# Stencil per interior point: 9 evaluations (C, E, W, N, S, EE, WW, NN, SS)
# with 5 AD first-derivative computations (at C, E, W, N, S). Convection uses
# can(uw2) (eq. 8/9), pressure gradient uses can(cd) (eq. 12/13), viscous
# Laplacian uses plain 2nd-order central difference. The /8 dispersion term
# is included for cd-pressure (matches paper eq. 12) and EXCLUDED for uw2
# convection (matches upstream demo notebook; eq. 11's /8 is a higher-order
# modified-equation correction the notebook drops).
# =============================================================================
def build_canpinn_grid_data(N_grid, device):
    """Build a uniform NxN grid on [0,1]^2 with interior/lid/wall classification.

    The grid is uniform (NOT Chebyshev) because the can-PINN stencil presupposes
    equispaced points. dx = dy = 1/(N_grid - 1). Boundary points are classified
    the same way as build_grid_data: top edge -> lid (u=1, v=0), other three
    edges -> wall (u=0, v=0), with corners assigned to lid via the
    `is_wall = is_boundary & ~is_lid` rule (matches upstream notebook's
    `_left & ~_top`, `_right & ~_top`).

    Returns a dict with the keys consumed by pde_residuals_canpinn_cavity().
    Stencil queries are computed inside the residual function from `xy_int`
    (interior collocation coordinates) and are NOT precomputed here, so this
    builder is intentionally minimal — neighbor coordinates can move outside
    [0,1]^2 (matches the upstream notebook's "no clamping" behavior; see
    Phase 1 spec gotcha 7.7).
    """
    dx = 1.0 / (N_grid - 1)
    dy = dx

    x_lin = np.linspace(0.0, 1.0, N_grid)
    xx, yy = np.meshgrid(x_lin, x_lin, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)
    is_lid = (yc > 1 - eps)
    is_wall = is_boundary & ~is_lid

    interior_idx = np.where(~is_boundary)[0]
    lid_idx = np.where(is_lid)[0]
    wall_idx = np.where(is_wall)[0]

    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_int = xy_all[interior_idx].contiguous()
    xy_lid = xy_all[lid_idx].contiguous()
    xy_wall = xy_all[wall_idx].contiguous()

    # Wall distance for Smagorinsky at the interior (C) points only.
    x_int = xy_int[:, 0:1]
    y_int = xy_int[:, 1:2]
    d_wall_int = torch.min(torch.min(x_int, 1.0 - x_int),
                           torch.min(y_int, 1.0 - y_int))
    Cs_d_sq_int = (Cs * d_wall_int) ** 2

    return {
        'xy_all': xy_all, 'xy_int': xy_int, 'xy_lid': xy_lid, 'xy_wall': xy_wall,
        'interior_idx': interior_idx, 'lid_idx': lid_idx, 'wall_idx': wall_idx,
        'd_wall_int': d_wall_int, 'Cs_d_sq_int': Cs_d_sq_int,
        'dx': dx, 'dy': dy,
        'N_grid': N_grid, 'N_int': len(interior_idx),
        'N_lid': len(lid_idx), 'N_wall': len(wall_idx),
        'nu_lam': float(nu_laminar),
    }


def _canpinn_stencil_offsets(dx, dy, device, dtype=torch.float32):
    """Return the 9 (Δx, Δy) offsets in the order [C, E, W, N, S, EE, WW, NN, SS]."""
    return torch.tensor([
        [ 0.0,    0.0   ],   # C
        [ dx,     0.0   ],   # E
        [-dx,     0.0   ],   # W
        [ 0.0,    dy    ],   # N
        [ 0.0,   -dy    ],   # S
        [ 2*dx,   0.0   ],   # EE
        [-2*dx,   0.0   ],   # WW
        [ 0.0,    2*dy  ],   # NN
        [ 0.0,   -2*dy  ],   # SS
    ], dtype=dtype, device=device)


def pde_residuals_canpinn_cavity(model, xy_int, dx, dy,
                                 Cs_d_sq_int=None, nu_lam=None,
                                 use_smagorinsky=True, return_components=False):
    """Compute the CAN-PINN PDE residuals at the given interior points.

    `xy_int` : (N_int, 2) tensor of interior collocation points. Boundary
        points must already be excluded by the caller (we do not re-mask here).
    `dx, dy` : scalar stencil spacings (Python floats, or 0-d tensors).
    `Cs_d_sq_int` : (N_int, 1) precomputed (Cs * d_wall)^2 at interior points.
        Required iff use_smagorinsky=True.
    `nu_lam` : scalar laminar viscosity. Defaults to module-level nu_laminar
        (= 1/Re_cavity = 1/1000).
    `use_smagorinsky` : if True (harness drop-in for cavity Re=1000+Smag),
        compute nu_eff = nu_lam + (Cs*d_wall)^2 |S| at C and use it in the
        viscous Laplacian via (1/Re_eff)·∇²U with Re_eff = 1/nu_eff. If False
        (paper-faithful Re=400 plain NS), use 1/Re·∇²U with Re = 1/nu_lam.

    Returns (continuity, mom_u, mom_v), each shape (N_int, 1).

    The implementation is a single forward pass over a 9*N_int batched input
    plus a single autograd.grad call to obtain (du/dx, du/dy, dv/dx, dv/dy,
    dp/dx, dp/dy) at every stencil location. The 5 AD-needed derivatives
    (at C, E, W, N, S) are then sliced from the full tensor — EE/WW/NN/SS
    AD gradients are computed but not used (cheap).
    """
    if nu_lam is None:
        nu_lam = nu_laminar

    N_int = xy_int.shape[0]
    device = xy_int.device
    dtype = xy_int.dtype

    # Build the 9*N_int stencil batch by broadcasting offsets onto each point.
    offs = _canpinn_stencil_offsets(dx, dy, device, dtype)        # (9, 2)
    xy_stencil = (xy_int.unsqueeze(0) + offs.unsqueeze(1))         # (9, N_int, 2)
    xy_stencil = xy_stencil.reshape(9 * N_int, 2)
    xy_stencil = xy_stencil.detach().requires_grad_(True)

    # Single forward pass over all stencil points.
    pred = model(xy_stencil)                                       # (9*N_int, 3)
    u_all = pred[:, 0:1]
    v_all = pred[:, 1:2]
    p_all = pred[:, 2:3]

    # Single AD call gets gradients of all three outputs at every stencil
    # location. We compute one gradient per output (3 calls), each accumulating
    # over the full 9*N_int batch — this gives us du, dv, dp at every neighbor
    # in three autograd.grad invocations rather than one per location.
    grad_u = torch.autograd.grad(u_all, xy_stencil,
                                 grad_outputs=torch.ones_like(u_all),
                                 create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(v_all, xy_stencil,
                                 grad_outputs=torch.ones_like(v_all),
                                 create_graph=True, retain_graph=True)[0]
    grad_p = torch.autograd.grad(p_all, xy_stencil,
                                 grad_outputs=torch.ones_like(p_all),
                                 create_graph=True, retain_graph=True)[0]

    # Reshape to (9, N_int, 1).
    u_s = u_all.reshape(9, N_int, 1)
    v_s = v_all.reshape(9, N_int, 1)
    p_s = p_all.reshape(9, N_int, 1)
    ux_s = grad_u[:, 0:1].reshape(9, N_int, 1)
    uy_s = grad_u[:, 1:2].reshape(9, N_int, 1)
    vx_s = grad_v[:, 0:1].reshape(9, N_int, 1)
    vy_s = grad_v[:, 1:2].reshape(9, N_int, 1)
    px_s = grad_p[:, 0:1].reshape(9, N_int, 1)
    py_s = grad_p[:, 1:2].reshape(9, N_int, 1)

    # Slice each stencil location.  Order: C, E, W, N, S, EE, WW, NN, SS.
    u_C, u_E, u_W, u_N, u_S, _, _, _, _ = (u_s[i] for i in range(9))
    v_C, v_E, v_W, v_N, v_S, _, _, _, _ = (v_s[i] for i in range(9))
    p_C, p_E, p_W, p_N, p_S, _, _, _, _ = (p_s[i] for i in range(9))
    u_x_C = ux_s[0]; u_x_E = ux_s[1]; u_x_W = ux_s[2]
    u_y_C = uy_s[0]; u_y_N = uy_s[3]; u_y_S = uy_s[4]
    v_x_C = vx_s[0]; v_x_E = vx_s[1]; v_x_W = vx_s[2]
    v_y_C = vy_s[0]; v_y_N = vy_s[3]; v_y_S = vy_s[4]
    p_x_C = px_s[0]; p_x_E = px_s[1]; p_x_W = px_s[2]
    p_y_C = py_s[0]; p_y_N = py_s[3]; p_y_S = py_s[4]

    # Face velocities (paper eq. 7 / notebook lines 363-365).
    u_face_e = 0.5 * (u_E + u_C)
    u_face_w = 0.5 * (u_W + u_C)
    v_face_n = 0.5 * (v_N + v_C)
    v_face_s = 0.5 * (v_S + v_C)

    # CAN(uw2) convection (paper eq. 8/9; notebook lines 444-476).
    # /8 dispersion correction commented out in upstream demo — match that.
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy

    Ue_minus = u_C + u_x_C * half_dx
    Ue_plus  = u_E - u_x_E * half_dx
    U_e = torch.where(u_face_e >= 0.0, Ue_minus, Ue_plus)

    Uw_minus = u_W + u_x_W * half_dx
    Uw_plus  = u_C - u_x_C * half_dx
    U_w = torch.where(u_face_w >= 0.0, Uw_minus, Uw_plus)

    Un_minus = u_C + u_y_C * half_dy
    Un_plus  = u_N - u_y_N * half_dy
    U_n = torch.where(v_face_n >= 0.0, Un_minus, Un_plus)

    Us_minus = u_S + u_y_S * half_dy
    Us_plus  = u_C - u_y_C * half_dy
    U_s = torch.where(v_face_s >= 0.0, Us_minus, Us_plus)

    Ve_minus = v_C + v_x_C * half_dx
    Ve_plus  = v_E - v_x_E * half_dx
    V_e = torch.where(u_face_e >= 0.0, Ve_minus, Ve_plus)

    Vw_minus = v_W + v_x_W * half_dx
    Vw_plus  = v_C - v_x_C * half_dx
    V_w = torch.where(u_face_w >= 0.0, Vw_minus, Vw_plus)

    Vn_minus = v_C + v_y_C * half_dy
    Vn_plus  = v_N - v_y_N * half_dy
    V_n = torch.where(v_face_n >= 0.0, Vn_minus, Vn_plus)

    Vs_minus = v_S + v_y_S * half_dy
    Vs_plus  = v_C - v_y_C * half_dy
    V_s = torch.where(v_face_s >= 0.0, Vs_minus, Vs_plus)

    UU_x = (u_face_e * U_e - u_face_w * U_w) / dx
    VU_y = (v_face_n * U_n - v_face_s * U_s) / dy
    UV_x = (u_face_e * V_e - u_face_w * V_w) / dx
    VV_y = (v_face_n * V_n - v_face_s * V_s) / dy

    # CAN(cd) pressure (paper eq. 12/13; notebook lines 478-485). /8 term ON.
    eighth_dx = dx / 8.0
    eighth_dy = dy / 8.0
    p_e = 0.5 * (p_C + p_E) - (p_x_E - p_x_C) * eighth_dx
    p_w = 0.5 * (p_W + p_C) - (p_x_C - p_x_W) * eighth_dx
    p_n = 0.5 * (p_C + p_N) - (p_y_N - p_y_C) * eighth_dy
    p_s = 0.5 * (p_S + p_C) - (p_y_C - p_y_S) * eighth_dy
    P_x = (p_e - p_w) / dx
    P_y = (p_n - p_s) / dy

    # Plain 2nd-order central difference for viscous Laplacian (notebook 402-405).
    Uxx = (u_E - 2.0 * u_C + u_W) / (dx * dx)
    Uyy = (u_N - 2.0 * u_C + u_S) / (dy * dy)
    Vxx = (v_E - 2.0 * v_C + v_W) / (dx * dx)
    Vyy = (v_N - 2.0 * v_C + v_S) / (dy * dy)

    # Continuity from staggered face velocities (notebook line 365).
    div = (u_face_e - u_face_w) / dx + (v_face_n - v_face_s) / dy

    # Effective viscosity. For Smagorinsky harness drop-in: nu_eff at C from
    # AD-S evaluated at C, then used as a local-constant in the FD Laplacian.
    # This is design choice b1 from the spec (§7.1). For paper-faithful plain
    # NS, use_smagorinsky=False -> nu_eff = nu_lam (constant).
    if use_smagorinsky:
        if Cs_d_sq_int is None:
            raise ValueError("Cs_d_sq_int must be provided when use_smagorinsky=True")
        Sxx = u_x_C
        Syy = v_y_C
        Sxy = 0.5 * (u_y_C + v_x_C)
        S_mag = torch.sqrt(2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2) + 1e-12)
        nu_eff = nu_lam + Cs_d_sq_int * S_mag
    else:
        nu_eff = nu_lam

    # PDE residuals (paper eq. 14 / notebook 487-489). Conservative form:
    # mom = (uU)_x + (vU)_y - nu·(U_xx + U_yy) - U·div + P_x.
    R_continuity = div
    R_mom_u = UU_x + VU_y - nu_eff * (Uxx + Uyy) - u_C * div + P_x
    R_mom_v = UV_x + VV_y - nu_eff * (Vxx + Vyy) - v_C * div + P_y

    if return_components:
        # For the stencil-sanity script: expose the can-PDE residuals and a
        # diagnostic of nu_eff so callers can compare them to AD residuals on
        # the same physics.
        return R_continuity, R_mom_u, R_mom_v, {'nu_eff': nu_eff}
    return R_continuity, R_mom_u, R_mom_v


# =============================================================================
# CAN-PINN extension: Kovasznay flow (Phase 5)
# =============================================================================
# Steady incompressible NS at Re=40 with the Kovasznay closed-form solution as
# Dirichlet BCs on [-0.5, 1.0] x [-0.5, 1.5]. Constant viscosity nu_kov = 1/40
# (no Smagorinsky term). The Taylor-coupled FD stencil (paper Chiu et al. 2022)
# is identical to the cavity scheme: can(uw2) for convection, can(cd) for the
# pressure gradient, and plain 2nd-order central FD for the viscous Laplacian.
# Domain is rectangular but non-square so dx and dy are independently sized.
# =============================================================================
def build_canpinn_grid_data_kovasznay(N_grid, device):
    """Build a uniform NxN grid on the Kovasznay domain [-0.5, 1.0] x [-0.5, 1.5].

    The grid is uniform (NOT Chebyshev) because the can-PINN stencil presupposes
    equispaced points. dx = Lx/(N_grid - 1), dy = Ly/(N_grid - 1) — the two are
    different because the domain is non-square (Lx=1.5, Ly=2.0).

    All boundary nodes carry Dirichlet BCs from `kovasznay_exact`; there is no
    distinction between "lid" and "wall" here. Stencil neighbors at points one
    cell from the boundary will land outside the domain — matches upstream
    notebook behavior (no clamping; see Phase 1 spec gotcha 7.7).
    """
    Lx, Ly = 1.5, 2.0
    x0, y0 = -0.5, -0.5

    dx = Lx / (N_grid - 1)
    dy = Ly / (N_grid - 1)

    x_lin = np.linspace(x0, x0 + Lx, N_grid)
    y_lin = np.linspace(y0, y0 + Ly, N_grid)
    xx, yy = np.meshgrid(x_lin, y_lin, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = ((xc < x0 + eps) | (xc > x0 + Lx - eps) |
                   (yc < y0 + eps) | (yc > y0 + Ly - eps))

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_int = xy_all[interior_idx].contiguous()
    xy_bc = xy_all[bc_idx].contiguous()

    # Exact BC values for u, v, p at boundary nodes.
    u_ex, v_ex, p_ex = kovasznay_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([u_ex, v_ex, p_ex], dim=1)  # (N_bc, 3)

    # Pressure reference at domain center (matches chebyshev variant).
    x_center = torch.tensor([[x0 + Lx / 2, y0 + Ly / 2]], dtype=torch.float32, device=device)
    _, _, p_ctr = kovasznay_exact(x_center[:, 0:1], x_center[:, 1:2])
    p_center_exact = p_ctr.item()

    return {
        'xy_all': xy_all, 'xy_int': xy_int, 'xy_bc': xy_bc,
        'xy_center': x_center,
        'interior_idx': interior_idx, 'bc_idx': bc_idx,
        'bc_target': bc_target, 'p_center_exact': p_center_exact,
        'dx': dx, 'dy': dy,
        'N_grid': N_grid, 'N_int': len(interior_idx), 'N_bc': len(bc_idx),
        'Lx': Lx, 'Ly': Ly, 'x0': x0, 'y0': y0,
        'nu_kov': float(nu_kov),
    }


def pde_residuals_canpinn_kov(model, xy_int, dx, dy, nu=None):
    """CAN-PINN PDE residuals for steady Kovasznay NS at the given interior points.

    `xy_int` : (N_int, 2) tensor of interior collocation points (boundary
        excluded by the caller).
    `dx, dy` : scalar stencil spacings (Python floats).
    `nu`     : kinematic viscosity. Defaults to module-level nu_kov (= 1/40).

    The scheme mirrors `pde_residuals_canpinn_cavity` but with:
      - constant viscosity (no Smagorinsky term),
      - no homogeneous-boundary specialization (BCs come from the exact
        Kovasznay solution and are imposed externally as Dirichlet residuals).

    Stencil layout (paper Fig 2): C, E, W, N, S, EE, WW, NN, SS — 9 forward-
    pass evaluations. AD gradients are taken at C, E, W, N, S only; EE/WW/NN/SS
    contribute through their direct values via the upwind branch selectors.
    Returns (continuity, mom_u, mom_v), each shape (N_int, 1).
    """
    if nu is None:
        nu = nu_kov

    N_int = xy_int.shape[0]
    device = xy_int.device
    dtype = xy_int.dtype

    offs = _canpinn_stencil_offsets(dx, dy, device, dtype)        # (9, 2)
    xy_stencil = (xy_int.unsqueeze(0) + offs.unsqueeze(1))         # (9, N_int, 2)
    xy_stencil = xy_stencil.reshape(9 * N_int, 2)
    xy_stencil = xy_stencil.detach().requires_grad_(True)

    pred = model(xy_stencil)                                       # (9*N_int, 3)
    u_all = pred[:, 0:1]
    v_all = pred[:, 1:2]
    p_all = pred[:, 2:3]

    grad_u = torch.autograd.grad(u_all, xy_stencil,
                                 grad_outputs=torch.ones_like(u_all),
                                 create_graph=True, retain_graph=True)[0]
    grad_v = torch.autograd.grad(v_all, xy_stencil,
                                 grad_outputs=torch.ones_like(v_all),
                                 create_graph=True, retain_graph=True)[0]
    grad_p = torch.autograd.grad(p_all, xy_stencil,
                                 grad_outputs=torch.ones_like(p_all),
                                 create_graph=True, retain_graph=True)[0]

    u_s = u_all.reshape(9, N_int, 1)
    v_s = v_all.reshape(9, N_int, 1)
    p_s = p_all.reshape(9, N_int, 1)
    ux_s = grad_u[:, 0:1].reshape(9, N_int, 1)
    uy_s = grad_u[:, 1:2].reshape(9, N_int, 1)
    vx_s = grad_v[:, 0:1].reshape(9, N_int, 1)
    vy_s = grad_v[:, 1:2].reshape(9, N_int, 1)
    px_s = grad_p[:, 0:1].reshape(9, N_int, 1)
    py_s = grad_p[:, 1:2].reshape(9, N_int, 1)

    u_C, u_E, u_W, u_N, u_S, _, _, _, _ = (u_s[i] for i in range(9))
    v_C, v_E, v_W, v_N, v_S, _, _, _, _ = (v_s[i] for i in range(9))
    p_C, p_E, p_W, p_N, p_S, _, _, _, _ = (p_s[i] for i in range(9))
    u_x_C = ux_s[0]; u_x_E = ux_s[1]; u_x_W = ux_s[2]
    u_y_C = uy_s[0]; u_y_N = uy_s[3]; u_y_S = uy_s[4]
    v_x_C = vx_s[0]; v_x_E = vx_s[1]; v_x_W = vx_s[2]
    v_y_C = vy_s[0]; v_y_N = vy_s[3]; v_y_S = vy_s[4]
    p_x_C = px_s[0]; p_x_E = px_s[1]; p_x_W = px_s[2]
    p_y_C = py_s[0]; p_y_N = py_s[3]; p_y_S = py_s[4]

    # Face velocities (paper eq. 7).
    u_face_e = 0.5 * (u_E + u_C)
    u_face_w = 0.5 * (u_W + u_C)
    v_face_n = 0.5 * (v_N + v_C)
    v_face_s = 0.5 * (v_S + v_C)

    # CAN(uw2) convection (paper eq. 8/9). /8 dispersion correction omitted to
    # match the upstream demo notebook (Phase 1 spec gotcha 7.2).
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy

    Ue_minus = u_C + u_x_C * half_dx
    Ue_plus  = u_E - u_x_E * half_dx
    U_e = torch.where(u_face_e >= 0.0, Ue_minus, Ue_plus)

    Uw_minus = u_W + u_x_W * half_dx
    Uw_plus  = u_C - u_x_C * half_dx
    U_w = torch.where(u_face_w >= 0.0, Uw_minus, Uw_plus)

    Un_minus = u_C + u_y_C * half_dy
    Un_plus  = u_N - u_y_N * half_dy
    U_n = torch.where(v_face_n >= 0.0, Un_minus, Un_plus)

    Us_minus = u_S + u_y_S * half_dy
    Us_plus  = u_C - u_y_C * half_dy
    U_s = torch.where(v_face_s >= 0.0, Us_minus, Us_plus)

    Ve_minus = v_C + v_x_C * half_dx
    Ve_plus  = v_E - v_x_E * half_dx
    V_e = torch.where(u_face_e >= 0.0, Ve_minus, Ve_plus)

    Vw_minus = v_W + v_x_W * half_dx
    Vw_plus  = v_C - v_x_C * half_dx
    V_w = torch.where(u_face_w >= 0.0, Vw_minus, Vw_plus)

    Vn_minus = v_C + v_y_C * half_dy
    Vn_plus  = v_N - v_y_N * half_dy
    V_n = torch.where(v_face_n >= 0.0, Vn_minus, Vn_plus)

    Vs_minus = v_S + v_y_S * half_dy
    Vs_plus  = v_C - v_y_C * half_dy
    V_s = torch.where(v_face_s >= 0.0, Vs_minus, Vs_plus)

    UU_x = (u_face_e * U_e - u_face_w * U_w) / dx
    VU_y = (v_face_n * U_n - v_face_s * U_s) / dy
    UV_x = (u_face_e * V_e - u_face_w * V_w) / dx
    VV_y = (v_face_n * V_n - v_face_s * V_s) / dy

    # CAN(cd) pressure (paper eq. 12/13). /8 term ON.
    eighth_dx = dx / 8.0
    eighth_dy = dy / 8.0
    p_e = 0.5 * (p_C + p_E) - (p_x_E - p_x_C) * eighth_dx
    p_w = 0.5 * (p_W + p_C) - (p_x_C - p_x_W) * eighth_dx
    p_n = 0.5 * (p_C + p_N) - (p_y_N - p_y_C) * eighth_dy
    p_s = 0.5 * (p_S + p_C) - (p_y_C - p_y_S) * eighth_dy
    P_x = (p_e - p_w) / dx
    P_y = (p_n - p_s) / dy

    # Plain 2nd-order central difference for viscous Laplacian.
    Uxx = (u_E - 2.0 * u_C + u_W) / (dx * dx)
    Uyy = (u_N - 2.0 * u_C + u_S) / (dy * dy)
    Vxx = (v_E - 2.0 * v_C + v_W) / (dx * dx)
    Vyy = (v_N - 2.0 * v_C + v_S) / (dy * dy)

    # Continuity from staggered face velocities.
    div = (u_face_e - u_face_w) / dx + (v_face_n - v_face_s) / dy

    # Steady incompressible NS at constant nu (paper eq. 14, conservative form).
    R_continuity = div
    R_mom_u = UU_x + VU_y - nu * (Uxx + Uyy) - u_C * div + P_x
    R_mom_v = UV_x + VV_y - nu * (Vxx + Vyy) - v_C * div + P_y

    return R_continuity, R_mom_u, R_mom_v


# =============================================================================
# CAN-PINN extension: 2D linear elasticity (Phase 5)
# =============================================================================
# 2D Navier-Cauchy with manufactured solution on [0,1]^2:
#   (lam+2*mu) * d^2 ux/dx^2 + mu * d^2 ux/dy^2 + (lam+mu) * d^2 uy/(dx*dy) + fx = 0
#   mu * d^2 uy/dx^2 + (lam+2*mu) * d^2 uy/dy^2 + (lam+mu) * d^2 ux/(dx*dy) + fy = 0
# 2 outputs (ux, uy), no pressure, no convection. The CAN-PINN coupling for
# 2nd-order PDEs is can(cd) — Taylor-augmented central-difference for the
# divergence-of-gradient term and plain central FD for the Laplacians (Phase 1
# spec §8.2). The cross derivative d^2 uy/(dx*dy) is computed by a 4-point
# stencil (corners not needed): central difference of the AD gradient `du/dy`
# along x at C, identical in form to the cd-pressure scheme — this preserves
# the Taylor coupling along x while leaving the y-derivative to AD.
# =============================================================================
def build_canpinn_grid_data_elasticity(N_grid, device):
    """Build a uniform NxN grid on [0,1]^2 for the elasticity manufactured PDE.

    All boundary nodes carry Dirichlet BCs from `elasticity_exact` (no lid/wall
    distinction). Body forces fx, fy are precomputed at every grid point so the
    residual function can index them with the interior mask.
    """
    dx = 1.0 / (N_grid - 1)
    dy = dx

    x_lin = np.linspace(0.0, 1.0, N_grid)
    xx, yy = np.meshgrid(x_lin, x_lin, indexing='xy')
    xy_grid = np.column_stack([xx.ravel(), yy.ravel()])

    eps = 1e-10
    xc, yc = xy_grid[:, 0], xy_grid[:, 1]
    is_boundary = (xc < eps) | (xc > 1 - eps) | (yc < eps) | (yc > 1 - eps)

    interior_idx = np.where(~is_boundary)[0]
    bc_idx = np.where(is_boundary)[0]

    xy_all = torch.tensor(xy_grid, dtype=torch.float32, device=device)
    xy_int = xy_all[interior_idx].contiguous()
    xy_bc = xy_all[bc_idx].contiguous()

    # Body forces at interior points (used inside the residual).
    fx_int, fy_int = elasticity_body_forces(xy_int[:, 0:1], xy_int[:, 1:2])

    # BC targets at boundary nodes.
    ux_ex, uy_ex = elasticity_exact(xy_bc[:, 0:1], xy_bc[:, 1:2])
    bc_target = torch.cat([ux_ex, uy_ex], dim=1)  # (N_bc, 2)

    return {
        'xy_all': xy_all, 'xy_int': xy_int, 'xy_bc': xy_bc,
        'interior_idx': interior_idx, 'bc_idx': bc_idx,
        'bc_target': bc_target,
        'fx_int': fx_int, 'fy_int': fy_int,
        'dx': dx, 'dy': dy,
        'N_grid': N_grid, 'N_int': len(interior_idx), 'N_bc': len(bc_idx),
        'lam_e': float(lam_e), 'mu_e': float(mu_e),
    }


def pde_residuals_canpinn_elas(model, xy_int, dx, dy, fx_int, fy_int,
                               lam=None, mu=None):
    """CAN-PINN PDE residuals for 2D linear elasticity at the given interior pts.

    Implements can(cd) for second-order PDE: plain central FD for the diagonal
    Laplacian terms (d2/dx2, d2/dy2), and a central-difference of the AD-gradient
    along the orthogonal axis for the cross derivative. This mirrors the
    cd-pressure construction (paper eq. 12/13) but applied to displacement
    derivatives — no upwind / convection branch, since elasticity has no
    advective term (Phase 1 spec §8.2).

    Stencil: C, E, W, N, S (5 points; EE/WW/NN/SS not needed for this PDE).
    AD gradients are taken at all 5 to feed the Taylor /8 correction in the
    cross-derivative term.
    Returns (eq_x, eq_y), each shape (N_int, 1).
    """
    if lam is None:
        lam = lam_e
    if mu is None:
        mu = mu_e

    N_int = xy_int.shape[0]
    device = xy_int.device
    dtype = xy_int.dtype

    # 5-point stencil — reuse the canonical 9-point offsets and slice the first
    # 5 (C, E, W, N, S) so callers and tooling can rely on a single offset
    # generator. The slicing is the only place the elasticity scheme deviates
    # from the cavity / Kovasznay 9-point evaluation.
    offs9 = _canpinn_stencil_offsets(dx, dy, device, dtype)        # (9, 2)
    offs = offs9[:5]                                                # (5, 2)
    xy_stencil = (xy_int.unsqueeze(0) + offs.unsqueeze(1))          # (5, N_int, 2)
    xy_stencil = xy_stencil.reshape(5 * N_int, 2)
    xy_stencil = xy_stencil.detach().requires_grad_(True)

    pred = model(xy_stencil)                                        # (5*N_int, 2)
    ux_all = pred[:, 0:1]
    uy_all = pred[:, 1:2]

    grad_ux = torch.autograd.grad(ux_all, xy_stencil,
                                  grad_outputs=torch.ones_like(ux_all),
                                  create_graph=True, retain_graph=True)[0]
    grad_uy = torch.autograd.grad(uy_all, xy_stencil,
                                  grad_outputs=torch.ones_like(uy_all),
                                  create_graph=True, retain_graph=True)[0]

    ux_s = ux_all.reshape(5, N_int, 1)
    uy_s = uy_all.reshape(5, N_int, 1)
    ux_x_s = grad_ux[:, 0:1].reshape(5, N_int, 1)
    ux_y_s = grad_ux[:, 1:2].reshape(5, N_int, 1)
    uy_x_s = grad_uy[:, 0:1].reshape(5, N_int, 1)
    uy_y_s = grad_uy[:, 1:2].reshape(5, N_int, 1)

    ux_C, ux_E, ux_W, ux_N, ux_S = (ux_s[i] for i in range(5))
    uy_C, uy_E, uy_W, uy_N, uy_S = (uy_s[i] for i in range(5))

    # AD gradients at C / E / W / N / S — used by the can(cd) cross-derivative.
    ux_y_C = ux_y_s[0]; ux_y_E = ux_y_s[1]; ux_y_W = ux_y_s[2]
    uy_x_C = uy_x_s[0]; uy_x_N = uy_x_s[3]; uy_x_S = uy_x_s[4]
    # ...also expose the raw AD second-order taylor terms used in /8 corrections
    ux_x_C = ux_x_s[0]; ux_x_E = ux_x_s[1]; ux_x_W = ux_x_s[2]
    uy_y_C = uy_y_s[0]; uy_y_N = uy_y_s[3]; uy_y_S = uy_y_s[4]

    # Diagonal Laplacian terms — plain 2nd-order central FD (no AD coupling).
    d2ux_dx2 = (ux_E - 2.0 * ux_C + ux_W) / (dx * dx)
    d2ux_dy2 = (ux_N - 2.0 * ux_C + ux_S) / (dy * dy)
    d2uy_dx2 = (uy_E - 2.0 * uy_C + uy_W) / (dx * dx)
    d2uy_dy2 = (uy_N - 2.0 * uy_C + uy_S) / (dy * dy)

    # Cross derivative d^2 uy / (dx dy) — can(cd) form: central difference of
    # the AD gradient `duy/dx` along y, with /8 Taylor correction in y. This
    # mirrors the cd-pressure construction in `pde_residuals_canpinn_cavity`
    # (paper eq. 12) — half-face values weight the AD second derivative onto
    # the interior point.
    eighth_dx = dx / 8.0
    eighth_dy = dy / 8.0
    # face-averaged uy_x along y, with Taylor correction:
    uy_x_n = 0.5 * (uy_x_C + uy_x_N) - (uy_y_N - uy_y_C) * eighth_dy
    uy_x_s_face = 0.5 * (uy_x_S + uy_x_C) - (uy_y_C - uy_y_S) * eighth_dy
    d2uy_dxdy = (uy_x_n - uy_x_s_face) / dy

    # Cross derivative d^2 ux / (dx dy) — can(cd) form: central difference of
    # the AD gradient `dux/dy` along x, with /8 Taylor correction in x.
    ux_y_e = 0.5 * (ux_y_C + ux_y_E) - (ux_x_E - ux_x_C) * eighth_dx
    ux_y_w = 0.5 * (ux_y_W + ux_y_C) - (ux_x_C - ux_x_W) * eighth_dx
    d2ux_dxdy = (ux_y_e - ux_y_w) / dx

    # Navier-Cauchy equilibrium with body forces.
    eq_x = ((lam + 2.0 * mu) * d2ux_dx2 + mu * d2ux_dy2
            + (lam + mu) * d2uy_dxdy + fx_int)
    eq_y = (mu * d2uy_dx2 + (lam + 2.0 * mu) * d2uy_dy2
            + (lam + mu) * d2ux_dxdy + fy_int)

    return eq_x, eq_y


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

    Standard incompressible NS with constant viscosity nu = 1/Re.
    nu is read from g['nu_kov'] if present (Program-B parametric family F2),
    otherwise falls back to the module-level nu_kov (Re=40).
    """
    nu = g['nu_kov'] if 'nu_kov' in g else nu_kov
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    du_dx = g['Dx'] @ u;  du_dy = g['Dy'] @ u
    dv_dx = g['Dx'] @ v;  dv_dy = g['Dy'] @ v
    dp_dx = g['Dx'] @ p;  dp_dy = g['Dy'] @ p

    # Second derivatives for viscous term (constant viscosity)
    d2u_dx2 = g['Dx'] @ du_dx;  d2u_dy2 = g['Dy'] @ du_dy
    d2v_dx2 = g['Dx'] @ dv_dx;  d2v_dy2 = g['Dy'] @ dv_dy

    continuity = du_dx + dv_dy
    mom_u = u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)

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
        'nu_kov': float(nu_kov),  # Program-B F2 parametric threading
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
    model_dtype = next(model.parameters()).dtype
    xy_t = torch.tensor(xy_eval, dtype=model_dtype, device=device, requires_grad=True)

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
        # Best-model tracking: save checkpoint at lowest PDE RMS
        self.best_pde_rms = float('inf')
        self.best_epoch = None
        self.best_state_dict = None

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

        # Track best model by PDE RMS
        pde_rms = metrics['pde_rms']
        if not math.isnan(pde_rms) and pde_rms < self.best_pde_rms:
            self.best_pde_rms = pde_rms
            self.best_epoch = epoch + 1
            self.best_state_dict = copy.deepcopy(base_model.state_dict())

        row = dict(self.metadata)
        row['epoch'] = epoch + 1
        row['train_loss'] = round(train_loss, 8) if not math.isnan(train_loss) else 'NaN'
        row['pde_rms'] = round(pde_rms, 6)
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


# --- Method: chebyshev-pinn ---
def train_chebyshev_pinn(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", grid_data=None, tracker=None):
    """Chebyshev-spectral PINN: dense Chebyshev differentiation matrices, autograd backward.
    (Previously misnamed `train_dtpinn` — the actual DT-PINN method of Sharma & Shankar 2022
    is a meshless RBF-FD method, now in `train_dtpinn` / `train_dtpinn_kovasznay` /
    `train_dtpinn_elasticity`. Preserved here as a separate baseline.)
    """
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


# --- Method: can-pinn-faithful (cavity) ---
def train_can_pinn_faithful(seed, device, n_epochs, lr, technique, grid_size,
                            model_name="mlp", tracker=None):
    """CAN-PINN faithful PyTorch port (cavity NS+Smagorinsky harness drop-in).

    Implements Chiu et al. 2022 (CMAME 395, 114909) coupled automatic-numerical
    differentiation. The PDE residual is computed by:
      - can(uw2) upwind for convection (notebook eq. 8/9; /8 dispersion term
        commented out in the upstream demo, MATCHED here),
      - can(cd) central-difference-with-AD for pressure gradient (eq. 12/13;
        /8 term INCLUDED, matching the upstream demo),
      - plain 2nd-order central FD for the viscous Laplacian.
    Smagorinsky is retained for parity with the harness's autodiff baseline
    (option b1 in the Phase-1 spec §7.1): nu_eff is computed via AD at C and
    used as a local-constant in the FD Laplacian. The paper-faithful Re=400
    plain-NS variant lives in scripts/can_pinn_paper_validation.py — this
    function is the harness drop-in for cavity Re=1000+Smag.

    Mirrors the structural pattern of train_chebyshev_pinn (cavity) so the
    tracker / CSV / dispatch wiring is byte-equivalent.
    """
    g = build_canpinn_grid_data(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = torch.amp.GradScaler('cuda') if technique == "amp" else None
    use_amp = technique == "amp"

    # Pre-extract the constants the residual function needs each iteration.
    xy_int = g['xy_int']
    Cs_d_sq_int = g['Cs_d_sq_int']
    dx = g['dx']
    dy = g['dy']
    nu_lam = g['nu_lam']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                continuity, mom_u, mom_v = pde_residuals_canpinn_cavity(
                    compiled_model, xy_int, dx, dy,
                    Cs_d_sq_int=Cs_d_sq_int, nu_lam=nu_lam,
                    use_smagorinsky=True)
                loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                           mse(mom_u, torch.zeros_like(mom_u)) + \
                           mse(mom_v, torch.zeros_like(mom_v))
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
            continuity, mom_u, mom_v = pde_residuals_canpinn_cavity(
                compiled_model, xy_int, dx, dy,
                Cs_d_sq_int=Cs_d_sq_int, nu_lam=nu_lam,
                use_smagorinsky=True)
            loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))
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


# --- Method: can-pinn-faithful (Kovasznay) ---
def train_can_pinn_faithful_kovasznay(seed, device, n_epochs, lr, technique, grid_size,
                                       model_name="mlp", tracker=None):
    """CAN-PINN faithful PyTorch port for Kovasznay flow (Phase 5 extension).

    Steady incompressible NS at Re=40 on [-0.5, 1.0] x [-0.5, 1.5] with the
    Kovasznay closed-form solution as Dirichlet BCs. Uses the can(uw2-conv,
    cd-p) scheme from Chiu et al. 2022 — convection terms are upwind-coupled,
    pressure gradient is cd-coupled with /8 Taylor correction, and the viscous
    Laplacian is plain 2nd-order central FD. Constant viscosity (no
    Smagorinsky). Mirrors the structural pattern of `train_chebyshev_pinn_kovasznay`.
    """
    g = build_canpinn_grid_data_kovasznay(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = torch.amp.GradScaler('cuda') if technique == "amp" else None
    use_amp = technique == "amp"

    xy_int = g['xy_int']
    xy_bc = g['xy_bc']
    bc_target = g['bc_target']
    xy_center = g['xy_center']
    p_center_exact = g['p_center_exact']
    dx = g['dx']
    dy = g['dy']
    nu = g['nu_kov']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                continuity, mom_u, mom_v = pde_residuals_canpinn_kov(
                    compiled_model, xy_int, dx, dy, nu=nu)
                loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                           mse(mom_u, torch.zeros_like(mom_u)) + \
                           mse(mom_v, torch.zeros_like(mom_v))
                pred_bc = compiled_model(xy_bc)
                loss_bc = mse(pred_bc, bc_target)
                pred_c = compiled_model(xy_center)
                p_target = torch.tensor([[p_center_exact]], dtype=torch.float32, device=device)
                loss_p = mse(pred_c[:, 2:3], p_target)
                loss = loss_pde + loss_bc + loss_p + model_reg_loss(model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            continuity, mom_u, mom_v = pde_residuals_canpinn_kov(
                compiled_model, xy_int, dx, dy, nu=nu)
            loss_pde = mse(continuity, torch.zeros_like(continuity)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))
            pred_bc = compiled_model(xy_bc)
            loss_bc = mse(pred_bc, bc_target)
            pred_c = compiled_model(xy_center)
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
    return model, train_time, final_loss


# --- Method: can-pinn-faithful (Elasticity) ---
def train_can_pinn_faithful_elasticity(seed, device, n_epochs, lr, technique, grid_size,
                                        model_name="mlp", tracker=None):
    """CAN-PINN faithful PyTorch port for 2D linear elasticity (Phase 5 extension).

    Manufactured-solution Navier-Cauchy on [0,1]^2 with Dirichlet BCs from the
    exact ux/uy. Implements can(cd) for the second-order PDE: plain central FD
    for the diagonal Laplacians and AD-coupled central difference for the
    cross-derivative gradient-of-divergence terms (Phase 1 spec §8.2). No
    convection term — there is nothing to upwind. Mirrors the structural
    pattern of `train_chebyshev_pinn_elasticity`, including the cosine-anneal
    LR schedule for late-epoch stability.
    """
    g = build_canpinn_grid_data_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    scaler = torch.amp.GradScaler('cuda') if technique == "amp" else None
    use_amp = technique == "amp"

    xy_int = g['xy_int']
    xy_bc = g['xy_bc']
    bc_target = g['bc_target']
    fx_int = g['fx_int']
    fy_int = g['fy_int']
    dx = g['dx']
    dy = g['dy']
    lam = g['lam_e']
    mu = g['mu_e']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    final_loss = float('nan')

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                eq_x, eq_y = pde_residuals_canpinn_elas(
                    compiled_model, xy_int, dx, dy, fx_int, fy_int, lam=lam, mu=mu)
                loss_pde = mse(eq_x, torch.zeros_like(eq_x)) + \
                           mse(eq_y, torch.zeros_like(eq_y))
                pred_bc = compiled_model(xy_bc)
                loss_bc = mse(pred_bc, bc_target)
                loss = loss_pde + loss_bc + model_reg_loss(model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            eq_x, eq_y = pde_residuals_canpinn_elas(
                compiled_model, xy_int, dx, dy, fx_int, fy_int, lam=lam, mu=mu)
            loss_pde = mse(eq_x, torch.zeros_like(eq_x)) + \
                       mse(eq_y, torch.zeros_like(eq_y))
            pred_bc = compiled_model(xy_bc)
            loss_bc = mse(pred_bc, bc_target)
            loss = loss_pde + loss_bc + model_reg_loss(model)
            loss.backward()
            optimizer.step()
        scheduler.step()

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
# Faithful DT-PINN (Sharma & Shankar 2022, arXiv:2205.09332)
#
# Meshless RBF-FD with PHS + Legendre polynomial basis, fp64 everywhere,
# L-BFGS with strong-Wolfe line search, 5K-epoch budget.
#
# Operator construction lives in src/rbf_fd_operators.py — this section just
# wires those operators to our PDE residuals (cavity NS+Smag, Kovasznay,
# elasticity).  Boundary condition formulation differs from the paper (the
# paper uses Robin BCs natural for the disk Poisson problem; our test PDEs are
# Dirichlet, applied via direct boundary-residual evaluation as in every
# other baseline in this file).  Method (RBF-FD + fp64 + L-BFGS) is faithful;
# BC type is matched to the test problem.
# =============================================================================


def _dtpinn_default_num_nodes(grid_size: int) -> int:
    """Pick a default scattered-node count comparable to the Chebyshev N×N
    tensor grid. The Chebyshev baseline uses N=50 (cavity) or N=30 (kov, elas)
    by default → 2500 / 900 nodes. Match those orders of magnitude.
    """
    return max(grid_size * grid_size, 64)


def _build_dtpinn_grid(problem, num_nodes, p, device, dtype, seed):
    """Generate scattered nodes + sparse RBF-FD operators for `problem`.

    Returns dict g containing torch fp64 tensors and sparse operators.

    For cavity / Kovasznay we build square (Nf, Nf) Dx, Dy so that the
    NS viscous-flux divergence ∂_x(ν_eff · ∂u/∂x) can be chained through the
    same operator. The Lap operator (Ni+Nb, Nf) and the elasticity Dxx/Dyy/Dxy
    operators stay rectangular (no chain needed there).
    """
    from src.rbf_fd_operators import gen_rectangle_nodes, build_operators, to_torch_sparse

    if problem == 'cavity' or problem == 'elasticity':
        xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    elif problem == 'kovasznay':
        xmin, xmax, ymin, ymax = -0.5, 1.0, -0.5, 1.5
    else:
        raise ValueError(f"unknown problem {problem!r}")

    Xi_np, Xb_np, normals_np, h = gen_rectangle_nodes(
        xmin, xmax, ymin, ymax, num_nodes, seed=seed
    )

    if problem == 'elasticity':
        # Lamé operator uses individual second derivatives, no chain → rectangular.
        ops_intbd = build_operators(
            Xi_np, Xb_np, normals_np, p=p,
            derivs=((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)),
            centres_kind='int_bd',
        )
        ops_full = None
    else:
        # NS+Smag (cavity) and Kovasznay need Dx, Dy at all Nf points to chain
        # the viscous-flux divergence; Lap stays rectangular for the no-chain
        # use case (e.g., logging / future Stokes-style residuals).
        ops_full = build_operators(
            Xi_np, Xb_np, normals_np, p=p,
            derivs=((1, 0), (0, 1)),
            centres_kind='full',
        )
        ops_intbd = build_operators(
            Xi_np, Xb_np, normals_np, p=p,
            derivs=('lap',),
            centres_kind='int_bd',
        )

    md = ops_intbd['__metadata__']
    Ni, Nb, Ng = md['Ni'], md['Nb'], md['Ng']
    Xf_np = md['Xf']
    Nf = Xf_np.shape[0]

    Xf = torch.tensor(Xf_np, dtype=dtype, device=device)
    Xb_t = torch.tensor(Xb_np, dtype=dtype, device=device)
    normals_t = torch.tensor(normals_np, dtype=dtype, device=device)

    g = {
        'Xf': Xf,
        'Xb': Xb_t,
        'normals': normals_t,
        'Ni': Ni, 'Nb': Nb, 'Ng': Ng, 'Nf': Nf,
        'h': h, 'p': p, 'problem': problem,
    }

    if problem == 'elasticity':
        g['Dx'] = to_torch_sparse(ops_intbd[(1, 0)], dtype=dtype, device=device)
        g['Dy'] = to_torch_sparse(ops_intbd[(0, 1)], dtype=dtype, device=device)
        g['Dxx'] = to_torch_sparse(ops_intbd[(2, 0)], dtype=dtype, device=device)
        g['Dyy'] = to_torch_sparse(ops_intbd[(0, 2)], dtype=dtype, device=device)
        g['Dxy'] = to_torch_sparse(ops_intbd[(1, 1)], dtype=dtype, device=device)
    else:
        g['Dx_full'] = to_torch_sparse(ops_full[(1, 0)], dtype=dtype, device=device)
        g['Dy_full'] = to_torch_sparse(ops_full[(0, 1)], dtype=dtype, device=device)
        g['Lap'] = to_torch_sparse(ops_intbd['lap'], dtype=dtype, device=device)

    if problem == 'cavity':
        eps = 1e-9
        is_lid = (Xb_np[:, 1] >= 1 - eps)
        lid_idx = np.where(is_lid)[0]
        wall_idx = np.where(~is_lid)[0]
        g['lid_idx'] = torch.from_numpy(lid_idx).to(device=device, dtype=torch.long)
        g['wall_idx'] = torch.from_numpy(wall_idx).to(device=device, dtype=torch.long)
        # Wall distance and Smagorinsky factor — evaluated at ALL Nf points
        # so they can multiply the (Nf,)-shape inner derivatives in the
        # viscous-flux chain. Clamp ≥ 0 to keep ghost points (slightly outside
        # the unit square) physically valid in the squared term.
        x_t, y_t = Xf[:, 0:1], Xf[:, 1:2]
        d_wall = torch.clamp_min(
            torch.min(torch.min(x_t, 1.0 - x_t), torch.min(y_t, 1.0 - y_t)),
            0.0,
        )
        g['Cs_d_sq_full'] = (Cs * d_wall) ** 2
        g['xy_center'] = torch.tensor([[0.5, 0.5]], dtype=dtype, device=device)
        g['nu_lam'] = float(nu_laminar)

    elif problem == 'kovasznay':
        u_b, v_b, p_b = kovasznay_exact(Xb_t[:, 0:1], Xb_t[:, 1:2])
        g['bc_target'] = torch.cat([u_b, v_b, p_b], dim=1)
        g['xy_center'] = torch.tensor([[0.0, 0.0]], dtype=dtype, device=device)
        _, _, p_c = kovasznay_exact(g['xy_center'][:, 0:1], g['xy_center'][:, 1:2])
        g['p_center_exact'] = float(p_c.item())
        g['nu_kov'] = float(nu_kov)

    elif problem == 'elasticity':
        ux_b, uy_b = elasticity_exact(Xb_t[:, 0:1], Xb_t[:, 1:2])
        g['bc_target'] = torch.cat([ux_b, uy_b], dim=1)
        # Body forces at the (Ni+Nb) PDE-residual points
        x_int_bd = Xf[: Ni + Nb, 0:1]
        y_int_bd = Xf[: Ni + Nb, 1:2]
        fx, fy = elasticity_body_forces(x_int_bd, y_int_bd)
        g['fx'] = fx
        g['fy'] = fy
        g['lam_e'] = float(lam_e)
        g['mu_e'] = float(mu_e)

    return g


def _compute_pde_dtpinn_cavity(pred_full, g):
    """NS+Smagorinsky cavity residual via square (Nf, Nf) RBF-FD operators.

    pred_full: (Nf, 3) network output at all interior+boundary+ghost points.
    Returns continuity, mom_u, mom_v of shape (Ni+Nb, 1) each — residuals at
    the (Ni+Nb) interior+boundary stencil centres (paper convention).
    """
    Dx, Dy = g['Dx_full'], g['Dy_full']
    Cs_d_sq = g['Cs_d_sq_full']
    nu_lam = g['nu_lam']
    nibd = g['Ni'] + g['Nb']

    u, v, p = pred_full[:, 0:1], pred_full[:, 1:2], pred_full[:, 2:3]
    du_dx = torch.sparse.mm(Dx, u); du_dy = torch.sparse.mm(Dy, u)
    dv_dx = torch.sparse.mm(Dx, v); dv_dy = torch.sparse.mm(Dy, v)
    Sxx, Syy = du_dx, dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2) + 1e-12)
    nu_eff = nu_lam + Cs_d_sq * S_mag  # (Nf, 1) at all points incl. ghost

    continuity = (du_dx + dv_dy)[:nibd]
    u_int = u[:nibd]; v_int = v[:nibd]
    u_conv = u_int * du_dx[:nibd] + v_int * du_dy[:nibd]
    v_conv = u_int * dv_dx[:nibd] + v_int * dv_dy[:nibd]
    dp_dx = torch.sparse.mm(Dx, p)[:nibd]
    dp_dy = torch.sparse.mm(Dy, p)[:nibd]
    visc_u = (
        torch.sparse.mm(Dx, nu_eff * du_dx)
        + torch.sparse.mm(Dy, nu_eff * du_dy)
    )[:nibd]
    visc_v = (
        torch.sparse.mm(Dx, nu_eff * dv_dx)
        + torch.sparse.mm(Dy, nu_eff * dv_dy)
    )[:nibd]
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    return continuity, mom_u, mom_v


def _compute_pde_dtpinn_kovasznay(pred_full, g):
    """Steady Kovasznay residual via square (Nf, Nf) RBF-FD operators."""
    Dx, Dy = g['Dx_full'], g['Dy_full']
    nu = g['nu_kov']
    nibd = g['Ni'] + g['Nb']

    u, v, p = pred_full[:, 0:1], pred_full[:, 1:2], pred_full[:, 2:3]
    du_dx = torch.sparse.mm(Dx, u); du_dy = torch.sparse.mm(Dy, u)
    dv_dx = torch.sparse.mm(Dx, v); dv_dy = torch.sparse.mm(Dy, v)
    continuity = (du_dx + dv_dy)[:nibd]
    u_int = u[:nibd]; v_int = v[:nibd]
    u_conv = u_int * du_dx[:nibd] + v_int * du_dy[:nibd]
    v_conv = u_int * dv_dx[:nibd] + v_int * dv_dy[:nibd]
    dp_dx = torch.sparse.mm(Dx, p)[:nibd]
    dp_dy = torch.sparse.mm(Dy, p)[:nibd]
    # constant ν → div(ν ∇u) = ν ∇²u, computed via Dx∘Dx + Dy∘Dy
    visc_u = nu * (
        torch.sparse.mm(Dx, du_dx) + torch.sparse.mm(Dy, du_dy)
    )[:nibd]
    visc_v = nu * (
        torch.sparse.mm(Dx, dv_dx) + torch.sparse.mm(Dy, dv_dy)
    )[:nibd]
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v
    return continuity, mom_u, mom_v


def _compute_pde_dtpinn_elasticity(pred_full, g):
    """Lamé/Navier-Cauchy residual via rectangular (Ni+Nb, Nf) RBF-FD ops.

    No chain needed — individual Dxx, Dyy, Dxy operators give the second
    derivatives directly.
    """
    lam = g['lam_e']; mu = g['mu_e']
    ux, uy = pred_full[:, 0:1], pred_full[:, 1:2]
    d2ux_dx2 = torch.sparse.mm(g['Dxx'], ux)
    d2ux_dy2 = torch.sparse.mm(g['Dyy'], ux)
    d2ux_dxdy = torch.sparse.mm(g['Dxy'], ux)
    d2uy_dx2 = torch.sparse.mm(g['Dxx'], uy)
    d2uy_dy2 = torch.sparse.mm(g['Dyy'], uy)
    d2uy_dxdy = torch.sparse.mm(g['Dxy'], uy)
    eq_x = ((lam + 2 * mu) * d2ux_dx2 + mu * d2ux_dy2
            + (lam + mu) * d2uy_dxdy + g['fx'])
    eq_y = (mu * d2uy_dx2 + (lam + 2 * mu) * d2uy_dy2
            + (lam + mu) * d2ux_dxdy + g['fy'])
    return eq_x, eq_y


def _dtpinn_train_loop(
    problem,
    seed,
    device,
    n_epochs,
    lr,
    grid_size,
    model_name,
    *,
    dtype,
    rbf_fd_order,
    num_nodes,
    optimizer_kind,
    tracker,
    match_protocol=False,
):
    """Common training loop shared by the three problem-specific train_dtpinn_*.

    Implements the paper-faithful auto-restart wrapper from
    temp/dt-pinn/src/dtpinn_cupy_fp64.py:494-503 — if the L-BFGS at the current
    lr fails to make progress, halve the lr and retry from the same seed. The
    paper checks `loss > 500` after i > 30; we use a relative two-checkpoint
    criterion (see ABORT_CHECKS below) because the absolute threshold 500 is
    paper-specific to its disk-Poisson scale (init ≈ 100) and misfires on PDEs
    with different residual magnitudes (e.g., elasticity init ≈ 2000 with the
    Q_e=4 manufactured solution).

    When ``match_protocol`` is True, the auto-restart wrapper is disabled so
    the run is apples-to-apples with the other matched-protocol comparators
    (autodiff / sage / chebyshev-pinn / can-pinn-faithful / sk-pinn-matched),
    none of which use a halve-lr-and-retry harness. RBF-FD operators are kept
    (the method's distinguishing feature). In matched mode the caller passes
    optimizer_kind='adam' / dtype=fp32 / lr=1e-3 / n_epochs=30000 to mirror
    the rest of the matched-protocol baselines.
    """
    if num_nodes is None:
        num_nodes = _dtpinn_default_num_nodes(grid_size)
    torch.manual_seed(seed)
    np.random.seed(seed)

    g = _build_dtpinn_grid(problem, num_nodes, rbf_fd_order, device, dtype, seed)
    Ni, Nb, Ng, Nf = g['Ni'], g['Nb'], g['Ng'], g['Nf']
    nibd = Ni + Nb
    print(f"  RBF-FD grid: Ni={Ni}, Nb={Nb}, Ng={Ng}, Nf={Nf}, h={g['h']:.4f}, p={rbf_fd_order}")

    if problem == 'elasticity':
        out_dim = 2
    else:
        out_dim = 3

    Xf = g['Xf']

    def compute_loss(model):
        pred_full = model(Xf)
        if problem == 'cavity':
            cont, mom_u, mom_v = _compute_pde_dtpinn_cavity(pred_full, g)
            zero = torch.zeros_like(cont)
            loss_pde = mse(cont, zero) + mse(mom_u, zero) + mse(mom_v, zero)
            # boundary: lid u=1, v=0; walls u=0, v=0
            pred_b = pred_full[Ni:Ni + Nb]  # boundary slice of Xf
            pred_lid = pred_b[g['lid_idx']]
            pred_wall = pred_b[g['wall_idx']]
            loss_lid = mse(pred_lid[:, 0:1], torch.ones_like(pred_lid[:, 0:1])) + \
                       mse(pred_lid[:, 1:2], torch.zeros_like(pred_lid[:, 1:2]))
            loss_wall = mse(pred_wall[:, 0:1], torch.zeros_like(pred_wall[:, 0:1])) + \
                        mse(pred_wall[:, 1:2], torch.zeros_like(pred_wall[:, 1:2]))
            pred_c = model(g['xy_center'])
            loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
            loss = loss_pde + loss_lid + loss_wall + loss_p + model_reg_loss(model)
        elif problem == 'kovasznay':
            cont, mom_u, mom_v = _compute_pde_dtpinn_kovasznay(pred_full, g)
            zero = torch.zeros_like(cont)
            loss_pde = mse(cont, zero) + mse(mom_u, zero) + mse(mom_v, zero)
            pred_b = pred_full[Ni:Ni + Nb]
            loss_bc = mse(pred_b, g['bc_target'])
            pred_c = model(g['xy_center'])
            p_target = torch.tensor(
                [[g['p_center_exact']]], dtype=dtype, device=device
            )
            loss_p = mse(pred_c[:, 2:3], p_target)
            loss = loss_pde + loss_bc + loss_p + model_reg_loss(model)
        elif problem == 'elasticity':
            eq_x, eq_y = _compute_pde_dtpinn_elasticity(pred_full, g)
            zero = torch.zeros_like(eq_x)
            loss_pde = mse(eq_x, zero) + mse(eq_y, zero)
            pred_b = pred_full[Ni:Ni + Nb]
            loss_bc = mse(pred_b, g['bc_target'])
            loss = loss_pde + loss_bc + model_reg_loss(model)
        else:
            raise ValueError(problem)
        return loss

    # Paper-faithful auto-restart: if L-BFGS overshoots and the run gets stuck
    # at initial loss OR converges to a clearly suboptimal basin, halve lr and
    # re-init the network with the same seed. Disabled for adam (Adam is robust
    # to lr; the check was only ever a paper artefact for raw L-BFGS without
    # line search). Two checkpoints because a single one cannot distinguish
    # stuck-at-init from slowly-converging-to-bad-minimum on problems with
    # different residual scales:
    #   - epoch 50 / ratio 0.5: catches "stuck at init" (e.g., elas lr=0.04
    #     stays at loss 2058 forever; aborts immediately).
    #   - epoch 200 / ratio 0.01: catches "converging to a worse minimum
    #     than lr/2 would" (e.g., elas lr=0.02 reaches loss ~30 at ep 200 vs
    #     lr=0.005 which reaches loss ~0.27; the former is 100× worse final
    #     accuracy and the auto-restart correctly halves lr to find a tighter
    #     basin). Note: for problems where lr=0.04 already works well
    #     (cavity, kovasznay), training reaches <<1% of initial loss by ep 200
    #     so this check is a no-op.
    MAX_RETRIES = 6
    ABORT_CHECKS = [
        (50, 0.5),    # stuck-at-init divergence
        (200, 0.01),  # slow-converging suboptimal minimum
    ]

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    cur_lr = lr
    final_loss = float('nan')
    final_model = None

    retry_attempts = 0
    while True:
        # Re-init network and optimizer. torch.manual_seed makes the network
        # init reproducible across retries; combined with stationary operators
        # this exactly matches the paper's retry semantics.
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make_model(model_name, output_dim=out_dim).to(device).to(dtype=dtype)

        if optimizer_kind == 'lbfgs':
            # Paper-faithful: temp/dt-pinn/src/dtpinn_cupy_fp64.py:117 builds
            # `optim.LBFGS(self.w.parameters(), lr=self.lr)` with PyTorch
            # defaults for line_search_fn (None), max_iter (20),
            # tolerance_grad (1e-7), tolerance_change (1e-9).
            optimizer = torch.optim.LBFGS(model.parameters(), lr=cur_lr)
        elif optimizer_kind == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr)
        else:
            raise ValueError(f"unknown optimizer {optimizer_kind!r}")

        # If retrying, reset the tracker so its CSV reflects only the
        # successful run. The TrainingTracker overwrites its CSV on
        # `_init_csv`, which is invoked when `_initialized` is False.
        if tracker is not None and retry_attempts > 0:
            tracker._initialized = False
            tracker.best_pde_rms = float('inf')
            tracker.best_epoch = None
            tracker.best_state_dict = None
            # Update the lr field so the tracker rows reflect the active lr
            # (otherwise tracker.metadata still says the original lr).
            tracker.metadata['lr'] = cur_lr

        initial_loss = float('nan')
        local_final_loss = float('nan')
        aborted = False

        for epoch in range(n_epochs):
            if optimizer_kind == 'lbfgs':
                def closure():
                    optimizer.zero_grad()
                    loss = compute_loss(model)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                local_final_loss = loss.item() if torch.is_tensor(loss) else float(loss)
            else:
                optimizer.zero_grad()
                loss = compute_loss(model)
                loss.backward()
                optimizer.step()
                local_final_loss = loss.item()

            if epoch == 0:
                initial_loss = local_final_loss

            if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}: loss={local_final_loss:.6e}")

            # Paper-faithful auto-restart check (lbfgs only; Adam doesn't need
            # it). If still stuck near / above initial loss at any checkpoint,
            # abort and retry with halved lr. Also disabled in match_protocol
            # mode (defensive — matched mode is Adam in practice, but this
            # makes the apples-to-apples intent explicit).
            triggered = False
            if (optimizer_kind == 'lbfgs'
                    and retry_attempts < MAX_RETRIES
                    and not match_protocol):
                for check_epoch, max_ratio in ABORT_CHECKS:
                    if epoch == check_epoch:
                        threshold = max_ratio * initial_loss
                        if math.isnan(local_final_loss) or local_final_loss > threshold:
                            print(
                                f"  No progress at epoch {epoch+1}: loss={local_final_loss:.3e}, "
                                f"initial={initial_loss:.3e}, threshold={threshold:.3e} "
                                f"({max_ratio:g}× initial); halving lr to {cur_lr/2.0} and retrying"
                            )
                            cur_lr = cur_lr / 2.0
                            aborted = True
                            triggered = True
                        break
            if triggered:
                break

            if tracker is not None:
                tracker.step(epoch, local_final_loss, model)

        if not aborted:
            final_model = model
            final_loss = local_final_loss
            break

        retry_attempts += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    if retry_attempts > 0:
        print(f"  Auto-restart succeeded after {retry_attempts} retries (final lr={cur_lr})")
    return final_model, train_time, final_loss


def train_dtpinn(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp",
                  grid_data=None, tracker=None, *,
                  dtype='fp64', rbf_fd_order=4, num_nodes=None,
                  optimizer_kind='lbfgs', match_protocol=False):
    """Faithful DT-PINN (Sharma & Shankar 2022) for the lid-driven cavity.

    RBF-FD spatial derivatives + fp64 + L-BFGS with strong-Wolfe line search.
    `grid_data` is ignored (kept for signature compatibility with the
    previous Chebyshev variant); operators are always built from scratch.

    When ``match_protocol`` is True, the auto-restart wrapper is disabled and
    the caller is expected to pass Adam-only / fp32 / lr=1e-3 / n_epochs=30000
    so the row is apples-to-apples with the other matched-protocol baselines.
    """
    torch_dtype = torch.float64 if dtype == 'fp64' else torch.float32
    return _dtpinn_train_loop(
        'cavity', seed, device, n_epochs, lr, grid_size, model_name,
        dtype=torch_dtype, rbf_fd_order=rbf_fd_order, num_nodes=num_nodes,
        optimizer_kind=optimizer_kind, tracker=tracker,
        match_protocol=match_protocol,
    )


def train_dtpinn_kovasznay(seed, device, n_epochs, lr, technique, grid_size,
                            model_name="mlp", tracker=None, *,
                            dtype='fp64', rbf_fd_order=4, num_nodes=None,
                            optimizer_kind='lbfgs', match_protocol=False):
    """Faithful DT-PINN for steady Kovasznay flow.

    See ``train_dtpinn`` for the matched-protocol semantics.
    """
    torch_dtype = torch.float64 if dtype == 'fp64' else torch.float32
    return _dtpinn_train_loop(
        'kovasznay', seed, device, n_epochs, lr, grid_size, model_name,
        dtype=torch_dtype, rbf_fd_order=rbf_fd_order, num_nodes=num_nodes,
        optimizer_kind=optimizer_kind, tracker=tracker,
        match_protocol=match_protocol,
    )


def train_dtpinn_elasticity(seed, device, n_epochs, lr, technique, grid_size,
                             model_name="mlp", tracker=None, *,
                             dtype='fp64', rbf_fd_order=4, num_nodes=None,
                             optimizer_kind='lbfgs', match_protocol=False):
    """Faithful DT-PINN for 2D linear elasticity (manufactured solution).

    See ``train_dtpinn`` for the matched-protocol semantics.
    """
    torch_dtype = torch.float64 if dtype == 'fp64' else torch.float32
    return _dtpinn_train_loop(
        'elasticity', seed, device, n_epochs, lr, grid_size, model_name,
        dtype=torch_dtype, rbf_fd_order=rbf_fd_order, num_nodes=num_nodes,
        optimizer_kind=optimizer_kind, tracker=tracker,
        match_protocol=match_protocol,
    )


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
        _is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track or _is_last:
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
                loss_pde = (cont[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                loss_lid = mse(pred_l[:, 0:1], torch.ones_like(pred_l[:, 0:1])) + \
                           mse(pred_l[:, 1:2], torch.zeros_like(pred_l[:, 1:2]))
                loss_wall = mse(pred_w[:, 0:1], torch.zeros_like(pred_w[:, 0:1])) + \
                            mse(pred_w[:, 1:2], torch.zeros_like(pred_w[:, 1:2]))
                loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                final_loss = loss_pde.item() + loss_lid.item() + loss_wall.item() + loss_p.item() + reg_val
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: sage (Symbolic Analytical Gradient Engine) ---
def train_sage(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None, grid_data=None):
    """SAGE: auto-generated backward via symbolic VJP engine."""
    g = grid_data if grid_data is not None else build_grid_data(grid_size, device)

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
        _is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track or _is_last:
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
                loss_pde = (cont[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                loss_lid = mse(pred_l[:, 0:1], torch.ones_like(pred_l[:, 0:1])) + \
                           mse(pred_l[:, 1:2], torch.zeros_like(pred_l[:, 1:2]))
                loss_wall = mse(pred_w[:, 0:1], torch.zeros_like(pred_w[:, 0:1])) + \
                            mse(pred_w[:, 1:2], torch.zeros_like(pred_w[:, 1:2]))
                loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                final_loss = loss_pde.item() + loss_lid.item() + loss_wall.item() + loss_p.item() + reg_val
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
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
    'best_epoch',
    'status', 'device', 'gpu_name', 'pytorch_version',
]


def append_csv_row(csv_path, row_dict):
    """Append a single row to CSV with file locking for concurrent safety.

    Header decision is made inside the lock by checking the file's current
    size via fstat, so two parallel writers racing on a fresh CSV won't both
    write a header.
    """
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if os.fstat(f.fileno()).st_size == 0:
                writer.writeheader()
            writer.writerow(row_dict)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# --- Method: sk-pinn ---
def train_sk_pinn(seed, device, n_epochs, lr, grid_size, model_name, grid_data, tracker=None,
                  match_protocol=False):
    """SK-PINN: sparse RKPM differentiation matrices, autograd backward.

    Uses model-specific weight decay to prevent the model from learning high-
    frequency features that exceed the RKPM operator's resolution (O(h^2)
    algebraic convergence).  More expressive architectures (PirateNet, TSA-PINN)
    need stronger regularization to avoid overfitting the discretization.

    When ``match_protocol`` is True, weight decay is set to 0.0 so the run is
    apples-to-apples with the rest of the matched-protocol baselines
    (autodiff/sage/chebyshev-pinn/can-pinn-faithful).
    """
    g = grid_data

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)
    wd = 0.0 if match_protocol else _SK_PINN_WD.get(model_name, 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

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
def train_sage_kovasznay(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None, grid_data=None):
    """SAGE: auto-generated backward for Kovasznay flow."""
    g = grid_data if grid_data is not None else build_grid_data_kovasznay(grid_size, device)

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
            n_out = pred_bc.shape[1]  # 3 for Kovasznay (u, v, p); MSELoss averages over numel
            grad_bc = 2.0 * (pred_bc - g['bc_target']) / (N_bc * n_out)

            grad_center = torch.zeros(1, 3, device=device)
            grad_center[:, 2:3] = 2.0 * (pred_c[:, 2:3] - g['p_center_exact'])

            upstream = torch.cat([grad_pde, grad_bc, grad_center], dim=0)
        pred_batch.backward(gradient=upstream)
        reg = model_reg_loss(model)
        if isinstance(reg, torch.Tensor):
            reg.backward()
        optimizer.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        _is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track or _is_last:
            with torch.no_grad():
                c, mu, mv = compute_pde_kovasznay(pred_pde, g)
                ii = g['interior_idx']
                loss_pde = (c[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                loss_bc = mse(pred_bc, g['bc_target'])
                p_target = torch.tensor([[g['p_center_exact']]], dtype=torch.float32, device=pred_c.device)
                loss_p = mse(pred_c[:, 2:3], p_target)
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                final_loss = loss_pde.item() + loss_bc.item() + loss_p.item() + reg_val
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: chebyshev-pinn (Kovasznay) ---
def train_chebyshev_pinn_kovasznay(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Chebyshev-spectral PINN for Kovasznay: dense spectral matrices, autograd backward."""
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


def train_sk_pinn_kovasznay(seed, device, n_epochs, lr, grid_size, model_name, grid_data, tracker=None,
                            match_protocol=False):
    """SK-PINN for Kovasznay flow: sparse RKPM matrices, autograd backward.

    Uses model-specific weight decay (see _SK_PINN_WD) to prevent the model
    from learning beyond RKPM resolution.

    When ``match_protocol`` is True, weight decay is set to 0.0 so the run is
    apples-to-apples with the rest of the matched-protocol baselines.
    """
    g = grid_data

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)
    wd = 0.0 if match_protocol else _SK_PINN_WD.get(model_name, 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

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
    model_dtype = next(model.parameters()).dtype
    xy_t = torch.tensor(xy_eval, dtype=model_dtype, device=device, requires_grad=True)

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

    Lamé constants are read from g['lam_e'] / g['mu_e'] if present
    (Program-B parametric family F3), otherwise fall back to module-level
    lam_e / mu_e.
    """
    lam = g['lam_e'] if 'lam_e' in g else lam_e
    mu = g['mu_e'] if 'mu_e' in g else mu_e
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
    eq_x = ((lam + 2 * mu) * d2ux_dx2 + mu * d2ux_dy2
            + (lam + mu) * d2uy_dxdy + g['fx'])
    eq_y = (mu * d2uy_dx2 + (lam + 2 * mu) * d2uy_dy2
            + (lam + mu) * d2ux_dxdy + g['fy'])

    return eq_x, eq_y


def compute_pde_elasticity_sparse(pred, g):
    """Elasticity PDE residuals via sparse RKPM differentiation matrices."""
    lam = g['lam_e'] if 'lam_e' in g else lam_e
    mu = g['mu_e'] if 'mu_e' in g else mu_e
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

    eq_x = ((lam + 2 * mu) * d2ux_dx2 + mu * d2ux_dy2
            + (lam + mu) * d2uy_dxdy + g['fx'])
    eq_y = (mu * d2uy_dx2 + (lam + 2 * mu) * d2uy_dy2
            + (lam + mu) * d2ux_dxdy + g['fy'])

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
        'lam_e': float(lam_e),  # Program-B F3 parametric threading
        'mu_e': float(mu_e),
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
    """Lazily generate and cache backward function for Elasticity PDE.

    Note: lam_e and mu_e are TracedVar constants so the emitted backward
    reads them from g at runtime. Without them, the generated backward
    bakes in the module-level defaults (lam_e=1.0, mu_e=0.5) and breaks
    Program-B F3 parametric threading via _reparam_elasticity_grid_.
    """
    global _generated_elasticity_backward
    if _generated_elasticity_backward is None:
        from src.symbolic_vjp import trace_pde_forward, emit_backward
        tape = []
        outputs, inputs = trace_pde_forward(
            compute_pde_elasticity, None, tape, sparse=False,
            constants=['Dxx', 'Dyy', 'Dxy', 'fx', 'fy', 'lam_e', 'mu_e'],
            input_names=['ux', 'uy'])
        _, _generated_elasticity_backward = emit_backward(
            tape, list(outputs), ['deq_x', 'deq_y'], inputs, sparse=False,
            func_name='generated_elasticity_grad',
            input_names=['ux', 'uy'])
    return _generated_elasticity_backward


# --- Method: sage (Elasticity) ---
def train_sage_elasticity(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None, grid_data=None):
    """SAGE: auto-generated backward for elasticity."""
    g = grid_data if grid_data is not None else build_grid_data_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

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
        scheduler.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        _is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track or _is_last:
            with torch.no_grad():
                eq_x, eq_y = compute_pde_elasticity(pred_pde, g)
                ii = g['interior_idx']
                loss_pde = (eq_x[ii]**2).mean() + (eq_y[ii]**2).mean()
                loss_bc = mse(pred_bc, g['bc_target'])
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                final_loss = loss_pde.item() + loss_bc.item() + reg_val
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: stencil-adjoint ---
#
# Forward path is identical to SAGE (exact dense spectral operators, so the
# observed residual and loss are exact). The reverse path replaces every
# D^T matrix-vector the generated backward would apply with a compact
# band-limited stencil approximation derived from local polynomial
# interpolation on the same node set. Nonlinear chain-rule factors are
# re-evaluated at the current network state, so the adjoint remains
# state-dependent even though the linear operators inside it are
# approximate. Phase 4 candidate C2; see research_log/04_design.md.
_STENCIL_HALF_BANDWIDTH = 3  # window = 7 nodes, effective order 6

def train_stencil_adjoint(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Cavity stencil-adjoint: dense forward, compact-FD adjoint."""
    g = build_grid_data(grid_size, device)
    _inject_stencil_adjoint(g, 'cavity', half_bandwidth=_STENCIL_HALF_BANDWIDTH)
    return train_sage(seed, device, n_epochs, lr, technique, grid_size,
                      model_name=model_name, tracker=tracker, grid_data=g)


def train_stencil_adjoint_kovasznay(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Kovasznay stencil-adjoint: dense forward, compact-FD adjoint."""
    g = build_grid_data_kovasznay(grid_size, device)
    _inject_stencil_adjoint(g, 'kovasznay', half_bandwidth=_STENCIL_HALF_BANDWIDTH)
    return train_sage_kovasznay(seed, device, n_epochs, lr, technique, grid_size,
                                model_name=model_name, tracker=tracker, grid_data=g)


def train_stencil_adjoint_elasticity(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Elasticity stencil-adjoint: dense forward, compact-FD adjoint."""
    g = build_grid_data_elasticity(grid_size, device)
    _inject_stencil_adjoint(g, 'elasticity', half_bandwidth=_STENCIL_HALF_BANDWIDTH)
    return train_sage_elasticity(seed, device, n_epochs, lr, technique, grid_size,
                                 model_name=model_name, tracker=tracker, grid_data=g)


# --- Method: chebyshev-pinn (Elasticity) ---
def train_chebyshev_pinn_elasticity(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """Chebyshev-spectral PINN for elasticity: dense spectral matrices, autograd backward."""
    g = build_grid_data_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

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
        scheduler.step()

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

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
        scheduler.step()

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5)

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
            scheduler.step()

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
def train_sk_pinn_elasticity(seed, device, n_epochs, lr, grid_size, model_name, grid_data, tracker=None,
                             match_protocol=False):
    """SK-PINN for elasticity: sparse RKPM matrices, autograd backward.

    Uses model-specific weight decay (see _SK_PINN_WD).

    When ``match_protocol`` is True, weight decay is set to 0.0 and the
    cosine-annealing scheduler is disabled (flat lr=lr throughout) so the run
    is apples-to-apples with the rest of the matched-protocol baselines, which
    all run flat-Adam at lr=1e-3.
    """
    g = grid_data

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)
    wd = 0.0 if match_protocol else _SK_PINN_WD.get(model_name, 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if match_protocol:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=1e-5)

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
        if scheduler is not None:
            scheduler.step()

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
    model_dtype = next(model.parameters()).dtype
    xy_t = torch.tensor(xy_eval, dtype=model_dtype, device=device, requires_grad=True)

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
# SLRM — Static Linear Residual Map surrogate gradient
# (Research phase 5; see llmdocs/research/research_log/04_design.md.)
#
# The PDE-gradient path through the residual DAG is replaced at every
# training step by a single precomputed linear map applied to the
# current residual tensor. One constant matrix is built at problem
# setup by linearising the residual DAG at a fixed reference input;
# the matrix never changes over the 30 000 training steps.
# =============================================================================

def _slrm_build_M_cavity(pred_ref, g):
    from src.slrm import build_slrm_operator
    def residual_fn(pred):
        c, mu, mv = compute_pde_terms(pred, g)
        return torch.cat([c, mu, mv], dim=1)  # (N, 3)
    return build_slrm_operator(residual_fn, pred_ref)


def _slrm_build_M_kovasznay(pred_ref, g):
    from src.slrm import build_slrm_operator
    def residual_fn(pred):
        c, mu, mv = compute_pde_kovasznay(pred, g)
        return torch.cat([c, mu, mv], dim=1)  # (N, 3)
    return build_slrm_operator(residual_fn, pred_ref)


def _slrm_build_M_elasticity(pred_ref, g):
    from src.slrm import build_slrm_operator
    def residual_fn(pred):
        ex, ey = compute_pde_elasticity(pred, g)
        return torch.cat([ex, ey], dim=1)  # (N, 2)
    return build_slrm_operator(residual_fn, pred_ref)


def _slrm_grad_from_residual(pred_pde, M_ref, residual_fn_nograd, interior_mask, M_int):
    """Apply the SLRM operator to the current (masked, flattened) residual.

    Returns the full-grid gradient tensor suitable as the upstream gradient
    of pred_pde.
    """
    with torch.no_grad():
        r = residual_fn_nograd(pred_pde)          # (N, k)
        r = r * interior_mask                     # zero boundary rows
        r_flat = r.reshape(-1)                    # (N*k,)
        g_flat = M_ref @ r_flat                   # (N*K,)
        N, K = pred_pde.shape
        return (2.0 / M_int) * g_flat.reshape(N, K)


# --- Method: slrm (Cavity) ---
def train_slrm(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """SLRM: constant-linear-map surrogate gradient for cavity NS."""
    g = build_grid_data(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    # Linearise the residual DAG once at the NN's initial full-grid
    # output. Smooth, non-degenerate, respects the NN's implicit
    # initialisation statistics. Cost: one jacobian build (~seconds).
    with torch.no_grad():
        pred_ref = compiled_model(g['xy_all']).detach().clone()
    M_ref = _slrm_build_M_cavity(pred_ref, g)

    def residual_fn_nograd(pred):
        c, mu, mv = compute_pde_terms(pred, g)
        return torch.cat([c, mu, mv], dim=1)

    interior_mask = g['interior_mask']
    M_int = g['M']

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

            grad_pde = _slrm_grad_from_residual(
                pred_pde, M_ref, residual_fn_nograd, interior_mask, M_int)

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
        _is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track or _is_last:
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
                loss_pde = (cont[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                loss_lid = mse(pred_l[:, 0:1], torch.ones_like(pred_l[:, 0:1])) + \
                           mse(pred_l[:, 1:2], torch.zeros_like(pred_l[:, 1:2]))
                loss_wall = mse(pred_w[:, 0:1], torch.zeros_like(pred_w[:, 0:1])) + \
                            mse(pred_w[:, 1:2], torch.zeros_like(pred_w[:, 1:2]))
                loss_p = mse(pred_c[:, 2:3], torch.zeros_like(pred_c[:, 2:3]))
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                final_loss = loss_pde.item() + loss_lid.item() + loss_wall.item() + loss_p.item() + reg_val
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: slrm (Kovasznay) ---
def train_slrm_kovasznay(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """SLRM: constant-linear-map surrogate gradient for Kovasznay flow."""
    g = build_grid_data_kovasznay(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    with torch.no_grad():
        pred_ref = compiled_model(g['xy_all']).detach().clone()
    M_ref = _slrm_build_M_kovasznay(pred_ref, g)

    def residual_fn_nograd(pred):
        c, mu, mv = compute_pde_kovasznay(pred, g)
        return torch.cat([c, mu, mv], dim=1)

    interior_mask = g['interior_mask']
    M_int = g['M']

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

            grad_pde = _slrm_grad_from_residual(
                pred_pde, M_ref, residual_fn_nograd, interior_mask, M_int)

            N_bc = g['N_bc']
            n_out = pred_bc.shape[1]
            grad_bc = 2.0 * (pred_bc - g['bc_target']) / (N_bc * n_out)

            grad_center = torch.zeros(1, 3, device=device)
            grad_center[:, 2:3] = 2.0 * (pred_c[:, 2:3] - g['p_center_exact'])

            upstream = torch.cat([grad_pde, grad_bc, grad_center], dim=0)
        pred_batch.backward(gradient=upstream)
        reg = model_reg_loss(model)
        if isinstance(reg, torch.Tensor):
            reg.backward()
        optimizer.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        _is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track or _is_last:
            with torch.no_grad():
                c, mu, mv = compute_pde_kovasznay(pred_pde, g)
                ii = g['interior_idx']
                loss_pde = (c[ii]**2).mean() + (mu[ii]**2).mean() + (mv[ii]**2).mean()
                loss_bc = mse(pred_bc, g['bc_target'])
                p_target = torch.tensor([[g['p_center_exact']]], dtype=torch.float32, device=pred_c.device)
                loss_p = mse(pred_c[:, 2:3], p_target)
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                final_loss = loss_pde.item() + loss_bc.item() + loss_p.item() + reg_val
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - start
    return model, train_time, final_loss


# --- Method: slrm (Elasticity) ---
def train_slrm_elasticity(seed, device, n_epochs, lr, technique, grid_size, model_name="mlp", tracker=None):
    """SLRM: constant-linear-map surrogate gradient for 2D elasticity.

    The elasticity residual is linear in the prediction, so the
    linearisation at any reference input is exact — SLRM reduces to
    the exact closed-form backward for this problem.
    """
    g = build_grid_data_elasticity(grid_size, device)

    torch.manual_seed(seed)
    model = make_model(model_name, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    if technique == "compile":
        compiled_model = torch.compile(model, mode='reduce-overhead')
    else:
        compiled_model = model

    with torch.no_grad():
        pred_ref = compiled_model(g['xy_all']).detach().clone()
    M_ref = _slrm_build_M_elasticity(pred_ref, g)

    def residual_fn_nograd(pred):
        ex, ey = compute_pde_elasticity(pred, g)
        return torch.cat([ex, ey], dim=1)

    interior_mask = g['interior_mask']
    M_int = g['M']

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

            grad_pde = _slrm_grad_from_residual(
                pred_pde, M_ref, residual_fn_nograd, interior_mask, M_int)

            N_bc = g['N_bc']
            n_out = pred_bc.shape[1]
            grad_bc = 2.0 * (pred_bc - g['bc_target']) / (N_bc * n_out)

            upstream = torch.cat([grad_pde, grad_bc], dim=0)
        pred_batch.backward(gradient=upstream)
        reg = model_reg_loss(model)
        if isinstance(reg, torch.Tensor):
            reg.backward()
        optimizer.step()
        scheduler.step()

        _should_track = tracker is not None and (epoch + 1) % tracker.interval == 0
        _is_last = (epoch == n_epochs - 1)
        if (epoch + 1) % LOG_INTERVAL == 0 or _should_track or _is_last:
            with torch.no_grad():
                eq_x, eq_y = compute_pde_elasticity(pred_pde, g)
                ii = g['interior_idx']
                loss_pde = (eq_x[ii]**2).mean() + (eq_y[ii]**2).mean()
                loss_bc = mse(pred_bc, g['bc_target'])
                reg = model_reg_loss(model)
                reg_val = reg.item() if isinstance(reg, torch.Tensor) else float(reg)
                final_loss = loss_pde.item() + loss_bc.item() + reg_val
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}: loss={final_loss:.6f}")
            if _should_track:
                tracker.step(epoch, final_loss, model)

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

    is_kovasznay = (args.problem == "kovasznay")
    is_elasticity = (args.problem == "elasticity")

    # Validate problem + method combinations
    if is_kovasznay and args.method not in ("sage", "dtpinn", "chebyshev-pinn", "autodiff", "ropinn", "sk-pinn", "jaxpinn", "sage-jax", "bfsa", "sdccg", "slrm", "slrm-jax", "stencil-adjoint", "can-pinn-faithful"):
        print(f"ERROR: Kovasznay problem only supports sage, dtpinn, autodiff, ropinn, sk-pinn, jaxpinn, sage-jax, slrm, slrm-jax, stencil-adjoint, can-pinn-faithful methods, "
              f"not '{args.method}'")
        sys.exit(1)
    if is_elasticity and args.method not in ("sage", "dtpinn", "chebyshev-pinn", "autodiff", "ropinn", "sk-pinn", "jaxpinn", "sage-jax", "bfsa", "sdccg", "slrm", "slrm-jax", "stencil-adjoint", "can-pinn-faithful"):
        print(f"ERROR: Elasticity problem only supports sage, dtpinn, autodiff, ropinn, sk-pinn, jaxpinn, sage-jax, slrm, slrm-jax, stencil-adjoint, can-pinn-faithful methods, "
              f"not '{args.method}'")
        sys.exit(1)
    if args.method == "slrm-jax" and args.model not in ("mlp", "pirate-net"):
        print(f"ERROR: slrm-jax method only supports --model mlp or pirate-net")
        sys.exit(1)
    if args.method == "slrm-jax" and args.track:
        print(f"WARNING: --track is not supported for slrm-jax; disabling tracker for this run.")
        args.track = False
    if args.method == "jaxpinn" and args.model not in ("mlp", "pirate-net"):
        print(f"ERROR: jaxpinn method only supports --model mlp or pirate-net (TSA-PINN originates in TensorFlow "
              f"and has no official JAX port; see llmdocs/CONTEXT.md)")
        sys.exit(1)
    if args.method in ("sage-jax", "bfsa", "sdccg") and args.model not in ("mlp", "pirate-net"):
        print(f"ERROR: sage-jax method only supports --model mlp or pirate-net (TSA-PINN originates in TensorFlow "
              f"and has no Flax port; see llmdocs/CONTEXT.md)")
        sys.exit(1)
    if args.method in ("jaxpinn", "sage-jax", "bfsa", "sdccg") and args.track:
        print(f"WARNING: --track is not supported for {args.method}; disabling tracker for this run.")
        args.track = False

    # Paper-faithful defaults for the new (Sharma & Shankar 2022) DT-PINN.
    # The repo's global --optimizer default is `adam` for backward compatibility
    # with the older Chebyshev variant, but the actual DT-PINN paper trains with
    # L-BFGS + 5K outer steps + fp64 + lr=0.04. Override here so that the
    # headline command `--method dtpinn` matches the paper without the user
    # having to remember four flags. Pass `--optimizer adam` / `--dtype fp32` /
    # `--epochs N` / `--lr X` explicitly to opt out for ablation; we detect
    # explicit passes via sys.argv (comparing parsed values to argparse defaults
    # would silently override `--lr 0.001` since the default is also 0.001).
    def _arg_passed(name: str) -> bool:
        return any(a == name or a.startswith(name + '=') for a in sys.argv)

    # Skip the paper-faithful overrides when --match-protocol is set so the
    # matched-DT-PINN row uses Adam-only / fp32 / lr=1e-3 / n_epochs from the
    # Adam defaults (i.e., the same protocol the rest of the matched-protocol
    # comparators run on). The matched-protocol caller is expected to pass
    # those four flags explicitly; this guard is defensive against partial
    # passes from interactive smoke tests.
    if args.method == "dtpinn" and not args.match_protocol:
        if not _arg_passed("--optimizer"):
            args.optimizer = "lbfgs"
        if not _arg_passed("--dtype"):
            args.dtype = "fp64"
        if not _arg_passed("--epochs"):
            args.epochs = 5000
        if not _arg_passed("--lr"):
            args.lr = 0.04  # temp/dt-pinn/src/dtpinn_cupy_fp64.py:372

    # Per-method default grid sizes (problem-aware)
    if args.grid_size is None:
        if is_kovasznay:
            if args.method == 'sk-pinn':
                # Matched-protocol: drop to Chebyshev-paired uniform grid (N=30).
                # Default (paper-faithful upstream): N=150 (dense uniform RKPM).
                args.grid_size = 30 if args.match_protocol else 150
            elif args.method == 'can-pinn-faithful':
                # Uniform grid for the FD stencil. 51 -> dx=Lx/50=0.03,
                # dy=Ly/50=0.04 on the [-0.5, 1.0] x [-0.5, 1.5] domain.
                args.grid_size = 51
            else:
                args.grid_size = 30
        elif is_elasticity:
            if args.method == 'sk-pinn':
                # Matched-protocol: drop to Chebyshev-paired uniform grid (N=30).
                # Default (paper-faithful upstream): N=100 (dense uniform RKPM).
                args.grid_size = 30 if args.match_protocol else 100
            elif args.method == 'can-pinn-faithful':
                # Uniform grid for the FD stencil. 51 -> dx=dy=0.02 on [0,1]^2,
                # matching the paper's cavity stencil-spacing convention.
                args.grid_size = 51
            else:
                args.grid_size = 30
        else:
            method_defaults = {
                'autodiff': 50, 'dtpinn': 50, 'analytical': 50, 'chebyshev-pinn': 50,
                'ropinn': 50, 'pielm': 50, 'sk-pinn': 200, 'sage': 50,
                'jaxpinn': 50, 'sage-jax': 50, 'bfsa': 50, 'sdccg': 50, 'slrm': 50, 'slrm-jax': 50,
                'stencil-adjoint': 50,
                'can-pinn-faithful': 50,
            }
            args.grid_size = method_defaults[args.method]
            # Cavity matched-protocol override for sk-pinn: use Chebyshev-paired
            # uniform grid (N=50) instead of the paper-faithful N=200. For
            # dtpinn the default is already N=50 (the Chebyshev-paired count),
            # so --match-protocol is a no-op on the cavity grid; the matched-
            # protocol semantics for dtpinn live in the trainer (no L-BFGS,
            # no auto-restart) and the four-flag override skip above.
            if args.method == 'sk-pinn' and args.match_protocol:
                args.grid_size = 50

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
    if args.method == 'sk-pinn' and args.match_protocol:
        method_suffix = ' (matched protocol: wd=0, no LR scheduler, paired grid)'
    elif args.method == 'dtpinn' and args.match_protocol:
        method_suffix = ' (matched protocol: Adam-only, fp32, no auto-restart, RBF-FD kept)'
    else:
        method_suffix = ''
    print(f"Method:    {args.method}{method_suffix}")
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
    best_epoch = ''
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
                    args.model, tracker=tracker,
                    dtype=args.dtype, rbf_fd_order=args.rbf_fd_order,
                    num_nodes=args.num_nodes,
                    optimizer_kind=args.optimizer,
                    match_protocol=args.match_protocol)
            elif args.method == "chebyshev-pinn":
                model, train_time, final_loss = train_chebyshev_pinn_elasticity(
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
                    args.model, g, tracker=tracker,
                    match_protocol=args.match_protocol)
            elif args.method == "jaxpinn":
                from src.jax_pinn import train_jaxpinn_elasticity
                model, train_time, final_loss = train_jaxpinn_elasticity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "sage-jax":
                from src.jax_pinn import train_sage_jax_elasticity
                model, train_time, final_loss = train_sage_jax_elasticity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "bfsa":
                from src.jax_pinn import train_bfsa_elasticity
                model, train_time, final_loss = train_bfsa_elasticity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "sdccg":
                from src.jax_pinn import train_sdccg_elasticity
                model, train_time, final_loss = train_sdccg_elasticity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "slrm":
                model, train_time, final_loss = train_slrm_elasticity(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "slrm-jax":
                from src.jax_pinn import train_slrm_jax_elasticity
                model, train_time, final_loss = train_slrm_jax_elasticity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "stencil-adjoint":
                model, train_time, final_loss = train_stencil_adjoint_elasticity(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "can-pinn-faithful":
                model, train_time, final_loss = train_can_pinn_faithful_elasticity(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
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
                    args.model, tracker=tracker,
                    dtype=args.dtype, rbf_fd_order=args.rbf_fd_order,
                    num_nodes=args.num_nodes,
                    optimizer_kind=args.optimizer,
                    match_protocol=args.match_protocol)
            elif args.method == "chebyshev-pinn":
                model, train_time, final_loss = train_chebyshev_pinn_kovasznay(
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
                    args.model, g, tracker=tracker,
                    match_protocol=args.match_protocol)
            elif args.method == "jaxpinn":
                from src.jax_pinn import train_jaxpinn_kovasznay
                model, train_time, final_loss = train_jaxpinn_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "sage-jax":
                from src.jax_pinn import train_sage_jax_kovasznay
                model, train_time, final_loss = train_sage_jax_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "bfsa":
                from src.jax_pinn import train_bfsa_kovasznay
                model, train_time, final_loss = train_bfsa_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "sdccg":
                from src.jax_pinn import train_sdccg_kovasznay
                model, train_time, final_loss = train_sdccg_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "slrm":
                model, train_time, final_loss = train_slrm_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "slrm-jax":
                from src.jax_pinn import train_slrm_jax_kovasznay
                model, train_time, final_loss = train_slrm_jax_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
            elif args.method == "stencil-adjoint":
                model, train_time, final_loss = train_stencil_adjoint_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
            elif args.method == "can-pinn-faithful":
                model, train_time, final_loss = train_can_pinn_faithful_kovasznay(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
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
                    args.model, tracker=tracker,
                    dtype=args.dtype, rbf_fd_order=args.rbf_fd_order,
                    num_nodes=args.num_nodes,
                    optimizer_kind=args.optimizer,
                    match_protocol=args.match_protocol)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "chebyshev-pinn":
                model, train_time, final_loss = train_chebyshev_pinn(
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
                    args.model, g, tracker=tracker,
                    match_protocol=args.match_protocol)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "sage":
                model, train_time, final_loss = train_sage(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "jaxpinn":
                from src.jax_pinn import train_jaxpinn_cavity
                model, train_time, final_loss = train_jaxpinn_cavity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "sage-jax":
                from src.jax_pinn import train_sage_jax_cavity
                model, train_time, final_loss = train_sage_jax_cavity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "bfsa":
                from src.jax_pinn import train_bfsa_cavity
                model, train_time, final_loss = train_bfsa_cavity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "sdccg":
                from src.jax_pinn import train_sdccg_cavity
                model, train_time, final_loss = train_sdccg_cavity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "slrm":
                model, train_time, final_loss = train_slrm(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "slrm-jax":
                from src.jax_pinn import train_slrm_jax_cavity
                model, train_time, final_loss = train_slrm_jax_cavity(
                    args.seed, device, args.epochs, args.lr, args.grid_size, args.model)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "stencil-adjoint":
                model, train_time, final_loss = train_stencil_adjoint(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

            elif args.method == "can-pinn-faithful":
                model, train_time, final_loss = train_can_pinn_faithful(
                    args.seed, device, args.epochs, args.lr, args.technique, args.grid_size,
                    args.model, tracker=tracker)
                n_params = sum(p.numel() for p in model.parameters())

        # ---- Evaluate final model ----
        print("\nEvaluating on 51x51 uniform grid...")
        if is_elasticity:
            metrics = evaluate_elasticity(model, device)
        elif is_kovasznay:
            metrics = evaluate_kovasznay(model, device)
        elif args.method == "pielm":
            metrics = evaluate_pielm(model)
        else:
            metrics = evaluate_model(model, device)

        # ---- Best-epoch model restoration ----
        # If tracker found a better model during training, restore and re-evaluate
        best_epoch = ''
        if tracker and tracker.best_state_dict is not None:
            if tracker.best_pde_rms < metrics['pde_rms']:
                best_epoch = tracker.best_epoch
                print(f"\nBest tracked model at epoch {best_epoch} (PDE {tracker.best_pde_rms:.6f}) "
                      f"< final (PDE {metrics['pde_rms']:.6f}). Restoring...")
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                base_model.load_state_dict(tracker.best_state_dict)
                if is_elasticity:
                    metrics = evaluate_elasticity(model, device)
                elif is_kovasznay:
                    metrics = evaluate_kovasznay(model, device)
                else:
                    metrics = evaluate_model(model, device)
                print(f"Re-evaluated best model: PDE RMS = {metrics['pde_rms']:.6f}")
            else:
                best_epoch = args.epochs
                print(f"\nFinal model is best (PDE {metrics['pde_rms']:.6f} "
                      f"<= tracked best {tracker.best_pde_rms:.6f})")

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
    if best_epoch:
        print(f"Best epoch:      {best_epoch}")
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
        'best_epoch': best_epoch,
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
