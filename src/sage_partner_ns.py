"""
SAGE-accelerated PINN for partner team's battery cooling NS problem.

Extends SAGE to 3D (x, y, t) using Chebyshev tensor product grid with
sparse Kronecker product differentiation matrices.

Domain: [0, 2] x [0, 0.5] x [0, 1.0] (matching partner's DeepXDE setup)
Network: FNN [3, 128x6, 3], tanh, Glorot normal
Training: Adam 20K + L-BFGS (matching partner schedule)

Usage:
  python -u src/sage_partner_ns.py --method sage --Nx 55 --Ny 15 --Nt 30
"""

import argparse
import copy
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# Physics constants (matching partner's setup)
# =============================================================================
NU = 1e-3       # kinematic viscosity
V0 = 1.0        # inlet velocity
ALPHA = 1e-3    # thermal diffusivity (Stage B)
T_IN = 25.0     # inlet temperature (Stage B)

# Domain bounds
X_MIN, X_MAX = 0.0, 2.0
Y_MIN, Y_MAX = 0.0, 0.5
T_MIN, T_MAX = 0.0, 1.0

LOG_INTERVAL = 1000
mse = nn.MSELoss()


# =============================================================================
# Network — matches partner's FNN [3, 128x6, 3], tanh, Glorot normal
# =============================================================================
class FNN_NS(nn.Module):
    """FNN [3, 128x6, 3] with tanh activation and Glorot normal init."""
    def __init__(self, input_dim=3, output_dim=3, hidden=128, n_layers=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.Tanh())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_glorot()

    def _init_glorot(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 1D Chebyshev differentiation
# =============================================================================
def chebyshev_points(N):
    """N Chebyshev-Gauss-Lobatto points on [-1, 1]."""
    return np.cos(np.pi * np.arange(N) / (N - 1))


def chebyshev_diff_matrix(N):
    """Standard Chebyshev differentiation matrix on [-1, 1]."""
    x = chebyshev_points(N)
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1.0) ** (i + j)) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    return D


# =============================================================================
# 3D Chebyshev grid with sparse Kronecker product D matrices
# =============================================================================
def build_3d_grid(Nx, Ny, Nt, device, domain=None):
    """Build 3D Chebyshev tensor product grid and sparse differentiation matrices.

    Ordering: x varies fastest, then y, then t.
    Total points: (Nx+1) * (Ny+1) * (Nt+1).

    Args:
        Nx, Ny, Nt: number of Chebyshev intervals per dimension
        device: torch device
        domain: dict with x_min, x_max, y_min, y_max, t_min, t_max
                (defaults to partner's domain)

    Returns:
        dict 'g' with grid points, D matrices, boundary indices, etc.
    """
    if domain is None:
        domain = {
            'x_min': X_MIN, 'x_max': X_MAX,
            'y_min': Y_MIN, 'y_max': Y_MAX,
            't_min': T_MIN, 't_max': T_MAX,
        }

    Lx = domain['x_max'] - domain['x_min']
    Ly = domain['y_max'] - domain['y_min']
    Lt = domain['t_max'] - domain['t_min']

    # 1D sizes (number of points = N+1)
    nx = Nx + 1
    ny = Ny + 1
    nt = Nt + 1
    N_all = nx * ny * nt

    # 1D Chebyshev differentiation matrices, scaled to physical domain
    # D_phys = D_ref * (2 / L)
    D1d_x = chebyshev_diff_matrix(nx) * (2.0 / Lx)  # float64
    D1d_y = chebyshev_diff_matrix(ny) * (2.0 / Ly)
    D1d_t = chebyshev_diff_matrix(nt) * (2.0 / Lt)

    Ix = np.eye(nx)
    Iy = np.eye(ny)
    It = np.eye(nt)

    # Kronecker products (x fastest, then y, then t):
    # Dx = I_t ⊗ I_y ⊗ D_x
    # Dy = I_t ⊗ D_y ⊗ I_x
    # Dt = D_t ⊗ I_y ⊗ I_x
    Dx_np = np.kron(It, np.kron(Iy, D1d_x))   # (N_all, N_all) float64
    Dy_np = np.kron(It, np.kron(D1d_y, Ix))
    Dt_np = np.kron(D1d_t, np.kron(Iy, Ix))

    # Precompute D² in float64 for better conditioning
    Dxx_np = Dx_np @ Dx_np
    Dyy_np = Dy_np @ Dy_np

    # Physical grid points
    x_ref = chebyshev_points(nx)
    y_ref = chebyshev_points(ny)
    t_ref = chebyshev_points(nt)

    x_phys = domain['x_min'] + Lx * 0.5 * (x_ref + 1.0)
    y_phys = domain['y_min'] + Ly * 0.5 * (y_ref + 1.0)
    t_phys = domain['t_min'] + Lt * 0.5 * (t_ref + 1.0)

    # 3D meshgrid: x fastest, then y, then t
    # Ordering: for each t, for each y, for each x (x varies fastest)
    # This matches the Kronecker product structure
    tt, yy, xx = np.meshgrid(t_phys, y_phys, x_phys, indexing='ij')
    xyt_grid = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])

    # Identify boundary faces
    eps = 1e-10
    xc = xyt_grid[:, 0]
    yc = xyt_grid[:, 1]
    tc = xyt_grid[:, 2]

    is_inlet  = (xc < domain['x_min'] + eps)         # x = x_min
    is_outlet = (xc > domain['x_max'] - eps)          # x = x_max
    is_wall_bot = (yc < domain['y_min'] + eps)        # y = y_min
    is_wall_top = (yc > domain['y_max'] - eps)        # y = y_max
    is_ic = (tc < domain['t_min'] + eps)              # t = 0

    is_wall = is_wall_bot | is_wall_top
    is_boundary = is_inlet | is_outlet | is_wall | is_ic

    interior_idx = np.where(~is_boundary)[0]
    inlet_idx = np.where(is_inlet & ~is_ic)[0]        # inlet, excluding IC overlap
    wall_idx = np.where(is_wall & ~is_ic & ~is_inlet & ~is_outlet)[0]
    wall_bot_idx = np.where(is_wall_bot & ~is_ic & ~is_inlet & ~is_outlet)[0]
    wall_top_idx = np.where(is_wall_top & ~is_ic & ~is_inlet & ~is_outlet)[0]
    outlet_idx = np.where(is_outlet & ~is_ic)[0]      # outlet, excluding IC overlap
    ic_idx = np.where(is_ic)[0]                        # initial condition (t=0)

    M = len(interior_idx)

    # Convert to sparse tensors on device
    def to_sparse(A_np):
        """Convert dense numpy to sparse COO tensor (float32), dropping zeros."""
        rows, cols = np.nonzero(A_np)
        vals = A_np[rows, cols].astype(np.float32)
        indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, size=A_np.shape, device=device
        ).coalesce()

    Dx_sp = to_sparse(Dx_np)
    Dy_sp = to_sparse(Dy_np)
    Dt_sp = to_sparse(Dt_np)
    Dxx_sp = to_sparse(Dxx_np)
    Dyy_sp = to_sparse(Dyy_np)

    xyt_all = torch.tensor(xyt_grid, dtype=torch.float32, device=device)

    interior_mask = torch.zeros(N_all, 1, device=device)
    interior_mask[interior_idx] = 1.0

    # Boundary point tensors
    xyt_inlet = xyt_all[inlet_idx]
    xyt_wall = xyt_all[wall_idx]
    xyt_wall_bot = xyt_all[wall_bot_idx]
    xyt_wall_top = xyt_all[wall_top_idx]
    xyt_outlet = xyt_all[outlet_idx]
    xyt_ic = xyt_all[ic_idx]

    # Batched input: [all, inlet, wall, outlet, IC]
    xyt_batched = torch.cat([xyt_all, xyt_inlet, xyt_wall, xyt_outlet, xyt_ic], dim=0)
    off_inlet = N_all
    off_wall = N_all + len(inlet_idx)
    off_outlet = off_wall + len(wall_idx)
    off_ic = off_outlet + len(outlet_idx)

    return {
        # Sparse D matrices
        'Dx': Dx_sp, 'Dy': Dy_sp, 'Dt': Dt_sp,
        'Dxx': Dxx_sp, 'Dyy': Dyy_sp,
        # Grid points
        'xyt_all': xyt_all, 'xyt_batched': xyt_batched,
        'xyt_inlet': xyt_inlet, 'xyt_wall': xyt_wall,
        'xyt_wall_bot': xyt_wall_bot, 'xyt_wall_top': xyt_wall_top,
        'xyt_outlet': xyt_outlet, 'xyt_ic': xyt_ic,
        # Indices
        'interior_idx': interior_idx, 'interior_mask': interior_mask,
        'inlet_idx': inlet_idx, 'wall_idx': wall_idx,
        'wall_bot_idx': wall_bot_idx, 'wall_top_idx': wall_top_idx,
        'outlet_idx': outlet_idx, 'ic_idx': ic_idx,
        # Counts
        'N_all': N_all, 'M': M,
        'N_inlet': len(inlet_idx), 'N_wall': len(wall_idx),
        'N_wall_bot': len(wall_bot_idx), 'N_wall_top': len(wall_top_idx),
        'N_outlet': len(outlet_idx), 'N_ic': len(ic_idx),
        # Offsets in batched tensor
        'off_inlet': off_inlet, 'off_wall': off_wall,
        'off_outlet': off_outlet, 'off_ic': off_ic,
        # Grid info
        'Nx': Nx, 'Ny': Ny, 'Nt': Nt,
        'nx': nx, 'ny': ny, 'nt': nt,
        'domain': domain,
    }


# =============================================================================
# PDE Forward — SAGE-traceable (sparse matmul)
# =============================================================================
def compute_pde_ns_3d(pred, g):
    """Time-dependent 2D incompressible NS residuals via spectral matrices.

    PDE:
      continuity: u_x + v_y = 0
      mom_u: u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
      mom_v: v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0

    All derivatives computed via sparse Kronecker-product Chebyshev matrices.
    Uses only operations supported by SAGE's 10 VJP rules.
    """
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    # Time derivatives
    u_t = torch.sparse.mm(g['Dt'], u)
    v_t = torch.sparse.mm(g['Dt'], v)

    # First spatial derivatives
    u_x = torch.sparse.mm(g['Dx'], u)
    u_y = torch.sparse.mm(g['Dy'], u)
    v_x = torch.sparse.mm(g['Dx'], v)
    v_y = torch.sparse.mm(g['Dy'], v)
    p_x = torch.sparse.mm(g['Dx'], p)
    p_y = torch.sparse.mm(g['Dy'], p)

    # Second spatial derivatives (precomputed D² for accuracy)
    u_xx = torch.sparse.mm(g['Dxx'], u)
    u_yy = torch.sparse.mm(g['Dyy'], u)
    v_xx = torch.sparse.mm(g['Dxx'], v)
    v_yy = torch.sparse.mm(g['Dyy'], v)

    # PDE residuals
    cont = u_x + v_y
    mom_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    mom_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)

    return cont, mom_u, mom_v


# =============================================================================
# PDE Forward — JVP (forward-mode AD through network layers)
# =============================================================================
def compute_pde_ns_3d_jvp(xyt, net):
    """NS PDE residuals via forward-mode AD (JVP tangent propagation).

    Computes exact network derivatives through manual forward-mode AD,
    replacing spectral D-matrix approximation with pointwise-exact derivatives.

    For a feedforward network Linear -> Tanh -> [Linear -> Tanh]x5 -> Linear:
      First order:  v_k = sigma'(z_k) * (W_k @ v_{k-1})
      Second order: w_k = sigma''(z_k) * (W_k @ v_{k-1})^2 + sigma'(z_k) * (W_k @ w_{k-1})

    Args:
        xyt: (N, 3) input points [x, y, t]
        net: nn.Sequential from FNN_NS (Linear-Tanh-...-Linear)

    Returns:
        (out, cont, mom_u, mom_v) -- network output and 3 PDE residual tensors
    """
    # Extract Linear layers from Sequential
    linears = [m for m in net if isinstance(m, nn.Linear)]

    # Forward pass with activation derivative caching
    h = xyt
    sp_list = []   # sigma'(z_k) = 1 - tanh^2(z_k)
    spp_list = []  # sigma''(z_k) = -2*tanh(z_k)*sigma'(z_k)

    for i in range(len(linears) - 1):
        z = h @ linears[i].weight.t() + linears[i].bias
        t = torch.tanh(z)
        sp = 1.0 - t * t
        spp = -2.0 * t * sp
        sp_list.append(sp)
        spp_list.append(spp)
        h = t

    # Output layer (no activation)
    out = h @ linears[-1].weight.t() + linears[-1].bias

    # Tangent propagation for input dimension d
    def propagate(d, need_second):
        W0_d = linears[0].weight[:, d]  # (hidden,)
        v = sp_list[0] * W0_d  # (N, hidden) via broadcast
        if need_second:
            w = spp_list[0] * (W0_d * W0_d)
        else:
            w = None

        for i in range(1, len(linears) - 1):
            Wv = v @ linears[i].weight.t()
            if need_second:
                Ww = w @ linears[i].weight.t()
                w = spp_list[i] * (Wv * Wv) + sp_list[i] * Ww
            v = sp_list[i] * Wv

        W_out = linears[-1].weight
        dy1 = v @ W_out.t()
        dy2 = w @ W_out.t() if need_second else None
        return dy1, dy2

    # x-direction: 1st + 2nd order
    dy_dx, d2y_dx2 = propagate(0, need_second=True)
    # y-direction: 1st + 2nd order
    dy_dy, d2y_dy2 = propagate(1, need_second=True)
    # t-direction: 1st order only
    dy_dt, _ = propagate(2, need_second=False)

    u = out[:, 0:1]
    v_val = out[:, 1:2]
    u_x, v_x, p_x = dy_dx[:, 0:1], dy_dx[:, 1:2], dy_dx[:, 2:3]
    u_y, v_y, p_y = dy_dy[:, 0:1], dy_dy[:, 1:2], dy_dy[:, 2:3]
    u_t, v_t = dy_dt[:, 0:1], dy_dt[:, 1:2]
    u_xx, v_xx = d2y_dx2[:, 0:1], d2y_dx2[:, 1:2]
    u_yy, v_yy = d2y_dy2[:, 0:1], d2y_dy2[:, 1:2]

    cont = u_x + v_y
    mom_u = u_t + u * u_x + v_val * u_y + p_x - NU * (u_xx + u_yy)
    mom_v = v_t + u * v_x + v_val * v_y + p_y - NU * (v_xx + v_yy)

    return out, cont, mom_u, mom_v


# =============================================================================
# SAGE backward generation
# =============================================================================
_generated_ns3d_backward = None


def _get_generated_backward_ns3d():
    """Lazily generate and cache SAGE backward function for 3D NS."""
    global _generated_ns3d_backward
    if _generated_ns3d_backward is None:
        from src.symbolic_vjp import trace_pde_forward, emit_backward
        tape = []
        outputs, inputs = trace_pde_forward(
            compute_pde_ns_3d, None, tape, sparse=True,
            constants=['Dx', 'Dy', 'Dt', 'Dxx', 'Dyy'],
            input_names=['u', 'v', 'p'],
        )
        source, fn = emit_backward(
            tape, list(outputs), ['dc', 'dmu', 'dmv'], inputs,
            sparse=True, func_name='generated_partner_ns_grad',
            input_names=['u', 'v', 'p'],
        )
        print(f"[SAGE] Generated backward function ({len(tape)} tape ops)")
        _generated_ns3d_backward = fn
    return _generated_ns3d_backward


# =============================================================================
# BC/IC gradient computation (analytical — no SAGE needed)
# =============================================================================
def compute_bc_ic_grad(pred_batch, g):
    """Compute upstream gradient for BC and IC losses.

    Each BC group uses MSE normalization matching DT-PINN's loss.backward():
      inlet: mse(pred[:, 0:2], target[:, 0:2]) -> grad = 2*(pred-target) / (N*2)
      wall:  mse(pred[:, 0:2], zeros)           -> grad = 2*pred / (N*2)
      outlet: mse(pred[:, 2:3], zeros)          -> grad = 2*pred / (N*1)
      IC:    mse(pred, zeros)                   -> grad = 2*pred / (N*3)

    Returns gradient tensor same shape as pred_batch's BC/IC portion.
    """
    off_inlet = g['off_inlet']
    off_wall = g['off_wall']
    off_outlet = g['off_outlet']
    off_ic = g['off_ic']
    N_inlet = g['N_inlet']
    N_wall = g['N_wall']
    N_outlet = g['N_outlet']
    N_ic = g['N_ic']

    pred_inlet = pred_batch[off_inlet:off_wall]
    pred_wall = pred_batch[off_wall:off_outlet]
    pred_outlet = pred_batch[off_outlet:off_ic]
    pred_ic = pred_batch[off_ic:]

    # Inlet: u=V0, v=0 — mse on (N_inlet, 2) -> divide by N_inlet*2
    grad_inlet = torch.zeros_like(pred_inlet)
    grad_inlet[:, 0:1] = 2.0 * (pred_inlet[:, 0:1] - V0) / (N_inlet * 2)
    grad_inlet[:, 1:2] = 2.0 * pred_inlet[:, 1:2] / (N_inlet * 2)

    # Walls: u=0, v=0 — mse on (N_wall, 2) -> divide by N_wall*2
    grad_wall = torch.zeros_like(pred_wall)
    grad_wall[:, 0:1] = 2.0 * pred_wall[:, 0:1] / (N_wall * 2)
    grad_wall[:, 1:2] = 2.0 * pred_wall[:, 1:2] / (N_wall * 2)

    # Outlet: p=0 — mse on (N_outlet, 1) -> divide by N_outlet
    grad_outlet = torch.zeros_like(pred_outlet)
    grad_outlet[:, 2:3] = 2.0 * pred_outlet[:, 2:3] / N_outlet

    # IC: u=0, v=0, p=0 — mse on (N_ic, 3) -> divide by N_ic*3
    grad_ic = 2.0 * pred_ic / (N_ic * 3)

    return torch.cat([grad_inlet, grad_wall, grad_outlet, grad_ic], dim=0)


# =============================================================================
# Loss computation (for logging and L-BFGS)
# =============================================================================
def compute_losses(pred_batch, g):
    """Compute individual loss terms for logging."""
    N_all = g['N_all']
    ii = g['interior_idx']

    pred_all = pred_batch[:N_all]
    c, mu, mv = compute_pde_ns_3d(pred_all, g)
    loss_cont = (c[ii] ** 2).mean()
    loss_mom_u = (mu[ii] ** 2).mean()
    loss_mom_v = (mv[ii] ** 2).mean()
    loss_pde = loss_cont + loss_mom_u + loss_mom_v

    pred_inlet = pred_batch[g['off_inlet']:g['off_wall']]
    pred_wall = pred_batch[g['off_wall']:g['off_outlet']]
    pred_outlet = pred_batch[g['off_outlet']:g['off_ic']]
    pred_ic = pred_batch[g['off_ic']:]

    # BC losses
    inlet_target = torch.zeros_like(pred_inlet)
    inlet_target[:, 0] = V0
    loss_inlet = mse(pred_inlet[:, 0:2], inlet_target[:, 0:2])

    loss_wall = mse(pred_wall[:, 0:2], torch.zeros_like(pred_wall[:, 0:2]))
    loss_outlet = mse(pred_outlet[:, 2:3], torch.zeros_like(pred_outlet[:, 2:3]))
    loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

    loss_bc = loss_inlet + loss_wall + loss_outlet + loss_ic
    loss_total = loss_pde + loss_bc

    return {
        'total': loss_total.item(),
        'pde': loss_pde.item(),
        'cont': loss_cont.item(),
        'mom_u': loss_mom_u.item(),
        'mom_v': loss_mom_v.item(),
        'bc': loss_bc.item(),
        'inlet': loss_inlet.item(),
        'wall': loss_wall.item(),
        'outlet': loss_outlet.item(),
        'ic': loss_ic.item(),
    }


# =============================================================================
# Training: SAGE (Adam + L-BFGS)
# =============================================================================
def train_sage_ns(model, g, args, device):
    """Train with SAGE-generated backward."""
    backward_fn = _get_generated_backward_ns3d()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    N_all = g['N_all']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    # === Adam phase ===
    print(f"\n[SAGE] Adam phase: {args.adam_epochs} epochs, lr={args.lr}")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()

        pred_batch = model(g['xyt_batched'])

        with torch.no_grad():
            pred_pde = pred_batch[:N_all]
            grad_pde = backward_fn(pred_pde, g)
            grad_bc = compute_bc_ic_grad(pred_batch, g)
            upstream = torch.cat([grad_pde, grad_bc], dim=0)

        pred_batch.backward(gradient=upstream)
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            with torch.no_grad():
                losses = compute_losses(model(g['xyt_batched']), g)
            print(f"  Epoch {epoch+1}: total={losses['total']:.6f} "
                  f"pde={losses['pde']:.6f} bc={losses['bc']:.6f}")

    adam_time = time.perf_counter() - start
    print(f"[SAGE] Adam done in {adam_time:.1f}s")

    # === L-BFGS phase ===
    if args.lbfgs:
        print(f"\n[SAGE] L-BFGS phase (max {args.lbfgs_steps} outer steps, "
              f"max_iter={args.lbfgs_max_iter})")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=args.lbfgs_max_iter,
            max_eval=int(args.lbfgs_max_iter * 1.25),
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=args.lbfgs_history,
            line_search_fn='strong_wolfe',
        )

        lbfgs_state = {'iter': 0, 'loss': float('inf'), 'plateau': 0}

        def closure():
            lbfgs.zero_grad()
            pred_batch = model(g['xyt_batched'])

            # Compute loss for L-BFGS line search
            with torch.no_grad():
                losses = compute_losses(pred_batch, g)
                loss_val = losses['total']

                grad_pde = backward_fn(pred_batch[:N_all], g)
                grad_bc = compute_bc_ic_grad(pred_batch, g)
                upstream = torch.cat([grad_pde, grad_bc], dim=0)

            pred_batch.backward(gradient=upstream)

            lbfgs_state['iter'] += 1
            if lbfgs_state['iter'] % 10 == 0:
                print(f"  L-BFGS iter {lbfgs_state['iter']}: loss={loss_val:.6f}")

            # Return scalar loss for line search
            return torch.tensor(loss_val, device=device, requires_grad=False)

        for step in range(args.lbfgs_steps):
            loss_t = lbfgs.step(closure)
            cur_loss = loss_t.item() if loss_t is not None else lbfgs_state['loss']
            # Early stopping: plateau detection
            if abs(lbfgs_state['loss'] - cur_loss) < 1e-10 * max(1.0, abs(cur_loss)):
                lbfgs_state['plateau'] += 1
            else:
                lbfgs_state['plateau'] = 0
            lbfgs_state['loss'] = cur_loss
            if lbfgs_state['plateau'] >= 50:
                print(f"  L-BFGS converged (plateau) at outer step {step+1}")
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    with torch.no_grad():
        losses = compute_losses(model(g['xyt_batched']), g)

    print(f"\n[SAGE] Adam time: {adam_time:.1f}s ({adam_time/60:.2f} min)")
    if args.lbfgs:
        print(f"[SAGE] L-BFGS time: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")
    print(f"[SAGE] Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"[SAGE] Peak GPU memory: {peak_mem:.2f} GB")
    print(f"[SAGE] Final loss: {losses['total']:.6f}")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': losses,
    }


# =============================================================================
# Training: DT-PINN (Chebyshev matrices + autograd backward)
# =============================================================================
def train_dtpinn_ns(model, g, args, device):
    """Train with spectral matrices but autograd backward."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    N_all = g['N_all']
    ii = g['interior_idx']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    print(f"\n[DT-PINN] Adam phase: {args.adam_epochs} epochs, lr={args.lr}")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()

        pred_batch = model(g['xyt_batched'])
        pred_all = pred_batch[:N_all]
        c, mu, mv = compute_pde_ns_3d(pred_all, g)
        loss_pde = (c[ii] ** 2).mean() + (mu[ii] ** 2).mean() + (mv[ii] ** 2).mean()

        pred_inlet = pred_batch[g['off_inlet']:g['off_wall']]
        pred_wall = pred_batch[g['off_wall']:g['off_outlet']]
        pred_outlet = pred_batch[g['off_outlet']:g['off_ic']]
        pred_ic = pred_batch[g['off_ic']:]

        inlet_target = torch.zeros_like(pred_inlet)
        inlet_target[:, 0] = V0
        loss_inlet = mse(pred_inlet[:, 0:2], inlet_target[:, 0:2])
        loss_wall = mse(pred_wall[:, 0:2], torch.zeros_like(pred_wall[:, 0:2]))
        loss_outlet = mse(pred_outlet[:, 2:3], torch.zeros_like(pred_outlet[:, 2:3]))
        loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

        loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f} "
                  f"pde={loss_pde.item():.6f}")

    adam_time = time.perf_counter() - start
    print(f"[DT-PINN] Adam done in {adam_time:.1f}s")

    # L-BFGS phase
    if args.lbfgs:
        print(f"\n[DT-PINN] L-BFGS phase (max {args.lbfgs_steps} outer steps, "
              f"max_iter={args.lbfgs_max_iter})")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0, max_iter=args.lbfgs_max_iter,
            max_eval=int(args.lbfgs_max_iter * 1.25),
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=args.lbfgs_history,
            line_search_fn='strong_wolfe',
        )
        lbfgs_state = {'iter': 0, 'loss': float('inf'), 'plateau': 0}

        def closure():
            lbfgs.zero_grad()
            pred_batch = model(g['xyt_batched'])
            pred_all = pred_batch[:N_all]
            c, mu, mv = compute_pde_ns_3d(pred_all, g)
            loss_pde = (c[ii] ** 2).mean() + (mu[ii] ** 2).mean() + (mv[ii] ** 2).mean()

            pred_inlet = pred_batch[g['off_inlet']:g['off_wall']]
            pred_wall = pred_batch[g['off_wall']:g['off_outlet']]
            pred_outlet = pred_batch[g['off_outlet']:g['off_ic']]
            pred_ic = pred_batch[g['off_ic']:]

            inlet_target = torch.zeros_like(pred_inlet)
            inlet_target[:, 0] = V0
            loss_inlet = mse(pred_inlet[:, 0:2], inlet_target[:, 0:2])
            loss_wall = mse(pred_wall[:, 0:2], torch.zeros_like(pred_wall[:, 0:2]))
            loss_outlet = mse(pred_outlet[:, 2:3], torch.zeros_like(pred_outlet[:, 2:3]))
            loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

            loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
            loss.backward()

            lbfgs_state['iter'] += 1
            if lbfgs_state['iter'] % 10 == 0:
                print(f"  L-BFGS iter {lbfgs_state['iter']}: loss={loss.item():.6f}")
            return loss

        for step in range(args.lbfgs_steps):
            loss_t = lbfgs.step(closure)
            cur_loss = loss_t.item() if loss_t is not None else lbfgs_state['loss']
            if abs(lbfgs_state['loss'] - cur_loss) < 1e-10 * max(1.0, abs(cur_loss)):
                lbfgs_state['plateau'] += 1
            else:
                lbfgs_state['plateau'] = 0
            lbfgs_state['loss'] = cur_loss
            if lbfgs_state['plateau'] >= 50:
                print(f"  L-BFGS converged (plateau) at outer step {step+1}")
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    with torch.no_grad():
        losses = compute_losses(model(g['xyt_batched']), g)

    print(f"\n[DT-PINN] Adam time: {adam_time:.1f}s ({adam_time/60:.2f} min)")
    if args.lbfgs:
        print(f"[DT-PINN] L-BFGS time: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")
    print(f"[DT-PINN] Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"[DT-PINN] Peak GPU memory: {peak_mem:.2f} GB")
    print(f"[DT-PINN] Final loss: {losses['total']:.6f}")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': losses,
    }


# =============================================================================
# Training: Pure Autodiff (same grid, autograd derivatives)
# =============================================================================
def gradients(y, x):
    """Compute dy/dx via autograd."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True)[0]


def pde_residuals_autodiff(model, xyt):
    """NS PDE residuals via autograd. xyt must have requires_grad=True."""
    pred = model(xyt)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = gradients(u, xyt)  # (N, 3) -> du/dx, du/dy, du/dt
    grad_v = gradients(v, xyt)
    grad_p = gradients(p, xyt)

    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

    # Second derivatives
    grad_u_x = gradients(u_x, xyt)
    grad_u_y = gradients(u_y, xyt)
    grad_v_x = gradients(v_x, xyt)
    grad_v_y = gradients(v_y, xyt)

    u_xx = grad_u_x[:, 0:1]
    u_yy = grad_u_y[:, 1:2]
    v_xx = grad_v_x[:, 0:1]
    v_yy = grad_v_y[:, 1:2]

    cont = u_x + v_y
    mom_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
    mom_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)

    return cont, mom_u, mom_v


def train_autodiff_ns(model, g, args, device):
    """Train with pure autograd on same Chebyshev collocation points."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Use interior points from the grid for PDE collocation
    xyt_int = g['xyt_all'][g['interior_idx']].clone().requires_grad_(True)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    print(f"\n[Autodiff] Adam phase: {args.adam_epochs} epochs, lr={args.lr}")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()

        # PDE on interior points
        cont, mom_u, mom_v = pde_residuals_autodiff(model, xyt_int)
        loss_pde = mse(cont, torch.zeros_like(cont)) + \
                   mse(mom_u, torch.zeros_like(mom_u)) + \
                   mse(mom_v, torch.zeros_like(mom_v))

        # BCs
        pred_inlet = model(g['xyt_inlet'])
        inlet_target = torch.zeros_like(pred_inlet)
        inlet_target[:, 0] = V0
        loss_inlet = mse(pred_inlet[:, 0:2], inlet_target[:, 0:2])

        pred_wall = model(g['xyt_wall'])
        loss_wall = mse(pred_wall[:, 0:2], torch.zeros_like(pred_wall[:, 0:2]))

        pred_outlet = model(g['xyt_outlet'])
        loss_outlet = mse(pred_outlet[:, 2:3], torch.zeros_like(pred_outlet[:, 2:3]))

        pred_ic = model(g['xyt_ic'])
        loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

        loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f} "
                  f"pde={loss_pde.item():.6f}")

    adam_time = time.perf_counter() - start
    print(f"[Autodiff] Adam done in {adam_time:.1f}s")

    # L-BFGS phase
    if args.lbfgs:
        print(f"\n[Autodiff] L-BFGS phase (max {args.lbfgs_steps} outer steps, "
              f"max_iter={args.lbfgs_max_iter}, history={args.lbfgs_history})")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0, max_iter=args.lbfgs_max_iter,
            max_eval=int(args.lbfgs_max_iter * 1.25),
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=args.lbfgs_history,
            line_search_fn='strong_wolfe',
        )
        lbfgs_state = {'iter': 0, 'loss': float('inf'), 'plateau': 0}

        def closure():
            lbfgs.zero_grad()
            cont, mom_u, mom_v = pde_residuals_autodiff(model, xyt_int)
            loss_pde = mse(cont, torch.zeros_like(cont)) + \
                       mse(mom_u, torch.zeros_like(mom_u)) + \
                       mse(mom_v, torch.zeros_like(mom_v))

            pred_inlet = model(g['xyt_inlet'])
            inlet_target = torch.zeros_like(pred_inlet)
            inlet_target[:, 0] = V0
            loss_inlet = mse(pred_inlet[:, 0:2], inlet_target[:, 0:2])
            pred_wall = model(g['xyt_wall'])
            loss_wall = mse(pred_wall[:, 0:2], torch.zeros_like(pred_wall[:, 0:2]))
            pred_outlet = model(g['xyt_outlet'])
            loss_outlet = mse(pred_outlet[:, 2:3], torch.zeros_like(pred_outlet[:, 2:3]))
            pred_ic = model(g['xyt_ic'])
            loss_ic = mse(pred_ic, torch.zeros_like(pred_ic))

            loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
            loss.backward()

            lbfgs_state['iter'] += 1
            if lbfgs_state['iter'] % 10 == 0:
                print(f"  L-BFGS iter {lbfgs_state['iter']}: loss={loss.item():.6f}")
            return loss

        for step in range(args.lbfgs_steps):
            loss_t = lbfgs.step(closure)
            cur_loss = loss_t.item() if loss_t is not None else lbfgs_state['loss']
            if abs(lbfgs_state['loss'] - cur_loss) < 1e-10 * max(1.0, abs(cur_loss)):
                lbfgs_state['plateau'] += 1
            else:
                lbfgs_state['plateau'] = 0
            lbfgs_state['loss'] = cur_loss
            if lbfgs_state['plateau'] >= 50:
                print(f"  L-BFGS converged (plateau) at outer step {step+1}")
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    # Final loss evaluation
    with torch.no_grad():
        pred_batch = model(g['xyt_batched'])
        losses = compute_losses(pred_batch, g)

    print(f"\n[Autodiff] Adam time: {adam_time:.1f}s ({adam_time/60:.2f} min)")
    if args.lbfgs:
        print(f"[Autodiff] L-BFGS time: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")
    print(f"[Autodiff] Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"[Autodiff] Peak GPU memory: {peak_mem:.2f} GB")
    print(f"[Autodiff] Final loss: {losses['total']:.6f}")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': losses,
    }


# =============================================================================
# Training: JVP (forward-mode AD through network layers)
# =============================================================================
def train_jvp_ns(model, g, args, device):
    """Train with JVP (forward-mode AD) derivatives on Chebyshev grid.

    KEY FIX: PDE residuals computed on ALL grid points (domain + boundary + IC),
    matching the DeepXDE training structure that was critical for PyTorch-AD accuracy.
    Uses per-component BC/IC loss (13 terms) to match DeepXDE's loss weighting.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # FIX: PDE on ALL grid points (not just interior) — matches DeepXDE approach
    xyt_pde = g['xyt_all']
    n_pde = xyt_pde.shape[0]

    pde_fn = compute_pde_ns_3d_jvp
    if getattr(args, 'compile', False):
        print("[JVP] Compiling PDE function with torch.compile...")
        pde_fn = torch.compile(compute_pde_ns_3d_jvp, mode='default')

    def compute_jvp_loss():
        """Compute 13-term loss: 3 PDE + 10 BC/IC (per-component, matching DeepXDE)."""
        # 3 PDE terms on ALL grid points
        out, cont, mom_u, mom_v = pde_fn(xyt_pde, model.net)
        l_cont = mse(cont, torch.zeros_like(cont))
        l_mom_u = mse(mom_u, torch.zeros_like(mom_u))
        l_mom_v = mse(mom_v, torch.zeros_like(mom_v))

        # 2 inlet terms (u=V0, v=0)
        pred_in = model(g['xyt_inlet'])
        l_in_u = mse(pred_in[:, 0:1], torch.full_like(pred_in[:, 0:1], V0))
        l_in_v = mse(pred_in[:, 1:2], torch.zeros_like(pred_in[:, 1:2]))

        # 4 wall terms (wall_bot u=0, v=0; wall_top u=0, v=0)
        pred_wb = model(g['xyt_wall_bot'])
        l_wb_u = mse(pred_wb[:, 0:1], torch.zeros_like(pred_wb[:, 0:1]))
        l_wb_v = mse(pred_wb[:, 1:2], torch.zeros_like(pred_wb[:, 1:2]))
        pred_wt = model(g['xyt_wall_top'])
        l_wt_u = mse(pred_wt[:, 0:1], torch.zeros_like(pred_wt[:, 0:1]))
        l_wt_v = mse(pred_wt[:, 1:2], torch.zeros_like(pred_wt[:, 1:2]))

        # 1 outlet term (p=0)
        pred_out = model(g['xyt_outlet'])
        l_out_p = mse(pred_out[:, 2:3], torch.zeros_like(pred_out[:, 2:3]))

        # 3 IC terms (u=0, v=0, p=0)
        pred_ic = model(g['xyt_ic'])
        l_ic_u = mse(pred_ic[:, 0:1], torch.zeros_like(pred_ic[:, 0:1]))
        l_ic_v = mse(pred_ic[:, 1:2], torch.zeros_like(pred_ic[:, 1:2]))
        l_ic_p = mse(pred_ic[:, 2:3], torch.zeros_like(pred_ic[:, 2:3]))

        # Sum all 13 terms (matching DeepXDE's loss structure)
        loss = (l_cont + l_mom_u + l_mom_v +
                l_in_u + l_in_v +
                l_wb_u + l_wb_v + l_wt_u + l_wt_v +
                l_out_p +
                l_ic_u + l_ic_v + l_ic_p)

        pde_val = (l_cont + l_mom_u + l_mom_v).item()
        return loss, pde_val

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    print(f"\n[JVP] Adam phase: {args.adam_epochs} epochs, lr={args.lr}")
    print(f"  PDE points: {n_pde} (ALL grid points, matching DeepXDE)")
    print(f"  Loss: 13 per-component terms (matching DeepXDE)")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()
        loss, pde_val = compute_jvp_loss()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f} "
                  f"pde={pde_val:.6f}")

    adam_time = time.perf_counter() - start
    print(f"[JVP] Adam done in {adam_time:.1f}s")

    # L-BFGS phase — match DeepXDE loop structure
    if args.lbfgs:
        iter_per_step = min(args.lbfgs_max_iter, args.lbfgs_steps)
        fun_per_step = int(args.lbfgs_steps * 1.25) * iter_per_step // args.lbfgs_steps
        print(f"\n[JVP] L-BFGS phase (maxiter={args.lbfgs_steps}, "
              f"iter_per_step={iter_per_step}, history={args.lbfgs_history}, "
              f"gtol={args.lbfgs_tolerance_grad}, ftol={args.lbfgs_tolerance_change})")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0, max_iter=iter_per_step,
            max_eval=fun_per_step,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=args.lbfgs_history,
            line_search_fn='strong_wolfe',
        )
        lbfgs_state = {'closure_calls': 0}

        def closure():
            lbfgs.zero_grad()
            loss, _ = compute_jvp_loss()
            loss.backward()
            lbfgs_state['closure_calls'] += 1
            if lbfgs_state['closure_calls'] % 100 == 0:
                print(f"  L-BFGS closure {lbfgs_state['closure_calls']}: "
                      f"loss={loss.item():.6f}")
            return loss

        # DeepXDE loop: track total optimizer iterations, stop at maxiter
        prev_n_iter = 0
        step = 0
        while prev_n_iter < args.lbfgs_steps:
            lbfgs.step(closure)
            state = lbfgs.state_dict()["state"]
            if state:
                n_iter = list(state.values())[0]["n_iter"]
            else:
                break
            if prev_n_iter == n_iter - 1:
                print(f"  L-BFGS converged at total iter {n_iter} "
                      f"(step {step+1}, closures={lbfgs_state['closure_calls']})")
                break
            if step % 5 == 0:
                print(f"  L-BFGS step {step+1}: n_iter={n_iter}, "
                      f"delta={n_iter - prev_n_iter}")
            prev_n_iter = n_iter
            step += 1
        else:
            print(f"  L-BFGS reached maxiter={args.lbfgs_steps} "
                  f"(steps={step}, closures={lbfgs_state['closure_calls']})")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    # Final loss eval via JVP (exact derivatives, not spectral)
    with torch.no_grad():
        out, cont, mom_u, mom_v = compute_pde_ns_3d_jvp(xyt_pde, model.net)
        loss_cont = (cont ** 2).mean().item()
        loss_mom_u = (mom_u ** 2).mean().item()
        loss_mom_v = (mom_v ** 2).mean().item()
        loss_pde = loss_cont + loss_mom_u + loss_mom_v

        pred_inlet = model(g['xyt_inlet'])
        inlet_tgt = torch.zeros_like(pred_inlet); inlet_tgt[:, 0] = V0
        loss_inlet = mse(pred_inlet[:, 0:2], inlet_tgt[:, 0:2]).item()
        loss_wall = mse(model(g['xyt_wall'])[:, 0:2],
                        torch.zeros(g['N_wall'], 2, device=device)).item()
        loss_outlet = mse(model(g['xyt_outlet'])[:, 2:3],
                          torch.zeros(g['N_outlet'], 1, device=device)).item()
        loss_ic = mse(model(g['xyt_ic']),
                      torch.zeros(g['N_ic'], 3, device=device)).item()
        loss_bc = loss_inlet + loss_wall + loss_outlet + loss_ic
        loss_total = loss_pde + loss_bc

    losses = {
        'total': loss_total, 'pde': loss_pde,
        'cont': loss_cont, 'mom_u': loss_mom_u, 'mom_v': loss_mom_v,
        'bc': loss_bc, 'inlet': loss_inlet, 'wall': loss_wall,
        'outlet': loss_outlet, 'ic': loss_ic,
    }

    print(f"\n[JVP] Adam time: {adam_time:.1f}s ({adam_time/60:.2f} min)")
    if args.lbfgs:
        print(f"[JVP] L-BFGS time: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")
    print(f"[JVP] Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"[JVP] Peak GPU memory: {peak_mem:.2f} GB")
    print(f"[JVP] Final loss: {losses['total']:.6f}")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': losses,
    }


# =============================================================================
# Training: JVP on random collocation (fair comparison with DeepXDE)
# =============================================================================
def train_jvp_random_ns(model, rpts, args, device):
    """Train with JVP derivatives on random collocation points (like DeepXDE).

    KEY FIX: PDE residuals computed on ALL points (domain + boundary + IC),
    matching DeepXDE. Uses 13 per-component loss terms.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # FIX: PDE on ALL points (domain + boundary + IC), matching DeepXDE
    xyt_pde = torch.cat([
        rpts['xyt_domain'],
        rpts['xyt_inlet'],
        rpts['xyt_outlet'],
        rpts['xyt_wall'],
        rpts['xyt_ic'],
    ], dim=0)
    n_pde = xyt_pde.shape[0]

    pde_fn = compute_pde_ns_3d_jvp
    if getattr(args, 'compile', False):
        print("[JVP-R] Compiling PDE function with torch.compile...")
        pde_fn = torch.compile(compute_pde_ns_3d_jvp, mode='default')

    def compute_jvp_random_loss():
        """Compute 13-term loss matching DeepXDE structure."""
        # 3 PDE terms on ALL points
        out, cont, mom_u, mom_v = pde_fn(xyt_pde, model.net)
        l_cont = mse(cont, torch.zeros_like(cont))
        l_mom_u = mse(mom_u, torch.zeros_like(mom_u))
        l_mom_v = mse(mom_v, torch.zeros_like(mom_v))

        # 2 inlet terms
        pred_in = model(rpts['xyt_inlet'])
        l_in_u = mse(pred_in[:, 0:1], torch.full_like(pred_in[:, 0:1], V0))
        l_in_v = mse(pred_in[:, 1:2], torch.zeros_like(pred_in[:, 1:2]))

        # 4 wall terms
        pred_wb = model(rpts['xyt_wall_bot'])
        l_wb_u = mse(pred_wb[:, 0:1], torch.zeros_like(pred_wb[:, 0:1]))
        l_wb_v = mse(pred_wb[:, 1:2], torch.zeros_like(pred_wb[:, 1:2]))
        pred_wt = model(rpts['xyt_wall_top'])
        l_wt_u = mse(pred_wt[:, 0:1], torch.zeros_like(pred_wt[:, 0:1]))
        l_wt_v = mse(pred_wt[:, 1:2], torch.zeros_like(pred_wt[:, 1:2]))

        # 1 outlet term
        pred_out = model(rpts['xyt_outlet'])
        l_out_p = mse(pred_out[:, 2:3], torch.zeros_like(pred_out[:, 2:3]))

        # 3 IC terms
        pred_ic = model(rpts['xyt_ic'])
        l_ic_u = mse(pred_ic[:, 0:1], torch.zeros_like(pred_ic[:, 0:1]))
        l_ic_v = mse(pred_ic[:, 1:2], torch.zeros_like(pred_ic[:, 1:2]))
        l_ic_p = mse(pred_ic[:, 2:3], torch.zeros_like(pred_ic[:, 2:3]))

        loss = (l_cont + l_mom_u + l_mom_v +
                l_in_u + l_in_v +
                l_wb_u + l_wb_v + l_wt_u + l_wt_v +
                l_out_p +
                l_ic_u + l_ic_v + l_ic_p)

        pde_val = (l_cont + l_mom_u + l_mom_v).item()
        return loss, pde_val

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    print(f"\n[JVP-R] Adam phase: {args.adam_epochs} epochs, lr={args.lr}")
    print(f"  PDE points: {n_pde} (domain+boundary+IC, matching DeepXDE)")
    print(f"  Loss: 13 per-component terms (matching DeepXDE)")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()
        loss, pde_val = compute_jvp_random_loss()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f} "
                  f"pde={pde_val:.6f}")

    adam_time = time.perf_counter() - start
    print(f"[JVP-R] Adam done in {adam_time:.1f}s")

    # L-BFGS phase — match DeepXDE loop structure
    if args.lbfgs:
        iter_per_step = min(args.lbfgs_max_iter, args.lbfgs_steps)
        fun_per_step = int(args.lbfgs_steps * 1.25) * iter_per_step // args.lbfgs_steps
        print(f"\n[JVP-R] L-BFGS phase (maxiter={args.lbfgs_steps}, "
              f"iter_per_step={iter_per_step}, history={args.lbfgs_history}, "
              f"gtol={args.lbfgs_tolerance_grad}, ftol={args.lbfgs_tolerance_change})")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0, max_iter=iter_per_step,
            max_eval=fun_per_step,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=args.lbfgs_history,
            line_search_fn='strong_wolfe',
        )
        lbfgs_state = {'closure_calls': 0}

        def closure():
            lbfgs.zero_grad()
            loss, _ = compute_jvp_random_loss()
            loss.backward()
            lbfgs_state['closure_calls'] += 1
            if lbfgs_state['closure_calls'] % 100 == 0:
                print(f"  L-BFGS closure {lbfgs_state['closure_calls']}: "
                      f"loss={loss.item():.6f}")
            return loss

        prev_n_iter = 0
        step = 0
        while prev_n_iter < args.lbfgs_steps:
            lbfgs.step(closure)
            state = lbfgs.state_dict()["state"]
            if state:
                n_iter = list(state.values())[0]["n_iter"]
            else:
                break
            if prev_n_iter == n_iter - 1:
                print(f"  L-BFGS converged at total iter {n_iter} "
                      f"(step {step+1}, closures={lbfgs_state['closure_calls']})")
                break
            if step % 5 == 0:
                print(f"  L-BFGS step {step+1}: n_iter={n_iter}, "
                      f"delta={n_iter - prev_n_iter}")
            prev_n_iter = n_iter
            step += 1
        else:
            print(f"  L-BFGS reached maxiter={args.lbfgs_steps} "
                  f"(steps={step}, closures={lbfgs_state['closure_calls']})")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    # Final loss eval
    with torch.no_grad():
        _, cont, mom_u, mom_v = compute_pde_ns_3d_jvp(xyt_pde, model.net)
    loss_cont = (cont ** 2).mean().item()
    loss_mom_u = (mom_u ** 2).mean().item()
    loss_mom_v = (mom_v ** 2).mean().item()
    loss_pde = loss_cont + loss_mom_u + loss_mom_v

    with torch.no_grad():
        pred_inlet = model(rpts['xyt_inlet'])
        inlet_tgt = torch.zeros_like(pred_inlet); inlet_tgt[:, 0] = V0
        loss_inlet = mse(pred_inlet[:, 0:2], inlet_tgt[:, 0:2]).item()
        loss_wall = mse(model(rpts['xyt_wall'])[:, 0:2],
                        torch.zeros(rpts['N_wall'], 2, device=device)).item()
        loss_outlet = mse(model(rpts['xyt_outlet'])[:, 2:3],
                          torch.zeros(rpts['N_outlet'], 1, device=device)).item()
        loss_ic = mse(model(rpts['xyt_ic']),
                      torch.zeros(rpts['N_ic'], 3, device=device)).item()
        loss_bc = loss_inlet + loss_wall + loss_outlet + loss_ic
        loss_total = loss_pde + loss_bc

    losses = {
        'total': loss_total, 'pde': loss_pde,
        'cont': loss_cont, 'mom_u': loss_mom_u, 'mom_v': loss_mom_v,
        'bc': loss_bc, 'inlet': loss_inlet, 'wall': loss_wall,
        'outlet': loss_outlet, 'ic': loss_ic,
    }

    print(f"\n[JVP-R] Adam time: {adam_time:.1f}s ({adam_time/60:.2f} min)")
    if args.lbfgs:
        print(f"[JVP-R] L-BFGS time: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")
    print(f"[JVP-R] Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"[JVP-R] Peak GPU memory: {peak_mem:.2f} GB")
    print(f"[JVP-R] Final loss: {loss_total:.6f}")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': losses,
    }


# =============================================================================
# Evaluation on dense uniform grid
# =============================================================================
def evaluate_ns(model, device, domain=None):
    """Evaluate on uniform grid via autograd PDE residuals."""
    if domain is None:
        domain = {
            'x_min': X_MIN, 'x_max': X_MAX,
            'y_min': Y_MIN, 'y_max': Y_MAX,
            't_min': T_MIN, 't_max': T_MAX,
        }
    nx_eval, ny_eval, nt_eval = 161, 81, 20
    xs = np.linspace(domain['x_min'], domain['x_max'], nx_eval)
    ys = np.linspace(domain['y_min'], domain['y_max'], ny_eval)
    ts = np.linspace(domain['t_min'], domain['t_max'], nt_eval)

    all_cont, all_mu, all_mv = [], [], []
    model.eval()

    for t_val in ts:
        X, Y = np.meshgrid(xs, ys)
        T_arr = np.full_like(X, t_val)
        xyt_np = np.column_stack([X.ravel(), Y.ravel(), T_arr.ravel()])
        xyt_t = torch.tensor(xyt_np, dtype=torch.float32, device=device,
                             requires_grad=True)

        pred = model(xyt_t)
        u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

        grad_u = gradients(u, xyt_t)
        grad_v = gradients(v, xyt_t)
        grad_p = gradients(p, xyt_t)

        u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
        v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
        p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

        grad_u_x = gradients(u_x, xyt_t)
        grad_u_y = gradients(u_y, xyt_t)
        grad_v_x = gradients(v_x, xyt_t)
        grad_v_y = gradients(v_y, xyt_t)
        u_xx = grad_u_x[:, 0:1]
        u_yy = grad_u_y[:, 1:2]
        v_xx = grad_v_x[:, 0:1]
        v_yy = grad_v_y[:, 1:2]

        cont = u_x + v_y
        mom_u = u_t + u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        mom_v = v_t + u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)

        all_cont.append(cont.detach().cpu().numpy().flatten())
        all_mu.append(mom_u.detach().cpu().numpy().flatten())
        all_mv.append(mom_v.detach().cpu().numpy().flatten())

    cont_all = np.concatenate(all_cont)
    mu_all = np.concatenate(all_mu)
    mv_all = np.concatenate(all_mv)

    pde_rms = float(np.sqrt(np.mean(cont_all**2 + mu_all**2 + mv_all**2)))
    cont_rms = float(np.sqrt(np.mean(cont_all**2)))
    mom_rms = float(np.sqrt(np.mean(mu_all**2 + mv_all**2)))

    model.train()
    return {
        'pde_rms': pde_rms,
        'continuity_rms': cont_rms,
        'momentum_rms': mom_rms,
    }


# =============================================================================
# Verification utilities
# =============================================================================
def verify_grid(g):
    """Verify D matrices against known derivatives."""
    print("\n" + "=" * 60)
    print("Grid Verification")
    print("=" * 60)

    N_all = g['N_all']
    xyt = g['xyt_all']
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]

    # Test: d(sin(pi*x))/dx = pi*cos(pi*x)
    f = torch.sin(math.pi * x)
    df_dx_exact = math.pi * torch.cos(math.pi * x)
    df_dx_num = torch.sparse.mm(g['Dx'], f)
    err_dx = (df_dx_num - df_dx_exact).abs().max().item()
    print(f"  Dx test (sin(pi*x)): max err = {err_dx:.2e}")

    # Test: d(sin(2*pi*y))/dy = 2*pi*cos(2*pi*y)
    f2 = torch.sin(2 * math.pi * y)
    df2_dy_exact = 2 * math.pi * torch.cos(2 * math.pi * y)
    df2_dy_num = torch.sparse.mm(g['Dy'], f2)
    err_dy = (df2_dy_num - df2_dy_exact).abs().max().item()
    print(f"  Dy test (sin(2*pi*y)): max err = {err_dy:.2e}")

    # Test: d(t^2)/dt = 2t
    f3 = t ** 2
    df3_dt_exact = 2 * t
    df3_dt_num = torch.sparse.mm(g['Dt'], f3)
    err_dt = (df3_dt_num - df3_dt_exact).abs().max().item()
    print(f"  Dt test (t^2): max err = {err_dt:.2e}")

    # Test: d²(sin(pi*x))/dx² = -pi²*sin(pi*x)
    d2f_dx2_exact = -(math.pi ** 2) * torch.sin(math.pi * x)
    d2f_dx2_num = torch.sparse.mm(g['Dxx'], f)
    err_dxx = (d2f_dx2_num - d2f_dx2_exact).abs().max().item()
    print(f"  Dxx test (sin(pi*x)): max err = {err_dxx:.2e}")

    # Test: d²(sin(2*pi*y))/dy² = -4*pi²*sin(2*pi*y)
    d2f2_dy2_exact = -(2 * math.pi) ** 2 * torch.sin(2 * math.pi * y)
    d2f2_dy2_num = torch.sparse.mm(g['Dyy'], f2)
    err_dyy = (d2f2_dy2_num - d2f2_dy2_exact).abs().max().item()
    print(f"  Dyy test (sin(2*pi*y)): max err = {err_dyy:.2e}")

    # Grid point counts
    print(f"\n  Grid: {g['nx']}x{g['ny']}x{g['nt']} = {N_all} points")
    print(f"  Interior: {g['M']}, Inlet: {g['N_inlet']}, "
          f"Wall: {g['N_wall']}, Outlet: {g['N_outlet']}, IC: {g['N_ic']}")

    ok = err_dx < 0.1 and err_dy < 0.1 and err_dt < 0.1
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_sage_backward(g, device):
    """Verify SAGE backward against autograd."""
    print("\n" + "=" * 60)
    print("SAGE Backward Verification")
    print("=" * 60)

    N_all = g['N_all']
    backward_fn = _get_generated_backward_ns3d()

    # Random prediction
    pred = torch.randn(N_all, 3, device=device, requires_grad=True)

    # Autograd reference
    c, mu, mv = compute_pde_ns_3d(pred, g)
    M = g['M']
    mask = g['interior_mask']
    loss = ((c**2 + mu**2 + mv**2) * mask).sum() / M
    auto_grad = torch.autograd.grad(loss, pred)[0]

    # SAGE
    sage_grad = backward_fn(pred.detach(), g)

    diff = (sage_grad - auto_grad).abs().max().item()
    rel_diff = diff / (auto_grad.abs().max().item() + 1e-12)
    print(f"  max |diff|: {diff:.2e}")
    print(f"  relative: {rel_diff:.2e}")
    ok = diff < 1e-3  # float32 tolerance
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_jvp_derivatives(model, device, n_test=200):
    """Verify JVP derivatives match autograd to ~1e-5 in float32."""
    print("\n" + "=" * 60)
    print("JVP Derivative Verification")
    print("=" * 60)

    torch.manual_seed(12345)
    xyt = torch.rand(n_test, 3, device=device)
    xyt[:, 0] = xyt[:, 0] * (X_MAX - X_MIN) + X_MIN
    xyt[:, 1] = xyt[:, 1] * (Y_MAX - Y_MIN) + Y_MIN
    xyt[:, 2] = xyt[:, 2] * (T_MAX - T_MIN) + T_MIN

    # JVP residuals
    with torch.no_grad():
        _, cont_jvp, mu_jvp, mv_jvp = compute_pde_ns_3d_jvp(xyt, model.net)

    # Autograd residuals
    xyt_ag = xyt.clone().requires_grad_(True)
    cont_ag, mu_ag, mv_ag = pde_residuals_autodiff(model, xyt_ag)

    err_cont = (cont_jvp - cont_ag.detach()).abs().max().item()
    err_mu = (mu_jvp - mu_ag.detach()).abs().max().item()
    err_mv = (mv_jvp - mv_ag.detach()).abs().max().item()
    max_err = max(err_cont, err_mu, err_mv)

    print(f"  Continuity max |err|: {err_cont:.2e}")
    print(f"  Momentum-u max |err|: {err_mu:.2e}")
    print(f"  Momentum-v max |err|: {err_mv:.2e}")

    ok = max_err < 1e-3
    print(f"  Status: {'PASS' if ok else 'FAIL'} (max err = {max_err:.2e})")
    return ok


# =============================================================================
# Training: PyTorch Autodiff with random collocation (matching DeepXDE points)
# =============================================================================
def _hammersley_sample(n, dim):
    """Generate Hammersley quasi-random points in [0,1]^dim using skopt (same as DeepXDE)."""
    from skopt.sampler import Hammersly
    sampler = Hammersly()
    space = [(0.0, 1.0)] * dim
    # Skip first point [0,0,...] like DeepXDE does
    pts = np.asarray(sampler.generate(space, n + 1)[1:], dtype=np.float32)
    return pts


def sample_random_collocation(n_domain, n_boundary, n_initial, device, seed=0,
                               sampling='random'):
    """Sample collocation points for training.

    Args:
        sampling: 'random' for pseudorandom, 'hammersley' for quasi-random (DeepXDE default).

    Returns dict with same interface as build_3d_grid for compatibility.
    """
    rng = np.random.RandomState(seed)
    use_qr = (sampling == 'hammersley')

    # Domain interior points — 3D Hammersley or pseudorandom
    if use_qr:
        pts3d = _hammersley_sample(n_domain, 3)
        xyt_domain = np.column_stack([
            pts3d[:, 0] * (X_MAX - X_MIN) + X_MIN,
            pts3d[:, 1] * (Y_MAX - Y_MIN) + Y_MIN,
            pts3d[:, 2] * (T_MAX - T_MIN) + T_MIN,
        ])
    else:
        x_dom = rng.uniform(X_MIN, X_MAX, (n_domain, 1))
        y_dom = rng.uniform(Y_MIN, Y_MAX, (n_domain, 1))
        t_dom = rng.uniform(T_MIN, T_MAX, (n_domain, 1))
        xyt_domain = np.hstack([x_dom, y_dom, t_dom])

    # Boundary points: distribute among 4 faces (equal per face, matching DeepXDE)
    n_per_face = n_boundary // 4
    n_extra = n_boundary - 4 * n_per_face

    # For boundary, DeepXDE uses Hammersley in 2D spatial + 1D time (permuted)
    # We approximate this with Hammersley on each face
    def _face_pts(n_pts, fixed_dim, fixed_val, free_dim, free_range):
        """Generate points on a boundary face."""
        if use_qr:
            pts2d = _hammersley_sample(n_pts, 2)
            free_vals = pts2d[:, 0:1] * (free_range[1] - free_range[0]) + free_range[0]
            t_vals = pts2d[:, 1:2] * (T_MAX - T_MIN) + T_MIN
        else:
            free_vals = rng.uniform(free_range[0], free_range[1], (n_pts, 1))
            t_vals = rng.uniform(T_MIN, T_MAX, (n_pts, 1))
        out = np.zeros((n_pts, 3), dtype=np.float32)
        out[:, fixed_dim] = fixed_val
        out[:, free_dim] = free_vals[:, 0]
        out[:, 2] = t_vals[:, 0]
        return out

    # Inlet (x=0)
    n_inlet = n_per_face + n_extra
    xyt_inlet = _face_pts(n_inlet, 0, X_MIN, 1, (Y_MIN, Y_MAX))

    # Outlet (x=2)
    xyt_outlet = _face_pts(n_per_face, 0, X_MAX, 1, (Y_MIN, Y_MAX))

    # Wall bottom (y=0)
    xyt_wall_bot = _face_pts(n_per_face, 1, Y_MIN, 0, (X_MIN, X_MAX))

    # Wall top (y=0.5)
    xyt_wall_top = _face_pts(n_per_face, 1, Y_MAX, 0, (X_MIN, X_MAX))

    xyt_wall = np.vstack([xyt_wall_bot, xyt_wall_top])

    # Initial condition points (t=0) — 2D Hammersley or pseudorandom
    if use_qr:
        pts2d = _hammersley_sample(n_initial, 2)
        xyt_ic = np.column_stack([
            pts2d[:, 0] * (X_MAX - X_MIN) + X_MIN,
            pts2d[:, 1] * (Y_MAX - Y_MIN) + Y_MIN,
            np.full(n_initial, T_MIN),
        ])
    else:
        x_ic = rng.uniform(X_MIN, X_MAX, (n_initial, 1))
        y_ic = rng.uniform(Y_MIN, Y_MAX, (n_initial, 1))
        xyt_ic = np.hstack([x_ic, y_ic, np.full((n_initial, 1), T_MIN)])

    # Convert to tensors
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    return {
        'xyt_domain': to_t(xyt_domain),
        'xyt_inlet': to_t(xyt_inlet),
        'xyt_outlet': to_t(xyt_outlet),
        'xyt_wall': to_t(xyt_wall),
        'xyt_wall_bot': to_t(xyt_wall_bot),
        'xyt_wall_top': to_t(xyt_wall_top),
        'xyt_ic': to_t(xyt_ic),
        'N_domain': n_domain,
        'N_inlet': n_inlet,
        'N_outlet': n_per_face,
        'N_wall': len(xyt_wall),
        'N_wall_bot': len(xyt_wall_bot),
        'N_wall_top': len(xyt_wall_top),
        'N_ic': n_initial,
        'N_total': n_domain + n_boundary + n_initial,
    }


def compute_loss_13term(model, xyt_int, rpts):
    """Compute 13 per-component loss terms matching DeepXDE's loss structure.

    DeepXDE computes each BC/IC component as a separate MSE term, giving
    ~5.7x more relative weight to BC/IC vs PDE compared to our grouped losses.
    This prevents overfitting to PDE at training points.

    Returns (total_loss, pde_loss_value) for logging.
    """
    zero = torch.zeros(1, device=xyt_int.device)

    # 3 PDE terms
    cont, mom_u, mom_v = pde_residuals_autodiff(model, xyt_int)
    l_cont  = mse(cont, torch.zeros_like(cont))
    l_mom_u = mse(mom_u, torch.zeros_like(mom_u))
    l_mom_v = mse(mom_v, torch.zeros_like(mom_v))

    # 2 inlet terms (u=V0, v=0)
    pred_in = model(rpts['xyt_inlet'])
    l_in_u = mse(pred_in[:, 0:1], torch.full_like(pred_in[:, 0:1], V0))
    l_in_v = mse(pred_in[:, 1:2], torch.zeros_like(pred_in[:, 1:2]))

    # 4 wall terms (wall_bot u=0, v=0; wall_top u=0, v=0)
    pred_wb = model(rpts['xyt_wall_bot'])
    l_wb_u = mse(pred_wb[:, 0:1], torch.zeros_like(pred_wb[:, 0:1]))
    l_wb_v = mse(pred_wb[:, 1:2], torch.zeros_like(pred_wb[:, 1:2]))
    pred_wt = model(rpts['xyt_wall_top'])
    l_wt_u = mse(pred_wt[:, 0:1], torch.zeros_like(pred_wt[:, 0:1]))
    l_wt_v = mse(pred_wt[:, 1:2], torch.zeros_like(pred_wt[:, 1:2]))

    # 1 outlet term (p=0)
    pred_out = model(rpts['xyt_outlet'])
    l_out_p = mse(pred_out[:, 2:3], torch.zeros_like(pred_out[:, 2:3]))

    # 3 IC terms (u=0, v=0, p=0)
    pred_ic = model(rpts['xyt_ic'])
    l_ic_u = mse(pred_ic[:, 0:1], torch.zeros_like(pred_ic[:, 0:1]))
    l_ic_v = mse(pred_ic[:, 1:2], torch.zeros_like(pred_ic[:, 1:2]))
    l_ic_p = mse(pred_ic[:, 2:3], torch.zeros_like(pred_ic[:, 2:3]))

    # Sum all 13 terms (matching DeepXDE's torch.sum(losses))
    loss = (l_cont + l_mom_u + l_mom_v +
            l_in_u + l_in_v +
            l_wb_u + l_wb_v + l_wt_u + l_wt_v +
            l_out_p +
            l_ic_u + l_ic_v + l_ic_p)

    pde_val = (l_cont + l_mom_u + l_mom_v).item()
    return loss, pde_val


def train_pytorch_ad_ns(model, rpts, args, device):
    """Train with pure PyTorch autograd on random collocation points (like DeepXDE)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Match DeepXDE: PDE residuals computed on domain + boundary + initial points (28K total).
    # DeepXDE's train_x_all = [initial; boundary; domain] and PDE loss uses ALL of it.
    # This enforces PDE consistency at boundary/IC locations, preventing overfitting.
    xyt_pde = torch.cat([
        rpts['xyt_domain'],
        rpts['xyt_inlet'],
        rpts['xyt_outlet'],
        rpts['xyt_wall'],
        rpts['xyt_ic'],
    ], dim=0).clone().requires_grad_(True)
    n_pde = xyt_pde.shape[0]

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    print(f"\n[PyTorch-AD] Adam phase: {args.adam_epochs} epochs, lr={args.lr}")
    print(f"  Points: {rpts['N_domain']} domain, {rpts['N_inlet']} inlet, "
          f"{rpts['N_wall']} wall ({rpts['N_wall_bot']}+{rpts['N_wall_top']}), "
          f"{rpts['N_outlet']} outlet, {rpts['N_ic']} IC")
    print(f"  PDE residuals on: {n_pde} points (domain+boundary+IC, matching DeepXDE)")
    print(f"  Loss: 13 per-component terms (matching DeepXDE)")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()

        loss, pde_val = compute_loss_13term(model, xyt_pde, rpts)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f} "
                  f"pde={pde_val:.6f}")

    adam_time = time.perf_counter() - start
    print(f"[PyTorch-AD] Adam done in {adam_time:.1f}s")

    # L-BFGS phase
    if args.lbfgs:
        # Match DeepXDE's L-BFGS loop structure exactly:
        # - iter_per_step = min(1000, maxiter) → max_iter per optimizer.step()
        # - Total iterations capped at lbfgs_steps (default 15000)
        # - Stop if a step does only 1 iteration (fully converged)
        iter_per_step = min(args.lbfgs_max_iter, args.lbfgs_steps)
        fun_per_step = int(args.lbfgs_steps * 1.25) * iter_per_step // args.lbfgs_steps
        print(f"\n[PyTorch-AD] L-BFGS phase (maxiter={args.lbfgs_steps}, "
              f"iter_per_step={iter_per_step}, history={args.lbfgs_history}, "
              f"gtol={args.lbfgs_tolerance_grad}, ftol={args.lbfgs_tolerance_change})")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0, max_iter=iter_per_step,
            max_eval=fun_per_step,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            history_size=args.lbfgs_history,
            line_search_fn='strong_wolfe',
        )
        lbfgs_state = {'closure_calls': 0}

        def closure():
            lbfgs.zero_grad()
            loss, _ = compute_loss_13term(model, xyt_pde, rpts)
            loss.backward()
            lbfgs_state['closure_calls'] += 1
            if lbfgs_state['closure_calls'] % 100 == 0:
                print(f"  L-BFGS closure {lbfgs_state['closure_calls']}: "
                      f"loss={loss.item():.6f}")
            return loss

        # DeepXDE loop: track total optimizer iterations, stop at maxiter
        prev_n_iter = 0
        step = 0
        while prev_n_iter < args.lbfgs_steps:
            lbfgs.step(closure)
            # Read cumulative iteration count from optimizer state
            state = lbfgs.state_dict()["state"]
            if state:
                n_iter = list(state.values())[0]["n_iter"]
            else:
                break
            if prev_n_iter == n_iter - 1:
                # Converged: optimizer only did 1 iteration
                print(f"  L-BFGS converged at total iter {n_iter} "
                      f"(step {step+1}, closures={lbfgs_state['closure_calls']})")
                break
            if step % 5 == 0:
                print(f"  L-BFGS step {step+1}: n_iter={n_iter}, "
                      f"delta={n_iter - prev_n_iter}")
            prev_n_iter = n_iter
            step += 1
        else:
            print(f"  L-BFGS reached maxiter={args.lbfgs_steps} "
                  f"(steps={step}, closures={lbfgs_state['closure_calls']})")

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0

    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    # Final loss eval — compute on PDE points (needs grad for PDE residuals)
    cont, mom_u, mom_v = compute_pde_ns_3d_nodiff(model, xyt_pde)
    loss_cont = (cont ** 2).mean().item()
    loss_mom_u = (mom_u ** 2).mean().item()
    loss_mom_v = (mom_v ** 2).mean().item()
    loss_pde = loss_cont + loss_mom_u + loss_mom_v

    with torch.no_grad():
        pred_inlet = model(rpts['xyt_inlet'])
        inlet_tgt = torch.zeros_like(pred_inlet); inlet_tgt[:, 0] = V0
        loss_inlet = mse(pred_inlet[:, 0:2], inlet_tgt[:, 0:2]).item()
        loss_wall = mse(model(rpts['xyt_wall'])[:, 0:2],
                        torch.zeros(rpts['N_wall'], 2, device=device)).item()
        loss_outlet = mse(model(rpts['xyt_outlet'])[:, 2:3],
                          torch.zeros(rpts['N_outlet'], 1, device=device)).item()
        loss_ic = mse(model(rpts['xyt_ic']),
                      torch.zeros(rpts['N_ic'], 3, device=device)).item()
        loss_bc = loss_inlet + loss_wall + loss_outlet + loss_ic
        loss_total = loss_pde + loss_bc

    losses = {
        'total': loss_total, 'pde': loss_pde,
        'cont': loss_cont, 'mom_u': loss_mom_u, 'mom_v': loss_mom_v,
        'bc': loss_bc, 'inlet': loss_inlet, 'wall': loss_wall,
        'outlet': loss_outlet, 'ic': loss_ic,
    }

    print(f"\n[PyTorch-AD] Adam time: {adam_time:.1f}s ({adam_time/60:.2f} min)")
    if args.lbfgs:
        print(f"[PyTorch-AD] L-BFGS time: {lbfgs_time:.1f}s ({lbfgs_time/60:.2f} min)")
    print(f"[PyTorch-AD] Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"[PyTorch-AD] Peak GPU memory: {peak_mem:.2f} GB")
    print(f"[PyTorch-AD] Final loss: {loss_total:.6f}")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': losses,
    }


def compute_pde_ns_3d_nodiff(model, xyt):
    """Compute NS PDE residuals via autograd (for evaluation, no_grad context ok for outer)."""
    xyt_g = xyt.detach().requires_grad_(True)
    pred = model(xyt_g)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = torch.autograd.grad(u.sum(), xyt_g, create_graph=False, retain_graph=True)[0]
    grad_v = torch.autograd.grad(v.sum(), xyt_g, create_graph=False, retain_graph=True)[0]
    grad_p = torch.autograd.grad(p.sum(), xyt_g, create_graph=False, retain_graph=True)[0]

    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    v_x, v_y, v_t = grad_v[:, 0:1], grad_v[:, 1:2], grad_v[:, 2:3]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

    # Second derivatives need create_graph=True for the first grad
    xyt_g2 = xyt.detach().requires_grad_(True)
    pred2 = model(xyt_g2)
    u2 = pred2[:, 0:1]
    grad_u2 = torch.autograd.grad(u2.sum(), xyt_g2, create_graph=True)[0]
    u2_x = grad_u2[:, 0:1]
    u2_y = grad_u2[:, 1:2]
    u_xx = torch.autograd.grad(u2_x.sum(), xyt_g2, create_graph=False, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u2_y.sum(), xyt_g2, create_graph=False)[0][:, 1:2]

    xyt_g3 = xyt.detach().requires_grad_(True)
    pred3 = model(xyt_g3)
    v3 = pred3[:, 1:2]
    grad_v3 = torch.autograd.grad(v3.sum(), xyt_g3, create_graph=True)[0]
    v3_x = grad_v3[:, 0:1]
    v3_y = grad_v3[:, 1:2]
    v_xx = torch.autograd.grad(v3_x.sum(), xyt_g3, create_graph=False, retain_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v3_y.sum(), xyt_g3, create_graph=False)[0][:, 1:2]

    cont = u_x + v_y
    mom_u = u_t + u.detach() * u_x + v.detach() * u_y + p_x - NU * (u_xx + u_yy)
    mom_v = v_t + u.detach() * v_x + v.detach() * v_y + p_y - NU * (v_xx + v_yy)

    return cont.detach(), mom_u.detach(), mom_v.detach()


# =============================================================================
# Stage B: Temperature PDE
# =============================================================================
def compute_pde_temp_3d(pred_T, g, u_frozen, v_frozen):
    """Temperature advection-diffusion via spectral matrices.

    T_t + u*T_x + v*T_y - alpha*(T_xx + T_yy) = 0
    """
    T = pred_T  # (N_all, 1)
    T_t = torch.sparse.mm(g['Dt'], T)
    T_x = torch.sparse.mm(g['Dx'], T)
    T_y = torch.sparse.mm(g['Dy'], T)
    T_xx = torch.sparse.mm(g['Dxx'], T)
    T_yy = torch.sparse.mm(g['Dyy'], T)

    res = T_t + u_frozen * T_x + v_frozen * T_y - ALPHA * (T_xx + T_yy)
    return res


def compute_temp_losses(pred_batch, g_temp):
    """Compute temperature loss components."""
    N_all = g_temp['N_all']
    ii = g_temp['interior_idx']

    pred_all = pred_batch[:N_all]
    u_frozen = g_temp['u_frozen']
    v_frozen = g_temp['v_frozen']

    res = compute_pde_temp_3d(pred_all, g_temp, u_frozen, v_frozen)
    loss_pde = (res[ii] ** 2).mean()

    pred_inlet = pred_batch[g_temp['off_inlet']:g_temp['off_wall']]
    pred_wall = pred_batch[g_temp['off_wall']:g_temp['off_outlet']]
    pred_outlet = pred_batch[g_temp['off_outlet']:g_temp['off_ic']]
    pred_ic = pred_batch[g_temp['off_ic']:]

    loss_inlet = mse(pred_inlet, torch.full_like(pred_inlet, T_IN))
    # Walls: Neumann dT/dy = 0 — enforced via Dy rows at wall points
    # For spectral: wall Dy rows applied to all-points prediction gives dT/dy at walls
    Dy_wall = extract_sparse_rows(g_temp['Dy'], g_temp['wall_idx'])
    dTdy_wall = torch.sparse.mm(Dy_wall, pred_all)
    loss_wall_neumann = (dTdy_wall ** 2).mean()
    # Outlet: Neumann dT/dx = 0
    Dx_outlet = extract_sparse_rows(g_temp['Dx'], g_temp['outlet_idx'])
    dTdx_outlet = torch.sparse.mm(Dx_outlet, pred_all)
    loss_outlet_neumann = (dTdx_outlet ** 2).mean()
    loss_ic = mse(pred_ic, torch.full_like(pred_ic, T_IN))

    loss_bc = loss_inlet + loss_wall_neumann + loss_outlet_neumann + loss_ic
    loss_total = loss_pde + loss_bc

    return {
        'total': loss_total.item(), 'pde': loss_pde.item(),
        'bc': loss_bc.item(), 'inlet': loss_inlet.item(),
        'wall_neumann': loss_wall_neumann.item(),
        'outlet_neumann': loss_outlet_neumann.item(),
        'ic': loss_ic.item(),
    }


def extract_sparse_rows(sp_matrix, row_indices):
    """Extract rows from a sparse matrix, returning a new sparse matrix."""
    indices = sp_matrix.coalesce().indices()
    values = sp_matrix.coalesce().values()
    rows, cols = indices[0], indices[1]
    N_all = sp_matrix.size(1)

    row_set = set(row_indices.tolist()) if isinstance(row_indices, np.ndarray) else set(row_indices)
    # Build index mapping: old_row -> new_row
    row_map = {old: new for new, old in enumerate(sorted(row_set))}

    mask = torch.zeros(len(rows), dtype=torch.bool, device=sp_matrix.device)
    new_rows = torch.zeros(len(rows), dtype=torch.long, device=sp_matrix.device)
    for i in range(len(rows)):
        r = rows[i].item()
        if r in row_map:
            mask[i] = True
            new_rows[i] = row_map[r]

    new_indices = torch.stack([new_rows[mask], cols[mask]])
    new_values = values[mask]
    return torch.sparse_coo_tensor(
        new_indices, new_values,
        size=(len(row_set), N_all),
        device=sp_matrix.device,
    ).coalesce()


def train_dtpinn_temp(model, g_temp, args, device):
    """Train Stage B (temperature) with DT-PINN (spectral matrices + autograd backward)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    N_all = g_temp['N_all']
    ii = g_temp['interior_idx']
    u_frozen = g_temp['u_frozen']
    v_frozen = g_temp['v_frozen']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    print(f"\n[DT-PINN-Temp] Adam phase: {args.adam_epochs} epochs")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()
        pred_batch = model(g_temp['xyt_batched'])
        pred_all = pred_batch[:N_all]

        res = compute_pde_temp_3d(pred_all, g_temp, u_frozen, v_frozen)
        loss_pde = (res[ii] ** 2).mean()

        pred_inlet = pred_batch[g_temp['off_inlet']:g_temp['off_wall']]
        loss_inlet = mse(pred_inlet, torch.full_like(pred_inlet, T_IN))

        # Neumann BCs via D matrix rows
        Dy_wall = g_temp['Dy_wall']
        dTdy_wall = torch.sparse.mm(Dy_wall, pred_all)
        loss_wall = (dTdy_wall ** 2).mean()

        Dx_outlet = g_temp['Dx_outlet']
        dTdx_outlet = torch.sparse.mm(Dx_outlet, pred_all)
        loss_outlet = (dTdx_outlet ** 2).mean()

        pred_ic = pred_batch[g_temp['off_ic']:]
        loss_ic = mse(pred_ic, torch.full_like(pred_ic, T_IN))

        loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6e} pde={loss_pde.item():.6e}")

    adam_time = time.perf_counter() - start
    print(f"[DT-PINN-Temp] Adam done in {adam_time:.1f}s")

    if args.lbfgs:
        print(f"\n[DT-PINN-Temp] L-BFGS phase")
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=20, max_eval=25,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=50, line_search_fn='strong_wolfe',
        )
        lbfgs_state = {'iter': 0, 'loss': float('inf'), 'plateau': 0}

        def closure():
            lbfgs.zero_grad()
            pred_batch = model(g_temp['xyt_batched'])
            pred_all = pred_batch[:N_all]
            res = compute_pde_temp_3d(pred_all, g_temp, u_frozen, v_frozen)
            loss_pde = (res[ii] ** 2).mean()
            pred_inlet = pred_batch[g_temp['off_inlet']:g_temp['off_wall']]
            loss_inlet = mse(pred_inlet, torch.full_like(pred_inlet, T_IN))
            dTdy = torch.sparse.mm(g_temp['Dy_wall'], pred_all)
            loss_wall = (dTdy ** 2).mean()
            dTdx = torch.sparse.mm(g_temp['Dx_outlet'], pred_all)
            loss_outlet = (dTdx ** 2).mean()
            pred_ic = pred_batch[g_temp['off_ic']:]
            loss_ic = mse(pred_ic, torch.full_like(pred_ic, T_IN))
            loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
            loss.backward()
            lbfgs_state['iter'] += 1
            if lbfgs_state['iter'] % 10 == 0:
                print(f"  L-BFGS iter {lbfgs_state['iter']}: loss={loss.item():.6e}")
            return loss

        for step in range(args.lbfgs_steps):
            loss_t = lbfgs.step(closure)
            cur_loss = loss_t.item() if loss_t is not None else lbfgs_state['loss']
            if abs(lbfgs_state['loss'] - cur_loss) < 1e-12 * max(1.0, abs(cur_loss)):
                lbfgs_state['plateau'] += 1
            else:
                lbfgs_state['plateau'] = 0
            lbfgs_state['loss'] = cur_loss
            if lbfgs_state['plateau'] >= 50:
                print(f"  L-BFGS converged at outer step {step+1}")
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    with torch.no_grad():
        losses = compute_temp_losses(model(g_temp['xyt_batched']), g_temp)

    print(f"[DT-PINN-Temp] Total: {total_time:.1f}s, loss={losses['total']:.6e}")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': losses,
    }


def train_pytorch_ad_temp(model, rpts_temp, args, device):
    """Train Stage B (temperature) with PyTorch autograd on random points."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    xyt_int = rpts_temp['xyt_domain'].clone().requires_grad_(True)
    u_interp = rpts_temp['u_frozen']  # (N_domain,1)
    v_interp = rpts_temp['v_frozen']

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()

    print(f"\n[PyTorch-AD-Temp] Adam phase: {args.adam_epochs} epochs")
    for epoch in range(args.adam_epochs):
        optimizer.zero_grad()
        T_pred = model(xyt_int)
        grad_T = gradients(T_pred, xyt_int)
        T_x, T_y, T_t = grad_T[:, 0:1], grad_T[:, 1:2], grad_T[:, 2:3]
        grad_Tx = gradients(T_x, xyt_int)
        grad_Ty = gradients(T_y, xyt_int)
        T_xx = grad_Tx[:, 0:1]
        T_yy = grad_Ty[:, 1:2]

        res = T_t + u_interp * T_x + v_interp * T_y - ALPHA * (T_xx + T_yy)
        loss_pde = mse(res, torch.zeros_like(res))

        pred_inlet = model(rpts_temp['xyt_inlet'])
        loss_inlet = mse(pred_inlet, torch.full_like(pred_inlet, T_IN))

        # Neumann BCs via autograd
        xyt_wall_g = rpts_temp['xyt_wall'].clone().requires_grad_(True)
        T_wall = model(xyt_wall_g)
        grad_Twall = gradients(T_wall, xyt_wall_g)
        dTdy_wall = grad_Twall[:, 1:2]
        loss_wall = mse(dTdy_wall, torch.zeros_like(dTdy_wall))

        xyt_out_g = rpts_temp['xyt_outlet'].clone().requires_grad_(True)
        T_out = model(xyt_out_g)
        grad_Tout = gradients(T_out, xyt_out_g)
        dTdx_out = grad_Tout[:, 0:1]
        loss_outlet = mse(dTdx_out, torch.zeros_like(dTdx_out))

        pred_ic = model(rpts_temp['xyt_ic'])
        loss_ic = mse(pred_ic, torch.full_like(pred_ic, T_IN))

        loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
        loss.backward()
        optimizer.step()

        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == args.adam_epochs - 1:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6e} pde={loss_pde.item():.6e}")

    adam_time = time.perf_counter() - start
    print(f"[PyTorch-AD-Temp] Adam done in {adam_time:.1f}s")

    if args.lbfgs:
        print(f"\n[PyTorch-AD-Temp] L-BFGS phase")
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=20, max_eval=25,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=50, line_search_fn='strong_wolfe',
        )
        lbfgs_state = {'iter': 0, 'loss': float('inf'), 'plateau': 0}

        def closure():
            lbfgs.zero_grad()
            T_pred = model(xyt_int)
            grad_T = gradients(T_pred, xyt_int)
            T_x, T_y, T_t = grad_T[:, 0:1], grad_T[:, 1:2], grad_T[:, 2:3]
            grad_Tx = gradients(T_x, xyt_int)
            grad_Ty = gradients(T_y, xyt_int)
            T_xx, T_yy = grad_Tx[:, 0:1], grad_Ty[:, 1:2]
            res = T_t + u_interp * T_x + v_interp * T_y - ALPHA * (T_xx + T_yy)
            loss_pde = mse(res, torch.zeros_like(res))
            pred_inlet = model(rpts_temp['xyt_inlet'])
            loss_inlet = mse(pred_inlet, torch.full_like(pred_inlet, T_IN))
            xyt_w = rpts_temp['xyt_wall'].clone().requires_grad_(True)
            T_w = model(xyt_w)
            gw = gradients(T_w, xyt_w)
            loss_wall = mse(gw[:, 1:2], torch.zeros_like(gw[:, 1:2]))
            xyt_o = rpts_temp['xyt_outlet'].clone().requires_grad_(True)
            T_o = model(xyt_o)
            go = gradients(T_o, xyt_o)
            loss_outlet = mse(go[:, 0:1], torch.zeros_like(go[:, 0:1]))
            pred_ic = model(rpts_temp['xyt_ic'])
            loss_ic = mse(pred_ic, torch.full_like(pred_ic, T_IN))
            loss = loss_pde + loss_inlet + loss_wall + loss_outlet + loss_ic
            loss.backward()
            lbfgs_state['iter'] += 1
            if lbfgs_state['iter'] % 10 == 0:
                print(f"  L-BFGS iter {lbfgs_state['iter']}: loss={loss.item():.6e}")
            return loss

        for step in range(args.lbfgs_steps):
            loss_t = lbfgs.step(closure)
            cur_loss = loss_t.item() if loss_t is not None else lbfgs_state['loss']
            if abs(lbfgs_state['loss'] - cur_loss) < 1e-12 * max(1.0, abs(cur_loss)):
                lbfgs_state['plateau'] += 1
            else:
                lbfgs_state['plateau'] = 0
            lbfgs_state['loss'] = cur_loss
            if lbfgs_state['plateau'] >= 50:
                print(f"  L-BFGS converged at outer step {step+1}")
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    lbfgs_time = total_time - adam_time if args.lbfgs else 0.0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0.0

    print(f"[PyTorch-AD-Temp] Total: {total_time:.1f}s")
    return model, {
        'total_time': total_time, 'adam_time': adam_time, 'lbfgs_time': lbfgs_time,
        'peak_mem_gb': peak_mem, 'losses': {'total': lbfgs_state['loss']},
    }


def evaluate_temp(model, device, domain=None):
    """Evaluate temperature model on uniform grid."""
    if domain is None:
        domain = {
            'x_min': X_MIN, 'x_max': X_MAX,
            'y_min': Y_MIN, 'y_max': Y_MAX,
            't_min': T_MIN, 't_max': T_MAX,
        }
    nx_eval, ny_eval, nt_eval = 161, 81, 20
    xs = np.linspace(domain['x_min'], domain['x_max'], nx_eval)
    ys = np.linspace(domain['y_min'], domain['y_max'], ny_eval)
    ts = np.linspace(domain['t_min'], domain['t_max'], nt_eval)

    all_T = []
    model.eval()
    with torch.no_grad():
        for t_val in ts:
            X, Y = np.meshgrid(xs, ys)
            T_arr = np.full_like(X, t_val)
            xyt_np = np.column_stack([X.ravel(), Y.ravel(), T_arr.ravel()])
            xyt_t = torch.tensor(xyt_np, dtype=torch.float32, device=device)
            T_pred = model(xyt_t).cpu().numpy().flatten()
            all_T.append(T_pred)

    T_all = np.concatenate(all_T)
    T_mean = float(np.mean(T_all))
    T_std = float(np.std(T_all))
    T_err_from_25 = float(np.sqrt(np.mean((T_all - T_IN) ** 2)))

    model.train()
    return {
        'T_mean': T_mean, 'T_std': T_std,
        'T_rms_error_from_25': T_err_from_25,
    }


# =============================================================================
# CLI / Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fair comparison: DT-PINN vs SAGE vs PyTorch-AD for partner NS problem")
    parser.add_argument("--method", choices=["sage", "dtpinn", "autodiff", "pytorch_ad", "jvp", "jvp_random", "all"],
                        default="sage")
    parser.add_argument("--stage", choices=["ns", "temp", "all"], default="all",
                        help="Stage A (NS), Stage B (Temp), or both")
    parser.add_argument("--Nx", type=int, default=55)
    parser.add_argument("--Ny", type=int, default=15)
    parser.add_argument("--Nt", type=int, default=30)
    parser.add_argument("--adam_epochs", type=int, default=20000)
    parser.add_argument("--lbfgs", action="store_true", default=True)
    parser.add_argument("--no_lbfgs", action="store_true")
    parser.add_argument("--lbfgs_steps", type=int, default=15000,
                        help="Total L-BFGS iteration budget (DeepXDE default: 15000)")
    parser.add_argument("--lbfgs_max_iter", type=int, default=1000,
                        help="Max iterations per optimizer.step() (DeepXDE: min(1000, maxiter))")
    parser.add_argument("--lbfgs_history", type=int, default=100,
                        help="L-BFGS history size (DeepXDE default: 100)")
    parser.add_argument("--lbfgs_tolerance_grad", type=float, default=1e-8,
                        help="L-BFGS gradient tolerance (DeepXDE default: 1e-8)")
    parser.add_argument("--lbfgs_tolerance_change", type=float, default=0,
                        help="L-BFGS function tolerance (DeepXDE default: 0 = disabled)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify_only", action="store_true",
                        help="Only run verification, no training")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for JVP PDE function (experimental)")
    parser.add_argument("--sampling", choices=["random", "hammersley"], default="random",
                        help="Point sampling method (hammersley = quasi-random, DeepXDE default)")
    parser.add_argument("--outdir", type=str, default="results/sage_partner")
    args = parser.parse_args()

    if args.no_lbfgs:
        args.lbfgs = False

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    do_ns = args.stage in ("ns", "all")
    do_temp = args.stage in ("temp", "all")

    # Determine which methods need the Chebyshev grid vs random collocation
    if args.method == "all":
        methods_list = ["sage", "dtpinn", "autodiff", "pytorch_ad", "jvp", "jvp_random"]
    else:
        methods_list = [args.method]
    needs_grid = any(m in ("sage", "dtpinn", "autodiff", "jvp") for m in methods_list)
    needs_rpts = any(m in ("pytorch_ad", "jvp_random") for m in methods_list)

    # Build Chebyshev grid (used by dtpinn, sage, autodiff-on-grid)
    g = None
    if needs_grid or args.verify_only:
        print(f"\nBuilding 3D Chebyshev grid: Nx={args.Nx}, Ny={args.Ny}, Nt={args.Nt}")
        g = build_3d_grid(args.Nx, args.Ny, args.Nt, device)
        print(f"Grid: {g['N_all']} total, {g['M']} interior, "
              f"{g['N_inlet']} inlet, {g['N_wall']} wall, "
              f"{g['N_outlet']} outlet, {g['N_ic']} IC")
    else:
        print("\nSkipping Chebyshev grid build (not needed for selected methods)")

    # Build random collocation (used by pytorch_ad, jvp_random)
    rpts = None
    if needs_rpts or args.method == "all":
        n_domain, n_boundary, n_initial = 20000, 4000, 4000
        rpts = sample_random_collocation(n_domain, n_boundary, n_initial, device, args.seed,
                                          sampling=args.sampling)
        print(f"Random collocation: {rpts['N_total']} total "
              f"({rpts['N_domain']} domain, {rpts['N_inlet']} inlet, "
              f"{rpts['N_wall']} wall, {rpts['N_outlet']} outlet, {rpts['N_ic']} IC)")

    # Verification (only when grid is built)
    if g is not None:
        grid_ok = verify_grid(g)
        if not grid_ok:
            print("ERROR: Grid verification failed!")
            return

        sage_ok = verify_sage_backward(g, device)
        if not sage_ok:
            print("WARNING: SAGE backward verification shows discrepancy")

        # Verify JVP derivatives
        tmp_model = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
        jvp_ok = verify_jvp_derivatives(tmp_model, device)
        del tmp_model
        if not jvp_ok:
            print("WARNING: JVP derivative verification shows discrepancy")

    if args.verify_only:
        print("\nVerification complete.")
        return

    # Use methods_list computed above
    methods = methods_list

    results_ns = {}
    ns_models = {}  # Store trained NS models for Stage B

    # =========================================================================
    # Stage A: Navier-Stokes
    # =========================================================================
    if do_ns:
        print(f"\n{'#'*60}")
        print(f"# STAGE A: NAVIER-STOKES")
        print(f"{'#'*60}")

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Training NS: {method.upper()}")
            print(f"{'='*60}")

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            model = FNN_NS(input_dim=3, output_dim=3, hidden=128, n_layers=6).to(device)
            print(f"Model: {sum(p.numel() for p in model.parameters())} params")

            if method == "sage":
                model, info = train_sage_ns(model, g, args, device)
            elif method == "dtpinn":
                model, info = train_dtpinn_ns(model, g, args, device)
            elif method == "autodiff":
                model, info = train_autodiff_ns(model, g, args, device)
            elif method == "pytorch_ad":
                model, info = train_pytorch_ad_ns(model, rpts, args, device)
            elif method == "jvp":
                model, info = train_jvp_ns(model, g, args, device)
            elif method == "jvp_random":
                model, info = train_jvp_random_ns(model, rpts, args, device)

            # Evaluate on 161x81x20 uniform grid
            print(f"\nEvaluating on 161x81x20 uniform grid...")
            eval_results = evaluate_ns(model, device)
            print(f"  PDE RMS: {eval_results['pde_rms']:.6f}")
            print(f"  Continuity RMS: {eval_results['continuity_rms']:.6f}")
            print(f"  Momentum RMS: {eval_results['momentum_rms']:.6f}")

            results_ns[method] = {
                'method': method,
                'stage': 'ns',
                'total_time_s': info['total_time'],
                'total_time_min': info['total_time'] / 60,
                'adam_time_s': info['adam_time'],
                'adam_time_min': info['adam_time'] / 60,
                'lbfgs_time_s': info['lbfgs_time'],
                'lbfgs_time_min': info['lbfgs_time'] / 60,
                'peak_mem_gb': info['peak_mem_gb'],
                'final_loss': info['losses']['total'],
                'loss_pde': info['losses']['pde'],
                'loss_cont': info['losses']['cont'],
                'loss_mom_u': info['losses']['mom_u'],
                'loss_mom_v': info['losses']['mom_v'],
                'loss_bc': info['losses']['bc'],
                'loss_inlet': info['losses']['inlet'],
                'loss_wall': info['losses']['wall'],
                'loss_outlet': info['losses']['outlet'],
                'loss_ic': info['losses']['ic'],
                'eval_pde_rms': eval_results['pde_rms'],
                'eval_cont_rms': eval_results['continuity_rms'],
                'eval_mom_rms': eval_results['momentum_rms'],
                'collocation': g['N_all'] if method in ('sage', 'dtpinn', 'autodiff', 'jvp') else rpts['N_total'],
                'point_type': 'chebyshev' if method in ('sage', 'dtpinn', 'autodiff', 'jvp') else 'random (20K+4K+4K)',
                'Nx': args.Nx, 'Ny': args.Ny, 'Nt': args.Nt,
                'adam_epochs': args.adam_epochs, 'seed': args.seed,
            }

            ns_models[method] = model

            # Save model
            ckpt_path = os.path.join(args.outdir, f"model_ns_{method}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

        # NS Summary
        print(f"\n{'='*60}")
        print("STAGE A RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<14} {'Total(min)':>10} {'Adam(min)':>10} {'LBFGS(min)':>10} "
              f"{'PDE RMS':>10} {'Loss':>10} {'Mem(GB)':>8}")
        print("-" * 74)
        for method, r in results_ns.items():
            print(f"{method:<14} {r['total_time_min']:>10.2f} {r['adam_time_min']:>10.2f} "
                  f"{r['lbfgs_time_min']:>10.2f} {r['eval_pde_rms']:>10.6f} "
                  f"{r['final_loss']:>10.6f} {r['peak_mem_gb']:>8.2f}")

    # =========================================================================
    # Stage B: Temperature (uses frozen velocity from Stage A)
    # =========================================================================
    results_temp = {}
    if do_temp and ns_models:
        print(f"\n{'#'*60}")
        print(f"# STAGE B: TEMPERATURE (frozen velocity from Stage A)")
        print(f"{'#'*60}")

        for method in methods:
            if method not in ns_models:
                print(f"Skipping {method} Stage B — no NS model available")
                continue

            print(f"\n{'='*60}")
            print(f"Training Temp: {method.upper()}")
            print(f"{'='*60}")

            ns_model = ns_models[method]
            ns_model.eval()

            torch.manual_seed(args.seed + 100)  # Different seed for Stage B
            np.random.seed(args.seed + 100)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()

            temp_model = FNN_NS(input_dim=3, output_dim=1, hidden=128, n_layers=6).to(device)
            print(f"Temp model: {sum(p.numel() for p in temp_model.parameters())} params")

            if method in ('sage', 'dtpinn', 'autodiff', 'jvp'):
                # Get frozen velocity on Chebyshev grid
                with torch.no_grad():
                    ns_pred = ns_model(g['xyt_all'])
                    u_frozen = ns_pred[:, 0:1].clone()
                    v_frozen = ns_pred[:, 1:2].clone()

                # Build temperature grid dict (reuse g, add frozen fields)
                g_temp = dict(g)
                g_temp['u_frozen'] = u_frozen
                g_temp['v_frozen'] = v_frozen
                # Precompute extracted D matrix rows for Neumann BCs
                g_temp['Dy_wall'] = extract_sparse_rows(g['Dy'], g['wall_idx'])
                g_temp['Dx_outlet'] = extract_sparse_rows(g['Dx'], g['outlet_idx'])

                temp_model, info = train_dtpinn_temp(temp_model, g_temp, args, device)
            elif method == 'pytorch_ad':
                # Get frozen velocity on random domain points
                with torch.no_grad():
                    ns_pred = ns_model(rpts['xyt_domain'])
                    u_interp = ns_pred[:, 0:1].clone()
                    v_interp = ns_pred[:, 1:2].clone()

                rpts_temp = dict(rpts)
                rpts_temp['u_frozen'] = u_interp
                rpts_temp['v_frozen'] = v_interp

                temp_model, info = train_pytorch_ad_temp(temp_model, rpts_temp, args, device)

            # Evaluate temperature
            print("Evaluating temperature on 161x81x20 grid...")
            eval_temp = evaluate_temp(temp_model, device)
            print(f"  T_mean: {eval_temp['T_mean']:.4f}")
            print(f"  T_std: {eval_temp['T_std']:.6f}")
            print(f"  T_rms_from_25: {eval_temp['T_rms_error_from_25']:.6f}")

            results_temp[method] = {
                'method': method,
                'stage': 'temp',
                'total_time_s': info['total_time'],
                'total_time_min': info['total_time'] / 60,
                'adam_time_s': info['adam_time'],
                'lbfgs_time_s': info['lbfgs_time'],
                'peak_mem_gb': info['peak_mem_gb'],
                'final_loss': info['losses']['total'],
                'T_mean': eval_temp['T_mean'],
                'T_std': eval_temp['T_std'],
                'T_rms_from_25': eval_temp['T_rms_error_from_25'],
            }

            ckpt_path = os.path.join(args.outdir, f"model_temp_{method}.pt")
            torch.save(temp_model.state_dict(), ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # =========================================================================
    # Save comprehensive results CSV
    # =========================================================================
    csv_path = os.path.join(args.outdir, "results.csv")
    all_fields = [
        'method', 'stage', 'total_time_s', 'total_time_min',
        'adam_time_s', 'adam_time_min', 'lbfgs_time_s', 'lbfgs_time_min',
        'peak_mem_gb', 'final_loss',
        'loss_pde', 'loss_cont', 'loss_mom_u', 'loss_mom_v',
        'loss_bc', 'loss_inlet', 'loss_wall', 'loss_outlet', 'loss_ic',
        'eval_pde_rms', 'eval_cont_rms', 'eval_mom_rms',
        'T_mean', 'T_std', 'T_rms_from_25',
        'collocation', 'point_type',
        'Nx', 'Ny', 'Nt', 'adam_epochs', 'seed',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction='ignore')
        writer.writeheader()
        for r in results_ns.values():
            writer.writerow(r)
        for r in results_temp.values():
            writer.writerow(r)
    print(f"\nResults saved: {csv_path}")

    # Combined time summary
    if results_ns and results_temp:
        print(f"\n{'='*60}")
        print("COMBINED SUMMARY (Stage A + Stage B)")
        print(f"{'='*60}")
        for method in methods:
            ns_t = results_ns.get(method, {}).get('total_time_min', 0)
            temp_t = results_temp.get(method, {}).get('total_time_min', 0)
            total = ns_t + temp_t
            print(f"  {method:<14}: NS={ns_t:.2f}min + Temp={temp_t:.2f}min = {total:.2f}min total")


if __name__ == "__main__":
    main()
