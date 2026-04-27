"""Phase-2 baseline runner for Program B (neural-operator preconditioning).

Runs B1 (SAGE+BFSA from scratch), B2 (diagonal-Jacobi pre-conditioner),
B3 (warm-start) on parametric PDE families F1 (cavity NS), F2 (Kovasznay),
F3 (elasticity) at K=10 instances per family, 3 seeds, 2 architectures.

B4 (from-scratch upper bound) is an alias for B1; same numbers are reported
under both labels (see 02_landscape.md § SG-7 commitment).

Each run records a per-probe trajectory (step, time, pde_rms, loss) at
N_probe=500 iteration intervals for N_cap=30,000 iterations.

Output: per-run JSON line to `results/progB_phase2/{tag}/{run_id}.json`,
aggregated into `results/progB_phase2_trajectories.jsonl`.

Usage (from project root):
    source env/bin/activate
    python -m src.phase2_runner --family F1 --method B1 --instance 0 --seed 42 --arch mlp
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from jax import jit

from src.jax_pinn import (
    _build_bfsa_backward,
    _enrich_g_jax_with_1d_matrices,
    _init_model,
    _make_sage_jax_cavity_train_step,
    _make_sage_jax_kovasznay_train_step,
    _make_sage_jax_elasticity_train_step,
    _reparam_elasticity_grid_,
    _reparam_kovasznay_grid_,
    _torch_g_to_jax,
    count_params,
    flax_params_to_torch,
)


# ============================================================================
# Instance loader
# ============================================================================
INSTANCES_JSON = Path(__file__).resolve().parent.parent / "llmdocs" / "research" / "research_log" / "02_landscape_instances.json"


def load_instances() -> dict:
    with open(INSTANCES_JSON) as f:
        return json.load(f)


# ============================================================================
# Evaluators — torch-autograd PDE-RMS on 51x51 uniform grid, matched to the
# v2 archived accuracy metric (01_contract.md § Accuracy metric).
# ============================================================================
def _gradients(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True)[0]


def eval_pde_rms_cavity(torch_model, re_param: float, device) -> float:
    """PDE-RMS for cavity NS with Smagorinsky at the given Re on 51x51 grid."""
    from src.lid_benchmark import Cs, U_lid
    nu_lam_val = U_lid / re_param
    nx, ny = 51, 51
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    torch_model.eval()
    pred = torch_model(xy_t)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    grad_u = _gradients(u, xy_t); grad_v = _gradients(v, xy_t)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]

    x_coord, y_coord = xy_t[:, 0:1], xy_t[:, 1:2]
    d = torch.min(torch.min(x_coord, 1.0 - x_coord),
                  torch.min(y_coord, 1.0 - y_coord))
    Sxx, Syy, Sxy = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    S_mag = torch.sqrt(2.0 * (Sxx ** 2 + Syy ** 2 + 2.0 * Sxy ** 2) + 1e-12)
    nu_eff = nu_lam_val + (Cs * d) ** 2 * S_mag
    continuity = du_dx + dv_dy
    u_conv = u * du_dx + v * du_dy
    v_conv = u * dv_dx + v * dv_dy
    grad_p = _gradients(p, xy_t)
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
    qx_u, qy_u = nu_eff * du_dx, nu_eff * du_dy
    qx_v, qy_v = nu_eff * dv_dx, nu_eff * dv_dy
    gqxu = _gradients(qx_u, xy_t); gqyu = _gradients(qy_u, xy_t)
    gqxv = _gradients(qx_v, xy_t); gqyv = _gradients(qy_v, xy_t)
    visc_u = gqxu[:, 0:1] + gqyu[:, 1:2]
    visc_v = gqxv[:, 0:1] + gqyv[:, 1:2]
    mom_u = u_conv + dp_dx - visc_u
    mom_v = v_conv + dp_dy - visc_v

    cont_np = continuity.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()
    torch_model.train()
    return float(np.sqrt(np.mean(cont_np ** 2 + mom_u_np ** 2 + mom_v_np ** 2)))


def eval_pde_rms_kovasznay(torch_model, re_param: float, device) -> float:
    """PDE-RMS for Kovasznay flow at the given Re on 51x51 grid over the
    Kov domain [-0.5,1.0] x [-0.5,1.5]."""
    nu_val = 1.0 / re_param
    nx, ny = 51, 51
    x = np.linspace(-0.5, 1.0, nx); y = np.linspace(-0.5, 1.5, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    torch_model.eval()
    pred = torch_model(xy_t)
    u, v, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    grad_u = _gradients(u, xy_t); grad_v = _gradients(v, xy_t); grad_p = _gradients(p, xy_t)
    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]
    d2u_dx = _gradients(du_dx, xy_t); d2u_dy = _gradients(du_dy, xy_t)
    d2v_dx = _gradients(dv_dx, xy_t); d2v_dy = _gradients(dv_dy, xy_t)
    lap_u = d2u_dx[:, 0:1] + d2u_dy[:, 1:2]
    lap_v = d2v_dx[:, 0:1] + d2v_dy[:, 1:2]
    cont = du_dx + dv_dy
    mom_u = u * du_dx + v * du_dy + dp_dx - nu_val * lap_u
    mom_v = u * dv_dx + v * dv_dy + dp_dy - nu_val * lap_v
    cont_np = cont.detach().cpu().numpy().flatten()
    mom_u_np = mom_u.detach().cpu().numpy().flatten()
    mom_v_np = mom_v.detach().cpu().numpy().flatten()
    torch_model.train()
    return float(np.sqrt(np.mean(cont_np ** 2 + mom_u_np ** 2 + mom_v_np ** 2)))


def eval_pde_rms_elasticity(torch_model, E_ratio: float, nu_poisson: float, device) -> float:
    """PDE-RMS for linear elasticity at (E/E_0, nu) on 51x51 grid."""
    E_0 = 4.0 / 3.0
    E = E_ratio * E_0
    mu = E / (2.0 * (1.0 + nu_poisson))
    lam = E * nu_poisson / ((1.0 + nu_poisson) * (1.0 - 2.0 * nu_poisson))
    Q_E_val = 4.0
    nx, ny = 51, 51
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    xy_eval = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_eval, dtype=torch.float32, device=device, requires_grad=True)

    torch_model.eval()
    pred = torch_model(xy_t)
    ux, uy = pred[:, 0:1], pred[:, 1:2]
    gux = _gradients(ux, xy_t); guy = _gradients(uy, xy_t)
    dux_dx, dux_dy = gux[:, 0:1], gux[:, 1:2]
    duy_dx, duy_dy = guy[:, 0:1], guy[:, 1:2]
    gdux_dx = _gradients(dux_dx, xy_t); gdux_dy = _gradients(dux_dy, xy_t)
    gduy_dx = _gradients(duy_dx, xy_t); gduy_dy = _gradients(duy_dy, xy_t)
    d2ux_dx2 = gdux_dx[:, 0:1]; d2ux_dy2 = gdux_dy[:, 1:2]
    d2uy_dx2 = gduy_dx[:, 0:1]; d2uy_dy2 = gduy_dy[:, 1:2]
    d2uy_dxdy = gduy_dx[:, 1:2]; d2ux_dxdy = gdux_dx[:, 1:2]
    # Body forces for manufactured solution (applied consistently with (lam, mu))
    x_coord = xy_t[:, 0:1]; y_coord = xy_t[:, 1:2]
    pi = math.pi
    ux_xx = -(2 * pi) ** 2 * torch.cos(2 * pi * x_coord) * torch.sin(pi * y_coord)
    ux_yy = -(pi ** 2) * torch.cos(2 * pi * x_coord) * torch.sin(pi * y_coord)
    ux_xy = -2 * pi ** 2 * torch.sin(2 * pi * x_coord) * torch.cos(pi * y_coord)
    uy_xx = -(pi ** 2) * torch.sin(pi * x_coord) * Q_E_val * y_coord ** 4 / 4.0
    uy_yy = torch.sin(pi * x_coord) * Q_E_val * 3.0 * y_coord ** 2
    uy_xy = pi * torch.cos(pi * x_coord) * Q_E_val * y_coord ** 3
    fx = -((lam + 2 * mu) * ux_xx + mu * ux_yy + (lam + mu) * uy_xy)
    fy = -(mu * uy_xx + (lam + 2 * mu) * uy_yy + (lam + mu) * ux_xy)
    eq_x = (lam + 2 * mu) * d2ux_dx2 + mu * d2ux_dy2 + (lam + mu) * d2uy_dxdy + fx
    eq_y = mu * d2uy_dx2 + (lam + 2 * mu) * d2uy_dy2 + (lam + mu) * d2ux_dxdy + fy
    eq_x_np = eq_x.detach().cpu().numpy().flatten()
    eq_y_np = eq_y.detach().cpu().numpy().flatten()
    torch_model.train()
    return float(np.sqrt(np.mean(eq_x_np ** 2 + eq_y_np ** 2)))


# ============================================================================
# Per-family grid + train-step builders
# ============================================================================
def _build_cavity_train_step(re_param: float, grid_size: int, model_name: str,
                             seed: int, lr: float, device,
                             preconditioner: Optional[jnp.ndarray] = None):
    """Build (train_step, optimizer, g_jax, params, net, setup_info) for cavity."""
    from src.lid_benchmark import build_grid_data, U_lid
    g_torch = build_grid_data(grid_size, device)
    g_torch['nu_lam'] = float(U_lid / re_param)
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Cs_d_sq', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_lid', 'xy_wall', 'nu_lam']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_lid = int(g_torch['N_lid'])
    N_wall = int(g_torch['N_wall'])
    M = int(g_torch['M'])
    off_lid = int(g_torch['off_lid'])
    off_wall = int(g_torch['off_wall'])
    off_center = int(g_torch['off_center'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M
    g_jax = _enrich_g_jax_with_1d_matrices(g_jax, grid_size, Lx=1.0, Ly=1.0)

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    bfsa_backward = _build_bfsa_backward('cavity')

    if preconditioner is None:
        train_step, optimizer = _make_sage_jax_cavity_train_step(
            net, lr, g_jax, bfsa_backward,
            N_all, off_lid, off_wall, off_center, N_lid, N_wall, M)
    else:
        train_step, optimizer = _make_cavity_precond_train_step(
            net, lr, g_jax, bfsa_backward,
            N_all, off_lid, off_wall, off_center, N_lid, N_wall, M,
            preconditioner)
    opt_state = optimizer.init(params)
    return train_step, optimizer, g_jax, params, opt_state, net, {
        'N_all': N_all, 'off_lid': off_lid, 'off_wall': off_wall,
        'off_center': off_center, 'N_lid': N_lid, 'N_wall': N_wall, 'M': M,
    }


def _build_kovasznay_train_step(re_param: float, grid_size: int, model_name: str,
                                seed: int, lr: float, device,
                                preconditioner: Optional[jnp.ndarray] = None):
    from src.lid_benchmark import build_grid_data_kovasznay
    g_torch = build_grid_data_kovasznay(grid_size, device)
    g_torch = _reparam_kovasznay_grid_(g_torch, float(re_param))
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'interior_mask',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target', 'nu_kov']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    off_center = int(g_torch['off_center'])
    p_center_exact = float(g_torch['p_center_exact'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M
    g_jax = _enrich_g_jax_with_1d_matrices(g_jax, grid_size, Lx=1.5, Ly=2.0)

    net, params = _init_model(model_name, out_dim=3, seed=seed)
    bfsa_backward = _build_bfsa_backward('kovasznay')
    if preconditioner is None:
        train_step, optimizer = _make_sage_jax_kovasznay_train_step(
            net, lr, g_jax, bfsa_backward,
            N_all, off_bc, off_center, N_bc, M, p_center_exact)
    else:
        train_step, optimizer = _make_kovasznay_precond_train_step(
            net, lr, g_jax, bfsa_backward,
            N_all, off_bc, off_center, N_bc, M, p_center_exact, preconditioner)
    opt_state = optimizer.init(params)
    return train_step, optimizer, g_jax, params, opt_state, net, {
        'N_all': N_all, 'off_bc': off_bc, 'off_center': off_center,
        'N_bc': N_bc, 'M': M,
    }


def _build_elasticity_train_step(E_ratio: float, nu_poisson: float,
                                 grid_size: int, model_name: str, seed: int,
                                 lr: float, n_epochs: int, device,
                                 preconditioner: Optional[jnp.ndarray] = None):
    from src.lid_benchmark import build_grid_data_elasticity
    g_torch = build_grid_data_elasticity(grid_size, device)
    g_torch = _reparam_elasticity_grid_(g_torch, float(E_ratio), float(nu_poisson))
    keys = ['Dx', 'Dy', 'DxT', 'DyT', 'Dxx', 'Dyy', 'Dxy',
            'DxxT', 'DyyT', 'DxyT', 'interior_mask', 'fx', 'fy',
            'xy_batched', 'xy_all', 'xy_bc', 'bc_target', 'lam_e', 'mu_e']
    g_jax = _torch_g_to_jax(g_torch, keys)
    N_all = int(g_torch['N_all'])
    N_bc = int(g_torch['N_bc'])
    M = int(g_torch['M'])
    off_bc = int(g_torch['off_bc'])
    g_jax['N_all'] = N_all
    g_jax['M'] = M
    g_jax = _enrich_g_jax_with_1d_matrices(g_jax, grid_size, Lx=1.0, Ly=1.0)

    net, params = _init_model(model_name, out_dim=2, seed=seed)
    bfsa_backward = _build_bfsa_backward('elasticity')
    if preconditioner is None:
        train_step, optimizer = _make_sage_jax_elasticity_train_step(
            net, lr, n_epochs, g_jax, bfsa_backward, N_all, off_bc, N_bc, M)
    else:
        train_step, optimizer = _make_elasticity_precond_train_step(
            net, lr, n_epochs, g_jax, bfsa_backward,
            N_all, off_bc, N_bc, M, preconditioner)
    opt_state = optimizer.init(params)
    return train_step, optimizer, g_jax, params, opt_state, net, {
        'N_all': N_all, 'off_bc': off_bc, 'N_bc': N_bc, 'M': M,
    }


# ============================================================================
# B2 preconditioner train-step variants — identical to the SAGE+BFSA
# train step EXCEPT the final parameter gradient is elementwise-scaled by
# the fixed 1/sqrt(D + eps) Jacobi preconditioner computed once at init.
# ============================================================================
def _precond_apply(param_grads, preconditioner):
    """Apply the diagonal preconditioner elementwise to the param-gradient pytree."""
    import jax
    return jax.tree.map(lambda g, p: g * p, param_grads, preconditioner)


def _make_cavity_precond_train_step(model, lr, g_jax, sage_backward,
                                    N_all, off_lid, off_wall, off_center,
                                    N_lid, N_wall, M, preconditioner):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    interior_mask = g_jax['interior_mask']
    optimizer = optax.adam(lr)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)
        pred_pde = pred_batch[:N_all]
        pred_lid = pred_batch[off_lid:off_wall]
        pred_wall = pred_batch[off_wall:off_center]
        pred_c = pred_batch[off_center:]
        grad_pde = sage_backward(pred_pde, g_jax)
        grad_lid = jnp.concatenate([2.0 * (pred_lid[:, 0:1] - 1.0) / N_lid,
                                    2.0 * pred_lid[:, 1:2] / N_lid,
                                    jnp.zeros_like(pred_lid[:, 2:3])], axis=1)
        grad_wall = jnp.concatenate([2.0 * pred_wall[:, 0:1] / N_wall,
                                     2.0 * pred_wall[:, 1:2] / N_wall,
                                     jnp.zeros_like(pred_wall[:, 2:3])], axis=1)
        grad_center = jnp.concatenate([jnp.zeros_like(pred_c[:, 0:1]),
                                       jnp.zeros_like(pred_c[:, 1:2]),
                                       2.0 * pred_c[:, 2:3]], axis=1)
        upstream = jnp.concatenate([grad_pde, grad_lid, grad_wall, grad_center], axis=0)
        (param_grads,) = vjp_fn(upstream)
        # --- Jacobi preconditioner applied BEFORE Adam ---
        param_grads = _precond_apply(param_grads, preconditioner)

        from src.jax_pinn import _compute_pde_cavity_jax
        cont, mom_u, mom_v = _compute_pde_cavity_jax(pred_pde, g_jax)
        loss_pde = (jnp.sum(cont ** 2 * interior_mask) / M
                    + jnp.sum(mom_u ** 2 * interior_mask) / M
                    + jnp.sum(mom_v ** 2 * interior_mask) / M)
        loss_lid = jnp.mean((pred_lid[:, 0:1] - 1.0) ** 2) + jnp.mean(pred_lid[:, 1:2] ** 2)
        loss_wall = jnp.mean(pred_wall[:, 0:1] ** 2) + jnp.mean(pred_wall[:, 1:2] ** 2)
        loss_c = jnp.mean(pred_c[:, 2:3] ** 2)
        loss_val = loss_pde + loss_lid + loss_wall + loss_c
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def _make_kovasznay_precond_train_step(model, lr, g_jax, sage_backward,
                                       N_all, off_bc, off_center, N_bc, M,
                                       p_center_exact, preconditioner):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    bc_target = g_jax['bc_target']
    interior_mask = g_jax['interior_mask']
    optimizer = optax.adam(lr)
    p_center_f = jnp.float32(p_center_exact)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)
        pred_pde = pred_batch[:N_all]
        pred_bc = pred_batch[off_bc:off_center]
        pred_c = pred_batch[off_center:]
        grad_pde = sage_backward(pred_pde, g_jax)
        n_out = 3
        grad_bc = 2.0 * (pred_bc - bc_target) / (N_bc * n_out)
        gc_p = 2.0 * (pred_c[:, 2:3] - p_center_f)
        gc_uv = jnp.zeros_like(pred_c[:, 0:2])
        grad_center = jnp.concatenate([gc_uv, gc_p], axis=1)
        upstream = jnp.concatenate([grad_pde, grad_bc, grad_center], axis=0)
        (param_grads,) = vjp_fn(upstream)
        param_grads = _precond_apply(param_grads, preconditioner)

        from src.jax_pinn import _compute_pde_kovasznay_jax
        cont, mom_u, mom_v = _compute_pde_kovasznay_jax(pred_pde, g_jax)
        loss_pde = (jnp.sum(cont ** 2 * interior_mask) / M
                    + jnp.sum(mom_u ** 2 * interior_mask) / M
                    + jnp.sum(mom_v ** 2 * interior_mask) / M)
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        loss_p = jnp.mean((pred_c[:, 2:3] - p_center_f) ** 2)
        loss_val = loss_pde + loss_bc + loss_p
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


def _make_elasticity_precond_train_step(model, lr, n_epochs, g_jax, sage_backward,
                                        N_all, off_bc, N_bc, M, preconditioner):
    apply_fn = model.apply
    xy_batched = g_jax['xy_batched']
    bc_target = g_jax['bc_target']
    interior_mask = g_jax['interior_mask']
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_epochs,
                                           alpha=1e-5 / lr)
    optimizer = optax.adam(learning_rate=schedule)

    def loss_and_grads(params):
        def forward(p):
            return apply_fn(p, xy_batched)
        pred_batch, vjp_fn = jax.vjp(forward, params)
        pred_pde = pred_batch[:N_all]
        pred_bc = pred_batch[off_bc:]
        grad_pde = sage_backward(pred_pde, g_jax)
        n_out = 2
        grad_bc = 2.0 * (pred_bc - bc_target) / (N_bc * n_out)
        upstream = jnp.concatenate([grad_pde, grad_bc], axis=0)
        (param_grads,) = vjp_fn(upstream)
        param_grads = _precond_apply(param_grads, preconditioner)

        from src.jax_pinn import _compute_pde_elasticity_jax
        eq_x, eq_y = _compute_pde_elasticity_jax(pred_pde, g_jax)
        loss_pde = (jnp.sum(eq_x ** 2 * interior_mask) / M
                    + jnp.sum(eq_y ** 2 * interior_mask) / M)
        loss_bc = jnp.mean((pred_bc - bc_target) ** 2)
        loss_val = loss_pde + loss_bc
        return loss_val, param_grads

    @jit
    def train_step(params, opt_state):
        loss_val, param_grads = loss_and_grads(params)
        updates, opt_state = optimizer.update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    return train_step, optimizer


# ============================================================================
# B2 diagonal Jacobi preconditioner — computed once at theta_0 via Hutchinson
# estimator: D_j ≈ mean_k (grad(<v_k, residual>) ^ 2)_j
#
# We estimate diag(J^T J) where J = d(R)/d(theta) by drawing K random
# v ~ N(0, I) in residual space, computing J^T v (one vjp per v), and
# averaging the squared entries. K=32 is a standard choice — variance
# decays as 1/K on each entry.
# ============================================================================
def _compute_jacobi_precond(family: str, params, forward_fn_builder, g_jax,
                            N_all: int, M: int, K: int = 32, key_seed: int = 0,
                            eps: float = 1e-6):
    """Return pytree matching params: p_j = 1 / sqrt(D_j + eps * mean(D))."""
    import jax
    # forward_fn_builder(params) -> (pred_batch)
    forward_fn = forward_fn_builder

    def res_dot(params, v):
        """<v, residual_vector(params)>.  residual_vector is flattened PDE
        residual over interior points only."""
        pred = forward_fn(params)[:N_all]
        if family == 'F1_cavity_NS':
            from src.jax_pinn import _compute_pde_cavity_jax
            cont, mu, mv = _compute_pde_cavity_jax(pred, g_jax)
            imask = g_jax['interior_mask']
            r = jnp.concatenate([cont * imask, mu * imask, mv * imask], axis=0).reshape(-1)
        elif family == 'F2_Kovasznay':
            from src.jax_pinn import _compute_pde_kovasznay_jax
            cont, mu, mv = _compute_pde_kovasznay_jax(pred, g_jax)
            imask = g_jax['interior_mask']
            r = jnp.concatenate([cont * imask, mu * imask, mv * imask], axis=0).reshape(-1)
        elif family == 'F3_elasticity':
            from src.jax_pinn import _compute_pde_elasticity_jax
            ex, ey = _compute_pde_elasticity_jax(pred, g_jax)
            imask = g_jax['interior_mask']
            r = jnp.concatenate([ex * imask, ey * imask], axis=0).reshape(-1)
        else:
            raise ValueError(family)
        return jnp.dot(v, r)

    # Size of residual vector
    if family in ('F1_cavity_NS', 'F2_Kovasznay'):
        D = N_all * 3
    else:
        D = N_all * 2

    rng = np.random.default_rng(key_seed)
    # Running sum of squared gradients (diag_estimate)
    # First grad call — get pytree structure
    v0 = jnp.asarray(rng.standard_normal(D).astype(np.float32))
    g_sample = jax.grad(res_dot, argnums=0)(params, v0)

    acc = jax.tree.map(lambda x: jnp.zeros_like(x), g_sample)
    acc = jax.tree.map(lambda a, g: a + g ** 2, acc, g_sample)
    for _ in range(K - 1):
        v = jnp.asarray(rng.standard_normal(D).astype(np.float32))
        g_sample = jax.grad(res_dot, argnums=0)(params, v)
        acc = jax.tree.map(lambda a, g: a + g ** 2, acc, g_sample)
    diag_est = jax.tree.map(lambda a: a / K, acc)
    # Global mean for regularisation
    total_sum = sum(jnp.sum(x) for x in jax.tree.leaves(diag_est))
    total_count = sum(x.size for x in jax.tree.leaves(diag_est))
    mean_diag = float(total_sum / total_count)
    # Preconditioner: 1 / sqrt(D_j + eps * mean_diag)
    scale = eps * mean_diag
    precond = jax.tree.map(lambda a: 1.0 / jnp.sqrt(a + scale), diag_est)
    return precond, float(mean_diag)


# ============================================================================
# Main run-one-instance driver
# ============================================================================
def run_baseline(family: str, method: str, instance_idx: int, seed: int,
                 arch: str, n_epochs: int = 30000, probe_every: int = 500,
                 lr: float = 1e-3, grid_size: int = 50,
                 warm_start_params=None, instances: Optional[dict] = None,
                 device=None) -> dict:
    """Run a single (family, method, instance, seed, arch) and return the
    trajectory + final summary.

    method: 'B1' (SAGE+BFSA from scratch), 'B2' (Jacobi preconditioner),
    'B3' (warm-start; warm_start_params must be provided), 'B4' (alias B1).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if instances is None:
        instances = load_instances()

    fam_spec = instances[family]
    # Decode instance parameters
    if family == 'F1_cavity_NS':
        re_param = fam_spec['values'][instance_idx]
        fam_params = {'re_param': float(re_param)}
    elif family == 'F2_Kovasznay':
        re_param = fam_spec['values'][instance_idx]
        fam_params = {'re_param': float(re_param)}
    elif family == 'F3_elasticity':
        E_ratio, nu_poisson = fam_spec['values'][instance_idx]
        fam_params = {'E_ratio': float(E_ratio), 'nu_poisson': float(nu_poisson)}
    else:
        raise ValueError(family)

    # Build train step (B1 / B4 / B3 share this; B2 reuses after computing precond)
    preconditioner = None

    if method in ('B1', 'B4', 'B3'):
        if family == 'F1_cavity_NS':
            train_step, optimizer, g_jax, params, opt_state, net, info = \
                _build_cavity_train_step(fam_params['re_param'], grid_size, arch,
                                         seed, lr, device, preconditioner=None)
        elif family == 'F2_Kovasznay':
            train_step, optimizer, g_jax, params, opt_state, net, info = \
                _build_kovasznay_train_step(fam_params['re_param'], grid_size, arch,
                                            seed, lr, device, preconditioner=None)
        else:
            train_step, optimizer, g_jax, params, opt_state, net, info = \
                _build_elasticity_train_step(fam_params['E_ratio'],
                                             fam_params['nu_poisson'],
                                             grid_size, arch, seed, lr, n_epochs,
                                             device, preconditioner=None)
        # Warm-start override
        if method == 'B3':
            if warm_start_params is None:
                raise ValueError("B3 requires warm_start_params")
            params = warm_start_params
            opt_state = optimizer.init(params)
    elif method == 'B2':
        # First build a NO-preconditioner step to compute initial params + forward
        if family == 'F1_cavity_NS':
            _, _, g_jax0, params0, _, net, info = \
                _build_cavity_train_step(fam_params['re_param'], grid_size, arch,
                                         seed, lr, device, preconditioner=None)
            forward_fn = lambda p: net.apply(p, g_jax0['xy_batched'])
        elif family == 'F2_Kovasznay':
            _, _, g_jax0, params0, _, net, info = \
                _build_kovasznay_train_step(fam_params['re_param'], grid_size, arch,
                                            seed, lr, device, preconditioner=None)
            forward_fn = lambda p: net.apply(p, g_jax0['xy_batched'])
        else:
            _, _, g_jax0, params0, _, net, info = \
                _build_elasticity_train_step(fam_params['E_ratio'],
                                             fam_params['nu_poisson'],
                                             grid_size, arch, seed, lr, n_epochs,
                                             device, preconditioner=None)
            forward_fn = lambda p: net.apply(p, g_jax0['xy_batched'])
        # Compute diag Jacobi at theta_0
        preconditioner, mean_diag = _compute_jacobi_precond(
            family, params0, forward_fn, g_jax0, info['N_all'], info['M'],
            K=32, key_seed=seed)
        print(f"  [B2] Jacobi precond computed, mean_diag={mean_diag:.3e}")
        # Rebuild with preconditioner
        if family == 'F1_cavity_NS':
            train_step, optimizer, g_jax, params, opt_state, net, info = \
                _build_cavity_train_step(fam_params['re_param'], grid_size, arch,
                                         seed, lr, device, preconditioner=preconditioner)
        elif family == 'F2_Kovasznay':
            train_step, optimizer, g_jax, params, opt_state, net, info = \
                _build_kovasznay_train_step(fam_params['re_param'], grid_size, arch,
                                            seed, lr, device, preconditioner=preconditioner)
        else:
            train_step, optimizer, g_jax, params, opt_state, net, info = \
                _build_elasticity_train_step(fam_params['E_ratio'],
                                             fam_params['nu_poisson'],
                                             grid_size, arch, seed, lr, n_epochs,
                                             device, preconditioner=preconditioner)
    else:
        raise ValueError(f"Unknown method {method!r}")

    n_params = count_params(params)
    print(f"  [{method}] arch={arch}, params={n_params}, fam_params={fam_params}, "
          f"n_epochs={n_epochs}, probe_every={probe_every}")

    # JIT warmup (1 step)
    t_warm0 = time.perf_counter()
    params, opt_state, loss = train_step(params, opt_state)
    loss.block_until_ready()
    t_warm = time.perf_counter() - t_warm0

    # Evaluate-fn picker
    if family == 'F1_cavity_NS':
        eval_fn = lambda tm: eval_pde_rms_cavity(tm, fam_params['re_param'], device)
        out_dim = 3
    elif family == 'F2_Kovasznay':
        eval_fn = lambda tm: eval_pde_rms_kovasznay(tm, fam_params['re_param'], device)
        out_dim = 3
    else:
        eval_fn = lambda tm: eval_pde_rms_elasticity(
            tm, fam_params['E_ratio'], fam_params['nu_poisson'], device)
        out_dim = 2

    # Probe immediately after warmup (step=1, trajectory entry for the init-level)
    trajectory = []
    t_start = time.perf_counter()
    # Initial probe (after warmup step)
    tm_init = flax_params_to_torch(arch, params, out_dim=out_dim, device=device)
    rms_init = eval_fn(tm_init)
    trajectory.append({'step': 1, 'time_s': 0.0, 'pde_rms': rms_init,
                       'loss': float(loss)})

    n_probes = n_epochs // probe_every
    steps_done = 1  # warmup counts as 1
    for probe_i in range(n_probes):
        steps_this_probe = probe_every if probe_i > 0 else (probe_every - 1)
        for _ in range(steps_this_probe):
            params, opt_state, loss = train_step(params, opt_state)
        loss.block_until_ready()
        steps_done += steps_this_probe
        t_probe = time.perf_counter() - t_start

        tm = flax_params_to_torch(arch, params, out_dim=out_dim, device=device)
        rms = eval_fn(tm)
        trajectory.append({'step': steps_done, 'time_s': float(t_probe),
                           'pde_rms': float(rms), 'loss': float(loss)})
        if (probe_i + 1) % 10 == 0 or (probe_i + 1) == n_probes:
            print(f"    probe {probe_i+1}/{n_probes} step={steps_done} "
                  f"t={t_probe:.1f}s rms={rms:.4e} loss={float(loss):.4e}")

    total_time = time.perf_counter() - t_start + t_warm
    return {
        'family': family, 'method': method, 'instance_idx': instance_idx,
        'seed': seed, 'arch': arch, 'n_params': n_params,
        'fam_params': fam_params, 'n_epochs': n_epochs,
        'probe_every': probe_every, 'lr': lr,
        't_warm_s': float(t_warm),
        't_total_s': float(total_time),
        'trajectory': trajectory,
        'final_pde_rms': float(trajectory[-1]['pde_rms']),
        'final_loss': float(trajectory[-1]['loss']),
        'final_params_pytree_structure': None,  # don't persist; too big
    }, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', required=True,
                        choices=['F1_cavity_NS', 'F2_Kovasznay', 'F3_elasticity'])
    parser.add_argument('--method', required=True, choices=['B1', 'B2', 'B3', 'B4'])
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arch', required=True, choices=['mlp', 'pirate-net'])
    parser.add_argument('--n_epochs', type=int, default=30000)
    parser.add_argument('--probe_every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--tag', default='progB_phase2')
    parser.add_argument('--warm_start_path', default=None,
                        help='For B3: path to pickled params pytree')
    parser.add_argument('--out_dir', default='results/progB_phase2')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = f"{args.family}_{args.method}_i{args.instance}_s{args.seed}_{args.arch}"
    out_path = os.path.join(args.out_dir, f"{run_id}.json")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} already exists")
        return

    warm_start_params = None
    if args.method == 'B3':
        import pickle
        with open(args.warm_start_path, 'rb') as f:
            warm_start_params = pickle.load(f)

    result, _ = run_baseline(
        args.family, args.method, args.instance, args.seed, args.arch,
        n_epochs=args.n_epochs, probe_every=args.probe_every, lr=args.lr,
        warm_start_params=warm_start_params)
    result['tag'] = args.tag
    with open(out_path, 'w') as f:
        json.dump(result, f)
    print(f"[done] {out_path} final_rms={result['final_pde_rms']:.4e} "
          f"t_total={result['t_total_s']:.1f}s")


if __name__ == '__main__':
    main()
