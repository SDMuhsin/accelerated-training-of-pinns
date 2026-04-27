"""Level-2 verification for V4 flow SAGE-JAX NS adjoint.

Compares parameter gradients on the same FD-based NS residual between:
- Autograd: ``jax.grad`` through the 5-stencil Flax forward graph.
- SAGE-JAX: 5 forward evaluations (via ``jax.vjp``) + SAGE analytic
  adjoint on the stacked (B, 15) pred → reshape → single ``vjp_fn``
  call for param grads.

Both paths compute the IDENTICAL quantity (FD NS residual mean-square
loss); only the adjoint mechanism differs. Agreement must be at
float32 noise level.

Usage::

    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 python scripts/verify_v4_flow_jax_sage.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
from jax import random

from sage_ns_v4_jax import (
    FlowNetFlax, init_flow_params,
    build_v4_flow_sage_jax_backward,
    reshape_flow_pred_to_stencil_stack, reshape_flow_adj_to_stencil_stack,
    flow_stencil_inputs,
)


def main() -> int:
    # Match the PyTorch verify script's setup as closely as possible.
    B = 4096
    dx, dy = 1.0e-3, 1.0e-3
    inv_Lx, inv_Ly = 1.69205e-3, 2.40964e-3
    rho, nu_stage = 1076.0, 1.0e-3
    w_cont = w_momx = w_momy = 1.0

    # Net: 12×512 SiLU Flax.
    key = random.PRNGKey(0)
    net, params = init_flow_params(key, hidden_layers=12, hidden_size=512)
    # Amplify init by 1.5 so FD residuals are non-trivial (mirrors the PyTorch
    # verify). Without this the residuals live near float32 noise and max-rel
    # is inflated by catastrophic cancellation.
    params = jax.tree_util.tree_map(lambda x: x * 1.5, params)
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"[verify_v4_flow_jax_sage] params = {n_params}")

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.random((B, 1), dtype=np.float32))
    y = jnp.asarray(rng.random((B, 1), dtype=np.float32))
    dw = jnp.asarray(rng.random((B, 1), dtype=np.float32) * 0.1)
    sin_ = jnp.asarray(rng.random((B, 1), dtype=np.float32))
    sout_ = jnp.asarray(rng.random((B, 1), dtype=np.float32))

    inv_rho = 1.0 / rho
    inv_Lx2 = inv_Lx * inv_Lx
    inv_Ly2 = inv_Ly * inv_Ly
    inv_2dx = 1.0 / (2.0 * dx); inv_2dy = 1.0 / (2.0 * dy)
    inv_dx2 = 1.0 / (dx * dx); inv_dy2 = 1.0 / (dy * dy)

    # Reference path: jax.grad through the whole stencil forward.
    def loss_auto(params_):
        xy_stack, _ = flow_stencil_inputs(x, y, dw, sin_, sout_, dx, dy)
        pred_all = net.apply(params_, xy_stack)
        pred_stack = reshape_flow_pred_to_stencil_stack(pred_all, B)
        u0 = pred_stack[:, 0:1]; v0 = pred_stack[:, 1:2]; p0 = pred_stack[:, 2:3]
        u_xp = pred_stack[:, 3:4]; v_xp = pred_stack[:, 4:5]; p_xp = pred_stack[:, 5:6]
        u_xm = pred_stack[:, 6:7]; v_xm = pred_stack[:, 7:8]; p_xm = pred_stack[:, 8:9]
        u_yp = pred_stack[:, 9:10]; v_yp = pred_stack[:, 10:11]; p_yp = pred_stack[:, 11:12]
        u_ym = pred_stack[:, 12:13]; v_ym = pred_stack[:, 13:14]; p_ym = pred_stack[:, 14:15]
        du_dx = (u_xp - u_xm) * inv_2dx * inv_Lx
        du_dy = (u_yp - u_ym) * inv_2dy * inv_Ly
        dv_dx = (v_xp - v_xm) * inv_2dx * inv_Lx
        dv_dy = (v_yp - v_ym) * inv_2dy * inv_Ly
        dp_dx = (p_xp - p_xm) * inv_2dx * inv_Lx
        dp_dy = (p_yp - p_ym) * inv_2dy * inv_Ly
        d2u_dx2 = (u_xp + u_xm - 2.0 * u0) * inv_dx2 * inv_Lx2
        d2u_dy2 = (u_yp + u_ym - 2.0 * u0) * inv_dy2 * inv_Ly2
        d2v_dx2 = (v_xp + v_xm - 2.0 * v0) * inv_dx2 * inv_Lx2
        d2v_dy2 = (v_yp + v_ym - 2.0 * v0) * inv_dy2 * inv_Ly2
        cont = du_dx + dv_dy
        mom_x = u0 * du_dx + v0 * du_dy + inv_rho * dp_dx - nu_stage * (d2u_dx2 + d2u_dy2)
        mom_y = u0 * dv_dx + v0 * dv_dy + inv_rho * dp_dy - nu_stage * (d2v_dx2 + d2v_dy2)
        return (w_cont * jnp.mean(cont ** 2)
                + w_momx * jnp.mean(mom_x ** 2)
                + w_momy * jnp.mean(mom_y ** 2))

    loss_val_auto, grad_auto = jax.value_and_grad(loss_auto)(params)
    auto_flat = jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(grad_auto)])

    # SAGE-JAX path.
    sage_backward = build_v4_flow_sage_jax_backward(dx, dy, inv_Lx, inv_Ly, rho)

    def grad_sage_fn(params_):
        xy_stack, _ = flow_stencil_inputs(x, y, dw, sin_, sout_, dx, dy)
        pred_all, vjp = jax.vjp(lambda p: net.apply(p, xy_stack), params_)
        pred_stack = reshape_flow_pred_to_stencil_stack(pred_all, B)
        u0 = pred_stack[:, 0:1]; v0 = pred_stack[:, 1:2]; p0 = pred_stack[:, 2:3]
        u_xp = pred_stack[:, 3:4]; v_xp = pred_stack[:, 4:5]; p_xp = pred_stack[:, 5:6]
        u_xm = pred_stack[:, 6:7]; v_xm = pred_stack[:, 7:8]; p_xm = pred_stack[:, 8:9]
        u_yp = pred_stack[:, 9:10]; v_yp = pred_stack[:, 10:11]; p_yp = pred_stack[:, 11:12]
        u_ym = pred_stack[:, 12:13]; v_ym = pred_stack[:, 13:14]; p_ym = pred_stack[:, 14:15]
        du_dx = (u_xp - u_xm) * inv_2dx * inv_Lx
        du_dy = (u_yp - u_ym) * inv_2dy * inv_Ly
        dv_dx = (v_xp - v_xm) * inv_2dx * inv_Lx
        dv_dy = (v_yp - v_ym) * inv_2dy * inv_Ly
        dp_dx = (p_xp - p_xm) * inv_2dx * inv_Lx
        dp_dy = (p_yp - p_ym) * inv_2dy * inv_Ly
        d2u_dx2 = (u_xp + u_xm - 2.0 * u0) * inv_dx2 * inv_Lx2
        d2u_dy2 = (u_yp + u_ym - 2.0 * u0) * inv_dy2 * inv_Ly2
        d2v_dx2 = (v_xp + v_xm - 2.0 * v0) * inv_dx2 * inv_Lx2
        d2v_dy2 = (v_yp + v_ym - 2.0 * v0) * inv_dy2 * inv_Ly2
        cont = du_dx + dv_dy
        mom_x = u0 * du_dx + v0 * du_dy + inv_rho * dp_dx - nu_stage * (d2u_dx2 + d2u_dy2)
        mom_y = u0 * dv_dx + v0 * dv_dy + inv_rho * dp_dy - nu_stage * (d2v_dx2 + d2v_dy2)
        dc = 2.0 * cont * w_cont / float(B)
        dmu = 2.0 * mom_x * w_momx / float(B)
        dmv = 2.0 * mom_y * w_momy / float(B)
        g = {"nu_stage": nu_stage, "N_all": B}
        adj_stack = sage_backward(pred_stack, g, dc, dmu, dmv)
        adj_pred_all = reshape_flow_adj_to_stencil_stack(adj_stack)
        (pg,) = vjp(adj_pred_all)
        return pg

    grad_sage = grad_sage_fn(params)
    sage_flat = jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(grad_sage)])

    diff = jnp.abs(auto_flat - sage_flat)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ref_max = float(jnp.abs(auto_flat).max())
    ref_mean = float(jnp.abs(auto_flat).mean()) + 1e-30
    rel_max = max_abs / (ref_max + 1e-30)
    rel_mean = mean_abs / ref_mean

    print(f"loss autograd           = {float(loss_val_auto):.6e}")
    print(f"Mean |grad_auto|        = {ref_mean:.3e}")
    print(f"Max  |grad_auto|        = {ref_max:.3e}")
    print(f"Max |diff|              = {max_abs:.3e}")
    print(f"Mean |diff|             = {mean_abs:.3e}")
    print(f"Max |diff| / Max |ref|  = {rel_max:.3e}")
    print(f"Mean |diff| / Mean|ref| = {rel_mean:.3e}")
    ok = (rel_max < 1.0e-2) and (rel_mean < 1.0e-2)
    print("Result:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
