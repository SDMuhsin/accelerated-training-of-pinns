"""Level-2 verification for V4 temp SAGE-JAX advection-diffusion adjoint.

Compares ``jax.grad`` (through the 7-stencil Flax forward graph) against
the SAGE-JAX external-seed backward (adjoint on the (B, 7) stacked
pred → single VJP through the forward).

Usage::

    source env/bin/activate
    CUDA_VISIBLE_DEVICES=1 python scripts/verify_v4_temp_jax_sage.py
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
    TempNetFlax, init_temp_params,
    build_v4_temp_sage_jax_backward,
    reshape_temp_pred_to_stencil_stack, reshape_temp_adj_to_stencil_stack,
    temp_stencil_inputs,
)


def main() -> int:
    B = 2048
    dx, dy, dt_fd = 1.0e-3, 1.0e-3, 5.0e-3
    D, Q = 1.0e-5, 0.0

    key = random.PRNGKey(0)
    net, params = init_temp_params(key, hidden_layers=12, hidden_size=256)
    params = jax.tree_util.tree_map(lambda x: x * 1.5, params)
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"[verify_v4_temp_jax_sage] params = {n_params}")

    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.random((B, 1), dtype=np.float32))
    y = jnp.asarray(rng.random((B, 1), dtype=np.float32))
    t = jnp.asarray(rng.random((B, 1), dtype=np.float32))
    u = jnp.asarray(rng.random((B, 1), dtype=np.float32) * 0.1)
    v = jnp.asarray(rng.random((B, 1), dtype=np.float32) * 0.05)

    def loss_auto(params_):
        xy_stack, _ = temp_stencil_inputs(x, y, t, u, v, dx, dy, dt_fd)
        pred = net.apply(params_, xy_stack)
        stack = reshape_temp_pred_to_stencil_stack(pred, B)
        T0 = stack[:, 0:1]; T_xp = stack[:, 1:2]; T_xm = stack[:, 2:3]
        T_yp = stack[:, 3:4]; T_ym = stack[:, 4:5]
        T_tp = stack[:, 5:6]; T_tm = stack[:, 6:7]
        T_x = (T_xp - T_xm) / (2.0 * dx)
        T_y = (T_yp - T_ym) / (2.0 * dy)
        T_t = (T_tp - T_tm) / (2.0 * dt_fd)
        T_xx = (T_xp + T_xm - 2.0 * T0) / (dx * dx)
        T_yy = (T_yp + T_ym - 2.0 * T0) / (dy * dy)
        r = T_t + u * T_x + v * T_y - D * (T_xx + T_yy) - Q
        return jnp.mean(r ** 2)

    loss_val_auto, grad_auto = jax.value_and_grad(loss_auto)(params)
    auto_flat = jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(grad_auto)])

    sage_backward = build_v4_temp_sage_jax_backward(dx, dy, dt_fd, D, Q)

    def grad_sage_fn(params_):
        xy_stack, _ = temp_stencil_inputs(x, y, t, u, v, dx, dy, dt_fd)
        pred, vjp = jax.vjp(lambda p: net.apply(p, xy_stack), params_)
        stack = reshape_temp_pred_to_stencil_stack(pred, B)
        T0 = stack[:, 0:1]; T_xp = stack[:, 1:2]; T_xm = stack[:, 2:3]
        T_yp = stack[:, 3:4]; T_ym = stack[:, 4:5]
        T_tp = stack[:, 5:6]; T_tm = stack[:, 6:7]
        T_x = (T_xp - T_xm) / (2.0 * dx)
        T_y = (T_yp - T_ym) / (2.0 * dy)
        T_t = (T_tp - T_tm) / (2.0 * dt_fd)
        T_xx = (T_xp + T_xm - 2.0 * T0) / (dx * dx)
        T_yy = (T_yp + T_ym - 2.0 * T0) / (dy * dy)
        r = T_t + u * T_x + v * T_y - D * (T_xx + T_yy) - Q
        dr = 2.0 * r / float(B)
        g = {"u": u, "v": v, "N_all": B}
        adj_stack = sage_backward(stack, g, dr)
        adj_pred = reshape_temp_adj_to_stencil_stack(adj_stack)
        (pg,) = vjp(adj_pred)
        return pg

    grad_sage = grad_sage_fn(params)
    sage_flat = jnp.concatenate([l.ravel() for l in jax.tree_util.tree_leaves(grad_sage)])

    diff = jnp.abs(auto_flat - sage_flat)
    max_abs = float(diff.max()); mean_abs = float(diff.mean())
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
