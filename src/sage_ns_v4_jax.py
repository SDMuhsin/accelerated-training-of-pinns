"""SAGE-JAX drop-in for V4 flow (steady NS) and V4 temp (adv-diff) PDEs.

Parallel to ``sage_ns_v4.py`` (PyTorch) and structurally similar to the
``train_sage_jax_*`` patterns in ``jax_pinn.py``. Provides:

- Flax modules that reproduce V4's ``FullyConnectedArch`` (flow:
  12×512 SiLU, inputs (x, y, dw, sin, sout) → (u, v, p)) and V4's
  ``TemperatureNet`` (12×256 SiLU, inputs (x, y, t, u, v) → T).
- Pure-function NS / adv-diff FD residuals traceable by SAGE.
- Cached SAGE-JAX external-seed backward kernels built via
  ``symbolic_vjp.emit_backward(..., backend='jax', external_seeds=True)``.
- JIT'd train-step closures for flow (5-stage nu schedule) and temp.

The backward kernel receives the stacked (B, 15) stencil pred (NS) or
(B, 7) stencil pred (temp) plus externally-supplied adjoint seeds from
the loss aggregator — MSE ∂loss/∂residual_k = 2·residual·λ_k / B —
and returns the per-column adjoint, which is then routed through a
``jax.vjp`` on the net forward to produce parameter gradients in one
JIT'd pass.

Apples-to-apples note: the only permitted JAX divergence is that
Flax init differs from PyTorch init. Every other knob (lr, schedule,
batch sizes, loss weights, physics constants, 5-stage schedule, sampler
seeds) is threaded in from ``partner_v4_config.yaml`` unchanged. See
``llmdocs/CONTEXT.md`` § 7 for the full constraint list.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

import jax
# Match the rest of the SAGE paper stream: force full fp32 matmul
# precision. (This is already set globally when ``jax_pinn`` is imported
# but set it here too for safety if this file is loaded first.)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
from jax import random
import flax.linen as fnn
import optax

from symbolic_vjp import emit_backward, trace_pde_forward


# ---------------------------------------------------------------------------
# FD stencil step sizes (mirrors the PyTorch SAGE path)
# ---------------------------------------------------------------------------
_SAGE_FLOW_DX = 1.0e-3
_SAGE_FLOW_DY = 1.0e-3
_SAGE_TEMP_DX = 1.0e-3
_SAGE_TEMP_DY = 1.0e-3
_SAGE_TEMP_DT = 5.0e-3

_FLOW_INPUT_NAMES = [
    "u0", "v0", "p0",
    "u_xp", "v_xp", "p_xp",
    "u_xm", "v_xm", "p_xm",
    "u_yp", "v_yp", "p_yp",
    "u_ym", "v_ym", "p_ym",
]

_TEMP_INPUT_NAMES = [
    "T0", "T_xp", "T_xm", "T_yp", "T_ym", "T_tp", "T_tm",
]


# ===========================================================================
# Flax modules mirroring V4 architectures
# ===========================================================================
class FlowNetFlax(fnn.Module):
    """Flax equivalent of V4 ``FullyConnectedArch`` for the flow net.

    Inputs (N, 5): (x, y, dw, sin, sout). Output (N, 3): (u, v, p).
    12 hidden SiLU layers of width 512; final linear to 3 outputs.

    Init differs from PyTorch's ``nn.Linear`` default (Kaiming-uniform);
    Flax's ``fnn.Dense`` defaults to LeCun-normal. That's the only
    permitted apples-to-apples divergence — everything else (depth, width,
    activation, I/O shape) is the baseline's.
    """

    hidden_layers: int = 12
    hidden_size: int = 512

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for _ in range(self.hidden_layers):
            h = fnn.Dense(features=self.hidden_size)(h)
            h = fnn.silu(h)
        h = fnn.Dense(features=3)(h)
        return h


class TempNetFlax(fnn.Module):
    """Flax equivalent of V4 ``TemperatureNet``.

    Inputs (N, 5): (x, y, t, u, v). Output (N, 1): T.
    12 hidden SiLU layers of width 256; final linear to 1 output.
    """

    hidden_layers: int = 12
    hidden_size: int = 256

    @fnn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for _ in range(self.hidden_layers):
            h = fnn.Dense(features=self.hidden_size)(h)
            h = fnn.silu(h)
        h = fnn.Dense(features=1)(h)
        return h


def init_flow_params(rng_key: jnp.ndarray, hidden_layers: int = 12,
                     hidden_size: int = 512):
    """Return (net, params) for the V4 flow architecture, fp32."""
    net = FlowNetFlax(hidden_layers=hidden_layers, hidden_size=hidden_size)
    dummy = jnp.zeros((1, 5), dtype=jnp.float32)
    params = net.init(rng_key, dummy)
    return net, params


def init_temp_params(rng_key: jnp.ndarray, hidden_layers: int = 12,
                     hidden_size: int = 256):
    """Return (net, params) for the V4 temp architecture, fp32."""
    net = TempNetFlax(hidden_layers=hidden_layers, hidden_size=hidden_size)
    dummy = jnp.zeros((1, 5), dtype=jnp.float32)
    params = net.init(rng_key, dummy)
    return net, params


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# ===========================================================================
# FD residual forwards (pure functions — SAGE-traceable and JAX-traceable)
# ===========================================================================
def _make_flow_ns_fd_forward(dx: float, dy: float, inv_Lx: float,
                             inv_Ly: float, rho: float) -> Callable:
    """Return a TracedVar-compatible V4 NS FD forward for SAGE tracing.

    ``nu`` comes through the ``g`` dict as a per-stage TracedVar
    constant so a single generated backward works across all 5 flow
    stages (nu_schedule = [1e-2, 5e-3, 1e-3]).
    """
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_rho = 1.0 / float(rho)
    inv_Lx2 = inv_Lx * inv_Lx
    inv_Ly2 = inv_Ly * inv_Ly

    def compute_fn(pred, g):
        u0 = pred[:, 0:1]; v0 = pred[:, 1:2]; p0 = pred[:, 2:3]
        u_xp = pred[:, 3:4]; v_xp = pred[:, 4:5]; p_xp = pred[:, 5:6]
        u_xm = pred[:, 6:7]; v_xm = pred[:, 7:8]; p_xm = pred[:, 8:9]
        u_yp = pred[:, 9:10]; v_yp = pred[:, 10:11]; p_yp = pred[:, 11:12]
        u_ym = pred[:, 12:13]; v_ym = pred[:, 13:14]; p_ym = pred[:, 14:15]

        nu = g["nu_stage"]

        du_dx = (u_xp - u_xm) * (inv_2dx * inv_Lx)
        du_dy = (u_yp - u_ym) * (inv_2dy * inv_Ly)
        dv_dx = (v_xp - v_xm) * (inv_2dx * inv_Lx)
        dv_dy = (v_yp - v_ym) * (inv_2dy * inv_Ly)
        dp_dx = (p_xp - p_xm) * (inv_2dx * inv_Lx)
        dp_dy = (p_yp - p_ym) * (inv_2dy * inv_Ly)

        d2u_dx2 = (u_xp + u_xm - 2.0 * u0) * (inv_dx2 * inv_Lx2)
        d2u_dy2 = (u_yp + u_ym - 2.0 * u0) * (inv_dy2 * inv_Ly2)
        d2v_dx2 = (v_xp + v_xm - 2.0 * v0) * (inv_dx2 * inv_Lx2)
        d2v_dy2 = (v_yp + v_ym - 2.0 * v0) * (inv_dy2 * inv_Ly2)

        continuity = du_dx + dv_dy
        mom_x = u0 * du_dx + v0 * du_dy + inv_rho * dp_dx \
                - nu * (d2u_dx2 + d2u_dy2)
        mom_y = u0 * dv_dx + v0 * dv_dy + inv_rho * dp_dy \
                - nu * (d2v_dx2 + d2v_dy2)
        return (continuity, mom_x, mom_y)

    return compute_fn


def _make_temp_adv_diff_fd_forward(dx: float, dy: float, dt_fd: float,
                                   D: float, Q: float) -> Callable:
    """Return a TracedVar-compatible V4 temp adv-diff FD forward."""
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_2dt = 1.0 / (2.0 * dt_fd)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    def compute_fn(pred, g):
        T0 = pred[:, 0:1]
        T_xp = pred[:, 1:2]
        T_xm = pred[:, 2:3]
        T_yp = pred[:, 3:4]
        T_ym = pred[:, 4:5]
        T_tp = pred[:, 5:6]
        T_tm = pred[:, 6:7]

        u = g["u"]
        v = g["v"]

        T_x = (T_xp - T_xm) * inv_2dx
        T_y = (T_yp - T_ym) * inv_2dy
        T_t = (T_tp - T_tm) * inv_2dt
        T_xx = (T_xp + T_xm - 2.0 * T0) * inv_dx2
        T_yy = (T_yp + T_ym - 2.0 * T0) * inv_dy2

        residual = T_t + u * T_x + v * T_y - D * (T_xx + T_yy) - Q
        return (residual,)

    return compute_fn


# ===========================================================================
# SAGE-JAX backward kernel builders (cached on FD/physics scalars)
# ===========================================================================
_cached_flow_sage_jax_backward: Dict[Tuple[float, ...], object] = {}
_cached_temp_sage_jax_backward: Dict[Tuple[float, ...], object] = {}


def build_v4_flow_sage_jax_backward(dx: float, dy: float, inv_Lx: float,
                                    inv_Ly: float, rho: float):
    """Build (and cache) the JAX-backend SAGE external-seed backward for V4 NS.

    Signature: ``fn(pred_stack, g, dc, dmu, dmv) -> adj_pred`` where
    ``pred_stack`` is (B, 15) with columns ordered per ``_FLOW_INPUT_NAMES``,
    ``g`` must contain ``'nu_stage'`` (scalar) and ``'N_all'`` (= B), and
    ``dc, dmu, dmv`` are the externally-supplied ∂loss/∂residual seeds.
    """
    key = (float(dx), float(dy), float(inv_Lx), float(inv_Ly), float(rho))
    cached = _cached_flow_sage_jax_backward.get(key)
    if cached is not None:
        return cached

    compute_fn = _make_flow_ns_fd_forward(*key)
    tape: list = []
    outputs, inputs = trace_pde_forward(
        compute_fn,
        N_all=None,
        tape=tape,
        sparse=False,
        constants=["nu_stage"],
        input_names=_FLOW_INPUT_NAMES,
    )
    _source, fn = emit_backward(
        tape,
        list(outputs),
        seed_names=["dc", "dmu", "dmv"],
        input_vars=inputs,
        sparse=False,
        func_name="generated_v4_ns_backward_jax",
        input_names=_FLOW_INPUT_NAMES,
        backend="jax",
        external_seeds=True,
    )
    print(
        f"[SAGE-JAX V4 NS] Built external-seed backward: {len(tape)} tape "
        f"ops; dx={dx:g}, dy={dy:g}, invLx={inv_Lx:g}, invLy={inv_Ly:g}, "
        f"rho={rho:g}"
    )
    _cached_flow_sage_jax_backward[key] = fn
    return fn


def build_v4_temp_sage_jax_backward(dx: float, dy: float, dt_fd: float,
                                    D: float, Q: float):
    """Build (and cache) the JAX-backend SAGE external-seed backward for V4 temp."""
    key = (float(dx), float(dy), float(dt_fd), float(D), float(Q))
    cached = _cached_temp_sage_jax_backward.get(key)
    if cached is not None:
        return cached

    compute_fn = _make_temp_adv_diff_fd_forward(*key)
    tape: list = []
    outputs, inputs = trace_pde_forward(
        compute_fn,
        N_all=None,
        tape=tape,
        sparse=False,
        constants=["u", "v"],
        input_names=_TEMP_INPUT_NAMES,
    )
    _source, fn = emit_backward(
        tape,
        list(outputs),
        seed_names=["dr"],
        input_vars=inputs,
        sparse=False,
        func_name="generated_v4_temp_backward_jax",
        input_names=_TEMP_INPUT_NAMES,
        backend="jax",
        external_seeds=True,
    )
    print(
        f"[SAGE-JAX V4 temp] Built external-seed backward: {len(tape)} tape "
        f"ops; dx={dx:g}, dy={dy:g}, dt={dt_fd:g}, D={D:g}, Q={Q:g}"
    )
    _cached_temp_sage_jax_backward[key] = fn
    return fn


# ===========================================================================
# Public helpers: stack stencil inputs + run SAGE adjoint
# ===========================================================================
def flow_stencil_inputs(x: jnp.ndarray, y: jnp.ndarray, dw: jnp.ndarray,
                        sin_: jnp.ndarray, sout_: jnp.ndarray,
                        dx: float, dy: float) -> jnp.ndarray:
    """Stack the 5 stencil (x, y, dw, sin, sout) inputs into a (5*B, 5) array.

    Order: centre, x+dx, x-dx, y+dy, y-dy.
    """
    B = x.shape[0]
    dw_rep = jnp.tile(dw, (5, 1))
    sin_rep = jnp.tile(sin_, (5, 1))
    sout_rep = jnp.tile(sout_, (5, 1))
    x_all = jnp.concatenate([x, x + dx, x - dx, x, x], axis=0)
    y_all = jnp.concatenate([y, y, y, y + dy, y - dy], axis=0)
    return jnp.concatenate([x_all, y_all, dw_rep, sin_rep, sout_rep], axis=1), B


def reshape_flow_pred_to_stencil_stack(pred_all: jnp.ndarray, B: int) -> jnp.ndarray:
    """Given (5B, 3) flow net output (ordered centre, xp, xm, yp, ym), return (B, 15).

    Column order matches ``_FLOW_INPUT_NAMES``.
    """
    out_c = pred_all[0:B]
    out_xp = pred_all[B:2 * B]
    out_xm = pred_all[2 * B:3 * B]
    out_yp = pred_all[3 * B:4 * B]
    out_ym = pred_all[4 * B:5 * B]
    return jnp.concatenate(
        [out_c, out_xp, out_xm, out_yp, out_ym], axis=1
    )


def reshape_flow_adj_to_stencil_stack(adj_stack: jnp.ndarray) -> jnp.ndarray:
    """Inverse of :func:`reshape_flow_pred_to_stencil_stack`.

    Input (B, 15) adjoint stack → (5B, 3) ordered (centre, xp, xm, yp, ym).
    """
    B = adj_stack.shape[0]
    adj_c = adj_stack[:, 0:3]
    adj_xp = adj_stack[:, 3:6]
    adj_xm = adj_stack[:, 6:9]
    adj_yp = adj_stack[:, 9:12]
    adj_ym = adj_stack[:, 12:15]
    return jnp.concatenate([adj_c, adj_xp, adj_xm, adj_yp, adj_ym], axis=0)


def temp_stencil_inputs(x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray,
                        u: jnp.ndarray, v: jnp.ndarray,
                        dx: float, dy: float, dt_fd: float) -> jnp.ndarray:
    """Stack the 7 stencil (x, y, t, u, v) inputs into a (7B, 5) array.

    Order: centre, x+dx, x-dx, y+dy, y-dy, t+dt, t-dt. ``u, v`` are held
    fixed at the centre value (matching the PyTorch SAGE semantics: they
    are data features, not differentiated against).
    """
    B = x.shape[0]
    u_rep = jnp.tile(u, (7, 1))
    v_rep = jnp.tile(v, (7, 1))
    x_all = jnp.concatenate([x, x + dx, x - dx, x, x, x, x], axis=0)
    y_all = jnp.concatenate([y, y, y, y + dy, y - dy, y, y], axis=0)
    t_all = jnp.concatenate([t, t, t, t, t, t + dt_fd, t - dt_fd], axis=0)
    return jnp.concatenate([x_all, y_all, t_all, u_rep, v_rep], axis=1), B


def reshape_temp_pred_to_stencil_stack(pred_all: jnp.ndarray, B: int) -> jnp.ndarray:
    """Given (7B, 1) temp net output, return (B, 7) in ``_TEMP_INPUT_NAMES`` order."""
    return jnp.concatenate(
        [pred_all[k * B:(k + 1) * B] for k in range(7)], axis=1
    )


def reshape_temp_adj_to_stencil_stack(adj_stack: jnp.ndarray) -> jnp.ndarray:
    """Inverse of :func:`reshape_temp_pred_to_stencil_stack`."""
    return jnp.concatenate(
        [adj_stack[:, k:k + 1] for k in range(7)], axis=0
    )


# ===========================================================================
# Optimizer helpers — match V4 baseline exactly
# ===========================================================================
def make_flow_optimizer(lr: float = 5.0e-5, lr_decay_rate: float = 0.997,
                        lr_decay_steps: int = 500, grad_clip: float = 1.0,
                        betas: Tuple[float, float] = (0.9, 0.999),
                        eps: float = 1.0e-8,
                        weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Build the V4 flow optimizer: grad-clip → Adam(ExponentialLR).

    PyTorch's ``ExponentialLR(gamma, step_size=500)`` decays the lr by
    ``gamma`` every ``step_size`` steps. ``optax.exponential_decay`` with
    ``transition_steps=500, decay_rate=gamma, staircase=True`` reproduces
    that exactly.
    """
    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=int(lr_decay_steps),
        decay_rate=float(lr_decay_rate),
        staircase=True,
    )
    if weight_decay > 0.0:
        opt = optax.chain(
            optax.clip_by_global_norm(float(grad_clip)),
            optax.adamw(learning_rate=schedule, b1=float(betas[0]),
                        b2=float(betas[1]), eps=float(eps),
                        weight_decay=float(weight_decay)),
        )
    else:
        opt = optax.chain(
            optax.clip_by_global_norm(float(grad_clip)),
            optax.adam(learning_rate=schedule, b1=float(betas[0]),
                       b2=float(betas[1]), eps=float(eps)),
        )
    return opt


def make_temp_optimizer(lr: float = 5.0e-4, lr_decay_rate: float = 0.99,
                        lr_decay_steps: int = 500, grad_clip: float = 1.0,
                        betas: Tuple[float, float] = (0.9, 0.999),
                        eps: float = 1.0e-8,
                        weight_decay: float = 0.0) -> optax.GradientTransformation:
    """Build the V4 temp optimizer: grad-clip → Adam(StepLR).

    PyTorch uses ``StepLR`` for temp (step every ``lr_decay_steps``,
    multiply by ``gamma``). This is the same staircase decay pattern as
    flow's ``ExponentialLR``, just with a different gamma.
    """
    return make_flow_optimizer(
        lr=lr, lr_decay_rate=lr_decay_rate,
        lr_decay_steps=lr_decay_steps, grad_clip=grad_clip,
        betas=betas, eps=eps, weight_decay=weight_decay,
    )


# ===========================================================================
# JIT'd train-step closures — flow and temp
# ===========================================================================
def make_flow_train_step(net: FlowNetFlax, optimizer: optax.GradientTransformation,
                         sage_backward, dx: float, dy: float,
                         inv_Lx: float, inv_Ly: float, rho: float,
                         w_pde_continuity: float = 1.0,
                         w_pde_momentum_x: float = 1.0,
                         w_pde_momentum_y: float = 1.0):
    """Build the per-step closure used by all 5 flow stages.

    Signature: ``train_step(params, opt_state, batch_dict) -> (params,
    opt_state, aux_losses)`` where ``batch_dict`` carries the full set
    of constraint minibatches plus ``nu_stage`` (per-stage scalar) and
    per-constraint lambda weights. The train step is JIT'd; lambda
    weights must be floats or broadcastable arrays in the pytree.

    This closure reimplements PhysicsNeMo's Solver role by hand:
    sampler draws are done outside (NumPy side) and passed as arrays;
    this function computes the forward, assembles upstream adjoints for
    each constraint (SAGE for NS PDE, MSE for the 14 non-PDE terms),
    routes them through ``jax.vjp`` once for efficient param-grad
    aggregation, then applies ``optax.adam + clip + exp_decay``.
    """
    apply_fn = net.apply
    inv_rho = 1.0 / float(rho)
    inv_Lx2 = inv_Lx * inv_Lx
    inv_Ly2 = inv_Ly * inv_Ly

    @jax.jit
    def train_step(params, opt_state, batch):
        # Pull per-step minibatches out of the batch pytree.
        pde = batch["pde"]                    # stacked 5-stencil, (5*B_pde, 5)
        B_pde = int(pde.shape[0]) // 5        # static after JIT trace
        wall = batch["wall"]                  # (B_wall, 5)
        inlet = batch["inlet"]                # (B_in, 5)
        init_soft = batch.get("init_soft")   # (B_init, 5) or None
        geo_dir = batch.get("geo_dir")       # (B_g, 5)
        geo_parallel = batch.get("geo_par")  # (B_g, 5)
        wall_guard = batch.get("wall_guard")  # (B_wg, 5)
        wall_guard_sep = batch.get("wall_guard_sep")  # (B_wgs, 5)
        inlet_p = batch.get("inlet_p")       # (B_in_p, 5) optional
        outlet_p = batch.get("outlet_p")     # (B_out_p, 5) optional

        # Targets and lambda weights.
        wall_t = batch["wall_target"]         # (B_wall, 2)  (u, v)
        inlet_t = batch["inlet_target"]       # (B_in, 2)
        nu_stage = batch["nu_stage"]          # scalar

        lam_cont = batch["lam_cont"]
        lam_momx = batch["lam_momx"]
        lam_momy = batch["lam_momy"]
        lam_wall = batch["lam_wall"]          # (B_wall, 2)
        lam_inlet = batch["lam_inlet"]        # (B_in, 2)

        # Build the concatenated forward batch.
        parts = [pde, wall, inlet]
        offsets = {"pde": (0, 5 * B_pde), "wall": (5 * B_pde, 5 * B_pde + wall.shape[0]),
                   "inlet": (5 * B_pde + wall.shape[0], 5 * B_pde + wall.shape[0] + inlet.shape[0])}
        cur = offsets["inlet"][1]

        if init_soft is not None:
            parts.append(init_soft)
            offsets["init_soft"] = (cur, cur + init_soft.shape[0])
            cur = offsets["init_soft"][1]
            init_soft_t = batch["init_soft_target"]       # (B_init, 3)
            lam_init_soft = batch["lam_init_soft"]        # (B_init, 3)

        if geo_dir is not None:
            parts.append(geo_dir)
            offsets["geo_dir"] = (cur, cur + geo_dir.shape[0])
            cur = offsets["geo_dir"][1]
            geo_dir_gxgy = batch["geo_dir_gxgy"]          # (B_g, 2)
            geo_dir_target = batch["geo_dir_target"]       # (B_g, 2)  [cross_t=0, cosine_t=1]
            lam_geo_cross = batch["lam_geo_cross"]        # (B_g, 1)
            lam_geo_cosine = batch["lam_geo_cosine"]      # (B_g, 1)

        if geo_parallel is not None:
            parts.append(geo_parallel)
            offsets["geo_par"] = (cur, cur + geo_parallel.shape[0])
            cur = offsets["geo_par"][1]
            geo_par_gxgy = batch["geo_par_gxgy"]          # (B_g, 2)
            geo_par_target = batch["geo_par_target"]       # (B_g, 1)
            lam_geo_parallel = batch["lam_geo_parallel"]  # (B_g, 1)

        if wall_guard is not None:
            parts.append(wall_guard)
            offsets["wall_guard"] = (cur, cur + wall_guard.shape[0])
            cur = offsets["wall_guard"][1]
            wall_guard_n = batch["wall_guard_n"]          # (B_wg, 2) (nx, ny)
            lam_wall_guard = batch["lam_wall_guard"]      # (B_wg, 1)

        if wall_guard_sep is not None:
            parts.append(wall_guard_sep)
            offsets["wall_guard_sep"] = (cur, cur + wall_guard_sep.shape[0])
            cur = offsets["wall_guard_sep"][1]
            wall_guard_sep_n = batch["wall_guard_sep_n"]
            lam_wall_guard_sep = batch["lam_wall_guard_sep"]

        if inlet_p is not None:
            parts.append(inlet_p)
            offsets["inlet_p"] = (cur, cur + inlet_p.shape[0])
            cur = offsets["inlet_p"][1]
            inlet_p_target = batch["inlet_p_target"]  # (B_in_p, 1)
            lam_inlet_p = batch["lam_inlet_p"]        # (B_in_p, 1)

        if outlet_p is not None:
            parts.append(outlet_p)
            offsets["outlet_p"] = (cur, cur + outlet_p.shape[0])
            cur = offsets["outlet_p"][1]
            outlet_p_target = batch["outlet_p_target"]
            lam_outlet_p = batch["lam_outlet_p"]

        xy_batched = jnp.concatenate(parts, axis=0)  # (Ntotal, 5)

        def forward(p):
            return apply_fn(p, xy_batched)

        pred_all, vjp_fn = jax.vjp(forward, params)  # (Ntotal, 3)

        # Assemble upstream adjoints. Same ordering as the forward batch.
        upstream_parts = []

        # ------------------- PDE (SAGE-JAX) -------------------
        pde_pred = pred_all[offsets["pde"][0]:offsets["pde"][1]]  # (5B, 3)
        pred_stack = reshape_flow_pred_to_stencil_stack(pde_pred, B_pde)  # (B, 15)

        # Re-compute NS residuals from the stacked pred (for loss logging + seed).
        u0 = pred_stack[:, 0:1]; v0 = pred_stack[:, 1:2]; p0 = pred_stack[:, 2:3]
        u_xp = pred_stack[:, 3:4]; v_xp = pred_stack[:, 4:5]; p_xp = pred_stack[:, 5:6]
        u_xm = pred_stack[:, 6:7]; v_xm = pred_stack[:, 7:8]; p_xm = pred_stack[:, 8:9]
        u_yp = pred_stack[:, 9:10]; v_yp = pred_stack[:, 10:11]; p_yp = pred_stack[:, 11:12]
        u_ym = pred_stack[:, 12:13]; v_ym = pred_stack[:, 13:14]; p_ym = pred_stack[:, 14:15]

        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        du_dx = (u_xp - u_xm) * (inv_2dx * inv_Lx)
        du_dy = (u_yp - u_ym) * (inv_2dy * inv_Ly)
        dv_dx = (v_xp - v_xm) * (inv_2dx * inv_Lx)
        dv_dy = (v_yp - v_ym) * (inv_2dy * inv_Ly)
        dp_dx = (p_xp - p_xm) * (inv_2dx * inv_Lx)
        dp_dy = (p_yp - p_ym) * (inv_2dy * inv_Ly)
        d2u_dx2 = (u_xp + u_xm - 2.0 * u0) * (inv_dx2 * inv_Lx2)
        d2u_dy2 = (u_yp + u_ym - 2.0 * u0) * (inv_dy2 * inv_Ly2)
        d2v_dx2 = (v_xp + v_xm - 2.0 * v0) * (inv_dx2 * inv_Lx2)
        d2v_dy2 = (v_yp + v_ym - 2.0 * v0) * (inv_dy2 * inv_Ly2)

        continuity = du_dx + dv_dy
        momentum_x = u0 * du_dx + v0 * du_dy + inv_rho * dp_dx \
                     - nu_stage * (d2u_dx2 + d2u_dy2)
        momentum_y = u0 * dv_dx + v0 * dv_dy + inv_rho * dp_dy \
                     - nu_stage * (d2v_dx2 + d2v_dy2)

        loss_pde_cont = jnp.mean(continuity ** 2 * lam_cont)
        loss_pde_momx = jnp.mean(momentum_x ** 2 * lam_momx)
        loss_pde_momy = jnp.mean(momentum_y ** 2 * lam_momy)

        # External seeds for SAGE: ∂(mean(r^2 * lam_k))/∂r_k = 2 * r_k * lam_k / B.
        dc = 2.0 * continuity * lam_cont / float(B_pde)
        dmu = 2.0 * momentum_x * lam_momx / float(B_pde)
        dmv = 2.0 * momentum_y * lam_momy / float(B_pde)

        # Apply top-level loss weights.
        dc = dc * float(w_pde_continuity)
        dmu = dmu * float(w_pde_momentum_x)
        dmv = dmv * float(w_pde_momentum_y)

        g_sage = {"nu_stage": nu_stage, "N_all": B_pde}
        adj_stack = sage_backward(pred_stack, g_sage, dc, dmu, dmv)  # (B, 15)
        adj_pde = reshape_flow_adj_to_stencil_stack(adj_stack)        # (5B, 3)
        upstream_parts.append(adj_pde)

        # ------------------- Wall (MSE on u, v) -------------------
        wall_pred = pred_all[offsets["wall"][0]:offsets["wall"][1]]  # (B_wall, 3)
        B_wall = wall.shape[0]
        wall_diff = wall_pred[:, 0:2] - wall_t  # (B_wall, 2) over (u, v)
        loss_wall = jnp.mean(wall_diff ** 2 * lam_wall)
        adj_wall_uv = 2.0 * wall_diff * lam_wall / float(B_wall)  # (B_wall, 2)
        adj_wall_p = jnp.zeros_like(wall_pred[:, 2:3])
        adj_wall = jnp.concatenate([adj_wall_uv, adj_wall_p], axis=1)
        upstream_parts.append(adj_wall)

        # ------------------- Inlet velocity (MSE on u, v) -------------------
        inlet_pred = pred_all[offsets["inlet"][0]:offsets["inlet"][1]]
        B_in = inlet.shape[0]
        inlet_diff = inlet_pred[:, 0:2] - inlet_t
        loss_inlet = jnp.mean(inlet_diff ** 2 * lam_inlet)
        adj_inlet_uv = 2.0 * inlet_diff * lam_inlet / float(B_in)
        adj_inlet_p = jnp.zeros_like(inlet_pred[:, 2:3])
        adj_inlet = jnp.concatenate([adj_inlet_uv, adj_inlet_p], axis=1)
        upstream_parts.append(adj_inlet)

        # ------------------- init_field_fit / soft_init (MSE on u, v, p) -------------------
        if init_soft is not None:
            is_pred = pred_all[offsets["init_soft"][0]:offsets["init_soft"][1]]
            B_is = init_soft.shape[0]
            is_diff = is_pred - init_soft_t  # (B_is, 3)
            adj_is = 2.0 * is_diff * lam_init_soft / float(B_is)  # (B_is, 3)
            upstream_parts.append(adj_is)
            loss_init_soft = jnp.mean(is_diff ** 2 * lam_init_soft)
        else:
            loss_init_soft = jnp.asarray(0.0, dtype=jnp.float32)

        # ------------------- geo_dir (MSE on cross=0, cosine=1) -------------------
        if geo_dir is not None:
            gd_pred = pred_all[offsets["geo_dir"][0]:offsets["geo_dir"][1]]
            B_gd = geo_dir.shape[0]
            u_g = gd_pred[:, 0:1]; v_g = gd_pred[:, 1:2]
            gx = geo_dir_gxgy[:, 0:1]; gy = geo_dir_gxgy[:, 1:2]
            cross = -u_g * gy + v_g * gx
            speed_eps2 = 1.0e-8  # matches speed_eps^2 from baseline (1e-4 default)
            speed = jnp.sqrt(u_g * u_g + v_g * v_g + speed_eps2)
            parallel = u_g * gx + v_g * gy
            cosine = parallel / speed

            cross_diff = cross - geo_dir_target[:, 0:1]
            cosine_diff = cosine - geo_dir_target[:, 1:2]
            loss_geo_cross = jnp.mean(cross_diff ** 2 * lam_geo_cross)
            loss_geo_cosine = jnp.mean(cosine_diff ** 2 * lam_geo_cosine)

            # ∂loss/∂u_g, ∂loss/∂v_g via chain rule on (cross, cosine).
            dcross_du = -gy
            dcross_dv = gx
            one_over_speed = 1.0 / speed
            dcosine_du = (gx * speed - parallel * (u_g * one_over_speed)) / (speed * speed)
            dcosine_dv = (gy * speed - parallel * (v_g * one_over_speed)) / (speed * speed)

            seed_cross = 2.0 * cross_diff * lam_geo_cross / float(B_gd)
            seed_cosine = 2.0 * cosine_diff * lam_geo_cosine / float(B_gd)
            adj_gd_u = seed_cross * dcross_du + seed_cosine * dcosine_du
            adj_gd_v = seed_cross * dcross_dv + seed_cosine * dcosine_dv
            adj_gd_p = jnp.zeros_like(gd_pred[:, 2:3])
            adj_gd = jnp.concatenate([adj_gd_u, adj_gd_v, adj_gd_p], axis=1)
            upstream_parts.append(adj_gd)
        else:
            loss_geo_cross = jnp.asarray(0.0, dtype=jnp.float32)
            loss_geo_cosine = jnp.asarray(0.0, dtype=jnp.float32)

        # ------------------- geo_parallel (MSE on parallel=|v_target|) -------------------
        if geo_parallel is not None:
            gp_pred = pred_all[offsets["geo_par"][0]:offsets["geo_par"][1]]
            B_gp = geo_parallel.shape[0]
            u_g = gp_pred[:, 0:1]; v_g = gp_pred[:, 1:2]
            gx = geo_par_gxgy[:, 0:1]; gy = geo_par_gxgy[:, 1:2]
            parallel = u_g * gx + v_g * gy  # scalar residual
            par_diff = parallel - geo_par_target  # (B_gp, 1)
            loss_geo_parallel = jnp.mean(par_diff ** 2 * lam_geo_parallel)
            seed_par = 2.0 * par_diff * lam_geo_parallel / float(B_gp)
            adj_gp_u = seed_par * gx
            adj_gp_v = seed_par * gy
            adj_gp_p = jnp.zeros_like(gp_pred[:, 2:3])
            adj_gp = jnp.concatenate([adj_gp_u, adj_gp_v, adj_gp_p], axis=1)
            upstream_parts.append(adj_gp)
        else:
            loss_geo_parallel = jnp.asarray(0.0, dtype=jnp.float32)

        # ------------------- wall_guard (MSE on u*nx + v*ny = 0) -------------------
        if wall_guard is not None:
            wg_pred = pred_all[offsets["wall_guard"][0]:offsets["wall_guard"][1]]
            B_wg = wall_guard.shape[0]
            u_w = wg_pred[:, 0:1]; v_w = wg_pred[:, 1:2]
            nx = wall_guard_n[:, 0:1]; ny = wall_guard_n[:, 1:2]
            guard_resid = u_w * nx + v_w * ny  # target = 0
            loss_wall_guard = jnp.mean(guard_resid ** 2 * lam_wall_guard)
            seed_g = 2.0 * guard_resid * lam_wall_guard / float(B_wg)
            adj_wg = jnp.concatenate(
                [seed_g * nx, seed_g * ny, jnp.zeros_like(wg_pred[:, 2:3])],
                axis=1,
            )
            upstream_parts.append(adj_wg)
        else:
            loss_wall_guard = jnp.asarray(0.0, dtype=jnp.float32)

        # ------------------- wall_guard_sep (same form, different batch) -------------------
        if wall_guard_sep is not None:
            wgs_pred = pred_all[offsets["wall_guard_sep"][0]:offsets["wall_guard_sep"][1]]
            B_wgs = wall_guard_sep.shape[0]
            u_w = wgs_pred[:, 0:1]; v_w = wgs_pred[:, 1:2]
            nx = wall_guard_sep_n[:, 0:1]; ny = wall_guard_sep_n[:, 1:2]
            guard_resid = u_w * nx + v_w * ny
            loss_wall_guard_sep = jnp.mean(guard_resid ** 2 * lam_wall_guard_sep)
            seed_g = 2.0 * guard_resid * lam_wall_guard_sep / float(B_wgs)
            adj_wgs = jnp.concatenate(
                [seed_g * nx, seed_g * ny, jnp.zeros_like(wgs_pred[:, 2:3])],
                axis=1,
            )
            upstream_parts.append(adj_wgs)
        else:
            loss_wall_guard_sep = jnp.asarray(0.0, dtype=jnp.float32)

        # ------------------- inlet_p anchor (MSE on p) -------------------
        if inlet_p is not None:
            ip_pred = pred_all[offsets["inlet_p"][0]:offsets["inlet_p"][1]]
            B_ip = inlet_p.shape[0]
            ip_diff = ip_pred[:, 2:3] - inlet_p_target
            loss_inlet_p = jnp.mean(ip_diff ** 2 * lam_inlet_p)
            adj_ip_p = 2.0 * ip_diff * lam_inlet_p / float(B_ip)
            adj_ip = jnp.concatenate(
                [jnp.zeros_like(ip_pred[:, 0:1]), jnp.zeros_like(ip_pred[:, 1:2]), adj_ip_p],
                axis=1,
            )
            upstream_parts.append(adj_ip)
        else:
            loss_inlet_p = jnp.asarray(0.0, dtype=jnp.float32)

        # ------------------- outlet_p anchor (MSE on p) -------------------
        if outlet_p is not None:
            op_pred = pred_all[offsets["outlet_p"][0]:offsets["outlet_p"][1]]
            B_op = outlet_p.shape[0]
            op_diff = op_pred[:, 2:3] - outlet_p_target
            loss_outlet_p = jnp.mean(op_diff ** 2 * lam_outlet_p)
            adj_op_p = 2.0 * op_diff * lam_outlet_p / float(B_op)
            adj_op = jnp.concatenate(
                [jnp.zeros_like(op_pred[:, 0:1]), jnp.zeros_like(op_pred[:, 1:2]), adj_op_p],
                axis=1,
            )
            upstream_parts.append(adj_op)
        else:
            loss_outlet_p = jnp.asarray(0.0, dtype=jnp.float32)

        upstream = jnp.concatenate(upstream_parts, axis=0)  # (Ntotal, 3)
        (param_grads,) = vjp_fn(upstream)

        # Total loss for logging — sum of per-term MSE values.
        total_loss = (
            float(w_pde_continuity) * loss_pde_cont
            + float(w_pde_momentum_x) * loss_pde_momx
            + float(w_pde_momentum_y) * loss_pde_momy
            + loss_wall
            + loss_inlet
            + loss_init_soft
            + loss_geo_cross + loss_geo_cosine
            + loss_geo_parallel
            + loss_wall_guard
            + loss_wall_guard_sep
            + loss_inlet_p
            + loss_outlet_p
        )

        updates, new_opt_state = optimizer.update(param_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        aux = {
            "total": total_loss,
            "pde_cont": loss_pde_cont,
            "pde_momx": loss_pde_momx,
            "pde_momy": loss_pde_momy,
            "wall": loss_wall,
            "inlet": loss_inlet,
            "init_soft": loss_init_soft,
            "geo_cross": loss_geo_cross,
            "geo_cosine": loss_geo_cosine,
            "geo_parallel": loss_geo_parallel,
            "wall_guard": loss_wall_guard,
            "wall_guard_sep": loss_wall_guard_sep,
            "inlet_p": loss_inlet_p,
            "outlet_p": loss_outlet_p,
        }
        return new_params, new_opt_state, aux

    return train_step


def make_flow_train_step_warmup_only(net: FlowNetFlax,
                                     optimizer: optax.GradientTransformation):
    """Stage −1 init-field-fit: only the pseudo-label regression is active.

    Matches baseline behaviour where the PDE node is pruned by Graph's
    necessary-nodes walk because its outputs are disjoint from the
    ``init_field_fit`` constraint's required outputs (u, v, p).
    """
    apply_fn = net.apply

    @jax.jit
    def train_step(params, opt_state, batch):
        init_soft = batch["init_soft"]            # (B, 5)
        init_soft_t = batch["init_soft_target"]   # (B, 3)
        lam = batch["lam_init_soft"]              # (B, 3)

        def loss_fn(p):
            pred = apply_fn(p, init_soft)
            diff = pred - init_soft_t
            return jnp.mean(diff ** 2 * lam)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, {"total": loss, "init_soft": loss}

    return train_step


def make_flow_train_step_bc_only(net: FlowNetFlax,
                                 optimizer: optax.GradientTransformation,
                                 w_wall_u: float = 1.0, w_wall_v: float = 1.0,
                                 w_inlet_u: float = 1.0, w_inlet_v: float = 1.0,
                                 w_outlet_p: float = 1.0, w_inlet_p: float = 1.0,
                                 w_soft_init_u: float = 0.2,
                                 w_soft_init_v: float = 0.2,
                                 w_soft_init_p: float = 0.2,
                                 w_geo_cross: float = 0.2,
                                 w_geo_cosine: float = 0.1,
                                 w_geo_parallel: float = 0.05,
                                 w_wall_guard_normal: float = 0.5,
                                 w_wall_guard_sep_normal: float = 0.5,
                                 has_outlet_p: bool = False,
                                 has_inlet_p: bool = False,
                                 has_soft_init: bool = True,
                                 has_geo_dir: bool = True,
                                 has_geo_parallel: bool = True,
                                 has_wall_guard: bool = True,
                                 has_wall_guard_sep: bool = True):
    """Stage 0 BC-warmup: all constraints EXCEPT the NS PDE are active.

    Used only for stage 0 where the PDE is not in the domain. Everything
    else (wall/inlet/outlet/soft-init/geo/wall-guard) behaves exactly
    like the stage-1+ closure.
    """
    apply_fn = net.apply

    @jax.jit
    def train_step(params, opt_state, batch):
        wall = batch["wall"]
        inlet = batch["inlet"]
        init_soft = batch["init_soft"] if has_soft_init else None
        geo_dir = batch["geo_dir"] if has_geo_dir else None
        geo_parallel = batch["geo_par"] if has_geo_parallel else None
        wall_guard = batch["wall_guard"] if has_wall_guard else None
        wall_guard_sep = batch["wall_guard_sep"] if has_wall_guard_sep else None
        inlet_p = batch["inlet_p"] if has_inlet_p else None
        outlet_p = batch["outlet_p"] if has_outlet_p else None

        parts = [wall, inlet]
        offs = {"wall": (0, wall.shape[0]), "inlet": (wall.shape[0], wall.shape[0] + inlet.shape[0])}
        cur = offs["inlet"][1]
        if init_soft is not None:
            parts.append(init_soft)
            offs["init_soft"] = (cur, cur + init_soft.shape[0]); cur = offs["init_soft"][1]
        if geo_dir is not None:
            parts.append(geo_dir)
            offs["geo_dir"] = (cur, cur + geo_dir.shape[0]); cur = offs["geo_dir"][1]
        if geo_parallel is not None:
            parts.append(geo_parallel)
            offs["geo_par"] = (cur, cur + geo_parallel.shape[0]); cur = offs["geo_par"][1]
        if wall_guard is not None:
            parts.append(wall_guard)
            offs["wall_guard"] = (cur, cur + wall_guard.shape[0]); cur = offs["wall_guard"][1]
        if wall_guard_sep is not None:
            parts.append(wall_guard_sep)
            offs["wall_guard_sep"] = (cur, cur + wall_guard_sep.shape[0]); cur = offs["wall_guard_sep"][1]
        if inlet_p is not None:
            parts.append(inlet_p)
            offs["inlet_p"] = (cur, cur + inlet_p.shape[0]); cur = offs["inlet_p"][1]
        if outlet_p is not None:
            parts.append(outlet_p)
            offs["outlet_p"] = (cur, cur + outlet_p.shape[0]); cur = offs["outlet_p"][1]

        xy_batched = jnp.concatenate(parts, axis=0)

        def loss_fn(p):
            pred = apply_fn(p, xy_batched)
            total = jnp.asarray(0.0, dtype=jnp.float32)
            aux = {}

            wall_pred = pred[offs["wall"][0]:offs["wall"][1]]
            wall_t = batch["wall_target"]
            lam_w = batch["lam_wall"]
            diff = wall_pred[:, 0:2] - wall_t
            l_wall = jnp.mean(diff ** 2 * lam_w)
            total = total + l_wall
            aux["wall"] = l_wall

            inlet_pred = pred[offs["inlet"][0]:offs["inlet"][1]]
            inlet_t = batch["inlet_target"]
            lam_i = batch["lam_inlet"]
            diff = inlet_pred[:, 0:2] - inlet_t
            l_inlet = jnp.mean(diff ** 2 * lam_i)
            total = total + l_inlet
            aux["inlet"] = l_inlet

            if init_soft is not None:
                is_pred = pred[offs["init_soft"][0]:offs["init_soft"][1]]
                diff = is_pred - batch["init_soft_target"]
                lam = batch["lam_init_soft"]
                l_is = jnp.mean(diff ** 2 * lam)
                total = total + l_is
                aux["init_soft"] = l_is
            else:
                aux["init_soft"] = jnp.asarray(0.0, dtype=jnp.float32)

            if geo_dir is not None:
                gd_pred = pred[offs["geo_dir"][0]:offs["geo_dir"][1]]
                u_g = gd_pred[:, 0:1]; v_g = gd_pred[:, 1:2]
                gx = batch["geo_dir_gxgy"][:, 0:1]; gy = batch["geo_dir_gxgy"][:, 1:2]
                cross = -u_g * gy + v_g * gx
                speed = jnp.sqrt(u_g * u_g + v_g * v_g + 1.0e-8)
                parallel = u_g * gx + v_g * gy
                cosine = parallel / speed
                lam_c = batch["lam_geo_cross"]; lam_co = batch["lam_geo_cosine"]
                tgt = batch["geo_dir_target"]
                l_cr = jnp.mean((cross - tgt[:, 0:1]) ** 2 * lam_c)
                l_co = jnp.mean((cosine - tgt[:, 1:2]) ** 2 * lam_co)
                total = total + l_cr + l_co
                aux["geo_cross"] = l_cr; aux["geo_cosine"] = l_co
            else:
                aux["geo_cross"] = jnp.asarray(0.0, dtype=jnp.float32)
                aux["geo_cosine"] = jnp.asarray(0.0, dtype=jnp.float32)

            if geo_parallel is not None:
                gp_pred = pred[offs["geo_par"][0]:offs["geo_par"][1]]
                u_g = gp_pred[:, 0:1]; v_g = gp_pred[:, 1:2]
                gx = batch["geo_par_gxgy"][:, 0:1]; gy = batch["geo_par_gxgy"][:, 1:2]
                par = u_g * gx + v_g * gy
                l_gp = jnp.mean((par - batch["geo_par_target"]) ** 2 * batch["lam_geo_parallel"])
                total = total + l_gp
                aux["geo_parallel"] = l_gp
            else:
                aux["geo_parallel"] = jnp.asarray(0.0, dtype=jnp.float32)

            if wall_guard is not None:
                wg_pred = pred[offs["wall_guard"][0]:offs["wall_guard"][1]]
                u_w = wg_pred[:, 0:1]; v_w = wg_pred[:, 1:2]
                nx = batch["wall_guard_n"][:, 0:1]; ny = batch["wall_guard_n"][:, 1:2]
                r = u_w * nx + v_w * ny
                l_wg = jnp.mean(r ** 2 * batch["lam_wall_guard"])
                total = total + l_wg
                aux["wall_guard"] = l_wg
            else:
                aux["wall_guard"] = jnp.asarray(0.0, dtype=jnp.float32)

            if wall_guard_sep is not None:
                wgs_pred = pred[offs["wall_guard_sep"][0]:offs["wall_guard_sep"][1]]
                u_w = wgs_pred[:, 0:1]; v_w = wgs_pred[:, 1:2]
                nx = batch["wall_guard_sep_n"][:, 0:1]; ny = batch["wall_guard_sep_n"][:, 1:2]
                r = u_w * nx + v_w * ny
                l_wgs = jnp.mean(r ** 2 * batch["lam_wall_guard_sep"])
                total = total + l_wgs
                aux["wall_guard_sep"] = l_wgs
            else:
                aux["wall_guard_sep"] = jnp.asarray(0.0, dtype=jnp.float32)

            if inlet_p is not None:
                ip_pred = pred[offs["inlet_p"][0]:offs["inlet_p"][1]]
                diff = ip_pred[:, 2:3] - batch["inlet_p_target"]
                l_ip = jnp.mean(diff ** 2 * batch["lam_inlet_p"])
                total = total + l_ip
                aux["inlet_p"] = l_ip
            else:
                aux["inlet_p"] = jnp.asarray(0.0, dtype=jnp.float32)

            if outlet_p is not None:
                op_pred = pred[offs["outlet_p"][0]:offs["outlet_p"][1]]
                diff = op_pred[:, 2:3] - batch["outlet_p_target"]
                l_op = jnp.mean(diff ** 2 * batch["lam_outlet_p"])
                total = total + l_op
                aux["outlet_p"] = l_op
            else:
                aux["outlet_p"] = jnp.asarray(0.0, dtype=jnp.float32)

            return total, aux

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        aux["total"] = loss
        return new_params, new_opt_state, aux

    return train_step


def make_temp_train_step(net: TempNetFlax,
                         optimizer: optax.GradientTransformation,
                         sage_backward, dx: float, dy: float, dt_fd: float,
                         D: float, Q: float,
                         w_pde: float = 1.0, w_ic: float = 1.0,
                         w_arrival: float = 1.0, w_pre: float = 1.0,
                         w_inlet: float = 1.0, w_outlet: float = 0.5):
    """Temp train step: SAGE-JAX PDE + autograd BCs.

    Each minibatch is drawn outside (NumPy + PRNGKey), passed in via
    ``batch``. PDE seed = 2 * residual * w_pde / B; ``sage_backward``
    returns the adjoint at the 7 stencil columns, which is composed
    with a ``jax.vjp`` on the net forward at those stencil inputs to
    accumulate parameter gradients. The IC / arrival / pre / inlet
    terms use standard ``jax.grad`` on their MSE formulation; the
    outlet term uses ``jax.grad(T_out, x)`` which JAX handles natively.
    """
    apply_fn = net.apply

    @jax.jit
    def train_step(params, opt_state, batch):
        pde_stencil = batch["pde_stencil"]    # (7B, 5) stacked
        u_pde = batch["u_pde"]                # (B, 1)
        v_pde = batch["v_pde"]                # (B, 1)
        B = u_pde.shape[0]

        ic = batch["ic"]                      # (B_ic, 5)
        ic_target = batch["ic_target"]        # scalar
        arr = batch["arr"]                    # (B_arr, 5)
        arr_target = batch["arr_target"]      # scalar
        pre = batch["pre"]                    # (B_pre, 5)
        pre_target = batch["pre_target"]      # scalar
        inlet = batch.get("inlet")            # (B_in, 5) or None
        inlet_target = batch.get("inlet_target", 0.0)
        outlet_fwd = batch.get("outlet_fwd")  # (B_out, 5) or None
        outlet_x_eval = batch.get("outlet_x_eval")  # (B_out, 1) – x at which to take ∂T/∂x
        outlet_other = batch.get("outlet_other")  # (B_out, 4) – y, t, u, v

        # --- PDE block via SAGE-JAX ---
        def pde_forward(p):
            return apply_fn(p, pde_stencil)

        pde_pred_all, pde_vjp = jax.vjp(pde_forward, params)  # (7B, 1)
        pred_stack = reshape_temp_pred_to_stencil_stack(pde_pred_all, B)  # (B, 7)

        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)
        inv_2dt = 1.0 / (2.0 * dt_fd)
        inv_dx2 = 1.0 / (dx * dx)
        inv_dy2 = 1.0 / (dy * dy)
        T0 = pred_stack[:, 0:1]
        T_xp = pred_stack[:, 1:2]
        T_xm = pred_stack[:, 2:3]
        T_yp = pred_stack[:, 3:4]
        T_ym = pred_stack[:, 4:5]
        T_tp = pred_stack[:, 5:6]
        T_tm = pred_stack[:, 6:7]
        T_x = (T_xp - T_xm) * inv_2dx
        T_y = (T_yp - T_ym) * inv_2dy
        T_t = (T_tp - T_tm) * inv_2dt
        T_xx = (T_xp + T_xm - 2.0 * T0) * inv_dx2
        T_yy = (T_yp + T_ym - 2.0 * T0) * inv_dy2
        residual = T_t + u_pde * T_x + v_pde * T_y - D * (T_xx + T_yy) - Q
        loss_pde = jnp.mean(residual ** 2)

        dr = 2.0 * residual * float(w_pde) / float(B)
        g_sage = {"u": u_pde, "v": v_pde, "N_all": B}
        adj_stack = sage_backward(pred_stack, g_sage, dr)  # (B, 7)
        adj_pde = reshape_temp_adj_to_stencil_stack(adj_stack)  # (7B, 1)

        (pde_param_grads,) = pde_vjp(adj_pde)

        # --- BC blocks via jax.grad ---
        def bc_loss_fn(p):
            total = jnp.asarray(0.0, dtype=jnp.float32)
            aux = {}

            T_ic = apply_fn(p, ic)
            l_ic = jnp.mean((T_ic - ic_target) ** 2)
            total = total + float(w_ic) * l_ic
            aux["ic"] = l_ic

            T_arr = apply_fn(p, arr)
            l_arr = jnp.mean((T_arr - arr_target) ** 2)
            total = total + float(w_arrival) * l_arr
            aux["arrival"] = l_arr

            T_pre = apply_fn(p, pre)
            l_pre = jnp.mean((T_pre - pre_target) ** 2)
            total = total + float(w_pre) * l_pre
            aux["pre"] = l_pre

            if inlet is not None:
                T_inlet = apply_fn(p, inlet)
                l_inlet = jnp.mean((T_inlet - inlet_target) ** 2)
                total = total + float(w_inlet) * l_inlet
                aux["inlet"] = l_inlet
            else:
                aux["inlet"] = jnp.asarray(0.0, dtype=jnp.float32)

            if outlet_fwd is not None:
                # Per-point ∂T/∂x at the outlet points — vmap a scalar grad.
                def T_at(x_scalar, other):
                    # other = (y, t, u, v), all shape (4,)
                    x_vec = x_scalar.reshape(1,)
                    inp = jnp.concatenate([x_vec, other], axis=0).reshape(1, 5)
                    return apply_fn(p, inp)[0, 0]
                grad_T_x = jax.vmap(jax.grad(T_at, argnums=0), in_axes=(0, 0))(
                    outlet_x_eval[:, 0], outlet_other
                )
                l_out = jnp.mean(grad_T_x ** 2)
                total = total + float(w_outlet) * l_out
                aux["outlet"] = l_out
            else:
                aux["outlet"] = jnp.asarray(0.0, dtype=jnp.float32)

            return total, aux

        (bc_loss_val, bc_aux), bc_grads = jax.value_and_grad(
            bc_loss_fn, has_aux=True)(params)

        # Combine PDE and BC grads.
        combined_grads = jax.tree_util.tree_map(lambda a, b: a + b,
                                                 pde_param_grads, bc_grads)

        updates, new_opt_state = optimizer.update(combined_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        total_loss = float(w_pde) * loss_pde + bc_loss_val
        aux = dict(bc_aux)
        aux["pde"] = loss_pde
        aux["total"] = total_loss
        return new_params, new_opt_state, aux

    return train_step
