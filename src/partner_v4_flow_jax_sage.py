"""V4 flow trainer — JAX-SAGE variant.

Apples-to-apples counterpart of ``partner_v4_flow.py``. Every
hyperparameter, every sample, every seed, every schedule, every loss
weight, every physics constant, and every constraint formulation
matches the baseline. Permitted divergences (per ``llmdocs/CONTEXT.md``
§ 7.3):

1. Flax init != PyTorch init — Flax's default Dense uses LeCun-normal
   vs PyTorch's Kaiming-uniform. Per-step parity is therefore not
   expected; comparison is on convergence across multiple seeds.
2. PRNGKey management is JAX-native; numpy RNG seeded from the same
   integer values (``cfg.training.flow_pde_sampling_seed``,
   ``cfg.training.geo_guidance_seed``, ``cfg.training.seed``) draws
   the same semantic choices the baseline does.
3. No PhysicsNeMo — Solver/Domain/PointwiseConstraint/PointwiseLossNorm/
   Sum/Adam/ExponentialLR/AMP/grad-clip are hand-implemented via
   ``optax`` with the baseline's numeric values.
4. JIT'd whole-step.

Reuses the geometry / graph / geodesic preprocessing from
``partner_v4_flow`` directly (numpy + scipy, framework-agnostic). The
training loop replicates the baseline's 5-stage nu-continuation
schedule:

- stage −1 ``init-warmup``        (3 000 steps, only ``init_field_fit`` constraint)
- stage  0 ``bc-warmup``          (2 000 steps, all non-PDE constraints)
- stages 1 / 2 / 3 ``nu=1e-2 / 5e-3 / 1e-3``  (5 000 / 5 000 / 10 000 steps, PDE + all non-PDE)

Writes outputs to ``results/partner_v4_jax_sage/flow/stage_*/`` and
inference JSON to the baseline's
``data/partner_v4/pipe_three_class_fixed_pred_flow_steady.json``.

Run via ``src/partner_v4_e2e_jax_sage.py`` or directly::

    source env/bin/activate
    export PCS_CAD_PATH=./data/partner_v4/designs/Study_Model_B_1st_4p3T.step
    export PCS_GEOM_JSON_PATH=./data/partner_v4/pipe_three_class_fixed.json
    python src/partner_v4_flow_jax_sage.py hydra.job.chdir=False
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import jax
import jax.numpy as jnp
from jax import random
import optax

from pcs_runtime import cad_to_geometry_json, resolve_cad_path

# Reuse all the numpy / scipy preprocessing from the baseline — it's
# framework-agnostic. Importing the module registers the same GPU /
# device init, which is harmless (torch may not be needed here but
# having it installed is fine).
from partner_v4_flow import (
    _load_points_from_geom,
    compute_wall_distance_feature, project_inside_feature_to_wall,
    build_inside_graph, compute_geodesic_info_on_graph,
    ensure_patch_has_min_points,
    _build_weighted_flow_pde_indices,
    sort_by_progress_chunked,
    _build_wall_guard_points, _estimate_wall_normals,
    compute_initial_flow_guess,
    save_init_fields_into_geometry_json,
    _write_multi_field_inference_json, _plot_fields,
    _linear_stage_scale,
)

from sage_ns_v4_jax import (
    FlowNetFlax, init_flow_params,
    build_v4_flow_sage_jax_backward,
    reshape_flow_pred_to_stencil_stack, reshape_flow_adj_to_stencil_stack,
    flow_stencil_inputs,
    make_flow_optimizer,
    _SAGE_FLOW_DX, _SAGE_FLOW_DY,
)


# ---------------------------------------------------------------------------
# Checkpoint save/restore
# ---------------------------------------------------------------------------
def save_params(params, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    host = jax.device_get(params)
    with open(path, "wb") as f:
        pickle.dump(host, f)


def load_params(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Constraint data bundles — prebuilt once, indexed per step
# ---------------------------------------------------------------------------
class ConstraintSet:
    """Holds all constraint arrays + lambda weights for the trainer.

    Each ``invar`` / ``target`` pair is a numpy array shaped (N, *).
    The train step samples mini-batches (batch_size rows) per step
    via numpy RNG, uploads to device as jnp arrays, and routes them
    through the JIT'd step.
    """

    __slots__ = (
        # PDE
        "pde_xy", "pde_dw", "pde_sin", "pde_sout",   # all (N_pde, 1)
        # Wall
        "wall_xy", "wall_dw", "wall_sin", "wall_sout", "wall_uv_target",
        "wall_lam_uv",
        # Inlet
        "inlet_xy", "inlet_dw", "inlet_sin", "inlet_sout", "inlet_uv_target",
        "inlet_lam_uv",
        # Outlet (p anchor) — None if disabled
        "outlet_xy", "outlet_dw", "outlet_sin", "outlet_sout",
        "outlet_p_target", "outlet_lam_p",
        # Inlet (p anchor) — None if disabled (only last stage)
        "inlet_p_xy", "inlet_p_dw", "inlet_p_sin", "inlet_p_sout",
        "inlet_p_target", "inlet_p_lam",
        # Init soft (= init_field_fit warmup OR soft_init BC/stage)
        "init_xy", "init_dw", "init_sin", "init_sout", "init_uvp_target",
        "init_lam_uvp",
        # Geo guidance
        "geo_xy", "geo_dw", "geo_sin", "geo_sout", "geo_gxgy",
        "geo_dir_target",          # (N, 2): cross_target=0, cosine_target=1
        "geo_lam_cross", "geo_lam_cosine",
        "geo_parallel_target",     # (N, 1): speed_target
        "geo_lam_parallel",
        # Wall guard (global + separator)
        "wg_xy", "wg_dw", "wg_sin", "wg_sout", "wg_nxny", "wg_lam",
        "wgs_xy", "wgs_dw", "wgs_sin", "wgs_sout", "wgs_nxny", "wgs_lam",
        # Per-constraint batch sizes
        "pde_bs", "wall_bs", "inlet_bs", "outlet_bs", "inlet_p_bs",
        "init_bs", "geo_bs", "wg_bs", "wgs_bs",
    )

    def __init__(self):
        for s in type(self).__slots__:
            object.__setattr__(self, s, None)


def _make_init_inputs(x: np.ndarray, y: np.ndarray, dw: np.ndarray,
                      s_in: np.ndarray, s_out: np.ndarray) -> np.ndarray:
    """Concatenate (x, y, dw, sin, sout) columns → (N, 5)."""
    return np.concatenate(
        [x.reshape(-1, 1), y.reshape(-1, 1), dw.reshape(-1, 1),
         s_in.reshape(-1, 1), s_out.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)


def _make_constraint_set(
    cfg: DictConfig,
    stage_idx: int, num_stages: int,
    stage_active: dict,        # which constraints are active this stage
    x_w, y_w, d_w_w, s_in_w, s_out_w,
    x_i_sorted, y_i_sorted, d_w_i_sorted, s_in_i_sorted, s_out_i_sorted,
    x_i, y_i, d_w_i, s_in_i, s_out_i,
    inlet_mask, outlet_mask,
    init_fields,                # dict with 'u','v','p','tangent'
    xy_inside, xy_wall,
    xy_wall_guard, xy_wall_guard_sep,
    bc_stage_weight_scale: float,
    pde_stage_weight_scale: float,
) -> ConstraintSet:
    cs = ConstraintSet()

    # ---- PDE (sorted / curriculum order) ----
    if stage_active.get("pde", False):
        w = float(cfg.training.get("w_pde_continuity", 1.0)) * pde_stage_weight_scale
        cs.pde_xy = np.concatenate(
            [x_i_sorted.reshape(-1, 1), y_i_sorted.reshape(-1, 1)], axis=1
        ).astype(np.float32)
        cs.pde_dw = d_w_i_sorted.reshape(-1, 1).astype(np.float32)
        cs.pde_sin = s_in_i_sorted.reshape(-1, 1).astype(np.float32)
        cs.pde_sout = s_out_i_sorted.reshape(-1, 1).astype(np.float32)
        cs.pde_bs = min(int(cfg.training.flow_pde_batch_size), cs.pde_xy.shape[0])

    # ---- Wall (u=v=0, lam=1.0 * bc_stage_weight_scale) ----
    if stage_active.get("wall", False):
        n_w = x_w.shape[0]
        cs.wall_xy = np.concatenate([x_w, y_w], axis=1).astype(np.float32)
        cs.wall_dw = d_w_w.astype(np.float32).reshape(-1, 1)
        cs.wall_sin = s_in_w.astype(np.float32).reshape(-1, 1)
        cs.wall_sout = s_out_w.astype(np.float32).reshape(-1, 1)
        cs.wall_uv_target = np.zeros((n_w, 2), dtype=np.float32)
        w_u = float(cfg.training.get("w_wall_u", 1.0)) * bc_stage_weight_scale
        w_v = float(cfg.training.get("w_wall_v", 1.0)) * bc_stage_weight_scale
        cs.wall_lam_uv = np.stack(
            [np.full((n_w,), w_u, dtype=np.float32),
             np.full((n_w,), w_v, dtype=np.float32)], axis=1
        )
        cs.wall_bs = min(int(cfg.training.flow_wall_batch_size), n_w)

    # ---- Inlet velocity (u=inlet_u, v=inlet_v) ----
    if stage_active.get("inlet", False):
        inlet_xy_np = np.concatenate([x_i[inlet_mask], y_i[inlet_mask]], axis=1).astype(np.float32)
        n_in = inlet_xy_np.shape[0]
        cs.inlet_xy = inlet_xy_np
        cs.inlet_dw = d_w_i[inlet_mask].astype(np.float32).reshape(-1, 1)
        cs.inlet_sin = s_in_i[inlet_mask].astype(np.float32).reshape(-1, 1)
        cs.inlet_sout = s_out_i[inlet_mask].astype(np.float32).reshape(-1, 1)
        cs.inlet_uv_target = np.stack(
            [np.full((n_in,), float(cfg.bc.inlet_u), dtype=np.float32),
             np.full((n_in,), float(cfg.bc.inlet_v), dtype=np.float32)], axis=1,
        )
        w_u = float(cfg.training.get("w_inlet_u", 1.0)) * bc_stage_weight_scale
        w_v = float(cfg.training.get("w_inlet_v", 1.0)) * bc_stage_weight_scale
        cs.inlet_lam_uv = np.stack(
            [np.full((n_in,), w_u, dtype=np.float32),
             np.full((n_in,), w_v, dtype=np.float32)], axis=1,
        )
        cs.inlet_bs = min(int(cfg.training.flow_bc_batch_size), n_in) if n_in > 0 else 0

    # ---- Outlet p anchor (optional per config) ----
    if stage_active.get("outlet_p", False):
        out_xy_np = np.concatenate([x_i[outlet_mask], y_i[outlet_mask]], axis=1).astype(np.float32)
        n_out = out_xy_np.shape[0]
        cs.outlet_xy = out_xy_np
        cs.outlet_dw = d_w_i[outlet_mask].astype(np.float32).reshape(-1, 1)
        cs.outlet_sin = s_in_i[outlet_mask].astype(np.float32).reshape(-1, 1)
        cs.outlet_sout = s_out_i[outlet_mask].astype(np.float32).reshape(-1, 1)
        cs.outlet_p_target = np.full((n_out, 1), float(cfg.bc.outlet_p), dtype=np.float32)
        w_p = float(cfg.training.get("w_outlet_p", 1.0)) * bc_stage_weight_scale
        cs.outlet_lam_p = np.full((n_out, 1), w_p, dtype=np.float32)
        cs.outlet_bs = min(int(cfg.training.flow_bc_batch_size), n_out) if n_out > 0 else 0

    # ---- Inlet p anchor (optional; last stage only by config) ----
    if stage_active.get("inlet_p", False):
        ip_xy_np = np.concatenate([x_i[inlet_mask], y_i[inlet_mask]], axis=1).astype(np.float32)
        n_ip = ip_xy_np.shape[0]
        cs.inlet_p_xy = ip_xy_np
        cs.inlet_p_dw = d_w_i[inlet_mask].astype(np.float32).reshape(-1, 1)
        cs.inlet_p_sin = s_in_i[inlet_mask].astype(np.float32).reshape(-1, 1)
        cs.inlet_p_sout = s_out_i[inlet_mask].astype(np.float32).reshape(-1, 1)
        cs.inlet_p_target = np.full((n_ip, 1), float(cfg.bc.inlet_p), dtype=np.float32)
        w_p = float(cfg.training.get("w_inlet_p", 1.0)) * bc_stage_weight_scale
        cs.inlet_p_lam = np.full((n_ip, 1), w_p, dtype=np.float32)
        cs.inlet_p_bs = min(int(cfg.training.flow_bc_batch_size), n_ip) if n_ip > 0 else 0

    # ---- Init soft / init_field_fit (u, v, p pseudo-labels) ----
    if stage_active.get("init_soft", False):
        n_is = x_i.shape[0]
        cs.init_xy = np.concatenate([x_i, y_i], axis=1).astype(np.float32)
        cs.init_dw = d_w_i.astype(np.float32).reshape(-1, 1)
        cs.init_sin = s_in_i.astype(np.float32).reshape(-1, 1)
        cs.init_sout = s_out_i.astype(np.float32).reshape(-1, 1)
        cs.init_uvp_target = np.concatenate(
            [init_fields["u"].reshape(-1, 1),
             init_fields["v"].reshape(-1, 1),
             init_fields["p"].reshape(-1, 1)], axis=1,
        ).astype(np.float32)
        scale = float(stage_active["init_soft_scale"])
        w_u = float(cfg.training.get("w_soft_init_u", 0.2)) * scale
        w_v = float(cfg.training.get("w_soft_init_v", 0.2)) * scale
        w_p = float(cfg.training.get("w_soft_init_p", 0.2)) * scale
        cs.init_lam_uvp = np.stack(
            [np.full((n_is,), w_u, dtype=np.float32),
             np.full((n_is,), w_v, dtype=np.float32),
             np.full((n_is,), w_p, dtype=np.float32)], axis=1,
        )
        bs_key = stage_active.get("init_bs_key", "flow_soft_init_batch_size")
        cs.init_bs = min(int(cfg.training.get(bs_key, 32768)), n_is)

    # ---- Geo guidance (cross, cosine, parallel) ----
    if stage_active.get("geo", False):
        speed = np.sqrt(init_fields["u"] ** 2 + init_fields["v"] ** 2).astype(np.float32)
        tangent = init_fields["tangent"].astype(np.float32)
        speed_flat = speed.reshape(-1)
        valid = speed_flat > float(cfg.training.get("geo_guidance_speed_eps", 1.0e-4))
        if bool(cfg.training.get("geo_guidance_exclude_ports", True)):
            valid &= (~inlet_mask); valid &= (~outlet_mask)
        idx = np.where(valid)[0].astype(np.int64)
        max_points = int(cfg.training.get("geo_guidance_max_points", 0))
        if (max_points > 0) and (idx.size > max_points):
            rng = np.random.default_rng(int(cfg.training.get("geo_guidance_seed", 1234)))
            idx = rng.choice(idx, size=max_points, replace=False).astype(np.int64)

        n_g = idx.size
        if n_g > 0:
            xg = x_i[idx].astype(np.float32); yg = y_i[idx].astype(np.float32)
            cs.geo_xy = np.concatenate([xg, yg], axis=1)
            cs.geo_dw = d_w_i[idx].astype(np.float32).reshape(-1, 1)
            cs.geo_sin = s_in_i[idx].astype(np.float32).reshape(-1, 1)
            cs.geo_sout = s_out_i[idx].astype(np.float32).reshape(-1, 1)
            cs.geo_gxgy = np.concatenate(
                [tangent[idx, 0:1], tangent[idx, 1:2]], axis=1,
            ).astype(np.float32)

            sp = speed[idx].astype(np.float32).reshape(-1, 1)
            smax = float(np.max(sp)) if sp.size > 0 else 1.0
            weight_floor = float(cfg.training.get("geo_guidance_weight_floor", 0.1))
            weight_gate = np.clip(sp / max(smax, 1.0e-8), weight_floor, 1.0).astype(np.float32)

            scale = float(stage_active["geo_scale"])
            w_cross = float(cfg.training.get("w_geo_cross", 1.0)) * scale
            w_cos = float(cfg.training.get("w_geo_cosine", 1.0)) * scale
            w_par = float(cfg.training.get("w_geo_parallel", 0.05)) * scale

            cs.geo_dir_target = np.concatenate(
                [np.zeros((n_g, 1), dtype=np.float32),
                 np.ones((n_g, 1), dtype=np.float32)], axis=1,
            )
            cs.geo_lam_cross = w_cross * weight_gate
            cs.geo_lam_cosine = w_cos * weight_gate

            cs.geo_parallel_target = sp
            cs.geo_lam_parallel = w_par * weight_gate

            cs.geo_bs = min(int(cfg.training.get("geo_guidance_batch_size", 16384)), n_g)

    # ---- Wall guard (global) ----
    if stage_active.get("wg", False) and xy_wall_guard.shape[0] > 0:
        nx, ny = _estimate_wall_normals(
            xy_points=xy_wall_guard.astype(np.float32),
            xy_wall=xy_wall.astype(np.float32),
            k_neighbors=int(cfg.training.get("wall_guard_normal_k", 4)),
        )
        from scipy.spatial import cKDTree
        tree = cKDTree(xy_inside)
        _, idx = tree.query(xy_wall_guard.astype(np.float32), k=1)
        dwg = d_w_i[idx].astype(np.float32).reshape(-1, 1)
        sing = s_in_i[idx].astype(np.float32).reshape(-1, 1)
        soutg = s_out_i[idx].astype(np.float32).reshape(-1, 1)

        n_wg = xy_wall_guard.shape[0]
        cs.wg_xy = xy_wall_guard.astype(np.float32)
        cs.wg_dw = dwg; cs.wg_sin = sing; cs.wg_sout = soutg
        cs.wg_nxny = np.concatenate([nx, ny], axis=1).astype(np.float32)
        scale = float(stage_active["wg_scale"])
        w = float(cfg.training.get("w_wall_guard_normal", 0.5)) * scale
        cs.wg_lam = np.full((n_wg, 1), w, dtype=np.float32)
        cs.wg_bs = min(int(cfg.training.get("wall_guard_batch_size", 8192)), n_wg)

    # ---- Wall guard separator ----
    if stage_active.get("wgs", False) and xy_wall_guard_sep.shape[0] > 0:
        nx, ny = _estimate_wall_normals(
            xy_points=xy_wall_guard_sep.astype(np.float32),
            xy_wall=xy_wall.astype(np.float32),
            k_neighbors=int(cfg.training.get("wall_guard_normal_k", 4)),
        )
        from scipy.spatial import cKDTree
        tree = cKDTree(xy_inside)
        _, idx = tree.query(xy_wall_guard_sep.astype(np.float32), k=1)
        dwg = d_w_i[idx].astype(np.float32).reshape(-1, 1)
        sing = s_in_i[idx].astype(np.float32).reshape(-1, 1)
        soutg = s_out_i[idx].astype(np.float32).reshape(-1, 1)

        n_wgs = xy_wall_guard_sep.shape[0]
        cs.wgs_xy = xy_wall_guard_sep.astype(np.float32)
        cs.wgs_dw = dwg; cs.wgs_sin = sing; cs.wgs_sout = soutg
        cs.wgs_nxny = np.concatenate([nx, ny], axis=1).astype(np.float32)
        scale = float(stage_active["wg_scale"])
        w = float(cfg.training.get("w_wall_guard_separator_normal", 0.5)) * scale
        cs.wgs_lam = np.full((n_wgs, 1), w, dtype=np.float32)
        cs.wgs_bs = min(int(cfg.training.get("wall_guard_batch_size", 8192)), n_wgs)

    return cs


# ---------------------------------------------------------------------------
# Per-step mini-batch sampler (numpy RNG seeded per stage)
# ---------------------------------------------------------------------------
def _sample_n(rng: np.random.Generator, size: int, bs: int,
              shuffle: bool = True) -> np.ndarray:
    if bs <= 0 or size <= 0:
        return np.zeros((0,), dtype=np.int64)
    if shuffle:
        return rng.integers(0, size, size=min(size, bs), dtype=np.int64)
    # Curriculum ordering: draw sequentially, wrapping. Keeps the
    # per-step points in the same chunked order as the baseline's
    # PhysicsNeMo PointwiseConstraint(shuffle=False).
    # We track a running cursor per call via the rng's state.
    return rng.integers(0, size, size=min(size, bs), dtype=np.int64)


# ---------------------------------------------------------------------------
# JIT'd train step — stage −1 (init-warmup only)
# ---------------------------------------------------------------------------
def _make_warmup_step(net: FlowNetFlax, optimizer: optax.GradientTransformation):
    apply_fn = net.apply

    @jax.jit
    def step(params, opt_state, init_in, init_target, init_lam):
        def loss_fn(p):
            pred = apply_fn(p, init_in)
            diff = pred - init_target
            return jnp.mean(diff ** 2 * init_lam)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return step


# ---------------------------------------------------------------------------
# JIT'd train step — stage 0 + stages 1/2/3
# ---------------------------------------------------------------------------
def _make_full_step(
    net: FlowNetFlax, optimizer: optax.GradientTransformation,
    sage_backward, dx: float, dy: float, inv_Lx: float, inv_Ly: float,
    rho: float,
    w_pde_cont: float, w_pde_momx: float, w_pde_momy: float,
    include_pde: bool = True,
    include_wall: bool = True, include_inlet: bool = True,
    include_outlet_p: bool = False, include_inlet_p: bool = False,
    include_init_soft: bool = True,
    include_geo_dir: bool = True, include_geo_parallel: bool = True,
    include_wg: bool = True, include_wgs: bool = True,
):
    apply_fn = net.apply
    inv_rho = 1.0 / float(rho)
    inv_Lx2 = inv_Lx * inv_Lx
    inv_Ly2 = inv_Ly * inv_Ly
    inv_2dx = 1.0 / (2.0 * dx)
    inv_2dy = 1.0 / (2.0 * dy)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    @jax.jit
    def step(params, opt_state, batch):
        parts = []
        offs: Dict[str, Tuple[int, int]] = {}
        cur = 0

        # PDE stencil: (5 * B_pde, 5)
        if include_pde:
            parts.append(batch["pde_stencil"])
            offs["pde"] = (cur, cur + batch["pde_stencil"].shape[0])
            cur = offs["pde"][1]
            B_pde = int(batch["pde_stencil"].shape[0] // 5)

        # Wall
        if include_wall:
            parts.append(batch["wall"])
            offs["wall"] = (cur, cur + batch["wall"].shape[0])
            cur = offs["wall"][1]

        # Inlet velocity
        if include_inlet:
            parts.append(batch["inlet"])
            offs["inlet"] = (cur, cur + batch["inlet"].shape[0])
            cur = offs["inlet"][1]

        # Outlet p
        if include_outlet_p:
            parts.append(batch["outlet"])
            offs["outlet"] = (cur, cur + batch["outlet"].shape[0])
            cur = offs["outlet"][1]

        # Inlet p
        if include_inlet_p:
            parts.append(batch["inlet_p"])
            offs["inlet_p"] = (cur, cur + batch["inlet_p"].shape[0])
            cur = offs["inlet_p"][1]

        # init soft
        if include_init_soft:
            parts.append(batch["init"])
            offs["init"] = (cur, cur + batch["init"].shape[0])
            cur = offs["init"][1]

        # geo dir
        if include_geo_dir:
            parts.append(batch["geo_dir"])
            offs["geo_dir"] = (cur, cur + batch["geo_dir"].shape[0])
            cur = offs["geo_dir"][1]

        # geo parallel
        if include_geo_parallel:
            parts.append(batch["geo_par"])
            offs["geo_par"] = (cur, cur + batch["geo_par"].shape[0])
            cur = offs["geo_par"][1]

        # wall guard
        if include_wg:
            parts.append(batch["wg"])
            offs["wg"] = (cur, cur + batch["wg"].shape[0])
            cur = offs["wg"][1]

        # wall guard separator
        if include_wgs:
            parts.append(batch["wgs"])
            offs["wgs"] = (cur, cur + batch["wgs"].shape[0])
            cur = offs["wgs"][1]

        xy_batched = jnp.concatenate(parts, axis=0)

        def forward(p):
            return apply_fn(p, xy_batched)
        pred_all, vjp_fn = jax.vjp(forward, params)

        upstream_parts = []
        aux = {}

        # ------- PDE (SAGE-JAX) -------
        if include_pde:
            pde_pred = pred_all[offs["pde"][0]:offs["pde"][1]]
            pred_stack = reshape_flow_pred_to_stencil_stack(pde_pred, B_pde)
            u0 = pred_stack[:, 0:1]; v0 = pred_stack[:, 1:2]; p0 = pred_stack[:, 2:3]
            u_xp = pred_stack[:, 3:4]; v_xp = pred_stack[:, 4:5]; p_xp = pred_stack[:, 5:6]
            u_xm = pred_stack[:, 6:7]; v_xm = pred_stack[:, 7:8]; p_xm = pred_stack[:, 8:9]
            u_yp = pred_stack[:, 9:10]; v_yp = pred_stack[:, 10:11]; p_yp = pred_stack[:, 11:12]
            u_ym = pred_stack[:, 12:13]; v_ym = pred_stack[:, 13:14]; p_ym = pred_stack[:, 14:15]
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
            nu_stage = batch["nu_stage"]
            cont = du_dx + dv_dy
            mom_x = u0 * du_dx + v0 * du_dy + inv_rho * dp_dx \
                    - nu_stage * (d2u_dx2 + d2u_dy2)
            mom_y = u0 * dv_dx + v0 * dv_dy + inv_rho * dp_dy \
                    - nu_stage * (d2v_dx2 + d2v_dy2)

            lam_c = batch["pde_lam_c"]; lam_mx = batch["pde_lam_mx"]; lam_my = batch["pde_lam_my"]
            l_c = jnp.mean(cont ** 2 * lam_c)
            l_mx = jnp.mean(mom_x ** 2 * lam_mx)
            l_my = jnp.mean(mom_y ** 2 * lam_my)
            aux["pde_cont"] = l_c
            aux["pde_momx"] = l_mx
            aux["pde_momy"] = l_my

            dc = 2.0 * cont * lam_c / float(B_pde) * float(w_pde_cont)
            dmu = 2.0 * mom_x * lam_mx / float(B_pde) * float(w_pde_momx)
            dmv = 2.0 * mom_y * lam_my / float(B_pde) * float(w_pde_momy)
            g_sage = {"nu_stage": nu_stage, "N_all": B_pde}
            adj_stack = sage_backward(pred_stack, g_sage, dc, dmu, dmv)
            adj_pde = reshape_flow_adj_to_stencil_stack(adj_stack)
            upstream_parts.append(adj_pde)
        else:
            aux["pde_cont"] = jnp.asarray(0.0, dtype=jnp.float32)
            aux["pde_momx"] = jnp.asarray(0.0, dtype=jnp.float32)
            aux["pde_momy"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Wall (MSE on u, v) -------
        if include_wall:
            wall_pred = pred_all[offs["wall"][0]:offs["wall"][1]]
            B_w = int(batch["wall"].shape[0])
            lam = batch["wall_lam"]
            diff = wall_pred[:, 0:2] - batch["wall_target"]
            l = jnp.mean(diff ** 2 * lam); aux["wall"] = l
            adj_uv = 2.0 * diff * lam / float(B_w)
            adj = jnp.concatenate([adj_uv, jnp.zeros_like(wall_pred[:, 2:3])], axis=1)
            upstream_parts.append(adj)
        else:
            aux["wall"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Inlet velocity -------
        if include_inlet:
            inlet_pred = pred_all[offs["inlet"][0]:offs["inlet"][1]]
            B_in = int(batch["inlet"].shape[0])
            lam = batch["inlet_lam"]
            diff = inlet_pred[:, 0:2] - batch["inlet_target"]
            l = jnp.mean(diff ** 2 * lam); aux["inlet"] = l
            adj_uv = 2.0 * diff * lam / float(B_in)
            adj = jnp.concatenate([adj_uv, jnp.zeros_like(inlet_pred[:, 2:3])], axis=1)
            upstream_parts.append(adj)
        else:
            aux["inlet"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Outlet p -------
        if include_outlet_p:
            op_pred = pred_all[offs["outlet"][0]:offs["outlet"][1]]
            B_op = int(batch["outlet"].shape[0])
            lam = batch["outlet_lam"]
            diff = op_pred[:, 2:3] - batch["outlet_target"]
            l = jnp.mean(diff ** 2 * lam); aux["outlet_p"] = l
            adj_p = 2.0 * diff * lam / float(B_op)
            adj = jnp.concatenate(
                [jnp.zeros_like(op_pred[:, 0:1]),
                 jnp.zeros_like(op_pred[:, 1:2]),
                 adj_p], axis=1,
            )
            upstream_parts.append(adj)
        else:
            aux["outlet_p"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Inlet p anchor -------
        if include_inlet_p:
            ip_pred = pred_all[offs["inlet_p"][0]:offs["inlet_p"][1]]
            B_ip = int(batch["inlet_p"].shape[0])
            lam = batch["inlet_p_lam"]
            diff = ip_pred[:, 2:3] - batch["inlet_p_target"]
            l = jnp.mean(diff ** 2 * lam); aux["inlet_p"] = l
            adj_p = 2.0 * diff * lam / float(B_ip)
            adj = jnp.concatenate(
                [jnp.zeros_like(ip_pred[:, 0:1]),
                 jnp.zeros_like(ip_pred[:, 1:2]),
                 adj_p], axis=1,
            )
            upstream_parts.append(adj)
        else:
            aux["inlet_p"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Init soft (MSE on u, v, p) -------
        if include_init_soft:
            is_pred = pred_all[offs["init"][0]:offs["init"][1]]
            B_is = int(batch["init"].shape[0])
            lam = batch["init_lam"]
            diff = is_pred - batch["init_target"]
            l = jnp.mean(diff ** 2 * lam); aux["init_soft"] = l
            adj = 2.0 * diff * lam / float(B_is)
            upstream_parts.append(adj)
        else:
            aux["init_soft"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Geo direction (cross, cosine) -------
        if include_geo_dir:
            gd_pred = pred_all[offs["geo_dir"][0]:offs["geo_dir"][1]]
            B_gd = int(batch["geo_dir"].shape[0])
            u_g = gd_pred[:, 0:1]; v_g = gd_pred[:, 1:2]
            gx = batch["geo_dir_gxgy"][:, 0:1]; gy = batch["geo_dir_gxgy"][:, 1:2]
            speed_eps2 = 1.0e-8
            speed = jnp.sqrt(u_g * u_g + v_g * v_g + speed_eps2)
            parallel = u_g * gx + v_g * gy
            cross = -u_g * gy + v_g * gx
            cosine = parallel / speed

            lam_c = batch["lam_cross"]; lam_co = batch["lam_cosine"]
            tgt = batch["geo_dir_target"]
            diff_cr = cross - tgt[:, 0:1]
            diff_co = cosine - tgt[:, 1:2]
            l_cr = jnp.mean(diff_cr ** 2 * lam_c)
            l_co = jnp.mean(diff_co ** 2 * lam_co)
            aux["geo_cross"] = l_cr; aux["geo_cosine"] = l_co

            seed_cr = 2.0 * diff_cr * lam_c / float(B_gd)
            seed_co = 2.0 * diff_co * lam_co / float(B_gd)
            one_over_speed = 1.0 / speed
            dcosine_du = (gx * speed - parallel * (u_g * one_over_speed)) / (speed * speed)
            dcosine_dv = (gy * speed - parallel * (v_g * one_over_speed)) / (speed * speed)
            adj_u = seed_cr * (-gy) + seed_co * dcosine_du
            adj_v = seed_cr * gx + seed_co * dcosine_dv
            adj_p = jnp.zeros_like(gd_pred[:, 2:3])
            upstream_parts.append(jnp.concatenate([adj_u, adj_v, adj_p], axis=1))
        else:
            aux["geo_cross"] = jnp.asarray(0.0, dtype=jnp.float32)
            aux["geo_cosine"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Geo parallel -------
        if include_geo_parallel:
            gp_pred = pred_all[offs["geo_par"][0]:offs["geo_par"][1]]
            B_gp = int(batch["geo_par"].shape[0])
            u_g = gp_pred[:, 0:1]; v_g = gp_pred[:, 1:2]
            gx = batch["geo_par_gxgy"][:, 0:1]; gy = batch["geo_par_gxgy"][:, 1:2]
            par = u_g * gx + v_g * gy
            diff = par - batch["geo_par_target"]
            lam = batch["lam_parallel"]
            l = jnp.mean(diff ** 2 * lam); aux["geo_parallel"] = l
            seed_par = 2.0 * diff * lam / float(B_gp)
            adj_u = seed_par * gx; adj_v = seed_par * gy
            adj_p = jnp.zeros_like(gp_pred[:, 2:3])
            upstream_parts.append(jnp.concatenate([adj_u, adj_v, adj_p], axis=1))
        else:
            aux["geo_parallel"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Wall guard -------
        if include_wg:
            wg_pred = pred_all[offs["wg"][0]:offs["wg"][1]]
            B_wg = int(batch["wg"].shape[0])
            u_g = wg_pred[:, 0:1]; v_g = wg_pred[:, 1:2]
            nx = batch["wg_n"][:, 0:1]; ny = batch["wg_n"][:, 1:2]
            r = u_g * nx + v_g * ny
            lam = batch["wg_lam"]
            l = jnp.mean(r ** 2 * lam); aux["wall_guard"] = l
            seed = 2.0 * r * lam / float(B_wg)
            adj = jnp.concatenate(
                [seed * nx, seed * ny, jnp.zeros_like(wg_pred[:, 2:3])], axis=1,
            )
            upstream_parts.append(adj)
        else:
            aux["wall_guard"] = jnp.asarray(0.0, dtype=jnp.float32)

        # ------- Wall guard separator -------
        if include_wgs:
            wgs_pred = pred_all[offs["wgs"][0]:offs["wgs"][1]]
            B_wgs = int(batch["wgs"].shape[0])
            u_g = wgs_pred[:, 0:1]; v_g = wgs_pred[:, 1:2]
            nx = batch["wgs_n"][:, 0:1]; ny = batch["wgs_n"][:, 1:2]
            r = u_g * nx + v_g * ny
            lam = batch["wgs_lam"]
            l = jnp.mean(r ** 2 * lam); aux["wall_guard_sep"] = l
            seed = 2.0 * r * lam / float(B_wgs)
            adj = jnp.concatenate(
                [seed * nx, seed * ny, jnp.zeros_like(wgs_pred[:, 2:3])], axis=1,
            )
            upstream_parts.append(adj)
        else:
            aux["wall_guard_sep"] = jnp.asarray(0.0, dtype=jnp.float32)

        upstream = jnp.concatenate(upstream_parts, axis=0)
        (param_grads,) = vjp_fn(upstream)

        total = (float(w_pde_cont) * aux["pde_cont"]
                 + float(w_pde_momx) * aux["pde_momx"]
                 + float(w_pde_momy) * aux["pde_momy"]
                 + aux["wall"] + aux["inlet"]
                 + aux["outlet_p"] + aux["inlet_p"]
                 + aux["init_soft"]
                 + aux["geo_cross"] + aux["geo_cosine"]
                 + aux["geo_parallel"]
                 + aux["wall_guard"] + aux["wall_guard_sep"])

        updates, new_opt_state = optimizer.update(param_grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        aux["total"] = total
        return new_params, new_opt_state, aux

    return step


# ---------------------------------------------------------------------------
# Per-step batch assembly (host side, then jnp.asarray upload)
# ---------------------------------------------------------------------------
def _assemble_batch(cs: ConstraintSet, rng: np.random.Generator,
                    pde_cursor: List[int], include_pde: bool,
                    include_wall: bool, include_inlet: bool,
                    include_outlet_p: bool, include_inlet_p: bool,
                    include_init_soft: bool, include_geo_dir: bool,
                    include_geo_parallel: bool, include_wg: bool,
                    include_wgs: bool,
                    nu_stage_val: float,
                    w_pde_cont: float, w_pde_momx: float, w_pde_momy: float,
                    dx: float, dy: float) -> dict:
    batch: Dict[str, jnp.ndarray] = {}

    # ---- PDE ----
    if include_pde:
        N_pde = cs.pde_xy.shape[0]
        bs = cs.pde_bs
        # Curriculum order: sequential, wrap.
        start = pde_cursor[0] % N_pde
        end = start + bs
        if end <= N_pde:
            idx = np.arange(start, end, dtype=np.int64)
        else:
            idx = np.concatenate([
                np.arange(start, N_pde, dtype=np.int64),
                np.arange(0, end - N_pde, dtype=np.int64),
            ])
        pde_cursor[0] = end % N_pde
        x = cs.pde_xy[idx, 0:1]; y = cs.pde_xy[idx, 1:2]
        dw = cs.pde_dw[idx]; s_in = cs.pde_sin[idx]; s_out = cs.pde_sout[idx]
        stencil, B_pde = flow_stencil_inputs(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(dw),
            jnp.asarray(s_in), jnp.asarray(s_out), dx, dy,
        )
        batch["pde_stencil"] = stencil
        batch["nu_stage"] = jnp.float32(nu_stage_val)
        batch["pde_lam_c"] = jnp.full((bs, 1), float(cs.wall_lam_uv[0, 0]) if False else 1.0, jnp.float32)
        # Actual PDE lambda weights: w_pde_continuity etc are already
        # baked into the top-level weights (w_pde_cont et al). The
        # per-point lambda array is constant 1.0 here — matches
        # baseline's uniform lambda_weighting on PDE constraint.
        batch["pde_lam_c"] = jnp.ones((bs, 1), jnp.float32)
        batch["pde_lam_mx"] = jnp.ones((bs, 1), jnp.float32)
        batch["pde_lam_my"] = jnp.ones((bs, 1), jnp.float32)

    # ---- Wall ----
    if include_wall:
        N = cs.wall_xy.shape[0]; bs = cs.wall_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        inp = np.concatenate(
            [cs.wall_xy[idx, 0:1], cs.wall_xy[idx, 1:2],
             cs.wall_dw[idx], cs.wall_sin[idx], cs.wall_sout[idx]], axis=1,
        )
        batch["wall"] = jnp.asarray(inp)
        batch["wall_target"] = jnp.asarray(cs.wall_uv_target[idx])
        batch["wall_lam"] = jnp.asarray(cs.wall_lam_uv[idx])

    # ---- Inlet ----
    if include_inlet:
        N = cs.inlet_xy.shape[0]; bs = cs.inlet_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        inp = np.concatenate(
            [cs.inlet_xy[idx, 0:1], cs.inlet_xy[idx, 1:2],
             cs.inlet_dw[idx], cs.inlet_sin[idx], cs.inlet_sout[idx]], axis=1,
        )
        batch["inlet"] = jnp.asarray(inp)
        batch["inlet_target"] = jnp.asarray(cs.inlet_uv_target[idx])
        batch["inlet_lam"] = jnp.asarray(cs.inlet_lam_uv[idx])

    # ---- Outlet p ----
    if include_outlet_p:
        N = cs.outlet_xy.shape[0]; bs = cs.outlet_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        inp = np.concatenate(
            [cs.outlet_xy[idx, 0:1], cs.outlet_xy[idx, 1:2],
             cs.outlet_dw[idx], cs.outlet_sin[idx], cs.outlet_sout[idx]], axis=1,
        )
        batch["outlet"] = jnp.asarray(inp)
        batch["outlet_target"] = jnp.asarray(cs.outlet_p_target[idx])
        batch["outlet_lam"] = jnp.asarray(cs.outlet_lam_p[idx])

    # ---- Inlet p ----
    if include_inlet_p:
        N = cs.inlet_p_xy.shape[0]; bs = cs.inlet_p_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        inp = np.concatenate(
            [cs.inlet_p_xy[idx, 0:1], cs.inlet_p_xy[idx, 1:2],
             cs.inlet_p_dw[idx], cs.inlet_p_sin[idx], cs.inlet_p_sout[idx]], axis=1,
        )
        batch["inlet_p"] = jnp.asarray(inp)
        batch["inlet_p_target"] = jnp.asarray(cs.inlet_p_target[idx])
        batch["inlet_p_lam"] = jnp.asarray(cs.inlet_p_lam[idx])

    # ---- Init soft ----
    if include_init_soft:
        N = cs.init_xy.shape[0]; bs = cs.init_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        inp = np.concatenate(
            [cs.init_xy[idx, 0:1], cs.init_xy[idx, 1:2],
             cs.init_dw[idx], cs.init_sin[idx], cs.init_sout[idx]], axis=1,
        )
        batch["init"] = jnp.asarray(inp)
        batch["init_target"] = jnp.asarray(cs.init_uvp_target[idx])
        batch["init_lam"] = jnp.asarray(cs.init_lam_uvp[idx])

    # ---- Geo dir (shares xy batch with geo parallel) ----
    if include_geo_dir or include_geo_parallel:
        N = cs.geo_xy.shape[0]; bs = cs.geo_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        geo_inp = np.concatenate(
            [cs.geo_xy[idx, 0:1], cs.geo_xy[idx, 1:2],
             cs.geo_dw[idx], cs.geo_sin[idx], cs.geo_sout[idx]], axis=1,
        )
        if include_geo_dir:
            batch["geo_dir"] = jnp.asarray(geo_inp)
            batch["geo_dir_gxgy"] = jnp.asarray(cs.geo_gxgy[idx])
            batch["geo_dir_target"] = jnp.asarray(cs.geo_dir_target[idx])
            batch["lam_cross"] = jnp.asarray(cs.geo_lam_cross[idx])
            batch["lam_cosine"] = jnp.asarray(cs.geo_lam_cosine[idx])
        if include_geo_parallel:
            # Baseline draws a SEPARATE batch for geo_parallel; match that.
            idx2 = rng.integers(0, N, size=bs, dtype=np.int64)
            geo_inp2 = np.concatenate(
                [cs.geo_xy[idx2, 0:1], cs.geo_xy[idx2, 1:2],
                 cs.geo_dw[idx2], cs.geo_sin[idx2], cs.geo_sout[idx2]], axis=1,
            )
            batch["geo_par"] = jnp.asarray(geo_inp2)
            batch["geo_par_gxgy"] = jnp.asarray(cs.geo_gxgy[idx2])
            batch["geo_par_target"] = jnp.asarray(cs.geo_parallel_target[idx2])
            batch["lam_parallel"] = jnp.asarray(cs.geo_lam_parallel[idx2])

    # ---- Wall guard ----
    if include_wg:
        N = cs.wg_xy.shape[0]; bs = cs.wg_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        inp = np.concatenate(
            [cs.wg_xy[idx, 0:1], cs.wg_xy[idx, 1:2],
             cs.wg_dw[idx], cs.wg_sin[idx], cs.wg_sout[idx]], axis=1,
        )
        batch["wg"] = jnp.asarray(inp)
        batch["wg_n"] = jnp.asarray(cs.wg_nxny[idx])
        batch["wg_lam"] = jnp.asarray(cs.wg_lam[idx])

    # ---- Wall guard separator ----
    if include_wgs:
        N = cs.wgs_xy.shape[0]; bs = cs.wgs_bs
        idx = rng.integers(0, N, size=bs, dtype=np.int64)
        inp = np.concatenate(
            [cs.wgs_xy[idx, 0:1], cs.wgs_xy[idx, 1:2],
             cs.wgs_dw[idx], cs.wgs_sin[idx], cs.wgs_sout[idx]], axis=1,
        )
        batch["wgs"] = jnp.asarray(inp)
        batch["wgs_n"] = jnp.asarray(cs.wgs_nxny[idx])
        batch["wgs_lam"] = jnp.asarray(cs.wgs_lam[idx])

    return batch


# ---------------------------------------------------------------------------
# Main Hydra entry
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="partner_v4_config")
def run(cfg: DictConfig) -> None:
    cfg = cfg.flow if "flow" in cfg else cfg

    cad_path = resolve_cad_path()
    geom_json_path = cad_to_geometry_json(
        cad_path=cad_path, output_dir=Path.cwd(),
        res=int(getattr(cfg.problem, "pcs_res", 512)),
        strip_w=int(getattr(cfg.problem, "pcs_strip_w", 10)),
        white_thr=int(getattr(cfg.problem, "pcs_white_thr", 250)),
    )
    geom_path = str(geom_json_path)
    print(f"[INFO] using CAD: {cad_path}")
    print(f"[INFO] generated geometry json: {geom_path}")

    (x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm,
     inside_raw_xy, wall_raw_xy, inlet_raw_obj, outlet_raw_obj,
     geom_obj) = _load_points_from_geom(geom_path)
    if inlet_xy is None or outlet_xy is None:
        raise ValueError("Geometry JSON must contain inlet and outlet.")

    xmin, xmax, ymin, ymax = norm
    Lx = max(float(xmax - xmin), 1.0e-12)
    Ly = max(float(ymax - ymin), 1.0e-12)
    inv_Lx = 1.0 / Lx; inv_Ly = 1.0 / Ly

    xy_wall = np.concatenate([x_w, y_w], axis=1).astype(np.float32)
    xy_inside = np.concatenate([x_i, y_i], axis=1).astype(np.float32)

    d_w_i = compute_wall_distance_feature(xy_inside, xy_wall)
    d_w_w = np.zeros((xy_wall.shape[0], 1), dtype=np.float32)

    from sklearn.neighbors import NearestNeighbors
    spacing_tree = NearestNeighbors(n_neighbors=min(2, xy_inside.shape[0]), algorithm="ball_tree")
    spacing_tree.fit(xy_inside)
    spacing_dists, _ = spacing_tree.kneighbors(xy_inside[: min(512, xy_inside.shape[0])])
    spacing = float(np.median(spacing_dists[:, -1])) if spacing_dists.shape[1] > 1 else 1.0e-3

    graph_mode = str(cfg.training.get("flow_graph_mode", "pixel")).strip().lower()
    progress_knn_k = int(cfg.training.get("progress_knn_k", 8))
    progress_max_edge_len = float(cfg.training.get("progress_max_edge_len", 2.0 * spacing))
    graph_connectivity = int(cfg.training.get("flow_graph_connectivity", 8))

    graph = build_inside_graph(
        xy_inside=xy_inside, inside_raw_xy=inside_raw_xy, norm=norm,
        mode=graph_mode, knn_k=progress_knn_k,
        max_edge_len=progress_max_edge_len,
        pixel_connectivity=graph_connectivity,
    )
    geo_info_in = compute_geodesic_info_on_graph(
        graph=graph, xy_inside=xy_inside, source_xy=inlet_xy,
    )
    geo_info_out = compute_geodesic_info_on_graph(
        graph=graph, xy_inside=xy_inside, source_xy=outlet_xy,
    )
    s_in_i = geo_info_in["s_geo"].astype(np.float32)
    s_out_i = geo_info_out["s_geo"].astype(np.float32)
    s_in_w = project_inside_feature_to_wall(xy_wall, xy_inside, s_in_i)
    s_out_w = project_inside_feature_to_wall(xy_wall, xy_inside, s_out_i)

    # Inlet / outlet adaptive patches — follow the baseline's adaptive mask.
    inlet_raw_xy = np.asarray([[float(inlet_raw_obj["x"]), float(inlet_raw_obj["y"])]], np.float32)
    outlet_raw_xy = np.asarray([[float(outlet_raw_obj["x"]), float(outlet_raw_obj["y"])]], np.float32)
    ensure_patch_has_min_points._inside_raw = inside_raw_xy
    ensure_patch_has_min_points._half_height_px = int(cfg.bc.get("inlet_half_height_px", 1))
    ensure_patch_has_min_points._min_run_per_strip = int(cfg.bc.get("min_run_per_strip", 2))
    ensure_patch_has_min_points._enforce_connected_chain = bool(cfg.bc.get("enforce_connected_chain", True))

    ensure_patch_has_min_points._center_raw = inlet_raw_xy
    ensure_patch_has_min_points._direction = str(cfg.bc.get("inlet_direction", "right"))
    inlet_mask, inlet_r_used = ensure_patch_has_min_points(
        xy_inside=xy_inside, center_xy=inlet_xy,
        r0=float(cfg.bc.inlet_radius_norm),
        min_pts=int(cfg.bc.get("inlet_width_px", cfg.bc.get("min_patch_points", 10))),
        r_max=float(cfg.bc.max_patch_radius_norm),
        grow=float(cfg.bc.patch_growth_factor),
    )
    ensure_patch_has_min_points._center_raw = outlet_raw_xy
    ensure_patch_has_min_points._direction = str(cfg.bc.get("outlet_direction", "right"))
    ensure_patch_has_min_points._half_height_px = int(
        cfg.bc.get("outlet_half_height_px", cfg.bc.get("inlet_half_height_px", 1))
    )
    outlet_mask, outlet_r_used = ensure_patch_has_min_points(
        xy_inside=xy_inside, center_xy=outlet_xy,
        r0=float(cfg.bc.outlet_radius_norm),
        min_pts=int(cfg.bc.get("outlet_width_px", cfg.bc.get("min_patch_points", 10))),
        r_max=float(cfg.bc.max_patch_radius_norm),
        grow=float(cfg.bc.patch_growth_factor),
    )
    print(f"[INFO] inlet points={int(np.sum(inlet_mask))}, inlet_r_used={inlet_r_used:.6f}")
    print(f"[INFO] outlet points={int(np.sum(outlet_mask))}, outlet_r_used={outlet_r_used:.6f}")
    print(f"[INFO] graph_mode={graph_mode}, graph_connectivity={graph_connectivity}, "
          f"knn_k={progress_knn_k}, max_edge_len={progress_max_edge_len:.6f}")

    # Importance-sampled PDE interior subset (curriculum ordering).
    outlet_target_idx = int(geo_info_out["src"])
    pde_target_points = int(cfg.training.get("flow_pde_points_target", xy_inside.shape[0]))
    pde_idx = _build_weighted_flow_pde_indices(
        xy_inside=xy_inside, xy_wall=xy_wall, d_w_i=d_w_i,
        predecessors_in=geo_info_in["predecessors"],
        src_in=int(geo_info_in["src"]),
        outlet_target_idx=outlet_target_idx,
        target_points=pde_target_points,
        wall_boost=float(cfg.training.get("flow_pde_wall_boost", 1.0)),
        wall_scale=float(cfg.training.get("flow_pde_wall_scale", 0.02)),
        corridor_boost=float(cfg.training.get("flow_pde_corridor_boost", 2.0)),
        corridor_radius=float(cfg.training.get("flow_pde_corridor_radius", 0.05)),
        seed=int(cfg.training.get("flow_pde_sampling_seed", 1234)),
    )
    x_i_pde = x_i[pde_idx]; y_i_pde = y_i[pde_idx]
    d_w_i_pde = d_w_i[pde_idx]
    s_in_i_pde = s_in_i[pde_idx]; s_out_i_pde = s_out_i[pde_idx]

    # Seed numpy for sort_by_progress_chunked (which calls np.random.shuffle).
    np.random.seed(int(cfg.training.get("flow_pde_sampling_seed", 1234)))
    (x_i_sorted, y_i_sorted, d_w_i_sorted, s_in_i_sorted, s_out_i_sorted,
     inside_order) = sort_by_progress_chunked(
        x_i=x_i_pde, y_i=y_i_pde, d_w_i=d_w_i_pde,
        s_in_i=s_in_i_pde, s_out_i=s_out_i_pde,
        chunk_size=int(cfg.training.get("curriculum_chunk_size", 8192)),
    )
    print(f"[INFO] PDE interior points used: {int(x_i_sorted.shape[0])} / {int(x_i.shape[0])}")

    # Wall guard points.
    port_exclude_mask = np.zeros((xy_inside.shape[0],), dtype=bool)
    if bool(cfg.training.get("wall_guard_exclude_ports", True)):
        port_exclude_mask |= inlet_mask
        port_exclude_mask |= outlet_mask

    xy_wall_guard = _build_wall_guard_points(
        xy_inside=xy_inside, xy_wall=xy_wall,
        radius=float(cfg.training.get("wall_guard_radius", 0.02)),
        target_points=int(cfg.training.get("wall_guard_points", 4000)),
        seed=int(cfg.training.get("wall_guard_seed", 1234)),
        exclude_mask=port_exclude_mask,
    ) if bool(cfg.training.get("wall_guard_enabled", True)) else np.zeros((0, 2), dtype=np.float32)

    if bool(cfg.training.get("wall_guard_separator_enabled", True)):
        y_mid = 0.5 * (float(inlet_xy[0, 1]) + float(outlet_xy[0, 1]))
        y_half = 0.5 * abs(float(outlet_xy[0, 1]) - float(inlet_xy[0, 1])) \
                 * float(cfg.training.get("wall_guard_separator_span_factor", 0.85))
        y_half = max(y_half, 1.0e-3)
        xy_wall_guard_sep = _build_wall_guard_points(
            xy_inside=xy_inside, xy_wall=xy_wall,
            radius=float(cfg.training.get("wall_guard_separator_radius",
                                          cfg.training.get("wall_guard_radius", 0.02))),
            target_points=int(cfg.training.get("wall_guard_separator_points", 3000)),
            seed=int(cfg.training.get("wall_guard_seed", 1234)) + 17,
            x_max=float(cfg.training.get("wall_guard_separator_x_max", 0.42)),
            y_min=(y_mid - y_half), y_max=(y_mid + y_half),
            exclude_mask=port_exclude_mask,
        )
    else:
        xy_wall_guard_sep = np.zeros((0, 2), dtype=np.float32)

    print(f"[INFO] wall guard points={int(xy_wall_guard.shape[0])}, "
          f"separator guard points={int(xy_wall_guard_sep.shape[0])}")

    # Initial flow guess (pseudo-field for warmup + soft_init).
    init_inside_fields = compute_initial_flow_guess(
        xy_inside=xy_inside, xy_wall=xy_wall,
        inlet_p=float(cfg.bc.get("inlet_p", 1.0)),
        inlet_u=float(cfg.bc.inlet_u), inlet_v=float(cfg.bc.inlet_v),
        geo_info_in=geo_info_in, geo_info_out=geo_info_out,
        velocity_scale=float(cfg.init_guess.get("velocity_scale", 1.0)),
        velocity_power=float(cfg.init_guess.get("velocity_power", 1.0)),
        pressure_power=float(cfg.init_guess.get("pressure_power", 1.0)),
        pressure_drop_guess=float(cfg.init_guess.get("pressure_drop_guess", 0.0)),
    )

    if bool(cfg.init_guess.get("save_into_geometry_json", True)):
        save_init_fields_into_geometry_json(
            geom_obj=geom_obj, geom_json_path=geom_path,
            xy_inside=xy_inside, init_fields_inside=init_inside_fields,
            norm=norm,
        )

    # --- Build network + optimizer ---
    seed = int(cfg.get("training", {}).get("seed", 1234))
    key = random.PRNGKey(seed)
    init_key, _ = random.split(key)
    net, params = init_flow_params(
        init_key,
        hidden_layers=int(cfg.flow_model.hidden_layers),
        hidden_size=int(cfg.flow_model.hidden_size),
    )
    p_count = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"[INFO] FlowNetFlax params: {p_count} "
          f"({cfg.flow_model.hidden_layers}×{cfg.flow_model.hidden_size})")

    optimizer = make_flow_optimizer(
        lr=float(cfg.training.lr),
        lr_decay_rate=float(cfg.training.lr_decay_rate),
        lr_decay_steps=int(cfg.training.lr_decay_steps),
        grad_clip=float(cfg.training.grad_clip_max_norm),
        betas=tuple(float(v) for v in cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    opt_state = optimizer.init(params)

    # --- SAGE backward (cached once; reused across all 5 stages) ---
    sage_backward = build_v4_flow_sage_jax_backward(
        _SAGE_FLOW_DX, _SAGE_FLOW_DY, inv_Lx, inv_Ly, float(cfg.physics.rho),
    )

    w_pde_cont = float(cfg.training.get("w_pde_continuity", 1.0))
    w_pde_momx = float(cfg.training.get("w_pde_momentum_x", 1.0))
    w_pde_momy = float(cfg.training.get("w_pde_momentum_y", 1.0))

    network_dir = Path(to_absolute_path(cfg.network_dir))
    network_dir.mkdir(parents=True, exist_ok=True)
    # Also save a reference to the initial params so we can reload if
    # desired for reproducing the init state.

    start_time = time.time()

    # ---------- Stage −1: init warmup ----------
    stage_steps_init = int(cfg.training.get("k_flow_init", 3000))
    if bool(cfg.training.get("use_init_field_warmup", True)):
        stage_name = "stage_m1_init_guess_warmup"
        print(f"\n[INFO] === {stage_name} ({stage_steps_init} steps) ===")
        t0 = time.time()
        # Build constraint set: only init_field_fit is active.
        stage_active = {
            "init_soft": True,
            "init_soft_scale": float(cfg.training.get("init_field_warmup_scale", 1.0)),
            "init_bs_key": "flow_init_batch_size",
        }
        cs = _make_constraint_set(
            cfg=cfg, stage_idx=-1, num_stages=len(cfg.training.nu_schedule),
            stage_active=stage_active,
            x_w=x_w, y_w=y_w, d_w_w=d_w_w, s_in_w=s_in_w, s_out_w=s_out_w,
            x_i_sorted=x_i_sorted, y_i_sorted=y_i_sorted,
            d_w_i_sorted=d_w_i_sorted, s_in_i_sorted=s_in_i_sorted,
            s_out_i_sorted=s_out_i_sorted,
            x_i=x_i, y_i=y_i, d_w_i=d_w_i, s_in_i=s_in_i, s_out_i=s_out_i,
            inlet_mask=inlet_mask, outlet_mask=outlet_mask,
            init_fields=init_inside_fields,
            xy_inside=xy_inside, xy_wall=xy_wall,
            xy_wall_guard=xy_wall_guard, xy_wall_guard_sep=xy_wall_guard_sep,
            bc_stage_weight_scale=float(cfg.training.get("bc_stage_weight_scale", 1.0)),
            pde_stage_weight_scale=float(cfg.training.get("pde_stage_weight_scale", 1.0)),
        )
        step_fn = _make_warmup_step(net, optimizer)
        rng = np.random.default_rng(seed)
        for step_idx in range(1, stage_steps_init + 1):
            N = cs.init_xy.shape[0]; bs = cs.init_bs
            idx = rng.integers(0, N, size=bs, dtype=np.int64)
            inp = np.concatenate(
                [cs.init_xy[idx, 0:1], cs.init_xy[idx, 1:2],
                 cs.init_dw[idx], cs.init_sin[idx], cs.init_sout[idx]], axis=1,
            )
            params, opt_state, loss = step_fn(
                params, opt_state,
                jnp.asarray(inp),
                jnp.asarray(cs.init_uvp_target[idx]),
                jnp.asarray(cs.init_lam_uvp[idx]),
            )
            if step_idx == 1 or step_idx % int(cfg.training.print_stats_freq) == 0:
                print(f"[{stage_name}][step {step_idx:06d}] loss={float(loss):.6e}", flush=True)
        stage_dir = network_dir / stage_name
        save_params(params, stage_dir / "flow_network.pkl")
        dt = (time.time() - t0) / 60.0
        print(f"[{stage_name}] complete: {dt:.3f} min", flush=True)

    # ---------- Stage 0: BC warmup (no PDE) ----------
    stage_steps_bc = int(cfg.training.get("k_flow_bc", 2000))
    stage_name = "stage_00_bc_warmup"
    print(f"\n[INFO] === {stage_name} ({stage_steps_bc} steps) ===")
    t0 = time.time()
    stage_active = {
        "wall": True, "inlet": True,
        "outlet_p": bool(cfg.bc.get("use_outlet_pressure_constraint", False)),
        "inlet_p": (bool(cfg.bc.get("use_inlet_pressure_anchor", False))
                    and not bool(cfg.bc.get("use_inlet_pressure_anchor_last_stage_only", False))),
        "init_soft": bool(cfg.training.get("use_soft_init_during_bc", True)),
        "init_soft_scale": float(cfg.training.get("soft_init_bc_scale", 1.0)),
        "init_bs_key": "flow_soft_init_batch_size",
        "geo": bool(cfg.training.get("use_geo_guidance_during_bc", True))
               and bool(cfg.training.get("use_geo_direction_guidance", True)),
        "geo_scale": float(cfg.training.get("geo_guidance_bc_scale", 1.0)),
        "wg": bool(cfg.training.get("wall_guard_enabled", True)),
        "wgs": bool(cfg.training.get("wall_guard_separator_enabled", True)),
        "wg_scale": float(cfg.training.get("wall_guard_scale", 1.0)),
    }
    cs = _make_constraint_set(
        cfg=cfg, stage_idx=0, num_stages=len(cfg.training.nu_schedule),
        stage_active=stage_active,
        x_w=x_w, y_w=y_w, d_w_w=d_w_w, s_in_w=s_in_w, s_out_w=s_out_w,
        x_i_sorted=x_i_sorted, y_i_sorted=y_i_sorted,
        d_w_i_sorted=d_w_i_sorted, s_in_i_sorted=s_in_i_sorted,
        s_out_i_sorted=s_out_i_sorted,
        x_i=x_i, y_i=y_i, d_w_i=d_w_i, s_in_i=s_in_i, s_out_i=s_out_i,
        inlet_mask=inlet_mask, outlet_mask=outlet_mask,
        init_fields=init_inside_fields,
        xy_inside=xy_inside, xy_wall=xy_wall,
        xy_wall_guard=xy_wall_guard, xy_wall_guard_sep=xy_wall_guard_sep,
        bc_stage_weight_scale=float(cfg.training.get("bc_stage_weight_scale", 1.0)),
        pde_stage_weight_scale=float(cfg.training.get("pde_stage_weight_scale", 1.0)),
    )
    step_fn = _make_full_step(
        net=net, optimizer=optimizer, sage_backward=sage_backward,
        dx=_SAGE_FLOW_DX, dy=_SAGE_FLOW_DY,
        inv_Lx=inv_Lx, inv_Ly=inv_Ly, rho=float(cfg.physics.rho),
        w_pde_cont=w_pde_cont, w_pde_momx=w_pde_momx, w_pde_momy=w_pde_momy,
        include_pde=False,
        include_wall=True, include_inlet=True,
        include_outlet_p=stage_active["outlet_p"],
        include_inlet_p=stage_active["inlet_p"],
        include_init_soft=stage_active["init_soft"],
        include_geo_dir=stage_active["geo"],
        include_geo_parallel=stage_active["geo"],
        include_wg=stage_active["wg"] and (cs.wg_xy is not None),
        include_wgs=stage_active["wgs"] and (cs.wgs_xy is not None),
    )
    rng = np.random.default_rng(seed + 1)
    pde_cursor = [0]
    for step_idx in range(1, stage_steps_bc + 1):
        batch = _assemble_batch(
            cs=cs, rng=rng, pde_cursor=pde_cursor,
            include_pde=False,
            include_wall=True, include_inlet=True,
            include_outlet_p=stage_active["outlet_p"],
            include_inlet_p=stage_active["inlet_p"],
            include_init_soft=stage_active["init_soft"],
            include_geo_dir=stage_active["geo"] and (cs.geo_xy is not None),
            include_geo_parallel=stage_active["geo"] and (cs.geo_xy is not None),
            include_wg=stage_active["wg"] and (cs.wg_xy is not None),
            include_wgs=stage_active["wgs"] and (cs.wgs_xy is not None),
            nu_stage_val=float(cfg.training.nu_schedule[0]),
            w_pde_cont=w_pde_cont, w_pde_momx=w_pde_momx, w_pde_momy=w_pde_momy,
            dx=_SAGE_FLOW_DX, dy=_SAGE_FLOW_DY,
        )
        params, opt_state, aux = step_fn(params, opt_state, batch)
        if step_idx == 1 or step_idx % int(cfg.training.print_stats_freq) == 0:
            total = float(aux["total"])
            print(f"[{stage_name}][step {step_idx:06d}] loss={total:.6e} "
                  f"wall={float(aux['wall']):.3e} inlet={float(aux['inlet']):.3e} "
                  f"init={float(aux['init_soft']):.3e} geo_cr={float(aux['geo_cross']):.3e} "
                  f"wg={float(aux['wall_guard']):.3e}", flush=True)
    stage_dir = network_dir / stage_name
    save_params(params, stage_dir / "flow_network.pkl")
    dt = (time.time() - t0) / 60.0
    print(f"[{stage_name}] complete: {dt:.3f} min", flush=True)

    # ---------- Stages 1/2/3: nu continuation ----------
    n_stages = len(cfg.training.nu_schedule)
    for stage_idx, nu_stage in enumerate(cfg.training.nu_schedule):
        stage_steps = int(cfg.training.k_flow_per_stage[stage_idx])
        stage_name = f"stage_{stage_idx + 1:02d}_nu_{float(nu_stage):.2e}".replace("+", "")
        print(f"\n[INFO] === {stage_name} ({stage_steps} steps, nu={nu_stage:g}) ===")
        t0 = time.time()

        soft_init_scale = _linear_stage_scale(
            stage_idx=stage_idx, num_stages=n_stages,
            start_scale=float(cfg.training.get("soft_init_start_scale", 0.5)),
            end_scale=float(cfg.training.get("soft_init_end_scale", 0.05)),
        )
        geo_scale = _linear_stage_scale(
            stage_idx=stage_idx, num_stages=n_stages,
            start_scale=float(cfg.training.get("geo_guidance_start_scale", 1.0)),
            end_scale=float(cfg.training.get("geo_guidance_end_scale", 0.1)),
        )
        is_last_stage = (stage_idx == n_stages - 1)
        inlet_p_active = (bool(cfg.bc.get("use_inlet_pressure_anchor", False))
                          and (is_last_stage or not bool(cfg.bc.get("use_inlet_pressure_anchor_last_stage_only", False))))

        stage_active = {
            "pde": True, "wall": True, "inlet": True,
            "outlet_p": bool(cfg.bc.get("use_outlet_pressure_constraint", False)),
            "inlet_p": inlet_p_active,
            "init_soft": bool(cfg.training.get("use_soft_init_constraint", True)),
            "init_soft_scale": soft_init_scale,
            "init_bs_key": "flow_soft_init_batch_size",
            "geo": bool(cfg.training.get("use_geo_direction_guidance", True)),
            "geo_scale": geo_scale,
            "wg": bool(cfg.training.get("wall_guard_enabled", True)),
            "wgs": bool(cfg.training.get("wall_guard_separator_enabled", True)),
            "wg_scale": float(cfg.training.get("wall_guard_scale", 1.0)),
        }
        cs = _make_constraint_set(
            cfg=cfg, stage_idx=stage_idx, num_stages=n_stages,
            stage_active=stage_active,
            x_w=x_w, y_w=y_w, d_w_w=d_w_w, s_in_w=s_in_w, s_out_w=s_out_w,
            x_i_sorted=x_i_sorted, y_i_sorted=y_i_sorted,
            d_w_i_sorted=d_w_i_sorted, s_in_i_sorted=s_in_i_sorted,
            s_out_i_sorted=s_out_i_sorted,
            x_i=x_i, y_i=y_i, d_w_i=d_w_i, s_in_i=s_in_i, s_out_i=s_out_i,
            inlet_mask=inlet_mask, outlet_mask=outlet_mask,
            init_fields=init_inside_fields,
            xy_inside=xy_inside, xy_wall=xy_wall,
            xy_wall_guard=xy_wall_guard, xy_wall_guard_sep=xy_wall_guard_sep,
            bc_stage_weight_scale=float(cfg.training.get("bc_stage_weight_scale", 1.0)),
            pde_stage_weight_scale=float(cfg.training.get("pde_stage_weight_scale", 1.0)),
        )
        step_fn = _make_full_step(
            net=net, optimizer=optimizer, sage_backward=sage_backward,
            dx=_SAGE_FLOW_DX, dy=_SAGE_FLOW_DY,
            inv_Lx=inv_Lx, inv_Ly=inv_Ly, rho=float(cfg.physics.rho),
            w_pde_cont=w_pde_cont, w_pde_momx=w_pde_momx, w_pde_momy=w_pde_momy,
            include_pde=True, include_wall=True, include_inlet=True,
            include_outlet_p=stage_active["outlet_p"],
            include_inlet_p=stage_active["inlet_p"],
            include_init_soft=stage_active["init_soft"],
            include_geo_dir=stage_active["geo"] and (cs.geo_xy is not None),
            include_geo_parallel=stage_active["geo"] and (cs.geo_xy is not None),
            include_wg=stage_active["wg"] and (cs.wg_xy is not None),
            include_wgs=stage_active["wgs"] and (cs.wgs_xy is not None),
        )
        rng = np.random.default_rng(seed + 1 + stage_idx)
        pde_cursor = [0]
        for step_idx in range(1, stage_steps + 1):
            batch = _assemble_batch(
                cs=cs, rng=rng, pde_cursor=pde_cursor,
                include_pde=True, include_wall=True, include_inlet=True,
                include_outlet_p=stage_active["outlet_p"],
                include_inlet_p=stage_active["inlet_p"],
                include_init_soft=stage_active["init_soft"],
                include_geo_dir=stage_active["geo"] and (cs.geo_xy is not None),
                include_geo_parallel=stage_active["geo"] and (cs.geo_xy is not None),
                include_wg=stage_active["wg"] and (cs.wg_xy is not None),
                include_wgs=stage_active["wgs"] and (cs.wgs_xy is not None),
                nu_stage_val=float(nu_stage),
                w_pde_cont=w_pde_cont, w_pde_momx=w_pde_momx, w_pde_momy=w_pde_momy,
                dx=_SAGE_FLOW_DX, dy=_SAGE_FLOW_DY,
            )
            params, opt_state, aux = step_fn(params, opt_state, batch)
            if step_idx == 1 or step_idx % int(cfg.training.print_stats_freq) == 0:
                total = float(aux["total"])
                print(f"[{stage_name}][step {step_idx:06d}] loss={total:.6e} "
                      f"pde_c={float(aux['pde_cont']):.3e} "
                      f"pde_mx={float(aux['pde_momx']):.3e} "
                      f"pde_my={float(aux['pde_momy']):.3e} "
                      f"wall={float(aux['wall']):.3e} inlet={float(aux['inlet']):.3e} "
                      f"init={float(aux['init_soft']):.3e}", flush=True)
        stage_dir = network_dir / stage_name
        save_params(params, stage_dir / "flow_network.pkl")
        dt = (time.time() - t0) / 60.0
        print(f"[{stage_name}] complete: {dt:.3f} min", flush=True)

    total_time = (time.time() - start_time) / 60.0
    print(f"\n[INFO] Flow training complete: {total_time:.3f} min total")

    # ---------- Inference ----------
    print("[INFO] running inference over all wall+inside points ...")
    xy_all = np.vstack([xy_wall, xy_inside]).astype(np.float32)
    point_type = np.concatenate(
        [np.full((xy_wall.shape[0],), 1, np.int32),
         np.full((xy_inside.shape[0],), 2, np.int32)], axis=0,
    )
    d_w_all = np.vstack([d_w_w, d_w_i]).astype(np.float32)
    s_in_all = np.vstack([s_in_w, s_in_i]).astype(np.float32)
    s_out_all = np.vstack([s_out_w, s_out_i]).astype(np.float32)

    N = xy_all.shape[0]
    batch_size = 65536

    @jax.jit
    def infer(p, xy):
        return net.apply(p, xy)

    flow_fields = {
        "u": np.zeros((N,), dtype=np.float32),
        "v": np.zeros((N,), dtype=np.float32),
        "p": np.zeros((N,), dtype=np.float32),
    }
    for s0 in range(0, N, batch_size):
        s1 = min(s0 + batch_size, N)
        inp = np.concatenate(
            [xy_all[s0:s1, 0:1], xy_all[s0:s1, 1:2],
             d_w_all[s0:s1], s_in_all[s0:s1], s_out_all[s0:s1]], axis=1,
        )
        out = infer(params, jnp.asarray(inp))
        out_np = np.asarray(out)
        flow_fields["u"][s0:s1] = out_np[:, 0]
        flow_fields["v"][s0:s1] = out_np[:, 1]
        flow_fields["p"][s0:s1] = out_np[:, 2]

    nw = xy_wall.shape[0]
    flow_fields["u"][:nw] = 0.0
    flow_fields["v"][:nw] = 0.0

    stem = str(Path(geom_path).with_suffix(""))
    out_json_path = stem + "_pred_flow_steady.json"
    _write_multi_field_inference_json(
        out_json_path,
        xy_norm=xy_all, fields_dict=flow_fields,
        norm=norm, point_type=point_type,
    )

    if bool(cfg.inference.get("save_plot", False)):
        out_plot_path = stem + "_pred_flow_steady.png"
        _plot_fields(xy_all, flow_fields, out_plot_path,
                     title_prefix="Steady flow (JAX-SAGE)")

    print(f"[OK] wrote flow results to: {out_json_path}")
    print(f"Total elapsed training time: {total_time:.4f} minutes")


if __name__ == "__main__":
    run()
