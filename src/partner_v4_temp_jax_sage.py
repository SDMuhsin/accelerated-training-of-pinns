"""V4 temperature trainer — JAX-SAGE variant.

Apples-to-apples counterpart of ``partner_v4_temp.py``. Every sample,
every seed, every schedule, every loss weight, every physics constant
matches the baseline. Permitted divergences (per ``llmdocs/CONTEXT.md``
§ 7.3):

1. Flax init != PyTorch init — so convergence is compared across seeds
   rather than per-step parity.
2. JAX PRNGKey management — numpy RNG seeded from ``cfg.training.seed``
   draws the same SEMANTIC choices at each step (same pde_idx /
   t_pde / ic_idx / arr_idx / pre_idx / frac / inlet / outlet) so the
   sampler is identical in kind, not just in seed.
3. No PhysicsNeMo — we hand-implement Adam + StepLR + grad clipping
   using ``optax``.
4. JIT'd whole-step.

Reads the same flow JSON the baseline consumes; writes outputs to
``results/partner_v4_jax_sage/temp/`` (new dir — does not touch
baseline or PyTorch SAGE checkpoints).

Run via ``src/partner_v4_e2e_jax_sage.py`` or directly::

    source env/bin/activate
    export PCS_CAD_PATH=./data/partner_v4/designs/Study_Model_B_1st_4p3T.step
    export PCS_GEOM_JSON_PATH=./data/partner_v4/pipe_three_class_fixed.json
    python src/partner_v4_temp_jax_sage.py hydra.job.chdir=False
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

import jax
import jax.numpy as jnp
from jax import random
import flax
import flax.linen as fnn
import optax

from pcs_runtime import cad_to_geometry_json, derive_flow_json_path, resolve_cad_path
from sage_ns_v4_jax import (
    TempNetFlax, init_temp_params,
    build_v4_temp_sage_jax_backward,
    reshape_temp_pred_to_stencil_stack, reshape_temp_adj_to_stencil_stack,
    temp_stencil_inputs,
    make_temp_optimizer,
    _SAGE_TEMP_DX, _SAGE_TEMP_DY, _SAGE_TEMP_DT,
)


# ---------------------------------------------------------------------------
# Preprocessing (byte-identical to partner_v4_temp.py)
# ---------------------------------------------------------------------------
def points_in_radius(xy: np.ndarray, center_xy: np.ndarray, r: float) -> np.ndarray:
    dx = xy[:, 0] - center_xy[0, 0]
    dy = xy[:, 1] - center_xy[0, 1]
    return (dx * dx + dy * dy) <= (r * r)


def _normalize_xy(
    x_raw: np.ndarray, y_raw: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float
) -> Tuple[np.ndarray, np.ndarray]:
    xden = (xmax - xmin) if (xmax > xmin) else 1.0
    yden = (ymax - ymin) if (ymax > ymin) else 1.0
    x = (x_raw - xmin) / xden
    y = (y_raw - ymin) / yden
    return x.astype(np.float32), y.astype(np.float32)


def _denormalize_xy(
    x: np.ndarray, y: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float
) -> Tuple[np.ndarray, np.ndarray]:
    xden = (xmax - xmin) if (xmax > xmin) else 1.0
    yden = (ymax - ymin) if (ymax > ymin) else 1.0
    xr = x * xden + xmin
    yr = y * yden + ymin
    return xr.astype(np.float32), yr.astype(np.float32)


def _load_geometry_ports(json_path: str):
    obj = json.loads(Path(json_path).read_text())
    pts = obj["points"]
    inlet = obj.get("inlet", None)
    outlet = obj.get("outlet", None)
    all_x, all_y = [], []
    for p in pts:
        xr, yr, typ = float(p[0]), float(p[1]), int(p[2])
        if typ == 0:
            continue
        all_x.append(xr); all_y.append(yr)
    if not all_x:
        raise ValueError("No non-background geometry points found.")
    norm = (float(min(all_x)), float(max(all_x)), float(min(all_y)), float(max(all_y)))
    return inlet, outlet, norm


@dataclass
class FlowFieldData:
    xy_norm: np.ndarray
    xy_raw: np.ndarray
    point_type: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: Optional[np.ndarray]
    norm: Tuple[float, float, float, float]
    flow_times: Optional[np.ndarray]


def _select_flow_snapshot(field_value, flow_time_index: int) -> np.ndarray:
    arr = np.asarray(field_value, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Expected flow field to have 1 or 2 dims, got shape {arr.shape}")
    if flow_time_index < 0 or flow_time_index >= arr.shape[0]:
        raise ValueError(f"flow_time_index={flow_time_index} out of range for shape {arr.shape}")
    return arr[flow_time_index]


def _load_flow_field_json(json_path: str, flow_time_index: int = 0) -> FlowFieldData:
    obj = json.loads(Path(json_path).read_text())
    if "fields" not in obj or "u" not in obj["fields"] or "v" not in obj["fields"]:
        raise ValueError("Flow JSON missing fields.u/v.")
    xy_norm = np.asarray(obj["xy"], dtype=np.float32)
    point_type = np.asarray(obj["point_type"], dtype=np.int32)
    norm = tuple(float(v) for v in obj["norm"])
    u = _select_flow_snapshot(obj["fields"]["u"], flow_time_index)
    v = _select_flow_snapshot(obj["fields"]["v"], flow_time_index)
    p = None
    if "p" in obj["fields"]:
        p = _select_flow_snapshot(obj["fields"]["p"], flow_time_index)
    if "xy_raw" in obj:
        xy_raw = np.asarray(obj["xy_raw"], dtype=np.float32)
    else:
        xr, yr = _denormalize_xy(xy_norm[:, 0:1], xy_norm[:, 1:2], norm[0], norm[1], norm[2], norm[3])
        xy_raw = np.concatenate([xr, yr], axis=1).astype(np.float32)
    flow_times = None
    if "times" in obj:
        flow_times = np.asarray(obj["times"], dtype=np.float32)
    return FlowFieldData(
        xy_norm=xy_norm, xy_raw=xy_raw, point_type=point_type,
        u=np.asarray(u, dtype=np.float32), v=np.asarray(v, dtype=np.float32),
        p=None if p is None else np.asarray(p, dtype=np.float32),
        norm=norm, flow_times=flow_times,
    )


def _normalize_port(port, norm):
    if port is None:
        return None
    xr = np.asarray([[float(port["x"])]], dtype=np.float32)
    yr = np.asarray([[float(port["y"])]], dtype=np.float32)
    xn, yn = _normalize_xy(xr, yr, norm[0], norm[1], norm[2], norm[3])
    return np.asarray([[float(xn[0, 0]), float(yn[0, 0])]], dtype=np.float32)


def compute_geodesic_distance_from_inlet(
    xy_inside: np.ndarray, inlet_xy: np.ndarray,
    k: int = 8, max_edge_len: Optional[float] = None,
) -> np.ndarray:
    npts = xy_inside.shape[0]
    dx = xy_inside[:, 0] - inlet_xy[0, 0]
    dy = xy_inside[:, 1] - inlet_xy[0, 1]
    source_idx = int(np.argmin(dx * dx + dy * dy))
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    knn.fit(xy_inside)
    distances, indices = knn.kneighbors(xy_inside)
    rows, cols, vals = [], [], []
    for i in range(npts):
        for j_idx in range(1, k + 1):
            j = int(indices[i, j_idx])
            d = float(distances[i, j_idx])
            if max_edge_len is not None and d > max_edge_len:
                continue
            rows.extend([i, j]); cols.extend([j, i]); vals.extend([d, d])
    graph = csr_matrix((vals, (rows, cols)), shape=(npts, npts))
    dist = shortest_path(graph, method="D", indices=source_idx, directed=False).astype(np.float32)
    if np.any(np.isinf(dist)):
        euc = np.sqrt(dx * dx + dy * dy).astype(np.float32)
        dist[np.isinf(dist)] = euc[np.isinf(dist)]
    return dist


def estimate_point_spacing(xy_inside: np.ndarray, sample_n: int = 500) -> float:
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(xy_inside)
    idx = np.random.choice(len(xy_inside), min(sample_n, len(xy_inside)), replace=False)
    dists, _ = knn.kneighbors(xy_inside[idx])
    return float(np.median(dists[:, 1]))


# ---------------------------------------------------------------------------
# Checkpoint save/restore (JAX-native: pickle of flax params pytree)
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
# H5 + visualization (reuse partner_v4_temp formats)
# ---------------------------------------------------------------------------
def _save_h5(path, xy_norm, xy_raw, point_type, times, temperature, u, v,
             norm, flow_json_path, geom_json_path):
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("xy_norm", data=xy_norm)
        h5f.create_dataset("xy_raw", data=xy_raw)
        h5f.create_dataset("point_type", data=point_type)
        h5f.create_dataset("times", data=times)
        h5f.create_dataset("temperature", data=temperature)
        h5f.create_dataset("u", data=u)
        h5f.create_dataset("v", data=v)
        h5f.attrs["xmin"] = float(norm[0]); h5f.attrs["xmax"] = float(norm[1])
        h5f.attrs["ymin"] = float(norm[2]); h5f.attrs["ymax"] = float(norm[3])
        h5f.attrs["flow_json_path"] = flow_json_path
        h5f.attrs["geom_json_path"] = geom_json_path


def _run_visualization(geom_path: str, pred_path: str, outdir: Path) -> None:
    viz_script = Path(__file__).resolve().parent / "visualize_partner_v4.py"
    if not viz_script.exists():
        print(f"[WARN] visualize script not found at: {viz_script}")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(viz_script), "--geom", geom_path,
           "--pred", pred_path, "--outdir", str(outdir),
           "--field", "temperature", "--gif"]
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] visualization artifacts written to: {outdir}")
    except Exception as exc:
        print(f"[WARN] visualization step failed: {exc}")


# ---------------------------------------------------------------------------
# Sampler helpers — use numpy RNG seeded the same way as the baseline.
# ---------------------------------------------------------------------------
def _sample_indices(rng: np.random.Generator, size: int, batch_size: int) -> np.ndarray:
    """Sample ``min(size, batch_size)`` indices with replacement.

    Mirrors torch.randint semantics used by ``partner_v4_temp.py`` — i.e.
    uniform-with-replacement, matching the baseline's pde_idx / bc_idx
    draws up to framework RNG kind.
    """
    n = min(size, batch_size)
    return rng.integers(0, size, size=n, dtype=np.int64)


def _make_time_samples_np(rng: np.random.Generator, n: int, t_min: float,
                          t_max: float) -> np.ndarray:
    """Same shape as ``torch.rand(n, 1).requires_grad_(True)``: u²-biased."""
    u = rng.random((n, 1), dtype=np.float32)
    return (t_min + (t_max - t_min) * (u ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Training step — custom-built to stay close to the baseline
# ---------------------------------------------------------------------------
def _build_train_step(net: TempNetFlax, optimizer: optax.GradientTransformation,
                      sage_backward, dx: float, dy: float, dt_fd: float,
                      D: float, Q: float,
                      w_pde: float, w_ic: float, w_arr: float, w_pre: float,
                      w_inlet: float, w_outlet: float,
                      has_inlet: bool, has_outlet: bool):
    apply_fn = net.apply

    def _outlet_dTdx(params, x_out, y_out, t_out, u_out, v_out):
        """Per-point ∂T/∂x for the outlet BC. Auto-batched with vmap."""
        def T_scalar(xs, ys, ts, us, vs):
            inp = jnp.stack([xs, ys, ts, us, vs]).reshape(1, 5)
            return apply_fn(params, inp)[0, 0]
        grad_T_x = jax.vmap(jax.grad(T_scalar, argnums=0))(
            x_out[:, 0], y_out[:, 0], t_out[:, 0], u_out[:, 0], v_out[:, 0]
        )
        return grad_T_x.reshape(-1, 1)

    @jax.jit
    def train_step(params, opt_state, batch):
        # PDE via SAGE-JAX
        pde_stencil = batch["pde_stencil"]   # (7B_pde, 5)
        u_pde = batch["u_pde"]; v_pde = batch["v_pde"]
        B_pde = batch["u_pde"].shape[0]

        def pde_forward(p):
            return apply_fn(p, pde_stencil)
        pde_pred_all, pde_vjp = jax.vjp(pde_forward, params)  # (7B, 1)
        pred_stack = reshape_temp_pred_to_stencil_stack(pde_pred_all, B_pde)  # (B, 7)
        T0 = pred_stack[:, 0:1]
        T_xp = pred_stack[:, 1:2]; T_xm = pred_stack[:, 2:3]
        T_yp = pred_stack[:, 3:4]; T_ym = pred_stack[:, 4:5]
        T_tp = pred_stack[:, 5:6]; T_tm = pred_stack[:, 6:7]
        T_x = (T_xp - T_xm) / (2.0 * dx)
        T_y = (T_yp - T_ym) / (2.0 * dy)
        T_t = (T_tp - T_tm) / (2.0 * dt_fd)
        T_xx = (T_xp + T_xm - 2.0 * T0) / (dx * dx)
        T_yy = (T_yp + T_ym - 2.0 * T0) / (dy * dy)
        residual = T_t + u_pde * T_x + v_pde * T_y - D * (T_xx + T_yy) - Q
        loss_pde = jnp.mean(residual ** 2)

        dr = 2.0 * residual * float(w_pde) / float(B_pde)
        g_sage = {"u": u_pde, "v": v_pde, "N_all": B_pde}
        adj_stack = sage_backward(pred_stack, g_sage, dr)  # (B, 7)
        adj_pde = reshape_temp_adj_to_stencil_stack(adj_stack)
        (pde_param_grads,) = pde_vjp(adj_pde)

        # BC block via jax.grad
        def bc_loss(p):
            total = jnp.asarray(0.0, dtype=jnp.float32)
            aux = {}

            T_ic = apply_fn(p, batch["ic"])
            l_ic = jnp.mean((T_ic - batch["ic_target"]) ** 2)
            total = total + float(w_ic) * l_ic
            aux["ic"] = l_ic

            T_arr = apply_fn(p, batch["arr"])
            l_arr = jnp.mean((T_arr - batch["arr_target"]) ** 2)
            total = total + float(w_arr) * l_arr
            aux["arrival"] = l_arr

            T_pre = apply_fn(p, batch["pre"])
            l_pre = jnp.mean((T_pre - batch["pre_target"]) ** 2)
            total = total + float(w_pre) * l_pre
            aux["pre"] = l_pre

            if has_inlet:
                T_inlet = apply_fn(p, batch["inlet"])
                l_in = jnp.mean((T_inlet - batch["inlet_target"]) ** 2)
                total = total + float(w_inlet) * l_in
                aux["inlet"] = l_in
            else:
                aux["inlet"] = jnp.asarray(0.0, dtype=jnp.float32)

            if has_outlet:
                grad_T_x = _outlet_dTdx(
                    p, batch["outlet_x"], batch["outlet_y"], batch["outlet_t"],
                    batch["outlet_u"], batch["outlet_v"],
                )
                l_out = jnp.mean(grad_T_x ** 2)
                total = total + float(w_outlet) * l_out
                aux["outlet"] = l_out
            else:
                aux["outlet"] = jnp.asarray(0.0, dtype=jnp.float32)

            return total, aux

        (bc_loss_val, bc_aux), bc_grads = jax.value_and_grad(
            bc_loss, has_aux=True)(params)
        combined = jax.tree_util.tree_map(lambda a, b: a + b,
                                           pde_param_grads, bc_grads)
        updates, new_opt_state = optimizer.update(combined, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        total_loss = float(w_pde) * loss_pde + bc_loss_val
        aux = dict(bc_aux)
        aux["pde"] = loss_pde
        aux["total"] = total_loss
        return new_params, new_opt_state, aux

    return train_step


# ---------------------------------------------------------------------------
# Hydra entry
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="partner_v4_config")
def run(cfg: DictConfig) -> None:
    cfg = cfg.temp if "temp" in cfg else cfg
    seed = int(cfg.training.seed)
    np.random.seed(seed)

    cad_path = resolve_cad_path()
    geom_json_path = cad_to_geometry_json(
        cad_path=cad_path,
        output_dir=Path.cwd(),
        res=int(getattr(cfg.problem, "pcs_res", 512)),
        strip_w=int(getattr(cfg.problem, "pcs_strip_w", 10)),
        white_thr=int(getattr(cfg.problem, "pcs_white_thr", 250)),
    )
    flow_json = derive_flow_json_path(geom_json_path)
    if not flow_json.exists():
        raise FileNotFoundError(
            f"Flow JSON not found: {flow_json}. "
            "Run partner_v4_flow{,_sage,_jax_sage}.py with the same CAD path first."
        )
    geom_path = str(geom_json_path)
    flow_json_path = str(flow_json)
    print(f"[INFO] using CAD: {cad_path}")
    print(f"[INFO] generated geometry json: {geom_path}")
    print(f"[INFO] using flow json: {flow_json_path}")

    network_dir = Path(to_absolute_path(cfg.network_dir))
    network_dir.mkdir(parents=True, exist_ok=True)

    inlet_port, outlet_port, geom_norm = _load_geometry_ports(geom_path)
    flow = _load_flow_field_json(flow_json_path, int(cfg.problem.flow_time_index))
    norm = flow.norm if len(flow.norm) == 4 else geom_norm

    inlet_xy = _normalize_port(inlet_port, norm)
    outlet_xy = _normalize_port(outlet_port, norm)

    xy = flow.xy_norm.astype(np.float32)
    xy_raw = flow.xy_raw.astype(np.float32)
    point_type = flow.point_type.astype(np.int32)
    u_all = flow.u.reshape(-1, 1).astype(np.float32)
    v_all = flow.v.reshape(-1, 1).astype(np.float32)

    wall_mask = point_type == 1
    inside_mask = point_type == 2

    if not np.any(inside_mask):
        raise ValueError("Flow JSON does not contain any inside points (point_type == 2).")
    if not np.any(wall_mask):
        raise ValueError("Flow JSON does not contain any wall points (point_type == 1).")
    if inlet_xy is None:
        raise ValueError("Geometry JSON must contain an inlet point.")

    xy_inside = xy[inside_mask]
    u_inside = u_all[inside_mask]
    v_inside = v_all[inside_mask]

    spacing = estimate_point_spacing(xy_inside)
    max_edge_len = float(cfg.problem.geodesic_max_edge_len)
    if max_edge_len <= 0.0:
        max_edge_len = 2.0 * spacing

    dist_inside = compute_geodesic_distance_from_inlet(
        xy_inside, inlet_xy, k=int(cfg.problem.geodesic_knn_k),
        max_edge_len=max_edge_len,
    )
    mean_flow_speed = max(float(cfg.problem.arrival_speed), 1e-6)
    t_arrive_inside = np.clip(
        dist_inside / mean_flow_speed,
        float(cfg.problem.t_min),
        float(cfg.problem.t_max),
    ).astype(np.float32).reshape(-1, 1)

    inlet_mask_inside = points_in_radius(xy_inside, inlet_xy, float(cfg.bc.inlet_radius_norm))
    outlet_mask_inside = (
        points_in_radius(xy_inside, outlet_xy, float(cfg.bc.outlet_radius_norm))
        if outlet_xy is not None
        else None
    )

    has_inlet = bool(np.any(inlet_mask_inside))
    has_outlet = (outlet_mask_inside is not None) and bool(np.any(outlet_mask_inside))
    inlet_idx = np.where(inlet_mask_inside)[0].astype(np.int64) if has_inlet else None
    outlet_idx = np.where(outlet_mask_inside)[0].astype(np.int64) if has_outlet else None

    # --- Network + optimizer ---
    key = random.PRNGKey(seed)
    init_key, _ = random.split(key)
    net, params = init_temp_params(
        init_key,
        hidden_layers=int(cfg.model.hidden_layers),
        hidden_size=int(cfg.model.hidden_size),
    )
    p_count = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print(f"[INFO] TempNetFlax params: {p_count} ({cfg.model.hidden_layers}×{cfg.model.hidden_size})")

    optimizer = make_temp_optimizer(
        lr=float(cfg.training.lr),
        lr_decay_rate=float(cfg.training.lr_decay_rate),
        lr_decay_steps=int(cfg.training.lr_decay_steps),
        grad_clip=float(cfg.training.grad_clip_max_norm),
        betas=tuple(float(v) for v in cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    opt_state = optimizer.init(params)

    # --- SAGE backward (cached once) ---
    D_phys = float(cfg.physics.D); Q_phys = float(cfg.physics.Q)
    sage_backward = build_v4_temp_sage_jax_backward(
        _SAGE_TEMP_DX, _SAGE_TEMP_DY, _SAGE_TEMP_DT, D_phys, Q_phys,
    )

    t_min = float(cfg.problem.t_min); t_max = float(cfg.problem.t_max)
    t_init_val = float(cfg.problem.T_init); inlet_t_val = float(cfg.bc.inlet_T)
    loss_weights = cfg.training.loss_weights

    train_step = _build_train_step(
        net=net, optimizer=optimizer, sage_backward=sage_backward,
        dx=_SAGE_TEMP_DX, dy=_SAGE_TEMP_DY, dt_fd=_SAGE_TEMP_DT,
        D=D_phys, Q=Q_phys,
        w_pde=float(loss_weights.pde),
        w_ic=float(loss_weights.ic),
        w_arr=float(loss_weights.arrival),
        w_pre=float(loss_weights.pre_arrival),
        w_inlet=float(loss_weights.inlet),
        w_outlet=float(loss_weights.outlet),
        has_inlet=has_inlet,
        has_outlet=has_outlet,
    )

    # --- Data on host (we copy mini-batches to device per step) ---
    x_all = xy[:, 0:1].astype(np.float32)
    y_all = xy[:, 1:2].astype(np.float32)
    x_inside = xy_inside[:, 0:1].astype(np.float32)
    y_inside = xy_inside[:, 1:2].astype(np.float32)

    # --- Run mode ---
    ckpt_path = network_dir / "temperature_net.pkl"
    if str(cfg.run_mode).lower() == "eval":
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for eval mode: {ckpt_path}")
        params = load_params(ckpt_path)
    else:
        rng = np.random.default_rng(seed)
        pde_bs = int(cfg.training.pde_batch_size)
        bc_bs = int(cfg.training.bc_batch_size)
        max_steps = int(cfg.training.max_steps)
        print_freq = int(cfg.training.print_stats_freq)
        save_freq = int(cfg.training.save_network_freq)

        t_wall0 = time.perf_counter()
        for step in range(1, max_steps + 1):
            # ---- draw pde batch (2048) ----
            pde_i = _sample_indices(rng, x_inside.shape[0], pde_bs)
            x_pde = x_inside[pde_i]; y_pde = y_inside[pde_i]
            t_pde = _make_time_samples_np(rng, x_pde.shape[0], t_min, t_max)
            u_p = u_inside[pde_i]; v_p = v_inside[pde_i]

            B_pde = x_pde.shape[0]
            # stack stencil inputs for SAGE: (7*B, 5)
            stencil_stack, _ = temp_stencil_inputs(
                jnp.asarray(x_pde), jnp.asarray(y_pde), jnp.asarray(t_pde),
                jnp.asarray(u_p), jnp.asarray(v_p),
                _SAGE_TEMP_DX, _SAGE_TEMP_DY, _SAGE_TEMP_DT,
            )

            # ---- IC (512) ----
            ic_i = _sample_indices(rng, x_all.shape[0], bc_bs)
            ic_inputs = np.concatenate(
                [x_all[ic_i], y_all[ic_i],
                 np.zeros((ic_i.shape[0], 1), dtype=np.float32),
                 u_all[ic_i], v_all[ic_i]],
                axis=1,
            )

            # ---- Arrival (512) ----
            arr_i = _sample_indices(rng, x_inside.shape[0], bc_bs)
            arr_inputs = np.concatenate(
                [x_inside[arr_i], y_inside[arr_i],
                 t_arrive_inside[arr_i],
                 u_inside[arr_i], v_inside[arr_i]],
                axis=1,
            )

            # ---- Pre-arrival (512) ----
            pre_i = _sample_indices(rng, x_inside.shape[0], bc_bs)
            frac = (0.95 * rng.random((pre_i.shape[0], 1), dtype=np.float32)).astype(np.float32)
            t_pre = np.clip(frac * t_arrive_inside[pre_i], t_min, None).astype(np.float32)
            pre_inputs = np.concatenate(
                [x_inside[pre_i], y_inside[pre_i], t_pre,
                 u_inside[pre_i], v_inside[pre_i]],
                axis=1,
            )

            batch = {
                "pde_stencil": stencil_stack,
                "u_pde": jnp.asarray(u_p),
                "v_pde": jnp.asarray(v_p),
                "ic": jnp.asarray(ic_inputs),
                "ic_target": jnp.asarray(t_init_val),
                "arr": jnp.asarray(arr_inputs),
                "arr_target": jnp.asarray(inlet_t_val),
                "pre": jnp.asarray(pre_inputs),
                "pre_target": jnp.asarray(t_init_val),
            }

            if has_inlet:
                ii = inlet_idx[rng.integers(0, inlet_idx.shape[0], size=min(inlet_idx.shape[0], bc_bs), dtype=np.int64)]
                t_in = _make_time_samples_np(rng, ii.shape[0], t_min, t_max)
                inlet_inputs = np.concatenate(
                    [x_inside[ii], y_inside[ii], t_in, u_inside[ii], v_inside[ii]],
                    axis=1,
                )
                batch["inlet"] = jnp.asarray(inlet_inputs)
                batch["inlet_target"] = jnp.asarray(inlet_t_val)

            if has_outlet:
                oi = outlet_idx[rng.integers(0, outlet_idx.shape[0], size=min(outlet_idx.shape[0], bc_bs), dtype=np.int64)]
                t_out = _make_time_samples_np(rng, oi.shape[0], t_min, t_max)
                batch["outlet_x"] = jnp.asarray(x_inside[oi])
                batch["outlet_y"] = jnp.asarray(y_inside[oi])
                batch["outlet_t"] = jnp.asarray(t_out)
                batch["outlet_u"] = jnp.asarray(u_inside[oi])
                batch["outlet_v"] = jnp.asarray(v_inside[oi])

            params, opt_state, aux = train_step(params, opt_state, batch)

            if step % print_freq == 0 or step == 1:
                total = float(aux["total"])
                lp = float(aux["pde"]); li = float(aux["ic"])
                la = float(aux["arrival"]); lpre = float(aux["pre"])
                lin = float(aux["inlet"]); lo = float(aux["outlet"])
                print(f"[step {step:06d}] loss={total:.6e} pde={lp:.6e} ic={li:.6e} "
                      f"arrival={la:.6e} pre={lpre:.6e} inlet={lin:.6e} outlet={lo:.6e}",
                      flush=True)

            if step % save_freq == 0 or step == max_steps:
                save_params(params, ckpt_path)
                print(f"[OK] saved checkpoint to {ckpt_path}", flush=True)

        train_time = time.perf_counter() - t_wall0
        print(f"[OK] temp training complete ({max_steps} steps, {train_time / 60.0:.3f} min)")

    # --- Inference ---
    times = np.arange(
        float(cfg.problem.infer_t_start),
        float(cfg.problem.infer_t_end) + 1e-9,
        float(cfg.problem.infer_dt),
        dtype=np.float32,
    )
    temperature = np.zeros((times.shape[0], xy.shape[0]), dtype=np.float32)
    batch = int(cfg.inference.batch_size)

    @jax.jit
    def infer(params, xyt):
        return net.apply(params, xyt)

    for ti, tval in enumerate(times):
        for s0 in range(0, xy.shape[0], batch):
            s1 = min(s0 + batch, xy.shape[0])
            B = s1 - s0
            inp = np.concatenate(
                [x_all[s0:s1], y_all[s0:s1],
                 np.full((B, 1), float(tval), dtype=np.float32),
                 u_all[s0:s1], v_all[s0:s1]],
                axis=1,
            )
            out = infer(params, jnp.asarray(inp))
            temperature[ti, s0:s1] = np.asarray(out).reshape(-1)

    output_path = network_dir / str(cfg.inference.output_filename)
    _save_h5(
        str(output_path),
        xy_norm=xy, xy_raw=xy_raw, point_type=point_type,
        times=times, temperature=temperature,
        u=flow.u.astype(np.float32), v=flow.v.astype(np.float32),
        norm=norm,
        flow_json_path=flow_json_path, geom_json_path=geom_path,
    )
    print(f"[OK] wrote temperature predictions to {output_path}")

    if str(cfg.run_mode).lower() != "eval":
        _run_visualization(
            geom_path=geom_path, pred_path=str(output_path),
            outdir=network_dir / "visualizations",
        )


if __name__ == "__main__":
    run()
