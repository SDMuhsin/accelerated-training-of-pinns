"""V4.1 3D steady Navier-Stokes PINN flow trainer.

This is the 3D extension of ``src/partner_v4_flow.py`` (V4 2D cooling-channel
steady flow PINN). V4.1 mirrors V4's Hydra-driven Stage-to-Solver pipeline,
constraint-builder layout, checkpoint warm-start across stages, AMP, and
gradient clipping, lifted to 3D by:

  - a 6-input network ``(x, y, z, dw, sin, sout) -> (u, v, w, p)``
  - 3D PDE classes ``SteadyNavierStokes3DScaled``, ``WallNormalNoPenetration3D``,
    and ``FlowTrajectoryGuidance3D`` imported from ``partner_v4_1_physics``
  - 3D geometry / geodesic / init-guess helpers imported from
    ``partner_v4_1_geometry``
  - a parabolic z-profile on the inlet BC
  - an optional cap boost in the weighted PDE sampler
  - a 3D cross-product guidance loss split across ``flow_geo_cross_{x,y,z}``

The training schedule is identical to V4 (stage -1 init-warmup, stage 0
BC warmup, stages 1..N with viscosity continuation). Each stage warm-starts
from the previous stage's checkpoint via PhysicsNeMo's
``initialization_network_dir`` config key.

Reference: ``llmdocs/stream_battery_consortium/V4_1_DESIGN.md``.
This module is pure PhysicsNeMo + PyTorch (no SAGE, no JAX).
"""

import os
import json
import time
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from pathlib import Path

from physicsnemo.sym.key import Key
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.domain.constraint import PointwiseConstraint

from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors

# V4.1 physics + geometry helpers (separate modules -- no CUDA init at import)
from partner_v4_1_physics import (
    SteadyNavierStokes3DScaled,
    WallNormalNoPenetration3D,
    FlowTrajectoryGuidance3D,
)  # 3D PDE classes
from partner_v4_1_geometry import (
    load_2d_geometry_json,
    build_3d_point_cloud,
    compute_wall_distance_3d,
    build_inside_graph_3d,
    compute_geodesic_info_3d,
    estimate_wall_normals_3d,
    compute_initial_flow_guess_3d,
)  # 3D geometry helpers


# -----------------------------
# GPU init (mirrors V4 partner_v4_flow.py lines 33-41)
# -----------------------------
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # read local rank
    torch.cuda.set_device(local_rank)  # set device
    torch.cuda.init()  # init cuda
    _ = torch.empty(1, device="cuda")  # warmup


# -----------------------------
# Small constants
# -----------------------------
_SMALL = 1.0e-8  # numerical safety floor used throughout
_DEFAULT_PATCH_RADIUS = 0.002  # normalized inlet / outlet disk radius
_DEFAULT_PATCH_MAX_RADIUS = 0.02  # maximum grow radius
_DEFAULT_PATCH_GROW = 1.5  # growth factor per radius expansion
_DEFAULT_PATCH_MIN_PTS = 12  # minimum points per 2D patch before extrusion


# -----------------------------
# Activation registry (copy of V4 get_activation; no cross-file import)
# -----------------------------
def get_activation(act_name: str):
    """Return a torch.nn activation module for the given string name."""

    act_name = str(act_name).lower()  # normalize
    if act_name in ("silu", "swish"):
        return torch.nn.SiLU()  # silu
    if act_name == "tanh":
        return torch.nn.Tanh()  # tanh
    if act_name == "relu":
        return torch.nn.ReLU()  # relu
    if act_name == "gelu":
        return torch.nn.GELU()  # gelu
    raise ValueError(f"Unknown activation: {act_name}")  # invalid


# -----------------------------
# 3D geometry JSON loader (matches build_v4_1_geometry.py output schema)
# -----------------------------
def load_3d_geometry_json(path: str) -> dict:
    """Load the 3D geometry JSON produced by ``build_v4_1_geometry.py``.

    Returns a dict with keys matching the V4.1 schema:
        z_aspect, z_slices, norm, inlet, outlet,
        inlet_center_xyz, outlet_center_xyz,
        points_inside (N,6), points_wall (N,7),
        points_inlet (N_il,3), points_outlet (N_ol,3),
        normals_wall (N_w,3), init_fields {xyz, xyz_raw, fields{...}}.

    The return arrays are numpy float32 / int arrays (not raw Python lists).
    """

    p = Path(path)  # path object
    if not p.exists():
        raise FileNotFoundError(f"3D geometry JSON not found: {p}")  # guard
    obj = json.loads(p.read_text())  # parse

    required = [
        "z_aspect", "z_slices", "norm",
        "inlet", "outlet",
        "inlet_center_xyz", "outlet_center_xyz",
        "points_inside", "points_wall",
        "points_inlet", "points_outlet",
        "normals_wall",
    ]  # required keys (init_fields is optional)
    missing = [k for k in required if k not in obj]  # check presence
    if missing:
        raise KeyError(f"3D geometry JSON missing keys {missing}: {p}")  # schema error

    # Points arrays
    pts_in = np.asarray(obj["points_inside"], dtype=np.float32)  # (N_in,6)
    pts_w = np.asarray(obj["points_wall"], dtype=np.float32)  # (N_w,7)
    pts_il = np.asarray(obj["points_inlet"], dtype=np.float32)  # (N_il,3)
    pts_ol = np.asarray(obj["points_outlet"], dtype=np.float32)  # (N_ol,3)
    n_w = np.asarray(obj["normals_wall"], dtype=np.float32)  # (N_w,3)

    if pts_in.ndim != 2 or pts_in.shape[1] != 6:
        raise ValueError(
            f"points_inside must have shape (N,6), got {pts_in.shape}"
        )  # schema error
    if pts_w.ndim != 2 or pts_w.shape[1] != 7:
        raise ValueError(
            f"points_wall must have shape (N,7), got {pts_w.shape}"
        )  # schema error
    if pts_il.ndim != 2 or pts_il.shape[1] != 3:
        raise ValueError(
            f"points_inlet must have shape (N,3), got {pts_il.shape}"
        )  # schema error
    if pts_ol.ndim != 2 or pts_ol.shape[1] != 3:
        raise ValueError(
            f"points_outlet must have shape (N,3), got {pts_ol.shape}"
        )  # schema error
    if n_w.shape[0] != pts_w.shape[0] or n_w.shape[1] != 3:
        raise ValueError(
            f"normals_wall must have shape ({pts_w.shape[0]},3), got {n_w.shape}"
        )  # schema error

    norm_list = obj["norm"]  # xmin,xmax,ymin,ymax,zmin,zmax
    if not isinstance(norm_list, (list, tuple)) or len(norm_list) != 6:
        raise ValueError(f"norm must be a 6-tuple, got {norm_list}")  # schema error
    norm_tuple = tuple(float(v) for v in norm_list)  # norm as floats

    inlet = {"x": float(obj["inlet"]["x"]), "y": float(obj["inlet"]["y"])}  # raw inlet pixel
    outlet = {"x": float(obj["outlet"]["x"]), "y": float(obj["outlet"]["y"])}  # raw outlet pixel

    inlet_center_xyz = np.asarray(obj["inlet_center_xyz"], dtype=np.float32).reshape(1, 3)  # (1,3)
    outlet_center_xyz = np.asarray(obj["outlet_center_xyz"], dtype=np.float32).reshape(1, 3)  # (1,3)

    # Decompose inside + wall columns (normalised xyz + raw voxel xyz)
    xyz_inside = pts_in[:, 0:3].astype(np.float32)  # normalized interior xyz
    xyz_inside_raw = pts_in[:, 3:6].astype(np.float32)  # raw voxel interior xyz
    xyz_wall = pts_w[:, 0:3].astype(np.float32)  # normalized wall xyz
    xyz_wall_raw = pts_w[:, 3:6].astype(np.float32)  # raw voxel wall xyz
    class_wall = pts_w[:, 6].astype(np.int8)  # per-wall class label (0=side,1=bot,2=top)

    out = {
        "z_aspect": float(obj["z_aspect"]),
        "z_slices": int(obj["z_slices"]),
        "norm": norm_tuple,
        "inlet": inlet,
        "outlet": outlet,
        "inlet_center_xyz": inlet_center_xyz,
        "outlet_center_xyz": outlet_center_xyz,
        "xyz_inside": xyz_inside,
        "xyz_inside_raw": xyz_inside_raw,
        "xyz_wall": xyz_wall,
        "xyz_wall_raw": xyz_wall_raw,
        "class_wall": class_wall,
        "n_wall": n_w,
        "xyz_inlet": pts_il,
        "xyz_outlet": pts_ol,
    }  # bundle

    # Init fields -- optional but expected for the V4.1 pipeline
    if "init_fields" in obj and obj["init_fields"] is not None:
        ifs = obj["init_fields"]  # init_fields subdict
        if not isinstance(ifs, dict) or "fields" not in ifs:
            raise ValueError(
                "init_fields must be a dict with a 'fields' subdict"
            )  # schema error
        fields = ifs["fields"]  # fields subdict
        N_in = int(xyz_inside.shape[0])  # interior count
        init_out = {}  # decoded init fields
        for key in ("u", "v", "w", "p", "dw", "geo_in", "s_in", "s_out"):
            if key not in fields:
                raise KeyError(f"init_fields.fields missing '{key}'")  # schema error
            arr = np.asarray(fields[key], dtype=np.float32).reshape(-1, 1)  # (N,1)
            if arr.shape[0] != N_in:
                raise ValueError(
                    f"init_fields.fields['{key}'] length {arr.shape[0]} "
                    f"does not match points_inside count {N_in}"
                )  # size mismatch
            init_out[key] = arr  # store

        # Tangent / predecessors / src are optional in JSON; reconstruct if missing.
        if "tangent" in fields:
            tangent = np.asarray(fields["tangent"], dtype=np.float32).reshape(-1, 3)  # (N,3)
            if tangent.shape[0] != N_in:
                raise ValueError(
                    f"init_fields.fields['tangent'] length {tangent.shape[0]} "
                    f"does not match points_inside count {N_in}"
                )  # size mismatch
            init_out["tangent"] = tangent  # store tangent
        if "predecessors" in fields:
            init_out["predecessors"] = np.asarray(
                fields["predecessors"], dtype=np.int64
            )  # predecessor tree
        if "src" in ifs:
            init_out["src"] = int(ifs["src"])  # source index
        out["init_fields"] = init_out  # attach
    else:
        out["init_fields"] = None  # no init fields in the file
    return out  # parsed bundle


# -----------------------------
# Patch mask helpers (inlet / outlet disks in 3D)
# -----------------------------
def _build_patch_mask_3d(
    xyz_inside: np.ndarray,
    xyz_patch: np.ndarray,
    radius: float = 0.01,
) -> np.ndarray:
    """Boolean mask over inside points for all interior points within
    ``radius`` of ANY point in ``xyz_patch`` (3D Euclidean).

    This matches how V4 builds inlet/outlet masks (a disk-in-3D) but in
    the 3D case the patch is already an extruded collection of points,
    not a single center.
    """

    n = int(xyz_inside.shape[0])  # total interior
    mask = np.zeros((n,), dtype=bool)  # output mask
    if xyz_patch.shape[0] <= 0 or n <= 0:
        return mask  # nothing to intersect

    tree = cKDTree(xyz_inside.astype(np.float64))  # interior kd-tree
    # query_ball_point returns a list of index arrays, one per query point
    neighbor_lists = tree.query_ball_point(
        xyz_patch.astype(np.float64), r=float(max(radius, 1.0e-8)), p=2.0,
    )  # radius query
    for idx_list in neighbor_lists:
        if len(idx_list) > 0:
            mask[np.asarray(idx_list, dtype=np.int64)] = True  # mark inside points
    return mask  # (N,) bool


def _nearest_inside_mask_3d(
    xyz_inside: np.ndarray,
    xyz_patch: np.ndarray,
) -> np.ndarray:
    """Mark the single nearest interior point to each patch point.

    Used as a fallback when the radius-based search finds zero points
    (e.g. in tiny smoke geometries where z-slice spacing exceeds the
    default search radius).
    """

    n = int(xyz_inside.shape[0])  # interior count
    mask = np.zeros((n,), dtype=bool)  # output mask
    if xyz_patch.shape[0] <= 0 or n <= 0:
        return mask  # nothing to mark

    tree = cKDTree(xyz_inside.astype(np.float64))  # interior kd-tree
    _, idx = tree.query(xyz_patch.astype(np.float64), k=1)  # nearest interior per patch
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)  # flat
    mask[idx] = True  # mark
    return mask  # (N,) bool


def _apply_parabolic_z_profile(
    z_norm: np.ndarray,
    z_max: float,
) -> np.ndarray:
    """Evaluate phi(z) = 4 (z/zmax) (1 - z/zmax) at each point.

    Parameters
    ----------
    z_norm : (N, 1) normalised z coordinate in [0, z_max].
    z_max : float, the top cap z in normalised units (== z_aspect).

    Returns
    -------
    (N, 1) float32 phi array, clipped to [0, 1] for safety.
    """

    zm = float(max(z_max, 1.0e-8))  # guard
    phi = 4.0 * (z_norm / zm) * (1.0 - z_norm / zm)  # peaks at z=zm/2
    phi = np.clip(phi, 0.0, 1.0).astype(np.float32)  # safety
    return phi  # (N,1)


# -----------------------------
# Project inside feature to wall via nearest neighbour (3D)
# -----------------------------
def project_inside_feature_to_wall_3d(
    xyz_wall: np.ndarray,
    xyz_inside: np.ndarray,
    feat_inside: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour projection of a per-inside feature to each wall point.

    Mirrors V4's ``project_inside_feature_to_wall`` but in 3D.
    """

    if xyz_inside.shape[0] <= 0 or xyz_wall.shape[0] <= 0:
        return np.zeros((xyz_wall.shape[0], feat_inside.shape[1]), dtype=np.float32)  # empty
    tree = cKDTree(xyz_inside.astype(np.float64))  # interior tree
    _, idx = tree.query(xyz_wall.astype(np.float64), k=1)  # nearest interior per wall
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)  # flatten
    return feat_inside[idx].astype(np.float32)  # projected


# -----------------------------
# Trace predecessor path in the 3D inside graph
# -----------------------------
def trace_path_to_source_3d(
    predecessors: np.ndarray, src: int, dst: int, max_steps: int = 1_000_000,
) -> List[int]:
    """Walk from ``dst`` back to ``src`` along the Dijkstra predecessor tree.

    Returns a list of indices ordered from source to destination. If the
    path is broken (``predecessors[...]`` == -9999), we stop and return
    what we have so far.
    """

    path = [int(dst)]  # start from destination
    cur = int(dst)  # running node
    seen = set([cur])  # cycle guard
    for _ in range(max_steps):
        if cur == int(src):
            break  # reached source
        p = int(predecessors[cur])  # predecessor
        if p < 0:
            break  # broken
        if p in seen:
            break  # unexpected cycle
        path.append(p)  # add predecessor
        seen.add(p)  # mark visited
        cur = p  # step back
    path.reverse()  # from source to destination
    return path  # node indices


def estimate_tangent_from_predecessor_tree_3d(
    xyz_inside: np.ndarray,
    predecessors: np.ndarray,
    src: int,
) -> np.ndarray:
    """Build a per-point 3D unit tangent along the Dijkstra predecessor tree.

    A local copy of the helper used inside ``compute_initial_flow_guess_3d``;
    duplicated here so the trainer can estimate a guidance tangent without
    reconstructing the full init guess. Falls back to +x on degenerate rows.
    """

    n = int(xyz_inside.shape[0])  # point count
    tangent = np.zeros((n, 3), dtype=np.float32)  # output

    # Build reverse tree: children[p] = [i, j, ...]
    children: List[List[int]] = [[] for _ in range(n)]  # reverse tree
    for i in range(n):
        p = int(predecessors[i])  # parent
        if p >= 0:
            children[p].append(i)  # i is child of p

    fallback_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # +x fallback
    for i in range(n):
        p = int(predecessors[i])  # parent
        if i == int(src):
            if len(children[i]) > 0:
                j = int(children[i][0])  # first child
                vec = xyz_inside[j] - xyz_inside[i]  # forward direction
            else:
                vec = fallback_vec.copy()  # lone source
        else:
            if p >= 0:
                vec = xyz_inside[i] - xyz_inside[p]  # along path
            else:
                vec = fallback_vec.copy()  # disconnected
        m = float(np.linalg.norm(vec))  # magnitude
        if m > 1.0e-12:
            tangent[i] = (vec / m).astype(np.float32)  # normalize
        else:
            tangent[i] = fallback_vec  # degenerate fallback
    return tangent  # (N,3)


# -----------------------------
# Weighted PDE sampler (3D: wall boost + corridor boost + cap boost)
# -----------------------------
def _build_weighted_flow_pde_indices_3d(
    xyz_inside: np.ndarray,
    d_w_i: np.ndarray,
    predecessors_in: np.ndarray,
    src_in: int,
    outlet_target_idx: int,
    z_max: float,
    target_points: int,
    wall_boost: float,
    wall_scale: float,
    corridor_boost: float,
    corridor_radius: float,
    cap_boost: float,
    cap_scale: float,
    seed: int,
) -> np.ndarray:
    """Mirror of V4's ``_build_weighted_flow_pde_indices``, extended to 3D.

    Adds a cap-boost term: the sampler spends extra budget on points near
    the top and bottom caps because the z-boundary layer is a new 3D
    phenomenon that the PDE loss should resolve. Cap distance is
    ``min(z, z_max - z)`` in normalised units.
    """

    n = int(xyz_inside.shape[0])  # total interior points
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)  # empty

    # If target matches or exceeds all points, keep them all (no sampling).
    if (int(target_points) <= 0) or (int(target_points) >= n):
        return np.arange(n, dtype=np.int64)  # keep all

    weights = np.ones((n,), dtype=np.float64)  # base weights

    # wall-distance boost (near-wall importance)
    if float(wall_boost) > 0.0:
        dw = d_w_i.reshape(-1).astype(np.float32)  # flat wall distance
        wall_importance = np.exp(
            -((dw / max(float(wall_scale), 1.0e-8)) ** 2)
        ).astype(np.float64)  # narrow Gaussian near 0
        weights *= (1.0 + float(wall_boost) * wall_importance)  # apply boost

    # corridor boost (near the inlet->outlet predecessor path)
    if float(corridor_boost) > 0.0:
        path_idx = trace_path_to_source_3d(
            predecessors=predecessors_in,
            src=int(src_in),
            dst=int(outlet_target_idx),
        )  # path from inlet to outlet in the 3D graph
        if len(path_idx) >= 2:
            path_xyz = xyz_inside[np.asarray(path_idx, dtype=np.int64)]  # path coords
            knn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")  # tree over path
            knn.fit(path_xyz)  # fit
            d_path, _ = knn.kneighbors(xyz_inside)  # nearest-path distance
            d_path = d_path[:, 0].astype(np.float32)  # flatten
            corridor_importance = np.exp(
                -((d_path / max(float(corridor_radius), 1.0e-8)) ** 2)
            ).astype(np.float64)  # boost along path
            weights *= (1.0 + float(corridor_boost) * corridor_importance)  # apply

    # cap boost (new 3D-specific: near top/bottom caps)
    if float(cap_boost) > 0.0:
        zn = xyz_inside[:, 2].astype(np.float32)  # z coords
        zm = float(max(z_max, 1.0e-8))  # top cap z
        d_cap = np.minimum(zn, zm - zn).astype(np.float32)  # distance to nearest cap
        cap_importance = np.exp(
            -((d_cap / max(float(cap_scale), 1.0e-8)) ** 2)
        ).astype(np.float64)  # Gaussian near caps
        weights *= (1.0 + float(cap_boost) * cap_importance)  # apply boost

    weights = np.maximum(weights, 1.0e-12)  # guard zero
    weights = weights / float(np.sum(weights))  # normalize
    rng = np.random.default_rng(int(seed))  # rng
    idx = rng.choice(
        np.arange(n),
        size=int(target_points),
        replace=False,
        p=weights,
    ).astype(np.int64)  # weighted sample without replacement
    return idx  # (target_points,) int64


# -----------------------------
# Progress-sorted chunking (preserves V4 curriculum-order behaviour in 3D)
# -----------------------------
def sort_by_progress_chunked_3d(
    x_i: np.ndarray,
    y_i: np.ndarray,
    z_i: np.ndarray,
    d_w_i: np.ndarray,
    s_in_i: np.ndarray,
    s_out_i: np.ndarray,
    chunk_size: int,
    seed: int = 1234,
):
    """Sort points by inlet progress then shuffle inside fixed-size chunks.

    Matches V4's ``sort_by_progress_chunked`` semantics, extended with a z
    column so the 3D trainer can keep the same curriculum-style ordering.
    """

    order = np.argsort(s_in_i.reshape(-1))  # ascending inlet progress
    rng = np.random.default_rng(int(seed))  # rng for chunk shuffling
    chunks = []  # chunk list
    for s in range(0, len(order), int(max(chunk_size, 1))):
        e = min(s + int(max(chunk_size, 1)), len(order))  # chunk end
        ch = order[s:e].copy()  # chunk copy
        rng.shuffle(ch)  # in-chunk shuffle
        chunks.append(ch)  # save
    if len(chunks) == 0:
        order2 = np.arange(0, dtype=np.int64)  # empty
    else:
        order2 = np.concatenate(chunks, axis=0)  # final order
    return (
        x_i[order2],
        y_i[order2],
        z_i[order2],
        d_w_i[order2],
        s_in_i[order2],
        s_out_i[order2],
        order2,
    )  # reordered arrays + original order


# -----------------------------
# Wall-guard point generator (3D)
# -----------------------------
def _build_wall_guard_points_3d(
    xyz_inside: np.ndarray,
    xyz_wall: np.ndarray,
    radius: float,
    target_points: int,
    seed: int,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return up to ``target_points`` interior points within ``radius`` of any
    wall (3D Euclidean).

    Mirrors V4's ``_build_wall_guard_points`` but:
      - uses 3D coordinates (x, y, z)
      - z is never filtered (the z-restriction lives in V4 where there was
        no z axis; in 3D we keep all depth slices for the wall guard)
    """

    n = int(xyz_inside.shape[0])  # interior count
    if (
        n <= 0
        or xyz_wall.shape[0] <= 0
        or int(target_points) <= 0
        or float(radius) <= 0.0
    ):
        return np.zeros((0, 3), dtype=np.float32)  # empty set

    knn_w = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")  # nearest wall
    knn_w.fit(xyz_wall.astype(np.float32))  # fit
    dw, _ = knn_w.kneighbors(xyz_inside.astype(np.float32))  # distances
    dw = dw[:, 0].astype(np.float32)  # flatten

    mask = (dw <= float(radius))  # near-wall mask
    if x_min is not None:
        mask &= (xyz_inside[:, 0] >= float(x_min))  # x lower
    if x_max is not None:
        mask &= (xyz_inside[:, 0] <= float(x_max))  # x upper
    if y_min is not None:
        mask &= (xyz_inside[:, 1] >= float(y_min))  # y lower
    if y_max is not None:
        mask &= (xyz_inside[:, 1] <= float(y_max))  # y upper
    if exclude_mask is not None and int(exclude_mask.shape[0]) == n:
        mask &= (~exclude_mask.astype(bool))  # exclude ports

    idx = np.where(mask)[0].astype(np.int64)  # candidates
    if idx.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)  # none selected
    if idx.size > int(target_points):
        rng = np.random.default_rng(int(seed))  # rng
        idx = rng.choice(idx, size=int(target_points), replace=False).astype(np.int64)  # downsample
    return xyz_inside[idx].astype(np.float32)  # (M,3)


# -----------------------------
# PointwiseConstraint builder (mirrors V4 _build_pointwise_constraint)
# -----------------------------
def _build_pointwise_constraint(
    *,
    nodes,
    invar: Dict[str, np.ndarray],
    outvar: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
    lambda_weights: Optional[Dict[str, object]] = None,
):
    """Create a ``PointwiseConstraint.from_numpy`` with optional per-equation
    lambda weights.

    Scalar lambdas are broadcast to the target shape; array lambdas are
    broadcast too. Falls back to an unweighted constraint if the installed
    PhysicsNeMo version doesn't accept ``lambda_weighting``.
    """

    kwargs = {}  # optional kwargs
    if lambda_weights is not None:
        lambda_weighting = {}  # per-key weight map
        for key, target in outvar.items():
            w_spec = lambda_weights.get(key, 1.0)  # scalar or array
            if np.isscalar(w_spec):
                lambda_weighting[key] = np.full_like(
                    target, float(w_spec), dtype=np.float32,
                )  # scalar broadcast
            else:
                w_arr = np.asarray(w_spec, dtype=np.float32)  # array
                if w_arr.shape != target.shape:
                    w_arr = np.broadcast_to(w_arr, target.shape).astype(np.float32)  # broadcast
                lambda_weighting[key] = w_arr  # attach
        kwargs["lambda_weighting"] = lambda_weighting  # forward to PhysicsNeMo

    # Number of rows is consistent across keys; use the first outvar key.
    rows = int(next(iter(outvar.values())).shape[0])  # row count
    bs = int(max(min(int(batch_size), rows), 1))  # clamp batch size

    try:
        return PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=invar,
            outvar=outvar,
            batch_size=bs,
            shuffle=shuffle,
            **kwargs,
        )  # with lambda weighting
    except TypeError:
        return PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=invar,
            outvar=outvar,
            batch_size=bs,
            shuffle=shuffle,
        )  # fallback unweighted


# -----------------------------
# Linear stage scale (identical semantics to V4 _linear_stage_scale)
# -----------------------------
def _linear_stage_scale(
    stage_idx: int, num_stages: int, start_scale: float, end_scale: float,
) -> float:
    """Linearly interpolate from start_scale to end_scale across num_stages.

    num_stages=1 returns ``end_scale`` (matches V4 behaviour).
    """

    if int(num_stages) <= 1:
        return float(end_scale)  # degenerate case
    frac = float(stage_idx) / float(num_stages - 1)  # stage fraction [0,1]
    return float(start_scale + frac * (end_scale - start_scale))  # interpolate


# -----------------------------
# Stage cfg helper (identical to V4 make_stage_cfg)
# -----------------------------
def make_stage_cfg(
    base_cfg: DictConfig,
    stage_name: str,
    max_steps: int,
    init_dir: str = "",
) -> DictConfig:
    """Copy the base config and set stage-specific fields.

    Returns a fresh ``DictConfig`` so mutating the copy does not affect
    the parent config.
    """

    c = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))  # deep copy
    c.training.max_steps = int(max_steps)  # set steps
    c.network_dir = str(Path(base_cfg.network_dir) / stage_name)  # stage dir
    c.initialization_network_dir = str(init_dir)  # init dir
    return c  # ready for Solver


# -----------------------------
# Solver logging silencer (mirrors V4 _disable_recording)
# -----------------------------
def _disable_recording(slv: Solver) -> None:
    """Silence PhysicsNeMo's TensorBoard/monitor recording hooks.

    V4 uses this to avoid flushing per-constraint history every summary
    step; we mirror that for V4.1.
    """

    slv.record_constraints = lambda *args, **kwargs: None  # disable
    slv.record_validators = lambda *args, **kwargs: None  # disable
    slv.record_inferencers = lambda *args, **kwargs: None  # disable
    slv.record_monitors = lambda *args, **kwargs: None  # disable


# -----------------------------
# Primary (flow PDE + wall no-slip + inlet vel + outlet/inlet pressure) in 3D
# -----------------------------
def build_primary_constraints_3d(
    flow_nodes,
    cfg,
    x_w: np.ndarray,
    y_w: np.ndarray,
    z_w: np.ndarray,
    d_w_w: np.ndarray,
    s_in_w: np.ndarray,
    s_out_w: np.ndarray,
    x_i_sorted: np.ndarray,
    y_i_sorted: np.ndarray,
    z_i_sorted: np.ndarray,
    d_w_i_sorted: np.ndarray,
    s_in_i_sorted: np.ndarray,
    s_out_i_sorted: np.ndarray,
    inlet_invar: Dict[str, np.ndarray],
    outlet_invar: Dict[str, np.ndarray],
    inlet_phi_z: np.ndarray,
    outlet_phi_z: np.ndarray,
    weight_scale: float = 1.0,
):
    """Build the 4 primary constraints:
        flow_pde     -- residuals of the four 3D NS equations
        wall_noslip  -- u=v=w=0 on every wall point (side + both caps)
        inlet_vel    -- u=inlet_u * phi(z), v=inlet_v * phi(z), w=0 on inlet patch
        outlet_p     -- (optional) p=outlet_p on outlet patch
        inlet_p      -- (optional) p=inlet_p on inlet patch

    Inputs shaped (N,1) float32; ``weight_scale`` scales every lambda.
    """

    w_scale = float(max(weight_scale, 0.0))  # constraint weight scale

    # -------- wall no-slip (u=v=w=0) --------
    wall_inv = {
        "x": x_w.astype(np.float32),
        "y": y_w.astype(np.float32),
        "z": z_w.astype(np.float32),
        "dw": d_w_w.astype(np.float32),
        "sin": s_in_w.astype(np.float32),
        "sout": s_out_w.astype(np.float32),
    }  # wall inputs
    wall_out = {
        "u": np.zeros((x_w.shape[0], 1), np.float32),
        "v": np.zeros((x_w.shape[0], 1), np.float32),
        "w": np.zeros((x_w.shape[0], 1), np.float32),
    }  # wall targets
    wall_constraint = _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=wall_inv,
        outvar=wall_out,
        batch_size=min(int(cfg.training.flow_wall_batch_size), int(x_w.shape[0])),
        shuffle=True,
        lambda_weights={
            "u": float(getattr(cfg.training, "w_wall_u", 1.0)) * w_scale,
            "v": float(getattr(cfg.training, "w_wall_v", 1.0)) * w_scale,
            "w": float(getattr(cfg.training, "w_wall_w", 1.0)) * w_scale,
        },
    )  # wall constraint

    # -------- flow PDE (continuity + 3 momentum) --------
    flow_int_inv = {
        "x": x_i_sorted.astype(np.float32),
        "y": y_i_sorted.astype(np.float32),
        "z": z_i_sorted.astype(np.float32),
        "dw": d_w_i_sorted.astype(np.float32),
        "sin": s_in_i_sorted.astype(np.float32),
        "sout": s_out_i_sorted.astype(np.float32),
    }  # PDE inputs
    flow_pde_out = {
        "continuity": np.zeros((x_i_sorted.shape[0], 1), np.float32),
        "momentum_x": np.zeros((x_i_sorted.shape[0], 1), np.float32),
        "momentum_y": np.zeros((x_i_sorted.shape[0], 1), np.float32),
        "momentum_z": np.zeros((x_i_sorted.shape[0], 1), np.float32),
    }  # PDE targets
    flow_pde_constraint = _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=flow_int_inv,
        outvar=flow_pde_out,
        batch_size=min(
            int(cfg.training.flow_pde_batch_size), int(x_i_sorted.shape[0])
        ),
        shuffle=False,
        lambda_weights={
            "continuity": float(getattr(cfg.training, "w_pde_continuity", 1.0)) * w_scale,
            "momentum_x": float(getattr(cfg.training, "w_pde_momentum_x", 1.0)) * w_scale,
            "momentum_y": float(getattr(cfg.training, "w_pde_momentum_y", 1.0)) * w_scale,
            "momentum_z": float(getattr(cfg.training, "w_pde_momentum_z", 1.0)) * w_scale,
        },
    )  # PDE constraint

    # -------- inlet velocity (parabolic z-profile modulator) --------
    x_in_rows = int(inlet_invar["x"].shape[0])  # row count
    inlet_u_target = float(cfg.bc.inlet_u) * inlet_phi_z.astype(np.float32)  # u target
    inlet_v_target = float(cfg.bc.inlet_v) * inlet_phi_z.astype(np.float32)  # v target
    inlet_out = {
        "u": inlet_u_target.astype(np.float32),
        "v": inlet_v_target.astype(np.float32),
        "w": np.zeros((x_in_rows, 1), np.float32),
    }  # inlet targets
    inlet_constraint = _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=inlet_invar,
        outvar=inlet_out,
        batch_size=min(int(cfg.training.flow_bc_batch_size), x_in_rows),
        shuffle=True,
        lambda_weights={
            "u": float(getattr(cfg.training, "w_inlet_u", 1.0)) * w_scale,
            "v": float(getattr(cfg.training, "w_inlet_v", 1.0)) * w_scale,
            "w": float(getattr(cfg.training, "w_inlet_w", 1.0)) * w_scale,
        },
    )  # inlet velocity constraint

    # -------- outlet pressure (optional) --------
    outlet_constraint = None  # default none
    if bool(getattr(cfg.bc, "use_outlet_pressure_constraint", False)):
        x_out_rows = int(outlet_invar["x"].shape[0])  # row count
        outlet_out = {
            "p": np.full((x_out_rows, 1), float(cfg.bc.outlet_p), np.float32),
        }  # outlet target
        outlet_constraint = _build_pointwise_constraint(
            nodes=flow_nodes,
            invar=outlet_invar,
            outvar=outlet_out,
            batch_size=min(int(cfg.training.flow_bc_batch_size), x_out_rows),
            shuffle=True,
            lambda_weights={
                "p": float(getattr(cfg.training, "w_outlet_p", 1.0)) * w_scale,
            },
        )  # outlet pressure anchor

    # -------- inlet pressure anchor (optional) --------
    inlet_p_constraint = None  # default none
    if bool(getattr(cfg.bc, "use_inlet_pressure_anchor", False)):
        x_in_rows_p = int(inlet_invar["x"].shape[0])  # row count
        inlet_p_out = {
            "p": np.full((x_in_rows_p, 1), float(cfg.bc.inlet_p), np.float32),
        }  # inlet pressure target
        inlet_p_constraint = _build_pointwise_constraint(
            nodes=flow_nodes,
            invar=inlet_invar,
            outvar=inlet_p_out,
            batch_size=min(int(cfg.training.flow_bc_batch_size), x_in_rows_p),
            shuffle=True,
            lambda_weights={
                "p": float(getattr(cfg.training, "w_inlet_p", 1.0)) * w_scale,
            },
        )  # inlet pressure anchor

    return (
        flow_pde_constraint,
        wall_constraint,
        inlet_constraint,
        outlet_constraint,
        inlet_p_constraint,
    )  # bundle


# -----------------------------
# Pseudo-init (soft_init) constraint in 3D
# -----------------------------
def build_pseudo_init_constraint_3d(
    flow_nodes,
    cfg,
    x_i: np.ndarray,
    y_i: np.ndarray,
    z_i: np.ndarray,
    d_w_i: np.ndarray,
    s_in_i: np.ndarray,
    s_out_i: np.ndarray,
    init_inside_fields: Dict[str, np.ndarray],
    batch_size_key: str,
    scale: float,
):
    """Regress the flow net toward the precomputed 3D pseudo-init fields
    (u_0, v_0, w_0, p_0) with per-field lambda weights multiplied by
    ``scale``.

    Mirrors V4's ``build_pseudo_init_constraint`` extended with a ``w``
    field target.
    """

    s = float(max(scale, 0.0))  # scale
    init_invar = {
        "x": x_i.astype(np.float32),
        "y": y_i.astype(np.float32),
        "z": z_i.astype(np.float32),
        "dw": d_w_i.astype(np.float32),
        "sin": s_in_i.astype(np.float32),
        "sout": s_out_i.astype(np.float32),
    }  # inputs
    init_outvar = {
        "u": init_inside_fields["u"].astype(np.float32),
        "v": init_inside_fields["v"].astype(np.float32),
        "w": init_inside_fields["w"].astype(np.float32),
        "p": init_inside_fields["p"].astype(np.float32),
    }  # pseudo targets

    return _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=init_invar,
        outvar=init_outvar,
        batch_size=min(
            int(getattr(cfg.training, batch_size_key, 32768)),
            int(x_i.shape[0]),
        ),
        shuffle=True,
        lambda_weights={
            "u": float(getattr(cfg.training, "w_soft_init_u", 0.2)) * s,
            "v": float(getattr(cfg.training, "w_soft_init_v", 0.2)) * s,
            "w": float(getattr(cfg.training, "w_soft_init_w", 0.2)) * s,
            "p": float(getattr(cfg.training, "w_soft_init_p", 0.2)) * s,
        },
    )  # soft-init regression constraint


# -----------------------------
# Geo guidance constraints (3D): cross (3 components), cosine, parallel, speed
# -----------------------------
def build_geo_guidance_constraints_3d(
    geo_nodes,
    cfg,
    xyz_inside: np.ndarray,
    d_w_i: np.ndarray,
    s_in_i: np.ndarray,
    s_out_i: np.ndarray,
    init_inside_fields: Dict[str, np.ndarray],
    inlet_mask: np.ndarray,
    outlet_mask: np.ndarray,
    scale: float,
):
    """Build the three guidance constraints used during BC and PDE stages:
        geo_dir       -- cross (3 components) = 0 AND cosine = 1
        geo_parallel  -- parallel = speed_target
        geo_speed     -- (optional) speed = speed_target

    Points are speed-gated (speed > ``geo_guidance_speed_eps``) and
    subsampled to at most ``geo_guidance_max_points``. Guidance tangent
    (gx, gy, gz) comes from ``init_inside_fields["tangent"]``.
    """

    if not bool(getattr(cfg.training, "use_geo_direction_guidance", True)):
        return None, None, None  # disabled by config

    s = float(max(scale, 0.0))  # stage scale
    if s <= 0.0:
        return None, None, None  # inactive

    # Speed target per interior point (from 3D init guess)
    speed = np.sqrt(
        init_inside_fields["u"] ** 2
        + init_inside_fields["v"] ** 2
        + init_inside_fields["w"] ** 2
    ).astype(np.float32)  # (N,1)
    tangent = init_inside_fields["tangent"].astype(np.float32)  # (N,3)
    speed_flat = speed.reshape(-1)  # flat

    valid = speed_flat > float(getattr(cfg.training, "geo_guidance_speed_eps", 1.0e-4))  # gate

    if bool(getattr(cfg.training, "geo_guidance_exclude_ports", True)):
        valid &= (~inlet_mask)  # exclude inlet patch
        valid &= (~outlet_mask)  # exclude outlet patch

    idx = np.where(valid)[0].astype(np.int64)  # valid indices
    if idx.size <= 0:
        return None, None, None  # nothing valid

    max_points = int(getattr(cfg.training, "geo_guidance_max_points", 0))  # cap
    if (max_points > 0) and (idx.size > max_points):
        rng = np.random.default_rng(int(getattr(cfg.training, "geo_guidance_seed", 1234)))  # rng
        idx = rng.choice(idx, size=max_points, replace=False).astype(np.int64)  # subsample

    xg = xyz_inside[idx, 0:1].astype(np.float32)  # x
    yg = xyz_inside[idx, 1:2].astype(np.float32)  # y
    zg = xyz_inside[idx, 2:3].astype(np.float32)  # z
    dwg = d_w_i[idx].astype(np.float32)  # dw
    sing = s_in_i[idx].astype(np.float32)  # sin
    soutg = s_out_i[idx].astype(np.float32)  # sout
    gx = tangent[idx, 0:1].astype(np.float32)  # tangent x (no underscore to match sympy Symbol "gx")
    gy = tangent[idx, 1:2].astype(np.float32)  # tangent y
    gz = tangent[idx, 2:3].astype(np.float32)  # tangent z
    sp = speed[idx].astype(np.float32)  # target speed

    smax = float(np.max(sp)) if sp.size > 0 else 1.0  # max speed for gating
    weight_floor = float(getattr(cfg.training, "geo_guidance_weight_floor", 0.1))  # floor
    weight_gate = np.clip(sp / max(smax, 1.0e-8), weight_floor, 1.0).astype(np.float32)  # (N,1)

    shared_invar = {
        "x": xg,
        "y": yg,
        "z": zg,
        "dw": dwg,
        "sin": sing,
        "sout": soutg,
        "gx": gx,
        "gy": gy,
        "gz": gz,
    }  # guidance inputs

    # -------- direction (cross_{x,y,z} = 0) + cosine = 1 --------
    # Split the V4 2D cross weight across the 3 cross components in 3D.
    w_cross = float(getattr(cfg.training, "w_geo_cross", 0.2)) / 3.0  # split across 3 components
    dir_constraint = _build_pointwise_constraint(
        nodes=geo_nodes,
        invar=shared_invar,
        outvar={
            "flow_geo_cross_x": np.zeros((xg.shape[0], 1), np.float32),
            "flow_geo_cross_y": np.zeros((xg.shape[0], 1), np.float32),
            "flow_geo_cross_z": np.zeros((xg.shape[0], 1), np.float32),
            "flow_geo_cosine": np.ones((xg.shape[0], 1), np.float32),
        },
        batch_size=min(
            int(getattr(cfg.training, "geo_guidance_batch_size", 16384)),
            int(xg.shape[0]),
        ),
        shuffle=True,
        lambda_weights={
            "flow_geo_cross_x": w_cross * s * weight_gate,
            "flow_geo_cross_y": w_cross * s * weight_gate,
            "flow_geo_cross_z": w_cross * s * weight_gate,
            "flow_geo_cosine": float(getattr(cfg.training, "w_geo_cosine", 0.1)) * s * weight_gate,
        },
    )  # direction strictness

    # -------- parallel magnitude = speed target --------
    parallel_constraint = _build_pointwise_constraint(
        nodes=geo_nodes,
        invar=shared_invar,
        outvar={
            "flow_geo_parallel": sp.astype(np.float32),
        },
        batch_size=min(
            int(getattr(cfg.training, "geo_guidance_batch_size", 16384)),
            int(xg.shape[0]),
        ),
        shuffle=True,
        lambda_weights={
            "flow_geo_parallel": float(getattr(cfg.training, "w_geo_parallel", 0.05)) * s * weight_gate,
        },
    )  # along-tangent magnitude

    # -------- speed magnitude (optional; default w=0) --------
    speed_constraint = None  # default none
    if float(getattr(cfg.training, "w_geo_speed", 0.0)) > 0.0:
        speed_constraint = _build_pointwise_constraint(
            nodes=geo_nodes,
            invar=shared_invar,
            outvar={
                "flow_geo_speed": sp.astype(np.float32),
            },
            batch_size=min(
                int(getattr(cfg.training, "geo_guidance_batch_size", 16384)),
                int(xg.shape[0]),
            ),
            shuffle=True,
            lambda_weights={
                "flow_geo_speed": float(getattr(cfg.training, "w_geo_speed", 0.0)) * s * weight_gate,
            },
        )  # speed guidance

    return dir_constraint, parallel_constraint, speed_constraint  # bundle


# -----------------------------
# Wall-guard constraints in 3D
# -----------------------------
def build_wall_guard_constraints_3d(
    wall_guard_nodes,
    cfg,
    xyz_wall_guard: np.ndarray,
    xyz_wall_guard_sep: np.ndarray,
    xyz_wall: np.ndarray,
    graph_feature_lookup_inside: Dict[str, np.ndarray],
    scale: float,
):
    """Build the two wall-guard no-penetration constraints.

    - ``wall_guard``      -- u n_x + v n_y + w n_z = 0 in a band around ANY wall
    - ``wall_guard_sep``  -- same, restricted to the separator y-band

    Normals are estimated with ``estimate_wall_normals_3d`` (kNN
    inverse-distance) so interior band points get a reasonable normal
    estimate without polluting the exact wall normals.
    """

    if not bool(getattr(cfg.training, "wall_guard_enabled", True)):
        return None, None  # disabled

    s = float(max(scale, 0.0))  # scale
    if s <= 0.0:
        return None, None  # inactive

    def _one_constraint(xyz_guard: np.ndarray, weight_name: str):
        if xyz_guard.shape[0] <= 0:
            return None  # nothing to build

        # kNN inverse-distance normal estimation in 3D
        n_vec = estimate_wall_normals_3d(
            xyz_points=xyz_guard.astype(np.float32),
            xyz_wall=xyz_wall.astype(np.float32),
            k_neighbors=int(getattr(cfg.training, "wall_guard_normal_k", 4)),
        )  # (N,3) unit normals

        # Project nearest interior features onto guard points
        tree = cKDTree(
            graph_feature_lookup_inside["xyz_inside"].astype(np.float64)
        )  # interior tree
        _, idx = tree.query(xyz_guard.astype(np.float64), k=1)  # nearest interior
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)  # flat
        dwg = graph_feature_lookup_inside["d_w_i"][idx].astype(np.float32)  # dw
        sing = graph_feature_lookup_inside["s_in_i"][idx].astype(np.float32)  # sin
        soutg = graph_feature_lookup_inside["s_out_i"][idx].astype(np.float32)  # sout

        return _build_pointwise_constraint(
            nodes=wall_guard_nodes,
            invar={
                "x": xyz_guard[:, 0:1].astype(np.float32),
                "y": xyz_guard[:, 1:2].astype(np.float32),
                "z": xyz_guard[:, 2:3].astype(np.float32),
                "dw": dwg,
                "sin": sing,
                "sout": soutg,
                "n_x": n_vec[:, 0:1].astype(np.float32),
                "n_y": n_vec[:, 1:2].astype(np.float32),
                "n_z": n_vec[:, 2:3].astype(np.float32),
            },
            outvar={
                "wall_normal_velocity": np.zeros(
                    (xyz_guard.shape[0], 1), np.float32,
                ),
            },
            batch_size=min(
                int(getattr(cfg.training, "wall_guard_batch_size", 8192)),
                int(xyz_guard.shape[0]),
            ),
            shuffle=True,
            lambda_weights={
                "wall_normal_velocity": float(getattr(cfg.training, weight_name, 0.5)) * s,
            },
        )  # wall no-penetration constraint

    global_guard = _one_constraint(
        xyz_guard=xyz_wall_guard,
        weight_name="w_wall_guard_normal",
    )  # global wall band
    sep_guard = _one_constraint(
        xyz_guard=xyz_wall_guard_sep,
        weight_name="w_wall_guard_separator_normal",
    )  # separator band

    return global_guard, sep_guard  # two wall guards


# -----------------------------
# Output JSON writer (V4.1 schema -- see task spec)
# -----------------------------
def _write_v4_1_flow_output_json(
    path: str,
    z_aspect: float,
    z_slices: int,
    norm: Tuple[float, float, float, float, float, float],
    xyz_all: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    p: np.ndarray,
    inlet_raw: Dict[str, float],
    outlet_raw: Dict[str, float],
) -> None:
    """Write the V4.1 3D flow prediction JSON.

    Schema (matches task spec):
        {
          "z_aspect": 0.10,
          "z_slices": 9,
          "norm": [xmin, xmax, ymin, ymax, zmin, zmax],
          "points": [[xn, yn, zn], ...],   // inside first, then walls
          "flow": {"u": [...], "v": [...], "w": [...], "p": [...], "t": 0.0},
          "inlet": {"x": 6, "y": 260},
          "outlet": {"x": 6, "y": 180}
        }
    """

    p_out = Path(path)  # path object
    p_out.parent.mkdir(parents=True, exist_ok=True)  # ensure parent

    payload = {
        "z_aspect": float(z_aspect),
        "z_slices": int(z_slices),
        "norm": [float(v) for v in norm],
        "points": xyz_all.astype(np.float32).tolist(),
        "flow": {
            "u": u.astype(np.float32).reshape(-1).tolist(),
            "v": v.astype(np.float32).reshape(-1).tolist(),
            "w": w.astype(np.float32).reshape(-1).tolist(),
            "p": p.astype(np.float32).reshape(-1).tolist(),
            "t": 0.0,
        },
        "inlet": {
            "x": float(inlet_raw["x"]),
            "y": float(inlet_raw["y"]),
        },
        "outlet": {
            "x": float(outlet_raw["x"]),
            "y": float(outlet_raw["y"]),
        },
    }  # V4.1 JSON payload

    with open(p_out, "w", encoding="utf-8") as f:
        json.dump(payload, f)  # compact (no indent) to keep file size down


# -----------------------------
# 3D inference helper (batched forward pass over all points)
# -----------------------------
def _run_inference_3d(
    flow_net: FullyConnectedArch,
    xyz_all: np.ndarray,
    dw_all: np.ndarray,
    sin_all: np.ndarray,
    sout_all: np.ndarray,
    batch: int = 65536,
):
    """Run the flow net over every point in ``xyz_all`` in batches.

    Returns (u, v, w, p) as flat float32 numpy arrays of length ``N``.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device
    flow_net = flow_net.to(device).eval()  # move to GPU + eval mode
    N = int(xyz_all.shape[0])  # point count

    u_out = np.zeros((N,), dtype=np.float32)  # u buffer
    v_out = np.zeros((N,), dtype=np.float32)  # v buffer
    w_out = np.zeros((N,), dtype=np.float32)  # w buffer
    p_out = np.zeros((N,), dtype=np.float32)  # p buffer

    with torch.no_grad():
        for s0 in range(0, N, int(max(batch, 1))):
            s1 = min(s0 + int(max(batch, 1)), N)  # batch end
            invar = {
                "x": torch.from_numpy(xyz_all[s0:s1, 0:1]).to(device),
                "y": torch.from_numpy(xyz_all[s0:s1, 1:2]).to(device),
                "z": torch.from_numpy(xyz_all[s0:s1, 2:3]).to(device),
                "dw": torch.from_numpy(dw_all[s0:s1]).to(device),
                "sin": torch.from_numpy(sin_all[s0:s1]).to(device),
                "sout": torch.from_numpy(sout_all[s0:s1]).to(device),
            }  # batch inputs
            out = flow_net(invar)  # forward
            u_out[s0:s1] = out["u"].detach().cpu().numpy().reshape(-1)  # u
            v_out[s0:s1] = out["v"].detach().cpu().numpy().reshape(-1)  # v
            w_out[s0:s1] = out["w"].detach().cpu().numpy().reshape(-1)  # w
            p_out[s0:s1] = out["p"].detach().cpu().numpy().reshape(-1)  # p

    return u_out, v_out, w_out, p_out  # flat arrays


# -----------------------------
# Main Hydra-driven trainer entry point
# -----------------------------
@hydra.main(version_base=None, config_path="conf", config_name="partner_v4_1_config")
def main(cfg: DictConfig) -> None:
    """Run the V4.1 3D steady Navier-Stokes flow trainer.

    Schedule: stage -1 (init field warmup) -> stage 0 (BC warmup) -> stages
    1..N (PDE stages with viscosity continuation). Each stage warm-starts
    from the previous stage's checkpoint via PhysicsNeMo's
    ``initialization_network_dir`` config key.

    The final flow field is inferred over every point (inside + wall) and
    written to ``<geom_stem>_pred_flow_steady.json`` (V4.1 schema).
    """

    # If the user passed the flow subtree, use it; else the top-level cfg is
    # already the flow config.
    cfg_flow = cfg.flow if "flow" in cfg else cfg  # accept either nesting

    # Resolve geometry path + load
    geom_path = str(cfg_flow.problem.geom_json_path)  # geometry JSON path
    print(f"[INFO] loading 3D geometry JSON: {geom_path}")  # log
    geom = load_3d_geometry_json(geom_path)  # parse JSON

    # Dimensions from the geometry JSON
    xmin, xmax, ymin, ymax, zmin_raw, zmax_raw = geom["norm"]  # norm tuple
    Lx = max(float(xmax - xmin), 1.0e-12)  # physical x length
    Ly = max(float(ymax - ymin), 1.0e-12)  # physical y length
    Lz = max(float(zmax_raw - zmin_raw), 1.0e-12)  # physical z length (raw units)
    z_aspect = float(geom["z_aspect"])  # normalized zmax
    z_slices = int(geom["z_slices"])  # slice count
    inlet_raw = geom["inlet"]  # raw inlet pixel
    outlet_raw = geom["outlet"]  # raw outlet pixel

    print(
        f"[INFO] geometry: inside={geom['xyz_inside'].shape[0]}, "
        f"wall={geom['xyz_wall'].shape[0]}, "
        f"inlet_patch={geom['xyz_inlet'].shape[0]}, "
        f"outlet_patch={geom['xyz_outlet'].shape[0]}, "
        f"z_aspect={z_aspect:.4f}, z_slices={z_slices}"
    )  # geometry summary
    print(
        f"[INFO] norm: xmin={xmin:.2f}, xmax={xmax:.2f}, "
        f"ymin={ymin:.2f}, ymax={ymax:.2f}, zmin={zmin_raw:.2f}, zmax={zmax_raw:.2f}"
    )  # norm summary
    print(f"[INFO] Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")  # scale summary

    # -------------------------------------------------
    # Interior / wall arrays (per-coord columns)
    # -------------------------------------------------
    xyz_inside = geom["xyz_inside"].astype(np.float32)  # (N_in,3)
    xyz_wall = geom["xyz_wall"].astype(np.float32)  # (N_w,3)
    n_wall = geom["n_wall"].astype(np.float32)  # (N_w,3) exact wall normals (unused directly here)

    # Per-axis columns for interior + wall (matches V4's x_i/y_i split)
    x_i = xyz_inside[:, 0:1].astype(np.float32)  # (N,1)
    y_i = xyz_inside[:, 1:2].astype(np.float32)  # (N,1)
    z_i = xyz_inside[:, 2:3].astype(np.float32)  # (N,1)
    x_w = xyz_wall[:, 0:1].astype(np.float32)  # (N,1)
    y_w = xyz_wall[:, 1:2].astype(np.float32)  # (N,1)
    z_w = xyz_wall[:, 2:3].astype(np.float32)  # (N,1)

    # -------------------------------------------------
    # Wall distance (dw): use precomputed from init_fields if available
    # -------------------------------------------------
    have_init_fields = geom["init_fields"] is not None  # init fields in JSON?
    if have_init_fields and "dw" in geom["init_fields"]:
        d_w_i = geom["init_fields"]["dw"].astype(np.float32)  # (N,1)
    else:
        print("[INFO] computing 3D wall distance on the fly")  # fallback log
        d_w_i = compute_wall_distance_3d(xyz_inside, xyz_wall).astype(np.float32)  # (N,1)
    d_w_w = np.zeros((xyz_wall.shape[0], 1), dtype=np.float32)  # exact wall distance = 0

    # -------------------------------------------------
    # Geodesic progress features: sin, sout
    # -------------------------------------------------
    # If init_fields carries s_in/s_out, use them; otherwise rebuild.
    adj = None  # may be needed if we recompute
    geo_info_in = None  # may be needed for init guess
    geo_info_out = None  # may be needed for init guess

    if have_init_fields and all(
        k in geom["init_fields"] for k in ("s_in", "s_out")
    ):
        s_in_i = geom["init_fields"]["s_in"].astype(np.float32)  # (N,1)
        s_out_i = geom["init_fields"]["s_out"].astype(np.float32)  # (N,1)
        print("[INFO] using precomputed 3D sin/sout from init_fields")  # log
    else:
        print("[INFO] computing 3D sin/sout on the fly (voxel+kNN fallback)")  # log
        adj = build_inside_graph_3d(
            xyz_inside=xyz_inside,
            raw_xyz_inside=geom["xyz_inside_raw"],
            mode="voxel",
        )  # voxel adjacency
        geo_info_in = compute_geodesic_info_3d(
            adj, xyz_inside, geom["inlet_center_xyz"]
        )  # inlet geodesic
        cov_in = float(
            np.sum(geo_info_in["predecessors"] != -9999)
        ) / float(max(xyz_inside.shape[0], 1))  # coverage
        if cov_in < 0.99:
            print(
                f"[INFO] voxel coverage {cov_in:.3f} < 0.99; rebuilding with kNN"
            )  # fallback log
            adj = build_inside_graph_3d(
                xyz_inside=xyz_inside,
                raw_xyz_inside=geom["xyz_inside_raw"],
                mode="knn",
                knn_k=14,
                max_edge_len=0.05,
            )  # kNN graph
            geo_info_in = compute_geodesic_info_3d(
                adj, xyz_inside, geom["inlet_center_xyz"]
            )  # inlet geodesic
        geo_info_out = compute_geodesic_info_3d(
            adj, xyz_inside, geom["outlet_center_xyz"]
        )  # outlet geodesic
        s_in_i = geo_info_in["s_geo"].astype(np.float32)  # (N,1)
        s_out_i = geo_info_out["s_geo"].astype(np.float32)  # (N,1)

    # Wall-projected geodesics
    s_in_w = project_inside_feature_to_wall_3d(
        xyz_wall=xyz_wall, xyz_inside=xyz_inside, feat_inside=s_in_i,
    )  # (N_w,1)
    s_out_w = project_inside_feature_to_wall_3d(
        xyz_wall=xyz_wall, xyz_inside=xyz_inside, feat_inside=s_out_i,
    )  # (N_w,1)

    # -------------------------------------------------
    # Initial flow guess (use precomputed if possible, else recompute)
    # -------------------------------------------------
    if have_init_fields and all(
        k in geom["init_fields"] for k in ("u", "v", "w", "p", "geo_in")
    ):
        init_inside_fields = dict(geom["init_fields"])  # shallow copy
        print("[INFO] using precomputed 3D init_fields (u,v,w,p,dw,s_in,s_out)")  # log
    else:
        print("[INFO] computing 3D initial flow guess on the fly")  # log
        if adj is None:
            # Need a graph for the init guess.
            adj = build_inside_graph_3d(
                xyz_inside=xyz_inside,
                raw_xyz_inside=geom["xyz_inside_raw"],
                mode="voxel",
            )  # voxel graph
            geo_info_in = compute_geodesic_info_3d(
                adj, xyz_inside, geom["inlet_center_xyz"]
            )  # inlet geodesic
            cov_in = float(
                np.sum(geo_info_in["predecessors"] != -9999)
            ) / float(max(xyz_inside.shape[0], 1))  # coverage
            if cov_in < 0.99:
                adj = build_inside_graph_3d(
                    xyz_inside=xyz_inside,
                    raw_xyz_inside=geom["xyz_inside_raw"],
                    mode="knn",
                    knn_k=14,
                    max_edge_len=0.05,
                )  # kNN
                geo_info_in = compute_geodesic_info_3d(
                    adj, xyz_inside, geom["inlet_center_xyz"]
                )  # inlet geodesic
            geo_info_out = compute_geodesic_info_3d(
                adj, xyz_inside, geom["outlet_center_xyz"]
            )  # outlet geodesic
        init_inside_fields = compute_initial_flow_guess_3d(
            xyz_inside=xyz_inside,
            xyz_wall=xyz_wall,
            z_max=z_aspect,
            inlet_u=float(cfg_flow.bc.inlet_u),
            inlet_v=float(cfg_flow.bc.inlet_v),
            inlet_p=float(getattr(cfg_flow.bc, "inlet_p", 1.0)),
            geo_info_in=geo_info_in,
            geo_info_out=geo_info_out,
            adj=adj,
            velocity_scale=float(getattr(cfg_flow.init_guess, "velocity_scale", 1.0)),
            velocity_power=float(getattr(cfg_flow.init_guess, "velocity_power", 1.0)),
            pressure_power=float(getattr(cfg_flow.init_guess, "pressure_power", 1.0)),
            pressure_drop_guess=float(
                getattr(cfg_flow.init_guess, "pressure_drop_guess", 0.0)
            ),
        )  # (re-)compute init guess

    # If tangent or predecessors are missing (loaded-from-JSON path), rebuild them
    # from the predecessor tree so the geo-guidance constraints have a tangent.
    if "tangent" not in init_inside_fields or init_inside_fields.get("tangent") is None:
        if "predecessors" in init_inside_fields and "src" in init_inside_fields:
            init_inside_fields["tangent"] = estimate_tangent_from_predecessor_tree_3d(
                xyz_inside=xyz_inside,
                predecessors=init_inside_fields["predecessors"],
                src=int(init_inside_fields["src"]),
            )  # (N,3) unit tangent
        else:
            print(
                "[INFO] init_fields has no predecessors/src; estimating tangent from u,v,w"
            )  # fallback log
            u_init = init_inside_fields["u"].reshape(-1)  # u
            v_init = init_inside_fields["v"].reshape(-1)  # v
            w_init = init_inside_fields["w"].reshape(-1)  # w
            mag = np.sqrt(u_init ** 2 + v_init ** 2 + w_init ** 2) + 1.0e-12  # magnitude
            tangent = np.stack(
                [u_init / mag, v_init / mag, w_init / mag], axis=1,
            ).astype(np.float32)  # (N,3)
            # Replace rows with near-zero magnitude with +x fallback
            zero_rows = (
                np.sqrt(u_init ** 2 + v_init ** 2 + w_init ** 2) < 1.0e-10
            )  # (N,) bool
            tangent[zero_rows] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # fallback
            init_inside_fields["tangent"] = tangent  # store

    if "predecessors" not in init_inside_fields:
        init_inside_fields["predecessors"] = np.full(
            (xyz_inside.shape[0],), -9999, dtype=np.int64,
        )  # default all-disconnected (only used for corridor trace)
    if "src" not in init_inside_fields:
        # Nearest inside point to the inlet center
        d2 = np.sum(
            (xyz_inside - geom["inlet_center_xyz"][0:1]) ** 2, axis=1,
        )  # sq distance
        init_inside_fields["src"] = int(np.argmin(d2))  # fallback src

    # -------------------------------------------------
    # Inlet / outlet masks (for soft-init weighting + guidance exclusion)
    # -------------------------------------------------
    inlet_mask = _build_patch_mask_3d(
        xyz_inside=xyz_inside,
        xyz_patch=geom["xyz_inlet"],
        radius=float(getattr(cfg_flow.bc, "inlet_radius_norm", 0.002)),
    )  # inlet mask
    outlet_mask = _build_patch_mask_3d(
        xyz_inside=xyz_inside,
        xyz_patch=geom["xyz_outlet"],
        radius=float(getattr(cfg_flow.bc, "outlet_radius_norm", 0.002)),
    )  # outlet mask

    # Safety: if masks are empty (very small geometry), grow the radius
    # or fall back to nearest-inside markings per patch point.
    if int(np.sum(inlet_mask)) == 0:
        inlet_mask = _nearest_inside_mask_3d(
            xyz_inside=xyz_inside, xyz_patch=geom["xyz_inlet"],
        )  # fallback
        print(
            f"[INFO] inlet radius patch was empty; fell back to nearest-inside "
            f"({int(np.sum(inlet_mask))} pts)"
        )  # log
    if int(np.sum(outlet_mask)) == 0:
        outlet_mask = _nearest_inside_mask_3d(
            xyz_inside=xyz_inside, xyz_patch=geom["xyz_outlet"],
        )  # fallback
        print(
            f"[INFO] outlet radius patch was empty; fell back to nearest-inside "
            f"({int(np.sum(outlet_mask))} pts)"
        )  # log
    print(
        f"[INFO] inlet mask points={int(np.sum(inlet_mask))}, "
        f"outlet mask points={int(np.sum(outlet_mask))}"
    )  # mask info

    # -------------------------------------------------
    # Inlet / outlet patch invars (for inlet_vel + optional outlet_p + inlet_p anchors)
    # -------------------------------------------------
    inlet_patch = geom["xyz_inlet"].astype(np.float32)  # (N_il,3) 3D inlet patch
    outlet_patch = geom["xyz_outlet"].astype(np.float32)  # (N_ol,3) 3D outlet patch

    # Project inside features onto inlet / outlet patch points (nearest).
    if inlet_patch.shape[0] > 0:
        inlet_dw = project_inside_feature_to_wall_3d(
            xyz_wall=inlet_patch, xyz_inside=xyz_inside, feat_inside=d_w_i,
        )  # (N_il,1)
        inlet_sin = project_inside_feature_to_wall_3d(
            xyz_wall=inlet_patch, xyz_inside=xyz_inside, feat_inside=s_in_i,
        )  # (N_il,1)
        inlet_sout = project_inside_feature_to_wall_3d(
            xyz_wall=inlet_patch, xyz_inside=xyz_inside, feat_inside=s_out_i,
        )  # (N_il,1)
        inlet_phi_z = _apply_parabolic_z_profile(
            z_norm=inlet_patch[:, 2:3].astype(np.float32), z_max=z_aspect,
        )  # (N_il,1) phi(z)
    else:
        inlet_dw = np.zeros((0, 1), dtype=np.float32)  # empty
        inlet_sin = np.zeros((0, 1), dtype=np.float32)  # empty
        inlet_sout = np.zeros((0, 1), dtype=np.float32)  # empty
        inlet_phi_z = np.zeros((0, 1), dtype=np.float32)  # empty
    inlet_invar = {
        "x": inlet_patch[:, 0:1].astype(np.float32),
        "y": inlet_patch[:, 1:2].astype(np.float32),
        "z": inlet_patch[:, 2:3].astype(np.float32),
        "dw": inlet_dw,
        "sin": inlet_sin,
        "sout": inlet_sout,
    }  # inlet patch inputs

    # If inlet_z_profile is "plug", override phi by 1.0.
    if str(getattr(cfg_flow.bc, "inlet_z_profile", "parabolic")).strip().lower() == "plug":
        inlet_phi_z = np.ones_like(inlet_phi_z).astype(np.float32)  # (N_il,1) plug profile
        print("[INFO] inlet z-profile: plug (phi(z)=1)")  # log
    else:
        print("[INFO] inlet z-profile: parabolic (phi(z) = 4 z/Zmax (1 - z/Zmax))")  # log

    if outlet_patch.shape[0] > 0:
        outlet_dw = project_inside_feature_to_wall_3d(
            xyz_wall=outlet_patch, xyz_inside=xyz_inside, feat_inside=d_w_i,
        )  # (N_ol,1)
        outlet_sin = project_inside_feature_to_wall_3d(
            xyz_wall=outlet_patch, xyz_inside=xyz_inside, feat_inside=s_in_i,
        )  # (N_ol,1)
        outlet_sout = project_inside_feature_to_wall_3d(
            xyz_wall=outlet_patch, xyz_inside=xyz_inside, feat_inside=s_out_i,
        )  # (N_ol,1)
        outlet_phi_z = _apply_parabolic_z_profile(
            z_norm=outlet_patch[:, 2:3].astype(np.float32), z_max=z_aspect,
        )  # (N_ol,1)
    else:
        outlet_dw = np.zeros((0, 1), dtype=np.float32)  # empty
        outlet_sin = np.zeros((0, 1), dtype=np.float32)  # empty
        outlet_sout = np.zeros((0, 1), dtype=np.float32)  # empty
        outlet_phi_z = np.zeros((0, 1), dtype=np.float32)  # empty
    outlet_invar = {
        "x": outlet_patch[:, 0:1].astype(np.float32),
        "y": outlet_patch[:, 1:2].astype(np.float32),
        "z": outlet_patch[:, 2:3].astype(np.float32),
        "dw": outlet_dw,
        "sin": outlet_sin,
        "sout": outlet_sout,
    }  # outlet patch inputs

    # -------------------------------------------------
    # Weighted PDE sampling (3D) + progress-sorted chunking
    # -------------------------------------------------
    pde_target = int(
        getattr(cfg_flow.training, "flow_pde_points_target", xyz_inside.shape[0])
    )  # target count
    outlet_nearest_idx = int(
        np.argmin(
            np.sum((xyz_inside - geom["outlet_center_xyz"][0:1]) ** 2, axis=1)
        )
    )  # nearest inside to outlet (for corridor trace endpoint)
    src_in_idx = int(init_inside_fields.get("src", 0))  # inlet source index

    pde_idx = _build_weighted_flow_pde_indices_3d(
        xyz_inside=xyz_inside,
        d_w_i=d_w_i,
        predecessors_in=init_inside_fields["predecessors"],
        src_in=src_in_idx,
        outlet_target_idx=outlet_nearest_idx,
        z_max=z_aspect,
        target_points=pde_target,
        wall_boost=float(getattr(cfg_flow.training, "flow_pde_wall_boost", 1.0)),
        wall_scale=float(getattr(cfg_flow.training, "flow_pde_wall_scale", 0.02)),
        corridor_boost=float(getattr(cfg_flow.training, "flow_pde_corridor_boost", 2.0)),
        corridor_radius=float(getattr(cfg_flow.training, "flow_pde_corridor_radius", 0.05)),
        cap_boost=float(getattr(cfg_flow.training, "flow_pde_cap_boost", 1.5)),
        cap_scale=float(getattr(cfg_flow.training, "flow_pde_cap_scale", 0.01)),
        seed=int(getattr(cfg_flow.training, "flow_pde_sampling_seed", 1234)),
    )  # (M,) indices

    x_i_pde = x_i[pde_idx].astype(np.float32)  # x
    y_i_pde = y_i[pde_idx].astype(np.float32)  # y
    z_i_pde = z_i[pde_idx].astype(np.float32)  # z
    d_w_i_pde = d_w_i[pde_idx].astype(np.float32)  # dw
    s_in_i_pde = s_in_i[pde_idx].astype(np.float32)  # sin
    s_out_i_pde = s_out_i[pde_idx].astype(np.float32)  # sout

    (
        x_i_sorted,
        y_i_sorted,
        z_i_sorted,
        d_w_i_sorted,
        s_in_i_sorted,
        s_out_i_sorted,
        inside_order,
    ) = sort_by_progress_chunked_3d(
        x_i=x_i_pde,
        y_i=y_i_pde,
        z_i=z_i_pde,
        d_w_i=d_w_i_pde,
        s_in_i=s_in_i_pde,
        s_out_i=s_out_i_pde,
        chunk_size=int(getattr(cfg_flow.training, "curriculum_chunk_size", 8192)),
        seed=int(getattr(cfg_flow.training, "flow_pde_sampling_seed", 1234)),
    )  # sort-by-progress chunking

    print(
        f"[INFO] PDE interior points used: {int(x_i_sorted.shape[0])} / {int(x_i.shape[0])}"
    )  # PDE subset info

    # -------------------------------------------------
    # Wall-guard point sets (3D)
    # -------------------------------------------------
    port_exclude_mask = np.zeros((xyz_inside.shape[0],), dtype=bool)  # exclusion mask
    if bool(getattr(cfg_flow.training, "wall_guard_exclude_ports", True)):
        port_exclude_mask |= inlet_mask  # exclude inlet patch
        port_exclude_mask |= outlet_mask  # exclude outlet patch

    xyz_wall_guard = (
        _build_wall_guard_points_3d(
            xyz_inside=xyz_inside,
            xyz_wall=xyz_wall,
            radius=float(getattr(cfg_flow.training, "wall_guard_radius", 0.02)),
            target_points=int(getattr(cfg_flow.training, "wall_guard_points", 4000)),
            seed=int(getattr(cfg_flow.training, "wall_guard_seed", 1234)) if hasattr(cfg_flow.training, "wall_guard_seed")
            else int(getattr(cfg_flow.training, "flow_pde_sampling_seed", 1234)),
            exclude_mask=port_exclude_mask,
        ) if bool(getattr(cfg_flow.training, "wall_guard_enabled", True))
        else np.zeros((0, 3), dtype=np.float32)
    )  # global wall band

    # Separator band (V4 uses y-filter around midline; z is left unrestricted in 3D)
    if bool(getattr(cfg_flow.training, "wall_guard_separator_enabled", True)):
        inlet_y_center = float(geom["inlet_center_xyz"][0, 1])  # inlet y center
        outlet_y_center = float(geom["outlet_center_xyz"][0, 1])  # outlet y center
        y_mid = 0.5 * (inlet_y_center + outlet_y_center)  # separator center y
        y_half = 0.5 * abs(outlet_y_center - inlet_y_center) * float(
            getattr(cfg_flow.training, "wall_guard_separator_span_factor", 0.85)
        )  # half-width
        y_half = max(y_half, 1.0e-3)  # guard
        xyz_wall_guard_sep = _build_wall_guard_points_3d(
            xyz_inside=xyz_inside,
            xyz_wall=xyz_wall,
            radius=float(getattr(
                cfg_flow.training,
                "wall_guard_separator_radius",
                float(getattr(cfg_flow.training, "wall_guard_radius", 0.02)),
            )),
            target_points=int(getattr(cfg_flow.training, "wall_guard_separator_points", 3000)),
            seed=int(getattr(cfg_flow.training, "flow_pde_sampling_seed", 1234)) + 17,
            x_max=float(getattr(cfg_flow.training, "wall_guard_separator_x_max", 0.42)),
            y_min=(y_mid - y_half),
            y_max=(y_mid + y_half),
            exclude_mask=port_exclude_mask,
        )  # separator band
    else:
        xyz_wall_guard_sep = np.zeros((0, 3), dtype=np.float32)  # none
    print(
        f"[INFO] wall guard points={int(xyz_wall_guard.shape[0])}, "
        f"separator guard points={int(xyz_wall_guard_sep.shape[0])}"
    )  # wall guard info

    # -------------------------------------------------
    # Feature lookup for wall-guard projection (3D)
    # -------------------------------------------------
    graph_feature_lookup_inside = {
        "xyz_inside": xyz_inside.astype(np.float32),
        "d_w_i": d_w_i.astype(np.float32),
        "s_in_i": s_in_i.astype(np.float32),
        "s_out_i": s_out_i.astype(np.float32),
    }  # lookup

    # -------------------------------------------------
    # Network (6 inputs, 4 outputs)
    # -------------------------------------------------
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("dw"), Key("sin"), Key("sout")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        layer_size=int(cfg_flow.flow_model.hidden_size),
        nr_layers=int(cfg_flow.flow_model.hidden_layers),
        activation_fn=get_activation(cfg_flow.flow_model.activation),
    )  # 3D flow net
    print(
        f"[INFO] flow net: "
        f"layers={int(cfg_flow.flow_model.hidden_layers)}, "
        f"width={int(cfg_flow.flow_model.hidden_size)}, "
        f"activation={cfg_flow.flow_model.activation}, "
        f"params={sum(p.numel() for p in flow_net.parameters())}"
    )  # net summary

    start_time = time.time()  # training timer

    # -------------------------------------------------
    # PDE/constraint classes (viscosity is continuation-scheduled below)
    # -------------------------------------------------
    geo_guidance_pde = FlowTrajectoryGuidance3D(
        speed_eps=float(getattr(cfg_flow.training, "geo_guidance_speed_eps", 1.0e-4)),
    )  # 3D guidance PDE
    wall_guard_pde = WallNormalNoPenetration3D(
        eq_name="wall_normal_velocity",
    )  # 3D wall no-penetration

    def make_flow_nodes(nu_value: float):
        """Build PhysicsNeMo nodes for a given viscosity value.

        Returns (flow_nodes, geo_nodes, wall_guard_nodes). Each node list
        includes the flow net (so the NS + guidance + wall-guard graphs
        share parameters).
        """

        ns = SteadyNavierStokes3DScaled(
            rho=float(cfg_flow.physics.rho),
            nu=float(nu_value),
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
        )  # 3D NS PDE
        flow_nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]  # NS graph
        geo_nodes = (
            geo_guidance_pde.make_nodes()
            + [flow_net.make_node(name="flow_network")]
        )  # guidance graph
        wall_guard_nodes = (
            wall_guard_pde.make_nodes()
            + [flow_net.make_node(name="flow_network")]
        )  # wall-guard graph
        return flow_nodes, geo_nodes, wall_guard_nodes  # bundle

    # -------------------------------------------------
    # Stage -1: pseudo-field warmup
    # -------------------------------------------------
    if bool(getattr(cfg_flow.training, "use_init_field_warmup", True)):
        flow_nodes_init, _, _ = make_flow_nodes(
            float(cfg_flow.training.nu_schedule[0])
        )  # stage -1 nodes

        init_constraint = build_pseudo_init_constraint_3d(
            flow_nodes=flow_nodes_init,
            cfg=cfg_flow,
            x_i=x_i,
            y_i=y_i,
            z_i=z_i,
            d_w_i=d_w_i,
            s_in_i=s_in_i,
            s_out_i=s_out_i,
            init_inside_fields=init_inside_fields,
            batch_size_key="flow_init_batch_size",
            scale=float(getattr(cfg_flow.training, "init_field_warmup_scale", 1.0)),
        )  # pseudo-field regression

        domain_init = Domain()  # stage -1 domain
        domain_init.add_constraint(init_constraint, "init_field_fit")  # single constraint

        cfg_init = make_stage_cfg(
            cfg_flow,
            "stage_m1_init_guess_warmup",
            int(getattr(cfg_flow.training, "k_flow_init", 3000)),
            init_dir="",
        )  # stage -1 cfg
        print(
            f"[INFO] stage -1 init_guess_warmup: steps={int(cfg_init.training.max_steps)}, "
            f"scale={float(getattr(cfg_flow.training, 'init_field_warmup_scale', 1.0))}"
        )  # stage log
        slv_init = Solver(cfg_init, domain_init)  # solver
        _disable_recording(slv_init)  # silence TB
        slv_init.solve()  # train warmup
        prev_dir = str(Path(cfg_init.network_dir))  # checkpoint dir for next stage
    else:
        prev_dir = ""  # no init checkpoint

    # -------------------------------------------------
    # Stage 0: BC warmup (no PDE)
    # -------------------------------------------------
    flow_nodes0, geo_nodes0, wall_guard_nodes0 = make_flow_nodes(
        float(cfg_flow.training.nu_schedule[0])
    )  # stage 0 nodes

    _pde_c0, wall_c0, inlet_c0, outlet_c0, inlet_p_c0 = build_primary_constraints_3d(
        flow_nodes=flow_nodes0,
        cfg=cfg_flow,
        x_w=x_w, y_w=y_w, z_w=z_w,
        d_w_w=d_w_w, s_in_w=s_in_w, s_out_w=s_out_w,
        x_i_sorted=x_i_sorted, y_i_sorted=y_i_sorted, z_i_sorted=z_i_sorted,
        d_w_i_sorted=d_w_i_sorted, s_in_i_sorted=s_in_i_sorted, s_out_i_sorted=s_out_i_sorted,
        inlet_invar=inlet_invar,
        outlet_invar=outlet_invar,
        inlet_phi_z=inlet_phi_z,
        outlet_phi_z=outlet_phi_z,
        weight_scale=float(getattr(cfg_flow.training, "bc_stage_weight_scale", 1.0)),
    )  # stage 0 primary constraints (pde constraint is discarded in stage 0)

    soft_init_bc = (
        build_pseudo_init_constraint_3d(
            flow_nodes=flow_nodes0,
            cfg=cfg_flow,
            x_i=x_i, y_i=y_i, z_i=z_i,
            d_w_i=d_w_i, s_in_i=s_in_i, s_out_i=s_out_i,
            init_inside_fields=init_inside_fields,
            batch_size_key="flow_soft_init_batch_size",
            scale=float(getattr(cfg_flow.training, "soft_init_bc_scale", 0.25)),
        )
        if bool(getattr(cfg_flow.training, "use_soft_init_during_bc", True))
        else None
    )  # BC soft init

    geo_dir_bc, geo_parallel_bc, geo_speed_bc = (
        build_geo_guidance_constraints_3d(
            geo_nodes=geo_nodes0,
            cfg=cfg_flow,
            xyz_inside=xyz_inside,
            d_w_i=d_w_i, s_in_i=s_in_i, s_out_i=s_out_i,
            init_inside_fields=init_inside_fields,
            inlet_mask=inlet_mask,
            outlet_mask=outlet_mask,
            scale=float(getattr(cfg_flow.training, "geo_guidance_bc_scale", 0.25)),
        )
        if bool(getattr(cfg_flow.training, "use_geo_guidance_during_bc", True))
        else (None, None, None)
    )  # BC geo guidance

    wall_guard_bc, wall_guard_sep_bc = build_wall_guard_constraints_3d(
        wall_guard_nodes=wall_guard_nodes0,
        cfg=cfg_flow,
        xyz_wall_guard=xyz_wall_guard,
        xyz_wall_guard_sep=xyz_wall_guard_sep,
        xyz_wall=xyz_wall,
        graph_feature_lookup_inside=graph_feature_lookup_inside,
        scale=float(getattr(cfg_flow.training, "wall_guard_scale", 1.0)),
    )  # BC wall guard

    domain_bc = Domain()  # stage 0 domain
    domain_bc.add_constraint(wall_c0, "wall_noslip")  # wall no-slip
    domain_bc.add_constraint(inlet_c0, "inlet_vel")  # inlet velocity
    if outlet_c0 is not None:
        domain_bc.add_constraint(outlet_c0, "outlet_p")  # optional outlet p
    if inlet_p_c0 is not None:
        domain_bc.add_constraint(inlet_p_c0, "inlet_p_anchor")  # optional inlet p
    if soft_init_bc is not None:
        domain_bc.add_constraint(soft_init_bc, "soft_init_bc")  # BC soft init
    if geo_dir_bc is not None:
        domain_bc.add_constraint(geo_dir_bc, "geo_dir_bc")  # BC direction guidance
    if geo_parallel_bc is not None:
        domain_bc.add_constraint(geo_parallel_bc, "geo_parallel_bc")  # BC parallel guidance
    if geo_speed_bc is not None:
        domain_bc.add_constraint(geo_speed_bc, "geo_speed_bc")  # BC speed guidance
    if wall_guard_bc is not None:
        domain_bc.add_constraint(wall_guard_bc, "wall_guard_bc")  # global wall guard
    if wall_guard_sep_bc is not None:
        domain_bc.add_constraint(wall_guard_sep_bc, "wall_guard_sep_bc")  # separator guard

    cfg_bc = make_stage_cfg(
        cfg_flow,
        "stage_00_bc_warmup",
        int(cfg_flow.training.k_flow_bc),
        init_dir=prev_dir,
    )  # stage 0 cfg
    print(
        f"[INFO] stage 0 bc_warmup: steps={int(cfg_bc.training.max_steps)}, "
        f"soft_init_scale={float(getattr(cfg_flow.training, 'soft_init_bc_scale', 0.25))}, "
        f"geo_scale={float(getattr(cfg_flow.training, 'geo_guidance_bc_scale', 0.25))}, "
        f"warm_start={prev_dir}"
    )  # stage log
    slv_bc = Solver(cfg_bc, domain_bc)  # solver
    _disable_recording(slv_bc)  # silence TB
    slv_bc.solve()  # train stage 0
    prev_dir = str(Path(cfg_bc.network_dir))  # checkpoint dir for next stage

    # -------------------------------------------------
    # PDE stages (viscosity continuation)
    # -------------------------------------------------
    n_stages = len(cfg_flow.training.nu_schedule)  # PDE stage count
    for stage_idx, nu_stage in enumerate(cfg_flow.training.nu_schedule):
        flow_nodes, geo_nodes, wall_guard_nodes = make_flow_nodes(float(nu_stage))  # nodes

        pde_weight_scale = float(
            getattr(cfg_flow.training, "pde_stage_weight_scale", 1.0)
        )  # PDE-stage weight scale

        (
            flow_pde_constraint,
            wall_constraint,
            inlet_constraint,
            outlet_constraint,
            inlet_p_constraint,
        ) = build_primary_constraints_3d(
            flow_nodes=flow_nodes,
            cfg=cfg_flow,
            x_w=x_w, y_w=y_w, z_w=z_w,
            d_w_w=d_w_w, s_in_w=s_in_w, s_out_w=s_out_w,
            x_i_sorted=x_i_sorted, y_i_sorted=y_i_sorted, z_i_sorted=z_i_sorted,
            d_w_i_sorted=d_w_i_sorted, s_in_i_sorted=s_in_i_sorted, s_out_i_sorted=s_out_i_sorted,
            inlet_invar=inlet_invar,
            outlet_invar=outlet_invar,
            inlet_phi_z=inlet_phi_z,
            outlet_phi_z=outlet_phi_z,
            weight_scale=pde_weight_scale,
        )  # primary constraints (including PDE)

        soft_init_scale = _linear_stage_scale(
            stage_idx=stage_idx,
            num_stages=n_stages,
            start_scale=float(getattr(cfg_flow.training, "soft_init_start_scale", 0.1)),
            end_scale=float(getattr(cfg_flow.training, "soft_init_end_scale", 0.01)),
        )  # soft-init decay schedule

        geo_guidance_scale = _linear_stage_scale(
            stage_idx=stage_idx,
            num_stages=n_stages,
            start_scale=float(getattr(cfg_flow.training, "geo_guidance_start_scale", 0.25)),
            end_scale=float(getattr(cfg_flow.training, "geo_guidance_end_scale", 0.02)),
        )  # geo-guidance decay schedule

        init_soft = (
            build_pseudo_init_constraint_3d(
                flow_nodes=flow_nodes,
                cfg=cfg_flow,
                x_i=x_i, y_i=y_i, z_i=z_i,
                d_w_i=d_w_i, s_in_i=s_in_i, s_out_i=s_out_i,
                init_inside_fields=init_inside_fields,
                batch_size_key="flow_soft_init_batch_size",
                scale=soft_init_scale,
            )
            if bool(getattr(cfg_flow.training, "use_soft_init_constraint", True))
            else None
        )  # stage soft init

        geo_dir_c, geo_parallel_c, geo_speed_c = (
            build_geo_guidance_constraints_3d(
                geo_nodes=geo_nodes,
                cfg=cfg_flow,
                xyz_inside=xyz_inside,
                d_w_i=d_w_i, s_in_i=s_in_i, s_out_i=s_out_i,
                init_inside_fields=init_inside_fields,
                inlet_mask=inlet_mask,
                outlet_mask=outlet_mask,
                scale=geo_guidance_scale,
            )
            if bool(getattr(cfg_flow.training, "use_geo_direction_guidance", True))
            else (None, None, None)
        )  # stage geo guidance

        wall_guard_c, wall_guard_sep_c = build_wall_guard_constraints_3d(
            wall_guard_nodes=wall_guard_nodes,
            cfg=cfg_flow,
            xyz_wall_guard=xyz_wall_guard,
            xyz_wall_guard_sep=xyz_wall_guard_sep,
            xyz_wall=xyz_wall,
            graph_feature_lookup_inside=graph_feature_lookup_inside,
            scale=float(getattr(cfg_flow.training, "wall_guard_scale", 1.0)),
        )  # stage wall guards

        domain_full = Domain()  # stage domain
        domain_full.add_constraint(flow_pde_constraint, "flow_pde")  # PDE
        domain_full.add_constraint(wall_constraint, "wall_noslip")  # wall no-slip
        domain_full.add_constraint(inlet_constraint, "inlet_vel")  # inlet velocity
        if outlet_constraint is not None:
            domain_full.add_constraint(outlet_constraint, "outlet_p")  # optional outlet p

        if inlet_p_constraint is not None:
            if bool(getattr(cfg_flow.bc, "use_inlet_pressure_anchor_last_stage_only", False)):
                if stage_idx == n_stages - 1:
                    domain_full.add_constraint(inlet_p_constraint, "inlet_p_anchor")  # last only
            else:
                domain_full.add_constraint(inlet_p_constraint, "inlet_p_anchor")  # every stage

        if init_soft is not None:
            domain_full.add_constraint(init_soft, f"soft_init_c{stage_idx + 1:02d}")  # soft init
        if geo_dir_c is not None:
            domain_full.add_constraint(geo_dir_c, f"geo_dir_c{stage_idx + 1:02d}")  # direction
        if geo_parallel_c is not None:
            domain_full.add_constraint(geo_parallel_c, f"geo_parallel_c{stage_idx + 1:02d}")  # parallel
        if geo_speed_c is not None:
            domain_full.add_constraint(geo_speed_c, f"geo_speed_c{stage_idx + 1:02d}")  # speed
        if wall_guard_c is not None:
            domain_full.add_constraint(wall_guard_c, f"wall_guard_c{stage_idx + 1:02d}")  # wall guard
        if wall_guard_sep_c is not None:
            domain_full.add_constraint(
                wall_guard_sep_c, f"wall_guard_sep_c{stage_idx + 1:02d}",
            )  # separator guard

        stage_steps = int(cfg_flow.training.k_flow_per_stage[stage_idx])  # stage steps
        stage_name = f"stage_{stage_idx + 1:02d}_nu_{nu_stage:.2e}".replace("+", "")  # stage name
        cfg_stage = make_stage_cfg(
            cfg_flow, stage_name, stage_steps, init_dir=prev_dir,
        )  # stage cfg

        print(
            f"[INFO] stage {stage_idx + 1}/{n_stages} nu={nu_stage:.4e} "
            f"steps={stage_steps} | soft_init_scale={soft_init_scale:.4f} "
            f"geo_guidance_scale={geo_guidance_scale:.4f} | warm_start={prev_dir}"
        )  # stage log

        slv = Solver(cfg_stage, domain_full)  # stage solver
        _disable_recording(slv)  # silence TB
        slv.solve()  # train stage

        prev_dir = str(Path(cfg_stage.network_dir))  # update checkpoint dir

    end_time = time.time()  # end timer
    total_min = (end_time - start_time) / 60.0  # minutes
    print(f"[OK] flow training complete in {total_min:.4f} minutes")  # timing

    # -------------------------------------------------
    # Inference over all points (inside first, then wall) + JSON output
    # -------------------------------------------------
    xyz_all = np.concatenate([xyz_inside, xyz_wall], axis=0).astype(np.float32)  # (N,3)
    dw_all = np.concatenate([d_w_i, d_w_w], axis=0).astype(np.float32)  # (N,1)
    sin_all = np.concatenate([s_in_i, s_in_w], axis=0).astype(np.float32)  # (N,1)
    sout_all = np.concatenate([s_out_i, s_out_w], axis=0).astype(np.float32)  # (N,1)

    print(f"[INFO] running inference on {int(xyz_all.shape[0])} points")  # inference log
    u_all, v_all, w_all, p_all = _run_inference_3d(
        flow_net=flow_net,
        xyz_all=xyz_all,
        dw_all=dw_all,
        sin_all=sin_all,
        sout_all=sout_all,
        batch=int(getattr(cfg_flow, "inference", {}).get("batch_size", 65536))
        if isinstance(getattr(cfg_flow, "inference", {}), dict)
        else int(getattr(getattr(cfg_flow, "inference", {}), "batch_size", 65536)),
    )  # batched forward

    # Exact no-slip on wall points (V4 pattern): inside first, then wall
    n_inside = int(xyz_inside.shape[0])  # wall starts at this offset
    u_all[n_inside:] = 0.0  # exact u=0 on walls
    v_all[n_inside:] = 0.0  # exact v=0 on walls
    w_all[n_inside:] = 0.0  # exact w=0 on walls

    # Output path: derive from the input geometry stem so smoke and full
    # runs with different input JSONs don't collide on the same output file.
    geom_p = Path(geom_path)  # input path
    out_json_path = str(
        geom_p.with_name(f"{geom_p.stem}_pred_flow_steady.json")
    )  # output next to input geometry, stem-derived
    _write_v4_1_flow_output_json(
        path=out_json_path,
        z_aspect=z_aspect,
        z_slices=z_slices,
        norm=geom["norm"],
        xyz_all=xyz_all,
        u=u_all,
        v=v_all,
        w=w_all,
        p=p_all,
        inlet_raw=inlet_raw,
        outlet_raw=outlet_raw,
    )  # write V4.1 schema JSON

    try:
        out_size = Path(out_json_path).stat().st_size  # file size
        out_size_mb = out_size / 1024.0 / 1024.0  # MB
        print(
            f"[OK] wrote flow results to: {out_json_path} ({out_size_mb:.1f} MB)"
        )  # success
    except OSError:
        print(f"[OK] wrote flow results to: {out_json_path}")  # success without size

    print(f"Total elapsed training time: {total_min:.4f} minutes")  # timing


if __name__ == "__main__":
    main()  # entry point
