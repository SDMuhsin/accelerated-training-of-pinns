"""V4.1 3D geometry utilities.

This module extrudes the V4 2D battery-cooling channel geometry to 3D for
the V4.1 PINN pipeline. It is autograd-only (no SAGE, no JAX). The public
API is a contract the V4.1 flow and temp trainers rely on; see
`llmdocs/stream_battery_consortium/V4_1_DESIGN.md`.

Key design points (locked in V4_1_DESIGN.md):
  - Depth axis z perpendicular to the 2D image plane.
  - Z_max = 0.10 * L_x (10:1 plate aspect ratio).
  - z-normalization uses L_x so z in [0, 0.10].
  - z_slices = 9 (2 caps + 7 interior fluid layers).
  - Side-wall normals: 2D (nx, ny) extruded, nz = 0.
  - Cap normals: (0, 0, -1) at z=0, (0, 0, +1) at z=zmax.
  - Interior-band normals: weighted inverse-distance kNN in 3D.
  - Inside graph: voxel 6-connectivity (fallback kNN k=14).
  - Inlet BC: parabolic z-profile phi(z) = 4 (z/Zmax)(1 - z/Zmax).

The helpers here mirror V4 style (short per-line comments, float32 output,
numpy-only) and duplicate small pieces of `partner_v4_flow.py` rather than
import from it — that file performs module-level CUDA init at import time.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Constants
# -----------------------------
_SMALL = 1.0e-8  # global small constant for guard / clamp


# -----------------------------
# 2D JSON loader
# -----------------------------
def load_2d_geometry_json(path):
    """Load the V4 2D JSON and validate schema.

    Parameters
    ----------
    path : str or Path
        Path to the V4 2D JSON. Schema:
        {width, height, inlet{x,y}, outlet{x,y}, legend, points:[[x,y,class]]}

    Returns
    -------
    dict
        Parsed JSON with validated keys. Returned as-is (no copy).
    """

    p = Path(path)  # path object
    if not p.exists():
        raise FileNotFoundError(f"2D geometry JSON not found: {p}")  # missing

    try:
        obj = json.loads(p.read_text())  # load JSON
    except json.JSONDecodeError as e:
        raise ValueError(f"2D geometry JSON not valid: {p}: {e}") from e  # bad JSON

    required = ["width", "height", "inlet", "outlet", "points"]  # required keys
    missing = [k for k in required if k not in obj]  # check presence
    if missing:
        raise KeyError(f"2D geometry JSON missing keys {missing}: {p}")  # schema error

    if not isinstance(obj["points"], list) or len(obj["points"]) == 0:
        raise ValueError(f"2D geometry JSON has empty 'points': {p}")  # empty

    for key in ("inlet", "outlet"):
        d = obj[key]  # port dict
        if not isinstance(d, dict) or "x" not in d or "y" not in d:
            raise ValueError(
                f"2D geometry JSON '{key}' must have x/y: {p}"
            )  # bad port
    return obj  # return parsed dict


# -----------------------------
# Normalization helpers (mirror V4)
# -----------------------------
def _normalize_xy_1d(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max normalise per-axis x/y (mirrors V4 _normalize_xy)."""

    xden = (xmax - xmin) if (xmax > xmin) else 1.0  # x range
    yden = (ymax - ymin) if (ymax > ymin) else 1.0  # y range
    x = (x_raw - xmin) / xden  # normalized x
    y = (y_raw - ymin) / yden  # normalized y
    return x.astype(np.float32), y.astype(np.float32)  # float32 cast


def _denormalize_xy_1d(
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of _normalize_xy_1d (mirrors V4 _denormalize_xy)."""

    xden = (xmax - xmin) if (xmax > xmin) else 1.0  # x range
    yden = (ymax - ymin) if (ymax > ymin) else 1.0  # y range
    xr = x * xden + xmin  # raw x
    yr = y * yden + ymin  # raw y
    return xr, yr  # raw coords


# -----------------------------
# 2D wall-normal estimator (local copy of V4 _estimate_wall_normals)
# -----------------------------
def _estimate_wall_normals_2d(
    xy_points: np.ndarray,
    xy_wall: np.ndarray,
    k_neighbors: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted inverse-distance kNN 2D wall-normal estimation.

    This is a local copy of V4's `_estimate_wall_normals` so we do not
    import from `partner_v4_flow.py` (which runs CUDA init at import time).
    """

    n = int(xy_points.shape[0])  # num points
    if (n <= 0) or (xy_wall.shape[0] <= 0):
        return (
            np.zeros((n, 1), dtype=np.float32),
            np.zeros((n, 1), dtype=np.float32),
        )  # empty
    k = int(max(min(int(k_neighbors), int(xy_wall.shape[0])), 1))  # clamp k
    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")  # knn
    knn.fit(xy_wall.astype(np.float32))  # fit
    dists, idx = knn.kneighbors(xy_points.astype(np.float32))  # neighbors
    wall_neighbors = xy_wall[idx]  # neighbor coords
    vec = xy_points[:, None, :] - wall_neighbors  # inward-ish vectors
    w = 1.0 / np.maximum(dists.astype(np.float32), 1.0e-6)  # inv-dist weights
    nvec = np.sum(vec * w[:, :, None], axis=1)  # weighted avg
    nrm = np.linalg.norm(nvec, axis=1, keepdims=True).astype(np.float32)  # magnitude
    fallback = vec[:, 0, :].astype(np.float32)  # nearest-wall fallback
    nvec = np.where(nrm > 1.0e-8, nvec, fallback)  # fallback if zero
    nrm = np.linalg.norm(nvec, axis=1, keepdims=True).astype(np.float32)  # recompute
    nvec = nvec / np.maximum(nrm, 1.0e-8)  # normalize
    return (
        nvec[:, 0:1].astype(np.float32),
        nvec[:, 1:2].astype(np.float32),
    )  # nx, ny


# -----------------------------
# 3D point cloud builder
# -----------------------------
def build_3d_point_cloud(
    geom_2d: dict,
    z_aspect: float = 0.10,
    z_slices: int = 9,
) -> dict:
    """Extrude the 2D geometry to 3D.

    All returned coordinate arrays are in NORMALIZED units (x, y in [0,1],
    z in [0, z_aspect]). Raw integer-pixel voxel coordinates are stored
    alongside so downstream voxel graphs can use them.

    Parameters
    ----------
    geom_2d : dict
        Parsed 2D JSON (from `load_2d_geometry_json`). Can be a downsampled
        version — the caller may have stripped rows from `points`.
    z_aspect : float
        Z_max / L_x ratio. Default 0.10 per design doc.
    z_slices : int
        Total z-levels including both caps. Must be >= 3. Default 9.

    Returns
    -------
    dict
        Full point-cloud bundle. See module-level docstring for keys.
    """

    if not isinstance(geom_2d, dict):
        raise TypeError("geom_2d must be the parsed 2D JSON dict")  # type guard
    if "points" not in geom_2d or "inlet" not in geom_2d or "outlet" not in geom_2d:
        raise KeyError("geom_2d missing required keys 'points'/'inlet'/'outlet'")  # schema
    if int(z_slices) < 3:
        raise ValueError("z_slices must be >= 3 (2 caps + >=1 interior)")  # guard
    if float(z_aspect) <= 0.0:
        raise ValueError("z_aspect must be > 0")  # guard

    pts = geom_2d["points"]  # raw point list
    inlet = geom_2d["inlet"]  # inlet dict
    outlet = geom_2d["outlet"]  # outlet dict

    # 2D partition into inside (type 2) and wall (type 1); ignore background (0)
    wall_x, wall_y, inside_x, inside_y = [], [], [], []  # class buffers
    all_x, all_y = [], []  # full coords for min/max
    for p in pts:
        xr, yr, typ = float(p[0]), float(p[1]), int(p[2])  # unpack row
        if typ == 0:
            continue  # skip background
        if typ == 1:
            wall_x.append(xr)  # wall x
            wall_y.append(yr)  # wall y
        elif typ == 2:
            inside_x.append(xr)  # inside x
            inside_y.append(yr)  # inside y
        else:
            raise ValueError(f"Unknown 2D class {typ}")  # bad class
        all_x.append(xr)  # contribute to bounds
        all_y.append(yr)  # contribute to bounds

    if len(wall_x) == 0:
        raise ValueError("No 2D wall points (class 1) in geom_2d")  # no wall
    if len(inside_x) == 0:
        raise ValueError("No 2D inside points (class 2) in geom_2d")  # no inside

    # 2D bounding box in raw pixel units
    xmin, xmax = float(min(all_x)), float(max(all_x))  # x bounds
    ymin, ymax = float(min(all_y)), float(max(all_y))  # y bounds
    Lx = max(xmax - xmin, 1.0e-6)  # physical x length (raw pixels)

    # Z bounds in normalized units (z is normalized by Lx so zmax = z_aspect)
    zmin_norm = 0.0  # bottom cap
    zmax_norm = float(z_aspect)  # top cap

    # Raw z-bounds in the same units as raw x/y (so 1 raw z-unit == 1 pixel)
    zmin_raw = 0.0  # raw bottom cap
    zmax_raw = float(z_aspect) * Lx  # raw top cap

    # Normalized 2D coordinates
    wx_raw = np.asarray(wall_x, np.float32).reshape(-1, 1)  # raw wall x
    wy_raw = np.asarray(wall_y, np.float32).reshape(-1, 1)  # raw wall y
    ix_raw = np.asarray(inside_x, np.float32).reshape(-1, 1)  # raw inside x
    iy_raw = np.asarray(inside_y, np.float32).reshape(-1, 1)  # raw inside y

    wx, wy = _normalize_xy_1d(wx_raw, wy_raw, xmin, xmax, ymin, ymax)  # wall norm
    ix, iy = _normalize_xy_1d(ix_raw, iy_raw, xmin, xmax, ymin, ymax)  # inside norm

    # z-levels (normalized) -- uniform on [0, z_aspect]
    Nz = int(z_slices)  # slice count
    z_levels = np.linspace(0.0, float(z_aspect), Nz, dtype=np.float32)  # z grid
    # Interior slices are the Nz-2 middle layers (caps are index 0 and Nz-1)
    interior_slices = z_levels[1:-1].astype(np.float32)  # interior z
    N_interior = int(interior_slices.shape[0])  # interior count
    if N_interior < 1:
        raise ValueError("z_slices too small: need >= 3 for interior layers")  # guard

    # ------- Inside fluid points: inside_2D x interior z-slices -------
    N_in_2d = int(ix.shape[0])  # 2D inside count
    xyz_in = np.empty((N_in_2d * N_interior, 3), dtype=np.float32)  # output buffer
    for s, zn in enumerate(interior_slices):
        s0 = s * N_in_2d  # slice start
        s1 = s0 + N_in_2d  # slice end
        xyz_in[s0:s1, 0] = ix[:, 0]  # x
        xyz_in[s0:s1, 1] = iy[:, 0]  # y
        xyz_in[s0:s1, 2] = float(zn)  # z (constant within slice)

    # ------- Side-wall points: wall_2D x ALL z-slices -------
    N_w_2d = int(wx.shape[0])  # 2D wall count
    xyz_side = np.empty((N_w_2d * Nz, 3), dtype=np.float32)  # output buffer
    for s, zn in enumerate(z_levels):
        s0 = s * N_w_2d  # slice start
        s1 = s0 + N_w_2d  # slice end
        xyz_side[s0:s1, 0] = wx[:, 0]  # x
        xyz_side[s0:s1, 1] = wy[:, 0]  # y
        xyz_side[s0:s1, 2] = float(zn)  # z
    class_side = np.zeros((xyz_side.shape[0],), dtype=np.int8)  # class 0 = side

    # ------- Cap walls: inside_2D at z=0 (bottom) and z=zmax (top) -------
    xyz_bot = np.empty((N_in_2d, 3), dtype=np.float32)  # bottom cap
    xyz_bot[:, 0] = ix[:, 0]  # x
    xyz_bot[:, 1] = iy[:, 0]  # y
    xyz_bot[:, 2] = zmin_norm  # z=0
    class_bot = np.full((N_in_2d,), 1, dtype=np.int8)  # class 1 = bottom cap

    xyz_top = np.empty((N_in_2d, 3), dtype=np.float32)  # top cap
    xyz_top[:, 0] = ix[:, 0]  # x
    xyz_top[:, 1] = iy[:, 0]  # y
    xyz_top[:, 2] = zmax_norm  # z=zmax
    class_top = np.full((N_in_2d,), 2, dtype=np.int8)  # class 2 = top cap

    # Combined wall set: side + bottom cap + top cap
    xyz_wall = np.concatenate([xyz_side, xyz_bot, xyz_top], axis=0).astype(
        np.float32
    )  # wall set
    class_wall = np.concatenate([class_side, class_bot, class_top], axis=0).astype(
        np.int8
    )  # per-point class label

    # ------- Wall normals per design doc --------
    # Side walls: extrude 2D estimator (nx, ny, 0) across all Nz slices
    nx_2d, ny_2d = _estimate_wall_normals_2d(
        xy_points=np.concatenate([wx, wy], axis=1),
        xy_wall=np.concatenate([wx, wy], axis=1),
        k_neighbors=4,
    )  # 2D wall-on-wall normals (self-kNN excludes same-point weight via inv-dist)
    nrm2 = np.sqrt(nx_2d ** 2 + ny_2d ** 2).astype(np.float32)  # in-plane magnitude
    nx_2d = (nx_2d / np.maximum(nrm2, 1.0e-8)).astype(np.float32)  # unit nx
    ny_2d = (ny_2d / np.maximum(nrm2, 1.0e-8)).astype(np.float32)  # unit ny
    n_side = np.empty((xyz_side.shape[0], 3), dtype=np.float32)  # side normals
    for s in range(Nz):
        s0 = s * N_w_2d  # slice start
        s1 = s0 + N_w_2d  # slice end
        n_side[s0:s1, 0] = nx_2d[:, 0]  # nx extruded
        n_side[s0:s1, 1] = ny_2d[:, 0]  # ny extruded
        n_side[s0:s1, 2] = 0.0  # nz = 0

    # Cap normals: exact (0,0,-1) at z=0, (0,0,+1) at z=zmax
    n_bot = np.zeros((N_in_2d, 3), dtype=np.float32)  # bottom cap normals
    n_bot[:, 2] = -1.0  # -z
    n_top = np.zeros((N_in_2d, 3), dtype=np.float32)  # top cap normals
    n_top[:, 2] = +1.0  # +z
    n_wall = np.concatenate([n_side, n_bot, n_top], axis=0).astype(np.float32)  # combined

    # Sanity: any zero-magnitude normals should not have slipped through.
    n_mag = np.linalg.norm(n_wall, axis=1)  # magnitudes
    if not np.all(np.isfinite(n_mag)):
        raise RuntimeError("Non-finite wall normals produced")  # defensive
    if not np.all(n_mag > 0.5):
        # Should be exactly unit or near-unit; warn if any got degenerate.
        raise RuntimeError(
            f"Degenerate wall normals produced (min |n|={float(np.min(n_mag)):.3e})"
        )

    # ------- Inlet and outlet patches -------
    # Build small 2D circular patches around the raw inlet/outlet pixels, then
    # extrude them across the 7 interior z-slices. This is the 3D analogue of
    # the V4 inlet/outlet disk. Radius matches V4's `inlet_radius_norm=0.002`
    # with growth to `max_patch_radius_norm=0.005`, and a minimum point count
    # sufficient to be picked up by the 3D trainer.
    inlet_xy_norm, _ = _normalize_xy_1d(
        np.asarray([[float(inlet["x"])]], np.float32),
        np.asarray([[float(inlet["y"])]], np.float32),
        xmin,
        xmax,
        ymin,
        ymax,
    )  # inlet xn -- (1,1)
    _, inlet_y_norm = _normalize_xy_1d(
        np.asarray([[float(inlet["x"])]], np.float32),
        np.asarray([[float(inlet["y"])]], np.float32),
        xmin,
        xmax,
        ymin,
        ymax,
    )  # inlet yn -- (1,1)
    inlet_center_2d = np.asarray(
        [[float(inlet_xy_norm[0, 0]), float(inlet_y_norm[0, 0])]],
        dtype=np.float32,
    )  # (1,2)

    outlet_xy_norm, _ = _normalize_xy_1d(
        np.asarray([[float(outlet["x"])]], np.float32),
        np.asarray([[float(outlet["y"])]], np.float32),
        xmin,
        xmax,
        ymin,
        ymax,
    )  # outlet xn
    _, outlet_y_norm = _normalize_xy_1d(
        np.asarray([[float(outlet["x"])]], np.float32),
        np.asarray([[float(outlet["y"])]], np.float32),
        xmin,
        xmax,
        ymin,
        ymax,
    )  # outlet yn
    outlet_center_2d = np.asarray(
        [[float(outlet_xy_norm[0, 0]), float(outlet_y_norm[0, 0])]],
        dtype=np.float32,
    )  # (1,2)

    xy_inside_2d = np.concatenate([ix, iy], axis=1).astype(np.float32)  # (N_in_2d,2)

    def _select_patch_2d(center_xy: np.ndarray, r0: float, r_max: float, grow: float,
                         min_pts: int) -> np.ndarray:
        """Return a boolean mask over 2D inside points for a circular patch.

        Grow radius until it contains at least `min_pts` points (or r_max).
        Falls back to k-nearest if radius growth fails."""
        r = float(max(r0, 1.0e-6))  # start radius
        d2 = np.sum((xy_inside_2d - center_xy[0:1]) ** 2, axis=1)  # sq distance
        mask = d2 <= (r * r)  # initial mask
        while int(np.sum(mask)) < int(min_pts) and r < r_max:
            r = min(r * grow, r_max)  # grow radius
            mask = d2 <= (r * r)  # recompute
        if int(np.sum(mask)) == 0:
            k = min(max(min_pts, 1), xy_inside_2d.shape[0])  # fallback k
            idx = np.argsort(d2)[:k]  # nearest-k
            mask = np.zeros((xy_inside_2d.shape[0],), dtype=bool)  # empty
            mask[idx] = True  # activate nearest
        return mask  # (N_in_2d,) bool

    inlet_mask_2d = _select_patch_2d(
        inlet_center_2d, r0=0.002, r_max=0.005, grow=1.5, min_pts=8
    )  # inlet patch mask in 2D
    outlet_mask_2d = _select_patch_2d(
        outlet_center_2d, r0=0.002, r_max=0.005, grow=1.5, min_pts=8
    )  # outlet patch mask in 2D

    def _extrude_patch(mask_2d: np.ndarray) -> np.ndarray:
        """Extrude a 2D inside-point mask across interior z-slices."""
        idx2d = np.where(mask_2d)[0]  # 2D point indices
        if idx2d.size == 0:
            return np.zeros((0, 3), dtype=np.float32)  # empty
        xys = xy_inside_2d[idx2d]  # (m,2)
        out = np.empty((xys.shape[0] * N_interior, 3), dtype=np.float32)  # buffer
        for s, zn in enumerate(interior_slices):
            s0 = s * xys.shape[0]  # slice start
            s1 = s0 + xys.shape[0]  # slice end
            out[s0:s1, 0] = xys[:, 0]  # x
            out[s0:s1, 1] = xys[:, 1]  # y
            out[s0:s1, 2] = float(zn)  # z
        return out  # (m*Nint, 3)

    xyz_inlet = _extrude_patch(inlet_mask_2d)  # 3D inlet patch
    xyz_outlet = _extrude_patch(outlet_mask_2d)  # 3D outlet patch

    # Inlet/outlet centers: 2D center xy + average interior z (approx half of zmax)
    mid_z = float(np.mean(interior_slices))  # interior midpoint
    inlet_center_xyz = np.asarray(
        [[float(inlet_center_2d[0, 0]), float(inlet_center_2d[0, 1]), mid_z]],
        dtype=np.float32,
    )  # (1,3)
    outlet_center_xyz = np.asarray(
        [[float(outlet_center_2d[0, 0]), float(outlet_center_2d[0, 1]), mid_z]],
        dtype=np.float32,
    )  # (1,3)

    # Build raw-voxel coordinates for the inside set (used by voxel graph).
    # We round the raw 2D (x,y) to integer pixels, then assign each interior
    # z-slice an integer index 1..Nz-2 (bottom cap would be 0, top cap Nz-1,
    # matching the z_levels indexing).
    xyz_in_raw = np.empty((xyz_in.shape[0], 3), dtype=np.float32)  # raw inside
    for s in range(N_interior):
        s0 = s * N_in_2d  # slice start
        s1 = s0 + N_in_2d  # slice end
        xyz_in_raw[s0:s1, 0] = np.rint(ix_raw[:, 0]).astype(np.float32)  # raw x
        xyz_in_raw[s0:s1, 1] = np.rint(iy_raw[:, 0]).astype(np.float32)  # raw y
        xyz_in_raw[s0:s1, 2] = float(s + 1)  # raw z index 1..N_interior

    # Raw wall voxel coords (integer pixels + z-index)
    xyz_wall_raw = np.empty((xyz_wall.shape[0], 3), dtype=np.float32)  # raw wall
    # side walls
    for s in range(Nz):
        s0 = s * N_w_2d  # slice start
        s1 = s0 + N_w_2d  # slice end
        xyz_wall_raw[s0:s1, 0] = np.rint(wx_raw[:, 0]).astype(np.float32)  # x
        xyz_wall_raw[s0:s1, 1] = np.rint(wy_raw[:, 0]).astype(np.float32)  # y
        xyz_wall_raw[s0:s1, 2] = float(s)  # z index 0..Nz-1
    # bottom cap
    off = N_w_2d * Nz  # offset after side
    xyz_wall_raw[off:off + N_in_2d, 0] = np.rint(ix_raw[:, 0]).astype(np.float32)
    xyz_wall_raw[off:off + N_in_2d, 1] = np.rint(iy_raw[:, 0]).astype(np.float32)
    xyz_wall_raw[off:off + N_in_2d, 2] = 0.0
    # top cap
    off2 = off + N_in_2d  # offset after bottom cap
    xyz_wall_raw[off2:off2 + N_in_2d, 0] = np.rint(ix_raw[:, 0]).astype(np.float32)
    xyz_wall_raw[off2:off2 + N_in_2d, 1] = np.rint(iy_raw[:, 0]).astype(np.float32)
    xyz_wall_raw[off2:off2 + N_in_2d, 2] = float(Nz - 1)

    # Bundle results -- all float32, all NORMALIZED coordinates where noted.
    out = {
        "xyz_inside": xyz_in.astype(np.float32),
        "xyz_wall": xyz_wall.astype(np.float32),
        "xyz_inlet": xyz_inlet.astype(np.float32),
        "xyz_outlet": xyz_outlet.astype(np.float32),
        "n_wall": n_wall.astype(np.float32),
        "class_wall": class_wall.astype(np.int8),
        "norm": (xmin, xmax, ymin, ymax, zmin_raw, zmax_raw),
        "z_aspect": float(z_aspect),
        "z_slices": int(Nz),
        "inlet_center_xyz": inlet_center_xyz.astype(np.float32),
        "outlet_center_xyz": outlet_center_xyz.astype(np.float32),
        "inlet_raw": {"x": float(inlet["x"]), "y": float(inlet["y"])},
        "outlet_raw": {"x": float(outlet["x"]), "y": float(outlet["y"])},
        "raw_2d_obj": geom_2d,
        # Private aux (not in public contract but needed by downstream callers)
        "xyz_inside_raw": xyz_in_raw.astype(np.float32),
        "xyz_wall_raw": xyz_wall_raw.astype(np.float32),
        "N_in_2d": int(N_in_2d),
        "N_w_2d": int(N_w_2d),
        "N_interior_slices": int(N_interior),
    }  # bundle
    return out  # point cloud dict


# -----------------------------
# Wall distance (3D)
# -----------------------------
def compute_wall_distance_3d(xyz_query: np.ndarray, xyz_wall: np.ndarray) -> np.ndarray:
    """3D Euclidean nearest-wall distance via cKDTree.

    Parameters
    ----------
    xyz_query : (N, 3) float
        Query points (normalised coordinates).
    xyz_wall : (M, 3) float
        Wall points (normalised coordinates).

    Returns
    -------
    (N, 1) float32 array.
    """

    if xyz_query.ndim != 2 or xyz_query.shape[1] != 3:
        raise ValueError("xyz_query must have shape (N,3)")  # guard
    if xyz_wall.ndim != 2 or xyz_wall.shape[1] != 3:
        raise ValueError("xyz_wall must have shape (M,3)")  # guard
    if xyz_wall.shape[0] == 0:
        return np.zeros((xyz_query.shape[0], 1), dtype=np.float32)  # no walls

    tree = cKDTree(xyz_wall.astype(np.float64))  # kd-tree in 3D
    d, _ = tree.query(xyz_query.astype(np.float64), k=1)  # nearest wall
    return d.astype(np.float32).reshape(-1, 1)  # (N,1)


# -----------------------------
# Inside graph (voxel 6-connectivity or kNN fallback)
# -----------------------------
def _build_voxel_graph_3d(
    xyz_inside: np.ndarray,
    raw_xyz_inside: np.ndarray,
) -> csr_matrix:
    """6-connectivity voxel graph over integer raw voxel coordinates.

    Edge weights are Euclidean distance in the *normalised* space. Any
    near-zero distance is clamped to `_SMALL`.
    """

    n = int(xyz_inside.shape[0])  # point count
    if n == 0:
        return csr_matrix((0, 0), dtype=np.float32)  # empty

    x_raw = np.rint(raw_xyz_inside[:, 0]).astype(np.int64)  # raw int x
    y_raw = np.rint(raw_xyz_inside[:, 1]).astype(np.int64)  # raw int y
    z_raw = np.rint(raw_xyz_inside[:, 2]).astype(np.int64)  # raw int z

    lookup = {}  # (x,y,z) -> index
    for i in range(n):
        key = (int(x_raw[i]), int(y_raw[i]), int(z_raw[i]))  # voxel key
        # In a well-formed cloud no two points should share a voxel. If they
        # do (e.g. downsampled lattice collision), keep the first occurrence.
        if key not in lookup:
            lookup[key] = i  # register

    # forward-half 6-connectivity offsets; we add reverse edges explicitly.
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 6-neighbor forward half

    rows, cols, vals = [], [], []  # buffers
    for i in range(n):
        xi, yi, zi = int(x_raw[i]), int(y_raw[i]), int(z_raw[i])  # voxel
        for dx, dy, dz in offsets:
            j = lookup.get((xi + dx, yi + dy, zi + dz), None)  # neighbor
            if j is None:
                continue  # no neighbor in this direction
            # Euclidean distance in NORMALIZED coords (for Dijkstra weights)
            diff = xyz_inside[i] - xyz_inside[j]  # vec
            w = float(np.sqrt(float(np.sum(diff * diff))))  # distance
            if not np.isfinite(w) or w < _SMALL:
                w = _SMALL  # clamp positive
            rows.append(i)  # forward
            cols.append(j)  # forward
            vals.append(w)  # weight
            rows.append(j)  # reverse
            cols.append(i)  # reverse
            vals.append(w)  # weight

    graph = csr_matrix(
        (vals, (rows, cols)),
        shape=(n, n),
        dtype=np.float32,
    )  # sparse adjacency
    return graph  # csr


def _build_knn_graph_3d(
    xyz_inside: np.ndarray,
    k: int = 14,
    max_edge_len: float = 0.02,
) -> csr_matrix:
    """kNN fallback graph in 3D.

    Prunes edges longer than `max_edge_len` (in normalised distance). All
    weights clamped to at least `_SMALL`.
    """

    n = int(xyz_inside.shape[0])  # point count
    if n == 0:
        return csr_matrix((0, 0), dtype=np.float32)  # empty

    k_eff = int(min(int(k) + 1, n))  # include self
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree")  # knn
    nbrs.fit(xyz_inside.astype(np.float32))  # fit
    dists, inds = nbrs.kneighbors(xyz_inside.astype(np.float32))  # neighbors

    rows, cols, vals = [], [], []  # buffers
    for i in range(n):
        for jpos in range(1, k_eff):
            j = int(inds[i, jpos])  # neighbor
            dij = float(dists[i, jpos])  # distance
            if max_edge_len is not None and dij > float(max_edge_len):
                continue  # prune long
            if not np.isfinite(dij) or dij < _SMALL:
                dij = _SMALL  # clamp positive
            rows.append(i)  # forward
            cols.append(j)  # forward
            vals.append(dij)  # weight
            rows.append(j)  # reverse
            cols.append(i)  # reverse
            vals.append(dij)  # weight

    graph = csr_matrix(
        (vals, (rows, cols)),
        shape=(n, n),
        dtype=np.float32,
    )  # sparse adjacency
    return graph  # csr


def build_inside_graph_3d(
    xyz_inside: np.ndarray,
    raw_xyz_inside: np.ndarray,
    mode: str = "voxel",
    knn_k: int = 14,
    max_edge_len: float = 0.02,
) -> csr_matrix:
    """Build sparse adjacency for shortest-path over inside points.

    Parameters
    ----------
    xyz_inside : (N, 3) float
        Normalized interior coordinates.
    raw_xyz_inside : (N, 3) float
        Integer-voxel coordinates aligned with `xyz_inside` (rounded).
        Used to detect 6-connectivity adjacency in voxel mode.
    mode : str
        "voxel" (default) or "knn". "voxel" uses 6-connectivity; "knn"
        uses NearestNeighbors with k=knn_k and prunes edges > max_edge_len.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric adjacency with positive weights.
    """

    mode_l = str(mode).strip().lower()  # normalize
    if mode_l == "voxel":
        return _build_voxel_graph_3d(xyz_inside, raw_xyz_inside)  # 6-conn
    if mode_l == "knn":
        return _build_knn_graph_3d(
            xyz_inside,
            k=int(knn_k),
            max_edge_len=float(max_edge_len),
        )  # knn fallback
    raise ValueError(f"Unknown graph mode: {mode}")  # invalid


# -----------------------------
# Geodesic info (3D)
# -----------------------------
def compute_geodesic_info_3d(
    adj,
    xyz_inside: np.ndarray,
    target_xyz: np.ndarray,
) -> dict:
    """Dijkstra over `adj` from the inside point closest to `target_xyz`.

    Returns
    -------
    dict with keys 'geo' (N,1), 's_geo' (N,1), 'src' int, 'predecessors' (N,) int64.
    Unreachable points get their Euclidean distance to the source as a
    fallback, and predecessor = -9999.
    """

    if xyz_inside.ndim != 2 or xyz_inside.shape[1] != 3:
        raise ValueError("xyz_inside must have shape (N,3)")  # guard
    if target_xyz.ndim != 2 or target_xyz.shape[0] != 1 or target_xyz.shape[1] != 3:
        raise ValueError("target_xyz must have shape (1,3)")  # guard

    d2 = np.sum((xyz_inside - target_xyz[0:1]) ** 2, axis=1)  # sq distance
    src = int(np.argmin(d2))  # nearest inside index
    geo, predecessors = shortest_path(
        adj,
        directed=False,
        indices=src,
        method="D",
        return_predecessors=True,
    )  # Dijkstra

    geo = np.asarray(geo, dtype=np.float32)  # cast
    predecessors = np.asarray(predecessors, dtype=np.int64)  # cast

    bad = np.isinf(geo)  # unreachable
    if np.any(bad):
        geo[bad] = np.sqrt(d2[bad]).astype(np.float32)  # fallback Euclidean
        predecessors[bad] = -9999  # sentinel
    geo = geo.reshape(-1, 1)  # (N,1)
    gmin = float(np.min(geo))  # min
    gmax = float(np.max(geo))  # max
    s_geo = (geo - gmin) / max(gmax - gmin, _SMALL)  # normalized progress

    return {
        "src": int(src),
        "geo": geo.astype(np.float32),
        "s_geo": s_geo.astype(np.float32),
        "predecessors": predecessors.astype(np.int64),
    }  # geodesic bundle


# -----------------------------
# 3D wall-normal estimator (for interior-band points)
# -----------------------------
def estimate_wall_normals_3d(
    xyz_points: np.ndarray,
    xyz_wall: np.ndarray,
    k_neighbors: int = 6,
) -> np.ndarray:
    """Weighted inverse-distance kNN 3D wall-normal estimation.

    Only for INTERIOR band points (wall_guard). For points already in
    `xyz_wall`, use the exact normals returned by `build_3d_point_cloud`.

    Returns
    -------
    (N, 3) float32 unit vectors.
    """

    n = int(xyz_points.shape[0])  # num query
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)  # empty
    if xyz_wall.ndim != 2 or xyz_wall.shape[1] != 3:
        raise ValueError("xyz_wall must have shape (M,3)")  # guard
    if xyz_wall.shape[0] <= 0:
        return np.zeros((n, 3), dtype=np.float32)  # no walls

    k = int(max(min(int(k_neighbors), int(xyz_wall.shape[0])), 1))  # clamp k
    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")  # 3D knn
    knn.fit(xyz_wall.astype(np.float32))  # fit
    dists, idx = knn.kneighbors(xyz_points.astype(np.float32))  # neighbors

    wall_neighbors = xyz_wall[idx]  # (N,k,3)
    vec = xyz_points[:, None, :] - wall_neighbors  # inward-ish
    w = 1.0 / np.maximum(dists.astype(np.float32), 1.0e-6)  # inv-dist weights
    nvec = np.sum(vec * w[:, :, None], axis=1)  # weighted average
    nrm = np.linalg.norm(nvec, axis=1, keepdims=True).astype(np.float32)  # magnitude
    fallback = vec[:, 0, :].astype(np.float32)  # nearest-wall fallback
    nvec = np.where(nrm > 1.0e-8, nvec, fallback)  # fallback if zero
    nrm = np.linalg.norm(nvec, axis=1, keepdims=True).astype(np.float32)  # re-magnitude
    nvec = nvec / np.maximum(nrm, 1.0e-8)  # normalize
    return nvec.astype(np.float32)  # (N,3)


# -----------------------------
# Tangent estimation from predecessor tree (3D analogue)
# -----------------------------
def _estimate_tangent_from_predecessor_tree_3d(
    xyz_inside: np.ndarray,
    predecessors: np.ndarray,
    src: int,
) -> np.ndarray:
    """Estimate a unit 3D tangent per point from Dijkstra predecessor tree.

    At the source, use the direction to one child. Elsewhere use
    (point - predecessor). Fallback to +x if degenerate.
    Mirrors V4 `estimate_tangent_from_predecessor_tree` with an added axis.
    """

    n = int(xyz_inside.shape[0])  # num points
    tangent = np.zeros((n, 3), dtype=np.float32)  # output

    children = [[] for _ in range(n)]  # reverse tree
    for i in range(n):
        p = int(predecessors[i])  # parent
        if p >= 0:
            children[p].append(i)  # child

    fallback_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # +x fallback
    for i in range(n):
        p = int(predecessors[i])  # parent
        if i == src:
            if len(children[i]) > 0:
                j = int(children[i][0])  # first child
                vec = xyz_inside[j] - xyz_inside[i]  # forward
            else:
                vec = fallback_vec.copy()  # lone source
        else:
            if p >= 0:
                vec = xyz_inside[i] - xyz_inside[p]  # along path
            else:
                vec = fallback_vec.copy()  # disconnected
        m = float(np.linalg.norm(vec))  # magnitude
        if m > 1.0e-12:
            tangent[i] = vec / m  # normalize
        else:
            tangent[i] = fallback_vec  # fallback
    return tangent.astype(np.float32)  # (N,3)


# -----------------------------
# Initial flow guess (3D)
# -----------------------------
def compute_initial_flow_guess_3d(
    xyz_inside: np.ndarray,
    xyz_wall: np.ndarray,
    z_max: float,
    inlet_u: float,
    inlet_v: float,
    inlet_p: float,
    geo_info_in: dict,
    geo_info_out: dict,
    adj,
    velocity_scale: float = 1.0,
    velocity_power: float = 1.0,
    pressure_power: float = 1.0,
    pressure_drop_guess: float = 0.0,
) -> dict:
    """Build 3D initial guess for (u, v, w, p) and auxiliary features.

    w0 is set to 0 (no bulk z-flow in a channel). u0/v0 use the in-plane
    geodesic tangent (xy components), scaled by wall profile, axial decay,
    and a parabolic z-profile phi(z) = 4 (z/Zmax)(1 - z/Zmax). Pressure
    anchored from inlet, same blending rule as V4.

    `adj` is the csr adjacency used only for reference; tangent uses the
    predecessor tree already stored in `geo_info_in`.
    """

    if xyz_inside.ndim != 2 or xyz_inside.shape[1] != 3:
        raise ValueError("xyz_inside must have shape (N,3)")  # guard
    if xyz_wall.ndim != 2 or xyz_wall.shape[1] != 3:
        raise ValueError("xyz_wall must have shape (M,3)")  # guard

    geo_in = geo_info_in["geo"]  # (N,1) raw inlet geodesic
    s_in = geo_info_in["s_geo"]  # (N,1) normalized inlet progress
    s_out = geo_info_out["s_geo"]  # (N,1) normalized outlet progress
    src = int(geo_info_in["src"])  # inlet source
    predecessors = geo_info_in["predecessors"]  # predecessor tree

    tangent = _estimate_tangent_from_predecessor_tree_3d(
        xyz_inside=xyz_inside,
        predecessors=predecessors,
        src=src,
    )  # (N,3) unit tangent -- includes z component

    d_wall = compute_wall_distance_3d(xyz_inside, xyz_wall)  # nearest wall distance
    dmax = float(np.max(d_wall)) if d_wall.size > 0 else 1.0  # max
    dnorm = d_wall / max(dmax, 1.0e-8)  # normalize to [0,1]

    # Parabolic z-profile phi(z) = 4 (z/zmax) (1 - z/zmax)
    zmax = float(max(z_max, 1.0e-8))  # guard
    z_here = xyz_inside[:, 2:3].astype(np.float32)  # (N,1)
    phi = 4.0 * (z_here / zmax) * (1.0 - z_here / zmax)  # (N,1) peaks at z=zmax/2
    phi = np.clip(phi, 0.0, 1.0).astype(np.float32)  # safety clamp

    inlet_speed = float(np.sqrt(float(inlet_u) ** 2 + float(inlet_v) ** 2))  # |U_in|
    if inlet_speed < 1.0e-12:
        inlet_speed = 1.0  # fallback if bc is zero

    wall_profile = np.clip(dnorm, 0.0, 1.0) ** float(velocity_power)  # slower near wall
    axial_decay = (1.0 - 0.10 * s_in).astype(np.float32)  # mild downstream decay
    speed = (
        float(velocity_scale)
        * inlet_speed
        * wall_profile
        * axial_decay
        * phi
    )  # scalar speed field (N,1)

    # Use ONLY the in-plane tangent components for u/v; w0 is zero per design.
    t_xy = tangent[:, 0:2]  # (N,2) xy components
    t_xy_norm = np.linalg.norm(t_xy, axis=1, keepdims=True).astype(np.float32)  # magnitude
    t_xy_unit = t_xy / np.maximum(t_xy_norm, 1.0e-8)  # renormalize in-plane
    u0 = (speed * t_xy_unit[:, 0:1]).astype(np.float32)  # (N,1)
    v0 = (speed * t_xy_unit[:, 1:2]).astype(np.float32)  # (N,1)
    w0 = np.zeros_like(u0, dtype=np.float32)  # no bulk z-flow

    # Pressure: blend from inlet anchor -- same formula as V4
    blend = np.clip(s_in / np.maximum(s_in + s_out, 1.0e-8), 0.0, 1.0)  # (N,1)
    p0 = (
        float(inlet_p)
        - float(pressure_drop_guess) * (blend ** float(pressure_power))
    ).astype(np.float32)  # (N,1)

    return {
        "u": u0.astype(np.float32),
        "v": v0.astype(np.float32),
        "w": w0.astype(np.float32),
        "p": p0.astype(np.float32),
        "dw": d_wall.astype(np.float32),
        "geo_in": geo_in.astype(np.float32),
        "s_in": s_in.astype(np.float32),
        "s_out": s_out.astype(np.float32),
        "tangent": tangent.astype(np.float32),
        "predecessors": predecessors.astype(np.int64),
        "src": int(src),
    }  # initial fields bundle


# -----------------------------
# JSON writer
# -----------------------------
def save_geometry_json_3d(
    path,
    point_cloud: dict,
    init_fields_inside: Optional[dict] = None,
):
    """Write the 3D geometry JSON.

    Keys match the spec:
      z_aspect, z_slices, norm,
      points_inside (N,6) [xn,yn,zn,xr,yr,zr],
      points_wall   (N,7) [xn,yn,zn,xr,yr,zr,class],
      points_inlet  (...,3) [xn,yn,zn],
      points_outlet (...,3),
      normals_wall  (N,3),
      inlet/outlet  (pass-through raw dicts),
      inlet_center_xyz / outlet_center_xyz,
      init_fields   (same schema as V4 but 3D; only if init_fields_inside
                     is provided).
    """

    p = Path(path)  # path object
    p.parent.mkdir(parents=True, exist_ok=True)  # ensure dir exists

    # Basic inputs
    xyz_in = np.asarray(point_cloud["xyz_inside"], dtype=np.float32)  # (N_in,3)
    xyz_in_raw = np.asarray(
        point_cloud.get(
            "xyz_inside_raw",
            np.zeros_like(xyz_in),
        ),
        dtype=np.float32,
    )  # (N_in,3) raw voxel

    xyz_w = np.asarray(point_cloud["xyz_wall"], dtype=np.float32)  # (N_w,3)
    xyz_w_raw = np.asarray(
        point_cloud.get(
            "xyz_wall_raw",
            np.zeros_like(xyz_w),
        ),
        dtype=np.float32,
    )  # (N_w,3) raw voxel
    class_w = np.asarray(point_cloud["class_wall"], dtype=np.int32).reshape(-1, 1)  # (N_w,1)
    n_w = np.asarray(point_cloud["n_wall"], dtype=np.float32)  # (N_w,3)

    xyz_il = np.asarray(point_cloud["xyz_inlet"], dtype=np.float32)  # (N_il,3)
    xyz_ol = np.asarray(point_cloud["xyz_outlet"], dtype=np.float32)  # (N_ol,3)

    # Compose per-point arrays
    points_inside = np.concatenate(
        [xyz_in, xyz_in_raw], axis=1
    ).astype(np.float32)  # (N_in,6)
    points_wall = np.concatenate(
        [xyz_w, xyz_w_raw, class_w.astype(np.float32)], axis=1
    ).astype(np.float32)  # (N_w,7)

    payload = {
        "z_aspect": float(point_cloud["z_aspect"]),
        "z_slices": int(point_cloud["z_slices"]),
        "norm": [float(v) for v in point_cloud["norm"]],
        "inlet": {
            "x": float(point_cloud["inlet_raw"]["x"]),
            "y": float(point_cloud["inlet_raw"]["y"]),
        },
        "outlet": {
            "x": float(point_cloud["outlet_raw"]["x"]),
            "y": float(point_cloud["outlet_raw"]["y"]),
        },
        "inlet_center_xyz": xyz_as_list(point_cloud["inlet_center_xyz"]),
        "outlet_center_xyz": xyz_as_list(point_cloud["outlet_center_xyz"]),
        "points_inside": points_inside.tolist(),
        "points_wall": points_wall.tolist(),
        "points_inlet": xyz_il.tolist(),
        "points_outlet": xyz_ol.tolist(),
        "normals_wall": n_w.tolist(),
    }  # base payload

    if init_fields_inside is not None:
        init_payload = {
            "xyz": xyz_in.tolist(),  # normalized interior coords
            "xyz_raw": xyz_in_raw.tolist(),  # raw voxel coords
            "fields": {
                "u": np.asarray(init_fields_inside["u"], np.float32).reshape(-1).tolist(),
                "v": np.asarray(init_fields_inside["v"], np.float32).reshape(-1).tolist(),
                "w": np.asarray(init_fields_inside["w"], np.float32).reshape(-1).tolist(),
                "p": np.asarray(init_fields_inside["p"], np.float32).reshape(-1).tolist(),
                "dw": np.asarray(init_fields_inside["dw"], np.float32).reshape(-1).tolist(),
                "geo_in": np.asarray(init_fields_inside["geo_in"], np.float32).reshape(-1).tolist(),
                "s_in": np.asarray(init_fields_inside["s_in"], np.float32).reshape(-1).tolist(),
                "s_out": np.asarray(init_fields_inside["s_out"], np.float32).reshape(-1).tolist(),
            },
        }  # 3D init-fields block
        payload["init_fields"] = init_payload  # attach

    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)  # compact JSON (no indent) to keep file size down


def xyz_as_list(arr):
    """Convert (1,3) or (3,) array to a Python list of 3 floats."""
    a = np.asarray(arr, dtype=np.float32).reshape(-1)  # flatten
    return [float(a[0]), float(a[1]), float(a[2])]  # three floats


# -----------------------------
# Self-test
# -----------------------------
def _self_test() -> None:
    """Lightweight end-to-end smoke test.

    - Loads the 2D JSON (path is hard-wired for the V4 geometry).
    - Downsamples the 2D `points` list by 8x for speed.
    - Builds a 3D cloud with z_slices=5.
    - Exercises every public API.
    - Asserts basic invariants (shapes, finiteness, normals unit to 1e-5,
      geodesic coverage >= 99%, initial-guess shapes, JSON round-trip).
    """

    repo_root = Path(__file__).resolve().parent.parent  # project root
    src_2d = repo_root / "data" / "partner_v4" / "pipe_three_class_fixed.json"  # input JSON
    if not src_2d.exists():
        raise FileNotFoundError(f"2D JSON missing for self-test: {src_2d}")  # guard

    print(f"[self-test] loading {src_2d}")  # progress
    obj_2d = load_2d_geometry_json(str(src_2d))  # parse

    # Downsample 2D points by 8x: keep every 8th point.
    ds = 8  # downsample factor
    orig_n = len(obj_2d["points"])  # original count
    obj_ds = dict(obj_2d)  # shallow copy
    obj_ds["points"] = obj_2d["points"][::ds]  # slice
    print(
        f"[self-test] downsampled 2D points: {orig_n} -> {len(obj_ds['points'])}"
    )

    # Build cloud at z_slices=5 (2 caps + 3 interior).
    print("[self-test] building 3D point cloud (z_slices=5, aspect=0.10)")
    pc = build_3d_point_cloud(obj_ds, z_aspect=0.10, z_slices=5)  # build

    # Basic shape checks
    assert pc["xyz_inside"].ndim == 2 and pc["xyz_inside"].shape[1] == 3
    assert pc["xyz_wall"].ndim == 2 and pc["xyz_wall"].shape[1] == 3
    assert pc["n_wall"].shape == pc["xyz_wall"].shape
    assert pc["class_wall"].shape[0] == pc["xyz_wall"].shape[0]
    assert pc["xyz_inside"].dtype == np.float32
    assert pc["xyz_wall"].dtype == np.float32
    assert pc["n_wall"].dtype == np.float32
    assert pc["class_wall"].dtype == np.int8
    assert len(pc["norm"]) == 6
    assert float(pc["z_aspect"]) == 0.10
    assert int(pc["z_slices"]) == 5

    # Check normalized bounds are in [0,1] for xy, and [0, z_aspect] for z.
    for xyz in (pc["xyz_inside"], pc["xyz_wall"], pc["xyz_inlet"], pc["xyz_outlet"]):
        assert np.all(np.isfinite(xyz))  # finite
        assert float(np.min(xyz[:, 0])) >= -1.0e-6
        assert float(np.max(xyz[:, 0])) <= 1.0 + 1.0e-6
        assert float(np.min(xyz[:, 1])) >= -1.0e-6
        assert float(np.max(xyz[:, 1])) <= 1.0 + 1.0e-6
        assert float(np.min(xyz[:, 2])) >= -1.0e-6
        assert float(np.max(xyz[:, 2])) <= 0.10 + 1.0e-6

    # Wall-normal magnitudes ~= 1 within 1e-5
    n_mag = np.linalg.norm(pc["n_wall"], axis=1)  # magnitudes
    assert np.max(np.abs(n_mag - 1.0)) < 1.0e-5, (
        f"wall normals not unit: max|n|-1|={np.max(np.abs(n_mag - 1.0)):.3e}"
    )

    # Cap normals exact
    side_mask = pc["class_wall"] == 0  # side
    bot_mask = pc["class_wall"] == 1  # bottom cap
    top_mask = pc["class_wall"] == 2  # top cap
    if np.any(bot_mask):
        assert np.allclose(pc["n_wall"][bot_mask, 2], -1.0, atol=1.0e-6)
    if np.any(top_mask):
        assert np.allclose(pc["n_wall"][top_mask, 2], +1.0, atol=1.0e-6)
    if np.any(side_mask):
        # Side nz must be exactly zero by construction
        assert np.max(np.abs(pc["n_wall"][side_mask, 2])) < 1.0e-6

    # compute_wall_distance_3d
    dw = compute_wall_distance_3d(pc["xyz_inside"], pc["xyz_wall"])  # (N_in,1)
    assert dw.shape[0] == pc["xyz_inside"].shape[0]
    assert dw.dtype == np.float32
    assert np.all(np.isfinite(dw))
    assert float(np.min(dw)) >= 0.0

    # build_inside_graph_3d (voxel) + geodesic
    adj = build_inside_graph_3d(
        pc["xyz_inside"],
        pc["xyz_inside_raw"],
        mode="voxel",
    )  # voxel adjacency
    assert adj.shape[0] == pc["xyz_inside"].shape[0]
    assert adj.shape[1] == pc["xyz_inside"].shape[0]
    assert adj.nnz > 0

    geo_info_in = compute_geodesic_info_3d(
        adj, pc["xyz_inside"], pc["inlet_center_xyz"]
    )  # inlet geodesic
    geo_info_out = compute_geodesic_info_3d(
        adj, pc["xyz_inside"], pc["outlet_center_xyz"]
    )  # outlet geodesic
    n_reached = int(np.sum(geo_info_in["predecessors"] != -9999))  # reached count
    coverage = float(n_reached) / float(pc["xyz_inside"].shape[0])  # coverage ratio

    if coverage < 0.99:
        print(
            f"[self-test] voxel coverage {coverage:.3f} < 0.99, retrying with kNN"
        )
        adj = build_inside_graph_3d(
            pc["xyz_inside"],
            pc["xyz_inside_raw"],
            mode="knn",
            knn_k=14,
            max_edge_len=0.05,
        )  # knn fallback
        geo_info_in = compute_geodesic_info_3d(
            adj, pc["xyz_inside"], pc["inlet_center_xyz"]
        )
        geo_info_out = compute_geodesic_info_3d(
            adj, pc["xyz_inside"], pc["outlet_center_xyz"]
        )
        n_reached = int(np.sum(geo_info_in["predecessors"] != -9999))
        coverage = float(n_reached) / float(pc["xyz_inside"].shape[0])
    assert coverage >= 0.99, f"geodesic coverage too low: {coverage:.3f}"

    # estimate_wall_normals_3d on interior points -- should be unit vectors
    n_i = estimate_wall_normals_3d(pc["xyz_inside"], pc["xyz_wall"], k_neighbors=6)
    assert n_i.shape == (pc["xyz_inside"].shape[0], 3)
    assert n_i.dtype == np.float32
    n_i_mag = np.linalg.norm(n_i, axis=1)  # magnitudes
    assert np.max(np.abs(n_i_mag - 1.0)) < 1.0e-5, (
        f"interior kNN normals not unit: max|n|-1|={np.max(np.abs(n_i_mag - 1.0)):.3e}"
    )

    # compute_initial_flow_guess_3d
    init_fields = compute_initial_flow_guess_3d(
        xyz_inside=pc["xyz_inside"],
        xyz_wall=pc["xyz_wall"],
        z_max=float(pc["z_aspect"]),
        inlet_u=1.0,
        inlet_v=0.0,
        inlet_p=1.0,
        geo_info_in=geo_info_in,
        geo_info_out=geo_info_out,
        adj=adj,
        velocity_scale=1.0,
        velocity_power=1.0,
        pressure_power=1.0,
        pressure_drop_guess=0.0,
    )  # init guess

    N_in = pc["xyz_inside"].shape[0]  # interior count
    for key in ("u", "v", "w", "p", "dw", "geo_in", "s_in", "s_out"):
        arr = init_fields[key]  # fetch
        assert arr.shape == (N_in, 1), (
            f"init_fields[{key}] has shape {arr.shape}, expected ({N_in},1)"
        )
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr)), f"init_fields[{key}] contains non-finite"
    assert init_fields["tangent"].shape == (N_in, 3)
    assert np.all(np.isfinite(init_fields["tangent"]))
    # Tangent should be unit length (barring rare fallback to +x which is unit)
    t_mag = np.linalg.norm(init_fields["tangent"], axis=1)
    assert np.max(np.abs(t_mag - 1.0)) < 1.0e-5

    # JSON round-trip to a temp path
    import tempfile  # local import

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False
    ) as tf:
        tmp_json_path = tf.name  # temp path
    try:
        save_geometry_json_3d(
            tmp_json_path,
            pc,
            init_fields_inside=init_fields,
        )  # write
        size = Path(tmp_json_path).stat().st_size  # file size
        # Reload and sanity check
        with open(tmp_json_path, "r", encoding="utf-8") as f:
            rt = json.load(f)  # re-read
        assert "points_inside" in rt
        assert "points_wall" in rt
        assert "normals_wall" in rt
        assert "init_fields" in rt
        assert len(rt["points_inside"]) == N_in
        assert len(rt["points_wall"]) == pc["xyz_wall"].shape[0]
        assert len(rt["normals_wall"]) == pc["xyz_wall"].shape[0]
        assert len(rt["init_fields"]["fields"]["u"]) == N_in
        print(f"[self-test] round-trip JSON size: {size / 1024.0:.1f} KB")
    finally:
        try:
            Path(tmp_json_path).unlink()  # cleanup
        except OSError:
            pass

    print(
        f"[self-test] interior={N_in}, wall={pc['xyz_wall'].shape[0]}, "
        f"inlet_patch={pc['xyz_inlet'].shape[0]}, outlet_patch={pc['xyz_outlet'].shape[0]}"
    )
    print(f"[self-test] geodesic coverage: {coverage:.4f}")
    print("self-test OK")


if __name__ == "__main__":
    _self_test()  # run smoke
