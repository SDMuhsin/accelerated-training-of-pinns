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

from sympy import Symbol, Function, Number, sqrt
from sympy import Derivative as D
from physicsnemo.sym.eq.pde import PDE

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from pcs_runtime import cad_to_geometry_json, resolve_cad_path
from sage_ns_v4 import SteadyNavierStokes2DScaledFDSAGE

# -----------------------------
# GPU init
# -----------------------------
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # read local rank
    torch.cuda.set_device(local_rank)  # set device
    torch.cuda.init()  # init cuda
    _ = torch.empty(1, device="cuda")  # warmup


# -----------------------------
# Geometry helpers
# -----------------------------
def _normalize_xy(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xden = (xmax - xmin) if (xmax > xmin) else 1.0  # x range
    yden = (ymax - ymin) if (ymax > ymin) else 1.0  # y range
    x = (x_raw - xmin) / xden  # normalized x
    y = (y_raw - ymin) / yden  # normalized y
    return x.astype(np.float32), y.astype(np.float32)  # float32


def _denormalize_xy(
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xden = (xmax - xmin) if (xmax > xmin) else 1.0  # x range
    yden = (ymax - ymin) if (ymax > ymin) else 1.0  # y range
    xr = x * xden + xmin  # raw x
    yr = y * yden + ymin  # raw y
    return xr, yr  # return raw coords


def _load_points_from_geom(json_path: str):
    obj = json.loads(Path(json_path).read_text())  # load geometry json
    pts = obj["points"]  # point list
    inlet = obj.get("inlet", None)  # inlet point
    outlet = obj.get("outlet", None)  # outlet point

    wall_x, wall_y = [], []  # wall buffers
    inside_x, inside_y = [], []  # inside buffers
    all_x, all_y = [], []  # valid coords buffers
    wall_raw_xy = []  # raw wall coords
    inside_raw_xy = []  # raw inside coords

    for p in pts:
        xr, yr, typ = float(p[0]), float(p[1]), int(p[2])  # unpack point
        if typ == 0:
            continue  # skip background
        if typ == 1:
            wall_x.append(xr)  # store wall x
            wall_y.append(yr)  # store wall y
            wall_raw_xy.append([xr, yr])  # store raw wall xy
        elif typ == 2:
            inside_x.append(xr)  # store inside x
            inside_y.append(yr)  # store inside y
            inside_raw_xy.append([xr, yr])  # store raw inside xy
        else:
            raise ValueError(f"Unknown point type {typ}")  # invalid type
        all_x.append(xr)  # append x
        all_y.append(yr)  # append y

    if len(wall_x) == 0:
        raise ValueError("No wall points found.")  # guard wall
    if len(inside_x) == 0:
        raise ValueError("No inside points found.")  # guard inside

    xmin, xmax = float(min(all_x)), float(max(all_x))  # x bounds
    ymin, ymax = float(min(all_y)), float(max(all_y))  # y bounds

    wall_x = np.asarray(wall_x, np.float32).reshape(-1, 1)  # wall x array
    wall_y = np.asarray(wall_y, np.float32).reshape(-1, 1)  # wall y array
    inside_x = np.asarray(inside_x, np.float32).reshape(-1, 1)  # inside x array
    inside_y = np.asarray(inside_y, np.float32).reshape(-1, 1)  # inside y array

    x_w, y_w = _normalize_xy(wall_x, wall_y, xmin, xmax, ymin, ymax)  # normalized walls
    x_i, y_i = _normalize_xy(inside_x, inside_y, xmin, xmax, ymin, ymax)  # normalized inside

    wall_raw_xy = np.asarray(wall_raw_xy, np.float32)  # raw wall array
    inside_raw_xy = np.asarray(inside_raw_xy, np.float32)  # raw inside array

    def norm_xy(d):
        if d is None:
            return None  # missing point
        xr = np.asarray([[float(d["x"])]], np.float32)  # raw x
        yr = np.asarray([[float(d["y"])]], np.float32)  # raw y
        xn, yn = _normalize_xy(xr, yr, xmin, xmax, ymin, ymax)  # normalize
        return np.asarray([[float(xn[0, 0]), float(yn[0, 0])]], np.float32)  # [1,2]

    inlet_xy = norm_xy(inlet)  # normalized inlet
    outlet_xy = norm_xy(outlet)  # normalized outlet
    norm = (xmin, xmax, ymin, ymax)  # normalization tuple

    return x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm, inside_raw_xy, wall_raw_xy, inlet, outlet, obj


def points_in_radius(xy: np.ndarray, center_xy: np.ndarray, r: float) -> np.ndarray:
    dx = xy[:, 0] - center_xy[0, 0]  # dx
    dy = xy[:, 1] - center_xy[0, 1]  # dy
    return (dx * dx + dy * dy) <= (r * r)  # mask


def build_type2_lookup_from_inside_raw(inside_raw: np.ndarray):
    inside_set = set()  # valid inside pixels
    for i in range(inside_raw.shape[0]):
        x = int(round(float(inside_raw[i, 0])))  # raw x pixel
        y = int(round(float(inside_raw[i, 1])))  # raw y pixel
        inside_set.add((x, y))  # add valid pixel
    return inside_set  # return lookup


def contiguous_segments(sorted_vals):
    if len(sorted_vals) == 0:
        return []  # no segments

    segs = []  # output segments
    start = sorted_vals[0]  # current segment start
    prev = sorted_vals[0]  # previous value

    for v in sorted_vals[1:]:
        if v == prev + 1:
            prev = v  # continue segment
        else:
            segs.append((start, prev))  # close segment
            start = v  # start new segment
            prev = v  # reset previous

    segs.append((start, prev))  # close final segment
    return segs  # return segments


def choose_best_segment(segs, y_center):
    if len(segs) == 0:
        return None  # no segment exists

    best_seg = None  # best segment
    best_score = None  # best distance score

    for y0, y1 in segs:
        mid = 0.5 * (y0 + y1)  # segment midpoint
        score = abs(mid - y_center)  # distance to current center
        if best_score is None or score < best_score:
            best_score = score  # update best score
            best_seg = (y0, y1)  # update best segment

    return best_seg  # return chosen segment


def adaptive_rect_mask_from_center(
    inside_set,
    center_xy_raw: np.ndarray,
    x_steps: int = 30,
    half_height: int = 5,
    direction: str = "right",
    min_run_per_strip: int = 2,
    enforce_connected_chain: bool = True,
):
    cx = int(round(float(center_xy_raw[0, 0])))  # center x
    cy0 = int(round(float(center_xy_raw[0, 1])))  # center y

    selected = []  # selected raw pixels
    prev_y0, prev_y1 = None, None  # previous strip y-range
    curr_center_y = cy0  # running center y

    for dx in range(x_steps + 1):
        if direction == "right":
            x = cx + dx  # move right
        elif direction == "left":
            x = cx - dx  # move left
        else:
            raise ValueError("direction must be 'right' or 'left'")  # invalid direction

        ys = []  # valid inside y's at this x
        for y in range(curr_center_y - half_height, curr_center_y + half_height + 1):
            if (x, y) in inside_set:
                ys.append(y)  # keep valid inside pixel

        if len(ys) == 0:
            continue  # nothing valid in this strip

        segs = contiguous_segments(sorted(ys))  # contiguous y-runs
        segs = [(a, b) for (a, b) in segs if (b - a + 1) >= min_run_per_strip]  # remove tiny runs
        if len(segs) == 0:
            continue  # no usable segment

        if enforce_connected_chain and prev_y0 is not None:
            overlap_segs = []  # overlapping segments
            for a, b in segs:
                if not (b < prev_y0 or a > prev_y1):
                    overlap_segs.append((a, b))  # keep overlapping segment
            if len(overlap_segs) > 0:
                segs = overlap_segs  # prefer continuity

        best = choose_best_segment(segs, curr_center_y)  # choose best segment
        if best is None:
            continue  # nothing valid to pick

        y0, y1 = best  # chosen range
        for y in range(y0, y1 + 1):
            selected.append((x, y))  # add selected raw pixel

        prev_y0, prev_y1 = y0, y1  # update previous strip
        curr_center_y = int(round(0.5 * (y0 + y1)))  # update running center

    if len(selected) == 0:
        return np.zeros((0, 2), dtype=np.int32)  # empty output

    return np.asarray(sorted(set(selected)), dtype=np.int32)  # unique selected pixels


def raw_pixels_to_inside_mask(inside_raw: np.ndarray, selected_pixels: np.ndarray) -> np.ndarray:
    selected_set = set((int(x), int(y)) for x, y in selected_pixels.tolist())  # selected lookup
    mask = np.zeros((inside_raw.shape[0],), dtype=bool)  # output mask

    for i in range(inside_raw.shape[0]):
        x = int(round(float(inside_raw[i, 0])))  # raw x
        y = int(round(float(inside_raw[i, 1])))  # raw y
        if (x, y) in selected_set:
            mask[i] = True  # mark selected

    return mask  # boolean mask


def ensure_patch_has_min_points(
    xy_inside: np.ndarray,
    center_xy: np.ndarray,
    r0: float,
    min_pts: int,
    r_max: float,
    grow: float,
):
    inside_raw = getattr(ensure_patch_has_min_points, "_inside_raw", None)  # cached raw inside points
    center_raw = getattr(ensure_patch_has_min_points, "_center_raw", None)  # cached raw center point
    direction = getattr(ensure_patch_has_min_points, "_direction", "right")  # cached growth direction
    half_height_px = int(getattr(ensure_patch_has_min_points, "_half_height_px", 5))  # cached half height
    min_run_per_strip = int(getattr(ensure_patch_has_min_points, "_min_run_per_strip", 2))  # cached minimum run
    enforce_connected_chain = bool(getattr(ensure_patch_has_min_points, "_enforce_connected_chain", True))  # cached chain flag

    if inside_raw is None or center_raw is None:
        r = float(max(r0, 1.0e-6))  # start radius
        mask = points_in_radius(xy_inside, center_xy, r)  # initial mask
        while int(np.sum(mask)) < int(min_pts) and r < r_max:
            r = min(r * grow, r_max)  # grow radius
            mask = points_in_radius(xy_inside, center_xy, r)  # recompute mask
        if int(np.sum(mask)) == 0:
            d2 = np.sum((xy_inside - center_xy[0:1]) ** 2, axis=1)  # squared distance
            k = min(max(min_pts, 1), xy_inside.shape[0])  # fallback count
            idx = np.argsort(d2)[:k]  # nearest points
            mask = np.zeros((xy_inside.shape[0],), dtype=bool)  # empty mask
            mask[idx] = True  # activate nearest points
        return mask, float(r)  # return fallback circular patch

    inside_set = build_type2_lookup_from_inside_raw(inside_raw)  # build inside lookup
    width_px = max(int(min_pts), 1)  # use min_pts as horizontal extent in pixels

    selected_pixels = adaptive_rect_mask_from_center(
        inside_set=inside_set,
        center_xy_raw=center_raw,
        x_steps=width_px,
        half_height=half_height_px,
        direction=direction,
        min_run_per_strip=min_run_per_strip,
        enforce_connected_chain=enforce_connected_chain,
    )  # build adaptive raw pixels

    mask = raw_pixels_to_inside_mask(inside_raw, selected_pixels)  # convert to inside mask

    if int(np.sum(mask)) == 0:
        d2 = np.sum((inside_raw - center_raw[0:1]) ** 2, axis=1)  # squared distance in raw space
        k = min(max(min_pts, 1), inside_raw.shape[0])  # fallback count
        idx = np.argsort(d2)[:k]  # nearest points
        mask = np.zeros((inside_raw.shape[0],), dtype=bool)  # empty mask
        mask[idx] = True  # activate nearest points

    return mask, float(selected_pixels.shape[0])  # return mask and selected pixel count


def compute_wall_distance_feature(xy_query: np.ndarray, xy_wall: np.ndarray) -> np.ndarray:
    tree = cKDTree(xy_wall)  # kd tree
    d, _ = tree.query(xy_query, k=1)  # nearest wall distance
    return d.astype(np.float32).reshape(-1, 1)  # [N,1]


def project_inside_feature_to_wall(
    xy_wall: np.ndarray,
    xy_inside: np.ndarray,
    feat_inside: np.ndarray,
) -> np.ndarray:
    tree = cKDTree(xy_inside)  # tree on inside points
    _, idx = tree.query(xy_wall, k=1)  # nearest inside point for each wall point
    return feat_inside[idx].astype(np.float32)  # projected feature


# -----------------------------
# Graph helpers
# -----------------------------
def build_knn_graph(
    xy_inside: np.ndarray,
    k: int = 8,
    max_edge_len: Optional[float] = None,
):
    n = xy_inside.shape[0]  # num points
    if n <= 0:
        return csr_matrix((0, 0), dtype=np.float32)  # empty graph

    k_eff = min(k + 1, n)  # include self in knn
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree")  # knn model
    nbrs.fit(xy_inside)  # fit
    dists, inds = nbrs.kneighbors(xy_inside)  # neighbors

    rows, cols, vals = [], [], []  # sparse graph buffers
    for i in range(n):
        for jpos in range(1, k_eff):
            j = int(inds[i, jpos])  # neighbor id
            dij = float(dists[i, jpos])  # euclidean distance
            if max_edge_len is not None and dij > max_edge_len:
                continue  # prune long edges
            rows.append(i)  # forward row
            cols.append(j)  # forward col
            vals.append(dij)  # forward weight
            rows.append(j)  # reverse row
            cols.append(i)  # reverse col
            vals.append(dij)  # reverse weight

    graph = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)  # sparse graph
    return graph  # return graph


def build_pixel_graph(
    inside_raw_xy: np.ndarray,
    norm: Tuple[float, float, float, float],
    connectivity: int = 8,
):
    n = int(inside_raw_xy.shape[0])  # num points
    if n <= 0:
        return csr_matrix((0, 0), dtype=np.float32)  # empty graph

    xmin, xmax, ymin, ymax = norm  # bounds
    xden = max(float(xmax - xmin), 1.0e-6)  # x span
    yden = max(float(ymax - ymin), 1.0e-6)  # y span

    x_pix = np.rint(inside_raw_xy[:, 0]).astype(np.int32)  # raw x pixels
    y_pix = np.rint(inside_raw_xy[:, 1]).astype(np.int32)  # raw y pixels

    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(x_pix, y_pix))}  # pixel->index map

    if int(connectivity) == 4:
        offsets = [(1, 0), (0, 1)]  # only forward half, add reverse explicitly
    else:
        offsets = [(1, 0), (0, 1), (1, 1), (1, -1)]  # forward half for 8-neighbor

    rows, cols, vals = [], [], []  # sparse graph buffers
    for i in range(n):
        xi = int(x_pix[i])  # current x
        yi = int(y_pix[i])  # current y
        for dx, dy in offsets:
            j = lookup.get((xi + dx, yi + dy), None)  # neighbor index
            if j is None:
                continue  # neighbor absent
            w = float(np.hypot(float(dx) / xden, float(dy) / yden))  # normalized step cost
            rows.append(i)  # forward row
            cols.append(j)  # forward col
            vals.append(w)  # forward weight
            rows.append(j)  # reverse row
            cols.append(i)  # reverse col
            vals.append(w)  # reverse weight

    graph = csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)  # sparse graph
    return graph  # return graph


def build_inside_graph(
    xy_inside: np.ndarray,
    inside_raw_xy: np.ndarray,
    norm: Tuple[float, float, float, float],
    mode: str,
    knn_k: int,
    max_edge_len: Optional[float],
    pixel_connectivity: int,
):
    mode = str(mode).strip().lower()  # normalize mode
    if mode == "pixel":
        return build_pixel_graph(
            inside_raw_xy=inside_raw_xy,
            norm=norm,
            connectivity=int(pixel_connectivity),
        )  # exact pixel graph
    return build_knn_graph(
        xy_inside=xy_inside,
        k=int(knn_k),
        max_edge_len=max_edge_len,
    )  # knn graph


def compute_geodesic_info_on_graph(
    graph: csr_matrix,
    xy_inside: np.ndarray,
    source_xy: np.ndarray,
):
    d2 = np.sum((xy_inside - source_xy[0:1]) ** 2, axis=1)  # squared distance to source
    src = int(np.argmin(d2))  # nearest inside point to source
    geo, predecessors = shortest_path(
        graph,
        directed=False,
        indices=src,
        method="D",
        return_predecessors=True,
    )  # geodesic + predecessor

    geo = np.asarray(geo, dtype=np.float32)  # cast
    predecessors = np.asarray(predecessors, dtype=np.int64)  # cast

    bad = np.isinf(geo)  # unreachable
    if np.any(bad):
        geo[bad] = np.sqrt(d2[bad]).astype(np.float32)  # fallback euclidean
        predecessors[bad] = -9999  # no predecessor

    geo = geo.reshape(-1, 1)  # [N,1]
    gmin = float(np.min(geo))  # min
    gmax = float(np.max(geo))  # max
    s_geo = (geo - gmin) / max(gmax - gmin, 1.0e-8)  # normalized progress

    return {
        "src": src,
        "geo": geo.astype(np.float32),
        "s_geo": s_geo.astype(np.float32),
        "predecessors": predecessors.astype(np.int64),
    }  # bundle


def trace_path_to_source(predecessors: np.ndarray, src: int, dst: int, max_steps: int = 100000) -> List[int]:
    path = [int(dst)]  # start from destination
    cur = int(dst)  # running node
    seen = set([cur])  # cycle guard

    for _ in range(max_steps):
        if cur == src:
            break  # reached source
        p = int(predecessors[cur])  # predecessor
        if p < 0:
            break  # path broken
        if p in seen:
            break  # unexpected cycle
        path.append(p)  # add predecessor
        seen.add(p)  # mark visited
        cur = p  # step back

    path.reverse()  # from source to destination
    return path  # return node list


def estimate_tangent_from_predecessor_tree(
    xy_inside: np.ndarray,
    predecessors: np.ndarray,
    src: int,
) -> np.ndarray:
    n = xy_inside.shape[0]  # number of points
    tangent = np.zeros((n, 2), dtype=np.float32)  # output tangent

    children = [[] for _ in range(n)]  # reverse tree
    for i in range(n):
        p = int(predecessors[i])  # predecessor of i
        if p >= 0:
            children[p].append(i)  # i is child of p

    for i in range(n):
        p = int(predecessors[i])  # predecessor
        if i == src:
            if len(children[i]) > 0:
                j = int(children[i][0])  # pick one child
                vec = xy_inside[j] - xy_inside[i]  # source forward direction
            else:
                vec = np.array([1.0, 0.0], dtype=np.float32)  # fallback
        else:
            if p >= 0:
                vec = xy_inside[i] - xy_inside[p]  # along path direction
            else:
                vec = np.array([1.0, 0.0], dtype=np.float32)  # fallback

        norm = float(np.linalg.norm(vec))  # magnitude
        if norm > 1.0e-12:
            tangent[i] = vec / norm  # normalize
        else:
            tangent[i] = np.array([1.0, 0.0], dtype=np.float32)  # fallback unit x

    return tangent  # [N,2]


def sort_by_progress_chunked(
    x_i: np.ndarray,
    y_i: np.ndarray,
    d_w_i: np.ndarray,
    s_in_i: np.ndarray,
    s_out_i: np.ndarray,
    chunk_size: int,
):
    order = np.argsort(s_in_i.reshape(-1))  # sort by inlet progress
    chunks = []  # chunk list
    for s in range(0, len(order), chunk_size):
        e = min(s + chunk_size, len(order))  # chunk end
        ch = order[s:e].copy()  # chunk copy
        np.random.shuffle(ch)  # local shuffle
        chunks.append(ch)  # save chunk
    order2 = np.concatenate(chunks, axis=0)  # final order
    return (
        x_i[order2],
        y_i[order2],
        d_w_i[order2],
        s_in_i[order2],
        s_out_i[order2],
        order2,
    )  # reordered


def _build_weighted_flow_pde_indices(
    xy_inside: np.ndarray,
    xy_wall: np.ndarray,
    d_w_i: np.ndarray,
    predecessors_in: np.ndarray,
    src_in: int,
    outlet_target_idx: int,
    target_points: int,
    wall_boost: float,
    wall_scale: float,
    corridor_boost: float,
    corridor_radius: float,
    seed: int,
):
    n = int(xy_inside.shape[0])  # total interior points
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)  # empty

    if (target_points <= 0) or (target_points >= n):
        return np.arange(n, dtype=np.int64)  # keep all

    weights = np.ones((n,), dtype=np.float64)  # base weights

    if wall_boost > 0.0:
        dw = d_w_i.reshape(-1).astype(np.float32)  # wall distance flat
        wall_importance = np.exp(-((dw / max(wall_scale, 1.0e-8)) ** 2)).astype(np.float64)  # near-wall boost
        weights *= (1.0 + wall_boost * wall_importance)  # apply boost

    if corridor_boost > 0.0:
        path_idx = trace_path_to_source(
            predecessors=predecessors_in,
            src=int(src_in),
            dst=int(outlet_target_idx),
        )  # path from inlet to outlet
        if len(path_idx) >= 2:
            path_xy = xy_inside[np.asarray(path_idx, dtype=np.int64)]  # path coords
            knn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")  # path tree
            knn.fit(path_xy)  # fit
            d_path, _ = knn.kneighbors(xy_inside)  # distance to path
            d_path = d_path[:, 0].astype(np.float32)  # flatten
            corridor_importance = np.exp(-((d_path / max(corridor_radius, 1.0e-8)) ** 2)).astype(np.float64)  # path boost
            weights *= (1.0 + corridor_boost * corridor_importance)  # apply boost

    weights = np.maximum(weights, 1.0e-12)  # guard zero
    weights = weights / float(np.sum(weights))  # normalize
    rng = np.random.default_rng(int(seed))  # rng
    idx = rng.choice(np.arange(n), size=int(target_points), replace=False, p=weights).astype(np.int64)  # sample
    return idx  # selected indices


def _build_wall_guard_points(
    xy_inside: np.ndarray,
    xy_wall: np.ndarray,
    radius: float,
    target_points: int,
    seed: int,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    exclude_mask: Optional[np.ndarray] = None,
):
    n = int(xy_inside.shape[0])  # num inside points
    if (n <= 0) or (xy_wall.shape[0] <= 0) or (target_points <= 0) or (radius <= 0.0):
        return np.zeros((0, 2), dtype=np.float32)  # empty set

    knn_w = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")  # nearest wall model
    knn_w.fit(xy_wall.astype(np.float32))  # fit
    dw, _ = knn_w.kneighbors(xy_inside.astype(np.float32))  # nearest wall distances
    dw = dw[:, 0].astype(np.float32)  # flatten

    mask = (dw <= float(radius))  # near-wall mask
    if x_min is not None:
        mask &= (xy_inside[:, 0] >= float(x_min))  # x lower bound
    if x_max is not None:
        mask &= (xy_inside[:, 0] <= float(x_max))  # x upper bound
    if y_min is not None:
        mask &= (xy_inside[:, 1] >= float(y_min))  # y lower bound
    if y_max is not None:
        mask &= (xy_inside[:, 1] <= float(y_max))  # y upper bound
    if exclude_mask is not None and int(exclude_mask.shape[0]) == n:
        mask &= (~exclude_mask.astype(bool))  # exclude ports

    idx = np.where(mask)[0].astype(np.int64)  # candidates
    if idx.size <= 0:
        return np.zeros((0, 2), dtype=np.float32)  # none selected

    if idx.size > int(target_points):
        rng = np.random.default_rng(int(seed))  # rng
        idx = rng.choice(idx, size=int(target_points), replace=False).astype(np.int64)  # sample
    return xy_inside[idx].astype(np.float32)  # return selected points


def _estimate_wall_normals(
    xy_points: np.ndarray,
    xy_wall: np.ndarray,
    k_neighbors: int = 4,
):
    n = int(xy_points.shape[0])  # num points
    if (n <= 0) or (xy_wall.shape[0] <= 0):
        return (
            np.zeros((n, 1), dtype=np.float32),
            np.zeros((n, 1), dtype=np.float32),
        )  # empty/fallback normals

    k = int(max(min(int(k_neighbors), int(xy_wall.shape[0])), 1))  # clamp k
    knn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")  # wall knn
    knn.fit(xy_wall.astype(np.float32))  # fit
    dists, idx = knn.kneighbors(xy_points.astype(np.float32))  # neighbors

    wall_neighbors = xy_wall[idx]  # wall neighbor coordinates
    vec = xy_points[:, None, :] - wall_neighbors  # inward-ish vectors
    w = 1.0 / np.maximum(dists.astype(np.float32), 1.0e-6)  # inverse-distance weights
    nvec = np.sum(vec * w[:, :, None], axis=1)  # weighted average

    nrm = np.linalg.norm(nvec, axis=1, keepdims=True).astype(np.float32)  # magnitude
    fallback = vec[:, 0, :].astype(np.float32)  # nearest-wall fallback
    nvec = np.where(nrm > 1.0e-8, nvec, fallback)  # fallback if zero
    nrm = np.linalg.norm(nvec, axis=1, keepdims=True).astype(np.float32)  # recompute
    nvec = nvec / np.maximum(nrm, 1.0e-8)  # normalize
    return nvec[:, 0:1].astype(np.float32), nvec[:, 1:2].astype(np.float32)  # nx, ny


# -----------------------------
# Initial pseudo-field generation
# -----------------------------
def compute_initial_flow_guess(
    xy_inside: np.ndarray,
    xy_wall: np.ndarray,
    inlet_p: float,
    inlet_u: float,
    inlet_v: float,
    geo_info_in: Dict[str, np.ndarray],
    geo_info_out: Dict[str, np.ndarray],
    velocity_scale: float = 1.0,
    velocity_power: float = 1.0,
    pressure_power: float = 1.0,
    pressure_drop_guess: float = 0.0,
):
    geo_in = geo_info_in["geo"]  # raw geodesic from inlet
    s_in = geo_info_in["s_geo"]  # normalized progress from inlet
    s_out = geo_info_out["s_geo"]  # normalized progress from outlet
    src = int(geo_info_in["src"])  # inlet source index
    predecessors = geo_info_in["predecessors"]  # predecessor tree from inlet

    tangent = estimate_tangent_from_predecessor_tree(
        xy_inside=xy_inside,
        predecessors=predecessors,
        src=src,
    )  # local direction

    d_wall = compute_wall_distance_feature(xy_inside, xy_wall)  # distance to nearest wall
    dmax = float(np.max(d_wall)) if d_wall.size > 0 else 1.0  # max wall distance
    dnorm = d_wall / max(dmax, 1.0e-8)  # normalize wall distance

    inlet_speed = float(np.sqrt(float(inlet_u) ** 2 + float(inlet_v) ** 2))  # inlet speed magnitude
    if inlet_speed < 1.0e-12:
        inlet_speed = 1.0  # fallback if inlet bc is zero

    wall_profile = np.clip(dnorm, 0.0, 1.0) ** float(velocity_power)  # slower near wall
    axial_decay = (1.0 - 0.10 * s_in).astype(np.float32)  # mild downstream decay
    speed = float(velocity_scale) * inlet_speed * wall_profile * axial_decay  # speed field

    u0 = speed * tangent[:, 0:1]  # projected x velocity
    v0 = speed * tangent[:, 1:2]  # projected y velocity

    blend = np.clip(s_in / np.maximum(s_in + s_out, 1.0e-8), 0.0, 1.0)  # blended progress
    p0 = float(inlet_p) - float(pressure_drop_guess) * (blend ** float(pressure_power))  # inlet-anchored pressure guess

    return {
        "u": u0.astype(np.float32),
        "v": v0.astype(np.float32),
        "p": p0.astype(np.float32),
        "dw": d_wall.astype(np.float32),
        "geo_in": geo_in.astype(np.float32),
        "s_in": s_in.astype(np.float32),
        "s_out": s_out.astype(np.float32),
        "tangent": tangent.astype(np.float32),
        "predecessors": predecessors.astype(np.int64),
        "src": int(src),
    }  # initial fields bundle


def plot_random_geodesic_paths(
    xy_inside: np.ndarray,
    inlet_xy: np.ndarray,
    outlet_xy: np.ndarray,
    predecessors: np.ndarray,
    src: int,
    save_path: str,
    num_paths: int = 12,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)  # rng
    n = xy_inside.shape[0]  # point count
    if n == 0:
        return  # nothing to plot

    num_paths = min(int(num_paths), n)  # cap number of paths
    sample_ids = rng.choice(np.arange(n), size=num_paths, replace=False)  # random destinations

    fig, ax = plt.subplots(figsize=(8, 8))  # figure
    ax.scatter(xy_inside[:, 0], xy_inside[:, 1], s=4, alpha=0.15, label="inside")  # inside cloud
    ax.scatter(inlet_xy[:, 0], inlet_xy[:, 1], s=80, marker="o", label="inlet")  # inlet
    ax.scatter(outlet_xy[:, 0], outlet_xy[:, 1], s=80, marker="x", label="outlet")  # outlet

    for idx in sample_ids:
        path = trace_path_to_source(predecessors=predecessors, src=src, dst=int(idx))  # recover path
        if len(path) < 2:
            continue  # skip degenerate path
        pts = xy_inside[np.asarray(path, dtype=np.int64)]  # path coordinates
        ax.plot(pts[:, 0], pts[:, 1], linewidth=1.5)  # draw path
        ax.scatter(pts[-1, 0], pts[-1, 1], s=24)  # end point

    ax.set_title("Random geodesic paths from inlet")  # title
    ax.set_xlabel("x")  # x label
    ax.set_ylabel("y")  # y label
    ax.set_aspect("equal")  # equal aspect
    ax.legend()  # legend
    plt.tight_layout()  # layout
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # save
    plt.close(fig)  # close


def save_init_fields_into_geometry_json(
    geom_obj: dict,
    geom_json_path: str,
    xy_inside: np.ndarray,
    init_fields_inside: Dict[str, np.ndarray],
    norm: Tuple[float, float, float, float],
):
    xmin, xmax, ymin, ymax = norm  # bounds
    xr, yr = _denormalize_xy(
        xy_inside[:, 0:1],
        xy_inside[:, 1:2],
        xmin,
        xmax,
        ymin,
        ymax,
    )  # raw inside coords

    geom_obj["init_fields"] = {
        "xy": xy_inside.tolist(),  # normalized inside coords
        "xy_raw": np.concatenate([xr, yr], axis=1).tolist(),  # raw inside coords
        "fields": {
            "u": init_fields_inside["u"].reshape(-1).tolist(),
            "v": init_fields_inside["v"].reshape(-1).tolist(),
            "p": init_fields_inside["p"].reshape(-1).tolist(),
            "dw": init_fields_inside["dw"].reshape(-1).tolist(),
            "geo_in": init_fields_inside["geo_in"].reshape(-1).tolist(),
            "s_in": init_fields_inside["s_in"].reshape(-1).tolist(),
            "s_out": init_fields_inside["s_out"].reshape(-1).tolist(),
        },
    }  # attach initial field section

    p = Path(geom_json_path)  # path object
    out_path = str(p.with_name(f"{p.stem}_with_initi_guess{p.suffix}"))  # output file path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geom_obj, f, indent=2)  # save augmented geometry json


# -----------------------------
# IO for output fields
# -----------------------------
def _write_multi_field_inference_json(
    path: str,
    xy_norm: np.ndarray,
    fields_dict: Dict[str, np.ndarray],
    norm: Optional[Tuple[float, float, float, float]] = None,
    point_type: Optional[np.ndarray] = None,
):
    payload = {
        "xy": xy_norm.tolist(),
        "fields": {k: v.tolist() for k, v in fields_dict.items()},
    }  # payload

    if norm is not None:
        payload["norm"] = list(norm)  # save norm
        xmin, xmax, ymin, ymax = norm  # unpack bounds
        xr, yr = _denormalize_xy(xy_norm[:, 0:1], xy_norm[:, 1:2], xmin, xmax, ymin, ymax)  # raw coords
        payload["xy_raw"] = np.concatenate([xr, yr], axis=1).tolist()  # save raw

    if point_type is not None:
        payload["point_type"] = point_type.tolist()  # save type

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)  # write json


def _plot_fields(xy, fields_dict, save_path, title_prefix="Flow field"):
    keys = [k for k in ["u", "v", "p"] if k in fields_dict]  # keys
    triang = Triangulation(xy[:, 0], xy[:, 1])  # triangulation
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)  # figure
    axes = axes[0]  # flatten
    for ax, key in zip(axes, keys):
        vals = fields_dict[key]  # values
        c = ax.tricontourf(triang, vals, levels=60)  # contour
        fig.colorbar(c, ax=ax)  # colorbar
        ax.set_title(f"{title_prefix}: {key}")  # title
        ax.set_aspect("equal")  # equal aspect
        ax.set_xlabel("x")  # x label
        ax.set_ylabel("y")  # y label
    plt.tight_layout()  # layout
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # save
    plt.close(fig)  # close


# -----------------------------
# PDEs
# -----------------------------
class SteadyNavierStokes2DScaled(PDE):
    name = "SteadyNavierStokes2DScaled"

    def __init__(self, rho=1.0, nu=1.0e-3, Lx=1.0, Ly=1.0):
        super().__init__()  # init base

        x = Symbol("x")  # normalized x
        y = Symbol("y")  # normalized y

        rhoN = Number(float(rho))  # density
        nuN = Number(float(nu))  # viscosity
        invLx = Number(1.0 / max(float(Lx), 1.0e-12))  # x derivative scale
        invLy = Number(1.0 / max(float(Ly), 1.0e-12))  # y derivative scale
        invLx2 = Number(float(invLx) ** 2)  # x second derivative scale
        invLy2 = Number(float(invLy) ** 2)  # y second derivative scale

        u = Function("u")(x, y)  # u
        v = Function("v")(x, y)  # v
        p = Function("p")(x, y)  # p

        ux = invLx * D(u, x)  # du/dx_phys
        uy = invLy * D(u, y)  # du/dy_phys
        vx = invLx * D(v, x)  # dv/dx_phys
        vy = invLy * D(v, y)  # dv/dy_phys
        px = invLx * D(p, x)  # dp/dx_phys
        py = invLy * D(p, y)  # dp/dy_phys
        uxx = invLx2 * D(u, x, 2)  # d2u/dx_phys2
        uyy = invLy2 * D(u, y, 2)  # d2u/dy_phys2
        vxx = invLx2 * D(v, x, 2)  # d2v/dx_phys2
        vyy = invLy2 * D(v, y, 2)  # d2v/dy_phys2

        self.equations = {
            "continuity": ux + vy,
            "momentum_x": u * ux + v * uy + (Number(1.0) / rhoN) * px - nuN * (uxx + uyy),
            "momentum_y": u * vx + v * vy + (Number(1.0) / rhoN) * py - nuN * (vxx + vyy),
        }  # equations


class WallNormalNoPenetration2D(PDE):
    name = "WallNormalNoPenetration2D"

    def __init__(self, eq_name: str = "wall_normal_velocity"):
        super().__init__()  # init base
        x = Symbol("x")  # x
        y = Symbol("y")  # y
        u = Function("u")(x, y)  # u
        v = Function("v")(x, y)  # v
        n_x = Symbol("n_x")  # wall normal x
        n_y = Symbol("n_y")  # wall normal y
        self.equations = {str(eq_name): u * n_x + v * n_y}  # no-penetration equation


class FlowTrajectoryGuidance2D(PDE):
    name = "FlowTrajectoryGuidance2D"

    def __init__(
        self,
        eq_parallel: str = "flow_geo_parallel",
        eq_cross: str = "flow_geo_cross",
        eq_cosine: str = "flow_geo_cosine",
        eq_speed: str = "flow_geo_speed",
        speed_eps: float = 1.0e-6,
    ):
        super().__init__()  # init base
        x = Symbol("x")  # x
        y = Symbol("y")  # y

        u = Function("u")(x, y)  # u
        v = Function("v")(x, y)  # v
        g_x = Symbol("g_x")  # desired direction x
        g_y = Symbol("g_y")  # desired direction y

        eps = Number(float(max(speed_eps, 1.0e-12)))  # speed epsilon
        speed = sqrt(u * u + v * v + eps)  # speed magnitude
        parallel = u * g_x + v * g_y  # along guidance direction
        cross = -u * g_y + v * g_x  # perpendicular component
        cosine = parallel / speed  # cosine alignment

        self.equations = {
            str(eq_parallel): parallel,
            str(eq_cross): cross,
            str(eq_cosine): cosine,
            str(eq_speed): speed,
        }  # guidance equations


# -----------------------------
# Training helpers
# -----------------------------
def get_activation(act_name: str):
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


def _disable_recording(slv: Solver):
    slv.record_constraints = lambda *args, **kwargs: None  # disable
    slv.record_validators = lambda *args, **kwargs: None  # disable
    slv.record_inferencers = lambda *args, **kwargs: None  # disable
    slv.record_monitors = lambda *args, **kwargs: None  # disable


def make_stage_cfg(base_cfg: DictConfig, stage_name: str, max_steps: int, init_dir: str = "") -> DictConfig:
    c = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))  # copy config
    c.training.max_steps = int(max_steps)  # set steps
    c.network_dir = str(Path(base_cfg.network_dir) / stage_name)  # stage dir
    c.initialization_network_dir = init_dir  # init dir
    return c  # return


def _build_pointwise_constraint(
    *,
    nodes,
    invar: Dict[str, np.ndarray],
    outvar: Dict[str, np.ndarray],
    batch_size: int,
    shuffle: bool = True,
    lambda_weights: Optional[Dict[str, object]] = None,
):
    kwargs = {}  # optional kwargs
    if lambda_weights is not None:
        lambda_weighting = {}  # lambda map
        for key, target in outvar.items():
            w_spec = lambda_weights.get(key, 1.0)  # scalar or array
            if np.isscalar(w_spec):
                lambda_weighting[key] = np.full_like(target, float(w_spec), dtype=np.float32)  # scalar->array
            else:
                w_arr = np.asarray(w_spec, dtype=np.float32)  # convert
                if w_arr.shape != target.shape:
                    w_arr = np.broadcast_to(w_arr, target.shape).astype(np.float32)  # broadcast
                lambda_weighting[key] = w_arr  # save
        kwargs["lambda_weighting"] = lambda_weighting  # attach lambda weights

    try:
        return PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=invar,
            outvar=outvar,
            batch_size=int(max(min(batch_size, next(iter(outvar.values())).shape[0]), 1)),
            shuffle=shuffle,
            **kwargs,
        )  # build constraint
    except TypeError:
        return PointwiseConstraint.from_numpy(
            nodes=nodes,
            invar=invar,
            outvar=outvar,
            batch_size=int(max(min(batch_size, next(iter(outvar.values())).shape[0]), 1)),
            shuffle=shuffle,
        )  # fallback unweighted


def _linear_stage_scale(stage_idx: int, num_stages: int, start_scale: float, end_scale: float) -> float:
    if num_stages <= 1:
        return float(end_scale)  # degenerate case
    frac = float(stage_idx) / float(num_stages - 1)  # stage fraction
    return float(start_scale + frac * (end_scale - start_scale))  # linear interpolation


# -----------------------------
# Constraint builders
# -----------------------------
def build_primary_constraints(
    flow_nodes,
    cfg,
    x_w,
    y_w,
    d_w_w,
    s_in_w,
    s_out_w,
    x_i_sorted,
    y_i_sorted,
    d_w_i_sorted,
    s_in_i_sorted,
    s_out_i_sorted,
    x_i,
    y_i,
    d_w_i,
    s_in_i,
    s_out_i,
    inlet_mask,
    outlet_mask,
    weight_scale: float = 1.0,
):
    w_scale = float(max(weight_scale, 0.0))  # constraint weight scale

    # wall inputs
    wall_invar = {
        "x": x_w.astype(np.float32),
        "y": y_w.astype(np.float32),
        "dw": d_w_w.astype(np.float32),
        "sin": s_in_w.astype(np.float32),
        "sout": s_out_w.astype(np.float32),
    }  # wall inputs
    wall_outvar = {
        "u": np.zeros((x_w.shape[0], 1), np.float32),
        "v": np.zeros((x_w.shape[0], 1), np.float32),
    }  # wall targets
    wall_constraint = _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=wall_invar,
        outvar=wall_outvar,
        batch_size=min(int(cfg.training.flow_wall_batch_size), x_w.shape[0]),
        shuffle=True,
        lambda_weights={
            "u": float(getattr(cfg.training, "w_wall_u", 1.0)) * w_scale,
            "v": float(getattr(cfg.training, "w_wall_v", 1.0)) * w_scale,
        },
    )  # wall constraint

    # PDE inputs
    flow_interior_invar = {
        "x": x_i_sorted.astype(np.float32),
        "y": y_i_sorted.astype(np.float32),
        "dw": d_w_i_sorted.astype(np.float32),
        "sin": s_in_i_sorted.astype(np.float32),
        "sout": s_out_i_sorted.astype(np.float32),
    }  # pde inputs
    flow_pde_outvar = {
        "continuity": np.zeros((x_i_sorted.shape[0], 1), np.float32),
        "momentum_x": np.zeros((x_i_sorted.shape[0], 1), np.float32),
        "momentum_y": np.zeros((x_i_sorted.shape[0], 1), np.float32),
    }  # pde targets
    flow_pde_constraint = _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=flow_interior_invar,
        outvar=flow_pde_outvar,
        batch_size=min(int(cfg.training.flow_pde_batch_size), x_i_sorted.shape[0]),
        shuffle=False,
        lambda_weights={
            "continuity": float(getattr(cfg.training, "w_pde_continuity", 1.0)) * w_scale,
            "momentum_x": float(getattr(cfg.training, "w_pde_momentum_x", 1.0)) * w_scale,
            "momentum_y": float(getattr(cfg.training, "w_pde_momentum_y", 1.0)) * w_scale,
        },
    )  # pde constraint

    # inlet velocity
    x_in = x_i[inlet_mask].astype(np.float32)
    y_in = y_i[inlet_mask].astype(np.float32)
    d_w_in = d_w_i[inlet_mask].astype(np.float32)
    s_in_in = s_in_i[inlet_mask].astype(np.float32)
    s_out_in = s_out_i[inlet_mask].astype(np.float32)
    inlet_invar = {
        "x": x_in,
        "y": y_in,
        "dw": d_w_in,
        "sin": s_in_in,
        "sout": s_out_in,
    }  # inlet inputs
    inlet_outvar = {
        "u": np.full((x_in.shape[0], 1), float(cfg.bc.inlet_u), np.float32),
        "v": np.full((x_in.shape[0], 1), float(cfg.bc.inlet_v), np.float32),
    }  # inlet targets
    inlet_constraint = _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=inlet_invar,
        outvar=inlet_outvar,
        batch_size=min(int(cfg.training.flow_bc_batch_size), x_in.shape[0]),
        shuffle=True,
        lambda_weights={
            "u": float(getattr(cfg.training, "w_inlet_u", 1.0)) * w_scale,
            "v": float(getattr(cfg.training, "w_inlet_v", 1.0)) * w_scale,
        },
    )  # inlet constraint

    # optional outlet pressure anchor
    outlet_constraint = None  # default none
    if bool(getattr(cfg.bc, "use_outlet_pressure_constraint", False)):
        x_out = x_i[outlet_mask].astype(np.float32)
        y_out = y_i[outlet_mask].astype(np.float32)
        d_w_out = d_w_i[outlet_mask].astype(np.float32)
        s_in_out = s_in_i[outlet_mask].astype(np.float32)
        s_out_out = s_out_i[outlet_mask].astype(np.float32)
        outlet_invar = {
            "x": x_out,
            "y": y_out,
            "dw": d_w_out,
            "sin": s_in_out,
            "sout": s_out_out,
        }  # outlet inputs
        outlet_outvar = {
            "p": np.full((x_out.shape[0], 1), float(cfg.bc.outlet_p), np.float32)
        }  # outlet target
        outlet_constraint = _build_pointwise_constraint(
            nodes=flow_nodes,
            invar=outlet_invar,
            outvar=outlet_outvar,
            batch_size=min(int(cfg.training.flow_bc_batch_size), x_out.shape[0]),
            shuffle=True,
            lambda_weights={
                "p": float(getattr(cfg.training, "w_outlet_p", 1.0)) * w_scale,
            },
        )  # outlet pressure anchor

    # optional inlet pressure anchor
    inlet_p_constraint = None  # default none
    if bool(getattr(cfg.bc, "use_inlet_pressure_anchor", False)):
        inlet_p_constraint = _build_pointwise_constraint(
            nodes=flow_nodes,
            invar=inlet_invar,
            outvar={"p": np.full((x_in.shape[0], 1), float(cfg.bc.inlet_p), np.float32)},
            batch_size=min(int(cfg.training.flow_bc_batch_size), x_in.shape[0]),
            shuffle=True,
            lambda_weights={
                "p": float(getattr(cfg.training, "w_inlet_p", 1.0)) * w_scale,
            },
        )  # inlet pressure anchor

    return flow_pde_constraint, wall_constraint, inlet_constraint, outlet_constraint, inlet_p_constraint  # all constraints


def build_pseudo_init_constraint(
    flow_nodes,
    cfg,
    x_i,
    y_i,
    d_w_i,
    s_in_i,
    s_out_i,
    init_inside_fields,
    batch_size_key: str,
    scale: float,
):
    s = float(max(scale, 0.0))  # scale
    init_invar = {
        "x": x_i.astype(np.float32),
        "y": y_i.astype(np.float32),
        "dw": d_w_i.astype(np.float32),
        "sin": s_in_i.astype(np.float32),
        "sout": s_out_i.astype(np.float32),
    }  # invar
    init_outvar = {
        "u": init_inside_fields["u"].astype(np.float32),
        "v": init_inside_fields["v"].astype(np.float32),
        "p": init_inside_fields["p"].astype(np.float32),
    }  # pseudo targets

    return _build_pointwise_constraint(
        nodes=flow_nodes,
        invar=init_invar,
        outvar=init_outvar,
        batch_size=min(int(getattr(cfg.training, batch_size_key, 32768)), x_i.shape[0]),
        shuffle=True,
        lambda_weights={
            "u": float(getattr(cfg.training, "w_soft_init_u", 1.0)) * s,
            "v": float(getattr(cfg.training, "w_soft_init_v", 1.0)) * s,
            "p": float(getattr(cfg.training, "w_soft_init_p", 1.0)) * s,
        },
    )  # init regression constraint


def build_geo_guidance_constraints(
    geo_nodes,
    cfg,
    x_i,
    y_i,
    d_w_i,
    s_in_i,
    s_out_i,
    init_inside_fields,
    inlet_mask,
    outlet_mask,
    scale: float,
):
    if not bool(getattr(cfg.training, "use_geo_direction_guidance", True)):
        return None, None, None  # disabled

    s = float(max(scale, 0.0))  # stage scale
    if s <= 0.0:
        return None, None, None  # inactive

    speed = np.sqrt(init_inside_fields["u"] ** 2 + init_inside_fields["v"] ** 2).astype(np.float32)  # init speed
    tangent = init_inside_fields["tangent"].astype(np.float32)  # init direction
    speed_flat = speed.reshape(-1)  # flat speed
    valid = speed_flat > float(getattr(cfg.training, "geo_guidance_speed_eps", 1.0e-4))  # valid direction mask

    if bool(getattr(cfg.training, "geo_guidance_exclude_ports", True)):
        valid &= (~inlet_mask)  # exclude inlet points
        valid &= (~outlet_mask)  # exclude outlet points

    idx = np.where(valid)[0].astype(np.int64)  # valid guidance points
    if idx.size <= 0:
        return None, None, None  # nothing to build

    max_points = int(getattr(cfg.training, "geo_guidance_max_points", 0))  # max guidance points
    if (max_points > 0) and (idx.size > max_points):
        rng = np.random.default_rng(int(getattr(cfg.training, "geo_guidance_seed", 1234)))  # rng
        idx = rng.choice(idx, size=max_points, replace=False).astype(np.int64)  # subsample

    xg = x_i[idx].astype(np.float32)  # x guidance
    yg = y_i[idx].astype(np.float32)  # y guidance
    dwg = d_w_i[idx].astype(np.float32)  # dw guidance
    sing = s_in_i[idx].astype(np.float32)  # sin guidance
    soutg = s_out_i[idx].astype(np.float32)  # sout guidance
    gx = tangent[idx, 0:1].astype(np.float32)  # desired direction x
    gy = tangent[idx, 1:2].astype(np.float32)  # desired direction y
    sp = speed[idx].astype(np.float32)  # target speed

    smax = float(np.max(sp)) if sp.size > 0 else 1.0  # max speed
    weight_floor = float(getattr(cfg.training, "geo_guidance_weight_floor", 0.1))  # min weight
    weight_gate = np.clip(sp / max(smax, 1.0e-8), weight_floor, 1.0).astype(np.float32)  # stronger in core flow

    shared_invar = {
        "x": xg,
        "y": yg,
        "dw": dwg,
        "sin": sing,
        "sout": soutg,
        "g_x": gx,
        "g_y": gy,
    }  # guidance inputs

    dir_constraint = _build_pointwise_constraint(
        nodes=geo_nodes,
        invar=shared_invar,
        outvar={
            "flow_geo_cross": np.zeros((xg.shape[0], 1), np.float32),
            "flow_geo_cosine": np.ones((xg.shape[0], 1), np.float32),
        },
        batch_size=min(int(getattr(cfg.training, "geo_guidance_batch_size", 16384)), xg.shape[0]),
        shuffle=True,
        lambda_weights={
            "flow_geo_cross": float(getattr(cfg.training, "w_geo_cross", 1.0)) * s * weight_gate,
            "flow_geo_cosine": float(getattr(cfg.training, "w_geo_cosine", 1.0)) * s * weight_gate,
        },
    )  # direction strictness

    parallel_constraint = _build_pointwise_constraint(
        nodes=geo_nodes,
        invar=shared_invar,
        outvar={
            "flow_geo_parallel": sp.astype(np.float32),
        },
        batch_size=min(int(getattr(cfg.training, "geo_guidance_batch_size", 16384)), xg.shape[0]),
        shuffle=True,
        lambda_weights={
            "flow_geo_parallel": float(getattr(cfg.training, "w_geo_parallel", 0.5)) * s * weight_gate,
        },
    )  # along-path magnitude

    speed_constraint = None  # optional
    if float(getattr(cfg.training, "w_geo_speed", 0.0)) > 0.0:
        speed_constraint = _build_pointwise_constraint(
            nodes=geo_nodes,
            invar={
                "x": xg,
                "y": yg,
                "dw": dwg,
                "sin": sing,
                "sout": soutg,
            },
            outvar={
                "flow_geo_speed": sp.astype(np.float32),
            },
            batch_size=min(int(getattr(cfg.training, "geo_guidance_batch_size", 16384)), xg.shape[0]),
            shuffle=True,
            lambda_weights={
                "flow_geo_speed": float(getattr(cfg.training, "w_geo_speed", 0.0)) * s * weight_gate,
            },
        )  # speed guidance

    return dir_constraint, parallel_constraint, speed_constraint  # all geo constraints


def build_wall_guard_constraints(
    wall_guard_nodes,
    cfg,
    xy_wall_guard: np.ndarray,
    xy_wall_guard_sep: np.ndarray,
    xy_wall: np.ndarray,
    graph_feature_lookup_inside: Dict[str, np.ndarray],
    scale: float,
):
    if not bool(getattr(cfg.training, "wall_guard_enabled", True)):
        return None, None  # disabled

    s = float(max(scale, 0.0))  # stage scale
    if s <= 0.0:
        return None, None  # inactive

    def _one_constraint(xy_guard: np.ndarray, weight_name: str):
        if xy_guard.shape[0] <= 0:
            return None  # nothing to build

        nx, ny = _estimate_wall_normals(
            xy_points=xy_guard.astype(np.float32),
            xy_wall=xy_wall.astype(np.float32),
            k_neighbors=int(getattr(cfg.training, "wall_guard_normal_k", 4)),
        )  # wall normals

        tree = cKDTree(graph_feature_lookup_inside["xy_inside"])  # nearest inside lookup
        _, idx = tree.query(xy_guard.astype(np.float32), k=1)  # project features
        dwg = graph_feature_lookup_inside["d_w_i"][idx].astype(np.float32)  # projected dw
        sing = graph_feature_lookup_inside["s_in_i"][idx].astype(np.float32)  # projected sin
        soutg = graph_feature_lookup_inside["s_out_i"][idx].astype(np.float32)  # projected sout

        return _build_pointwise_constraint(
            nodes=wall_guard_nodes,
            invar={
                "x": xy_guard[:, 0:1].astype(np.float32),
                "y": xy_guard[:, 1:2].astype(np.float32),
                "dw": dwg,
                "sin": sing,
                "sout": soutg,
                "n_x": nx,
                "n_y": ny,
            },
            outvar={
                "wall_normal_velocity": np.zeros((xy_guard.shape[0], 1), np.float32),
            },
            batch_size=min(int(getattr(cfg.training, "wall_guard_batch_size", 8192)), xy_guard.shape[0]),
            shuffle=True,
            lambda_weights={
                "wall_normal_velocity": float(getattr(cfg.training, weight_name, 1.0)) * s,
            },
        )  # wall no-penetration
    global_guard = _one_constraint(
        xy_guard=xy_wall_guard,
        weight_name="w_wall_guard_normal",
    )  # global wall band

    sep_guard = _one_constraint(
        xy_guard=xy_wall_guard_sep,
        weight_name="w_wall_guard_separator_normal",
    )  # separator-focused wall band

    return global_guard, sep_guard  # all wall guards


# -----------------------------
# Main
# -----------------------------
@hydra.main(version_base=None, config_path="conf", config_name="partner_v4_config")
def run(cfg: DictConfig) -> None:
    cfg = cfg.flow if "flow" in cfg else cfg
    cad_path = resolve_cad_path()  # required CAD source
    geom_json_path = cad_to_geometry_json(
        cad_path=cad_path,
        output_dir=Path.cwd(),
        res=int(getattr(cfg.problem, "pcs_res", 512)),
        strip_w=int(getattr(cfg.problem, "pcs_strip_w", 10)),
        white_thr=int(getattr(cfg.problem, "pcs_white_thr", 250)),
    )  # convert CAD to geometry json via PCS
    geom_path = str(geom_json_path)  # geometry json path
    print(f"[INFO] using CAD: {cad_path}")  # cad info
    print(f"[INFO] generated geometry json: {geom_path}")  # geometry info
    x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm, inside_raw_xy, wall_raw_xy, inlet_raw_obj, outlet_raw_obj, geom_obj = _load_points_from_geom(geom_path)  # geometry
    if inlet_xy is None or outlet_xy is None:
        raise ValueError("Geometry JSON must contain inlet and outlet.")  # require endpoints

    xmin, xmax, ymin, ymax = norm  # bounds
    Lx = max(float(xmax - xmin), 1.0e-12)  # physical x size
    Ly = max(float(ymax - ymin), 1.0e-12)  # physical y size

    xy_wall = np.concatenate([x_w, y_w], axis=1).astype(np.float32)  # wall coords
    xy_inside = np.concatenate([x_i, y_i], axis=1).astype(np.float32)  # inside coords

    d_w_i = compute_wall_distance_feature(xy_inside, xy_wall)  # inside wall distance
    d_w_w = np.zeros((xy_wall.shape[0], 1), dtype=np.float32)  # exact wall distance on walls

    spacing_tree = NearestNeighbors(n_neighbors=min(2, xy_inside.shape[0]), algorithm="ball_tree")  # spacing estimator
    spacing_tree.fit(xy_inside)  # fit
    spacing_dists, _ = spacing_tree.kneighbors(xy_inside[: min(512, xy_inside.shape[0])])  # sample spacing
    spacing = float(np.median(spacing_dists[:, -1])) if spacing_dists.shape[1] > 1 else 1.0e-3  # spacing

    # -------------------------------------------------
    # Build inside graph and geodesics
    # -------------------------------------------------
    graph_mode = str(getattr(cfg.training, "flow_graph_mode", "pixel")).strip().lower()  # graph mode
    progress_knn_k = int(getattr(cfg.training, "progress_knn_k", 8))  # graph knn
    progress_max_edge_len = float(getattr(cfg.training, "progress_max_edge_len", 2.0 * spacing))  # graph prune threshold
    graph_connectivity = int(getattr(cfg.training, "flow_graph_connectivity", 8))  # pixel connectivity

    graph = build_inside_graph(
        xy_inside=xy_inside,
        inside_raw_xy=inside_raw_xy,
        norm=norm,
        mode=graph_mode,
        knn_k=progress_knn_k,
        max_edge_len=progress_max_edge_len,
        pixel_connectivity=graph_connectivity,
    )  # inside graph

    geo_info_in = compute_geodesic_info_on_graph(
        graph=graph,
        xy_inside=xy_inside,
        source_xy=inlet_xy,
    )  # geodesic from inlet

    geo_info_out = compute_geodesic_info_on_graph(
        graph=graph,
        xy_inside=xy_inside,
        source_xy=outlet_xy,
    )  # geodesic from outlet

    s_in_i = geo_info_in["s_geo"].astype(np.float32)  # inlet progress
    s_out_i = geo_info_out["s_geo"].astype(np.float32)  # outlet progress

    s_in_w = project_inside_feature_to_wall(
        xy_wall=xy_wall,
        xy_inside=xy_inside,
        feat_inside=s_in_i,
    )  # wall inlet progress

    s_out_w = project_inside_feature_to_wall(
        xy_wall=xy_wall,
        xy_inside=xy_inside,
        feat_inside=s_out_i,
    )  # wall outlet progress

    # -------------------------------------------------
    # Inlet / outlet adaptive patches
    # -------------------------------------------------
    inlet_raw_xy = np.asarray([[float(inlet_raw_obj["x"]), float(inlet_raw_obj["y"])]], np.float32)  # inlet raw center
    outlet_raw_xy = np.asarray([[float(outlet_raw_obj["x"]), float(outlet_raw_obj["y"])]], np.float32)  # outlet raw center

    ensure_patch_has_min_points._inside_raw = inside_raw_xy  # cache raw inside points
    ensure_patch_has_min_points._half_height_px = int(getattr(cfg.bc, "inlet_half_height_px", 1))  # inlet half height
    ensure_patch_has_min_points._min_run_per_strip = int(getattr(cfg.bc, "min_run_per_strip", 2))  # min segment size
    ensure_patch_has_min_points._enforce_connected_chain = bool(getattr(cfg.bc, "enforce_connected_chain", True))  # chain flag

    ensure_patch_has_min_points._center_raw = inlet_raw_xy  # inlet raw center
    ensure_patch_has_min_points._direction = str(getattr(cfg.bc, "inlet_direction", "right"))  # inlet direction
    inlet_mask, inlet_r_used = ensure_patch_has_min_points(
        xy_inside=xy_inside,
        center_xy=inlet_xy,
        r0=float(cfg.bc.inlet_radius_norm),
        min_pts=int(getattr(cfg.bc, "inlet_width_px", getattr(cfg.bc, "min_patch_points", 10))),
        r_max=float(cfg.bc.max_patch_radius_norm),
        grow=float(cfg.bc.patch_growth_factor),
    )  # inlet adaptive patch

    ensure_patch_has_min_points._center_raw = outlet_raw_xy  # outlet raw center
    ensure_patch_has_min_points._direction = str(getattr(cfg.bc, "outlet_direction", "right"))  # outlet direction
    ensure_patch_has_min_points._half_height_px = int(getattr(cfg.bc, "outlet_half_height_px", getattr(cfg.bc, "inlet_half_height_px", 1)))  # outlet half height
    outlet_mask, outlet_r_used = ensure_patch_has_min_points(
        xy_inside=xy_inside,
        center_xy=outlet_xy,
        r0=float(cfg.bc.outlet_radius_norm),
        min_pts=int(getattr(cfg.bc, "outlet_width_px", getattr(cfg.bc, "min_patch_points", 10))),
        r_max=float(cfg.bc.max_patch_radius_norm),
        grow=float(cfg.bc.patch_growth_factor),
    )  # outlet adaptive patch

    print(f"[INFO] inlet points={int(np.sum(inlet_mask))}, inlet_r_used={inlet_r_used:.6f}")  # inlet info
    print(f"[INFO] outlet points={int(np.sum(outlet_mask))}, outlet_r_used={outlet_r_used:.6f}")  # outlet info
    print(f"[INFO] graph_mode={graph_mode}, graph_connectivity={graph_connectivity}, knn_k={progress_knn_k}, max_edge_len={progress_max_edge_len:.6f}")  # graph info

    # -------------------------------------------------
    # Importance-sampled PDE interior subset
    # -------------------------------------------------
    outlet_target_idx = int(geo_info_out["src"])  # nearest inside point to outlet
    pde_target_points = int(getattr(cfg.training, "flow_pde_points_target", xy_inside.shape[0]))  # subset size
    pde_idx = _build_weighted_flow_pde_indices(
        xy_inside=xy_inside,
        xy_wall=xy_wall,
        d_w_i=d_w_i,
        predecessors_in=geo_info_in["predecessors"],
        src_in=int(geo_info_in["src"]),
        outlet_target_idx=outlet_target_idx,
        target_points=pde_target_points,
        wall_boost=float(getattr(cfg.training, "flow_pde_wall_boost", 1.0)),
        wall_scale=float(getattr(cfg.training, "flow_pde_wall_scale", 0.02)),
        corridor_boost=float(getattr(cfg.training, "flow_pde_corridor_boost", 2.0)),
        corridor_radius=float(getattr(cfg.training, "flow_pde_corridor_radius", 0.05)),
        seed=int(getattr(cfg.training, "flow_pde_sampling_seed", 1234)),
    )  # PDE subset indices

    x_i_pde = x_i[pde_idx].astype(np.float32)  # sampled x
    y_i_pde = y_i[pde_idx].astype(np.float32)  # sampled y
    d_w_i_pde = d_w_i[pde_idx].astype(np.float32)  # sampled dw
    s_in_i_pde = s_in_i[pde_idx].astype(np.float32)  # sampled sin
    s_out_i_pde = s_out_i[pde_idx].astype(np.float32)  # sampled sout

    x_i_sorted, y_i_sorted, d_w_i_sorted, s_in_i_sorted, s_out_i_sorted, inside_order = sort_by_progress_chunked(
        x_i=x_i_pde,
        y_i=y_i_pde,
        d_w_i=d_w_i_pde,
        s_in_i=s_in_i_pde,
        s_out_i=s_out_i_pde,
        chunk_size=int(getattr(cfg.training, "curriculum_chunk_size", 8192)),
    )  # sorted PDE subset

    print(f"[INFO] PDE interior points used: {int(x_i_sorted.shape[0])} / {int(x_i.shape[0])}")  # PDE subset info

    # -------------------------------------------------
    # Wall guard point sets
    # -------------------------------------------------
    port_exclude_mask = np.zeros((xy_inside.shape[0],), dtype=bool)  # exclusion mask
    if bool(getattr(cfg.training, "wall_guard_exclude_ports", True)):
        port_exclude_mask |= inlet_mask  # exclude inlet patch
        port_exclude_mask |= outlet_mask  # exclude outlet patch

    xy_wall_guard = _build_wall_guard_points(
        xy_inside=xy_inside,
        xy_wall=xy_wall,
        radius=float(getattr(cfg.training, "wall_guard_radius", 0.02)),
        target_points=int(getattr(cfg.training, "wall_guard_points", 4000)),
        seed=int(getattr(cfg.training, "wall_guard_seed", 1234)),
        exclude_mask=port_exclude_mask,
    ) if bool(getattr(cfg.training, "wall_guard_enabled", True)) else np.zeros((0, 2), dtype=np.float32)  # global wall band

    if bool(getattr(cfg.training, "wall_guard_separator_enabled", True)):
        y_mid = 0.5 * (float(inlet_xy[0, 1]) + float(outlet_xy[0, 1]))  # separator center y
        y_half = 0.5 * abs(float(outlet_xy[0, 1]) - float(inlet_xy[0, 1])) * float(getattr(cfg.training, "wall_guard_separator_span_factor", 0.85))  # separator y half-width
        y_half = max(y_half, 1.0e-3)  # guard
        xy_wall_guard_sep = _build_wall_guard_points(
            xy_inside=xy_inside,
            xy_wall=xy_wall,
            radius=float(getattr(cfg.training, "wall_guard_separator_radius", float(getattr(cfg.training, "wall_guard_radius", 0.02)))),
            target_points=int(getattr(cfg.training, "wall_guard_separator_points", 3000)),
            seed=int(getattr(cfg.training, "wall_guard_seed", 1234)) + 17,
            x_max=float(getattr(cfg.training, "wall_guard_separator_x_max", 0.42)),
            y_min=(y_mid - y_half),
            y_max=(y_mid + y_half),
            exclude_mask=port_exclude_mask,
        )  # separator-focused wall band
    else:
        xy_wall_guard_sep = np.zeros((0, 2), dtype=np.float32)  # no separator band

    print(f"[INFO] wall guard points={int(xy_wall_guard.shape[0])}, separator guard points={int(xy_wall_guard_sep.shape[0])}")  # guard info

    # -------------------------------------------------
    # Build geodesic-based initial guess
    # -------------------------------------------------
    init_inside_fields = compute_initial_flow_guess(
        xy_inside=xy_inside,
        xy_wall=xy_wall,
        inlet_p=float(getattr(cfg.bc, "inlet_p", 1.0)),
        inlet_u=float(cfg.bc.inlet_u),
        inlet_v=float(cfg.bc.inlet_v),
        geo_info_in=geo_info_in,
        geo_info_out=geo_info_out,
        velocity_scale=float(getattr(cfg.init_guess, "velocity_scale", 1.0)),
        velocity_power=float(getattr(cfg.init_guess, "velocity_power", 1.0)),
        pressure_power=float(getattr(cfg.init_guess, "pressure_power", 1.0)),
        pressure_drop_guess=float(getattr(cfg.init_guess, "pressure_drop_guess", 0.0)),
    )  # pseudo field

    if bool(getattr(cfg.init_guess, "save_random_paths_plot", False)):
        path_plot = str(Path(geom_path).with_suffix("")) + "_geodesic_paths.png"  # path figure
        plot_random_geodesic_paths(
            xy_inside=xy_inside,
            inlet_xy=inlet_xy,
            outlet_xy=outlet_xy,
            predecessors=init_inside_fields["predecessors"],
            src=int(init_inside_fields["src"]),
            save_path=path_plot,
            num_paths=int(getattr(cfg.init_guess, "num_random_paths", 12)),
            seed=int(getattr(cfg.init_guess, "random_seed", 123)),
        )  # save path plot
        print(f"[OK] wrote geodesic path plot to: {path_plot}")  # log

    if bool(getattr(cfg.init_guess, "save_into_geometry_json", True)):
        save_init_fields_into_geometry_json(
            geom_obj=geom_obj,
            geom_json_path=geom_path,
            xy_inside=xy_inside,
            init_fields_inside=init_inside_fields,
            norm=norm,
        )  # save init fields into new geometry json
        print(f"[OK] wrote init_fields into geometry json copy from: {geom_path}")  # log

    if bool(getattr(cfg.init_guess, "save_init_preview_json", True)):
        init_preview_json = str(Path(geom_path).with_suffix("")) + "_init_guess_flow.json"  # preview json
        init_point_type = np.full((xy_inside.shape[0],), 2, dtype=np.int32)  # inside only type
        init_fields_to_save = {
            "u": init_inside_fields["u"].reshape(-1),
            "v": init_inside_fields["v"].reshape(-1),
            "p": init_inside_fields["p"].reshape(-1),
            "dw": init_inside_fields["dw"].reshape(-1),
            "geo_in": init_inside_fields["geo_in"].reshape(-1),
            "s_in": init_inside_fields["s_in"].reshape(-1),
            "s_out": init_inside_fields["s_out"].reshape(-1),
        }  # init preview fields
        _write_multi_field_inference_json(
            init_preview_json,
            xy_norm=xy_inside,
            fields_dict=init_fields_to_save,
            norm=norm,
            point_type=init_point_type,
        )  # save preview
        print(f"[OK] wrote initial guess preview json to: {init_preview_json}")  # log

    if bool(getattr(cfg.init_guess, "save_init_preview_plot", False)):
        init_preview_plot = str(Path(geom_path).with_suffix("")) + "_init_guess_flow.png"  # preview plot
        _plot_fields(
            xy_inside,
            {
                "u": init_inside_fields["u"].reshape(-1),
                "v": init_inside_fields["v"].reshape(-1),
                "p": init_inside_fields["p"].reshape(-1),
            },
            init_preview_plot,
            title_prefix="Initial geodesic guess",
        )  # plot preview
        print(f"[OK] wrote initial guess preview plot to: {init_preview_plot}")  # log

    # -------------------------------------------------
    # Network + nodes
    # -------------------------------------------------
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("dw"), Key("sin"), Key("sout")],  # topology-aware model inputs
        output_keys=[Key("u"), Key("v"), Key("p")],  # outputs
        layer_size=int(cfg.flow_model.hidden_size),  # hidden width
        nr_layers=int(cfg.flow_model.hidden_layers),  # hidden depth
        activation_fn=get_activation(cfg.flow_model.activation),  # activation
    )  # model

    start_time = time.time()  # timer

    geo_guidance_pde = FlowTrajectoryGuidance2D(
        speed_eps=float(getattr(cfg.training, "geo_guidance_speed_eps", 1.0e-6))
    )  # flow guidance PDE

    wall_guard_pde = WallNormalNoPenetration2D(
        eq_name="wall_normal_velocity"
    )  # wall no-penetration PDE

    def make_flow_nodes(nu_value: float):
        ns = SteadyNavierStokes2DScaledFDSAGE(
            rho=float(cfg.physics.rho),
            nu=float(nu_value),
            Lx=Lx,
            Ly=Ly,
            flow_net=flow_net,
            dx=1.0e-3,
            dy=1.0e-3,
        )  # steady NS PDE (SAGE drop-in: FD stencil + analytic adjoint)
        flow_nodes = ns.make_nodes() + [flow_net.make_node(name="flow_network")]  # standard nodes
        geo_nodes = geo_guidance_pde.make_nodes() + [flow_net.make_node(name="flow_network")]  # guidance nodes
        wall_guard_nodes = wall_guard_pde.make_nodes() + [flow_net.make_node(name="flow_network")]  # wall guard nodes
        return flow_nodes, geo_nodes, wall_guard_nodes  # return nodes

    graph_feature_lookup_inside = {
        "xy_inside": xy_inside.astype(np.float32),
        "d_w_i": d_w_i.astype(np.float32),
        "s_in_i": s_in_i.astype(np.float32),
        "s_out_i": s_out_i.astype(np.float32),
    }  # feature lookup for projected guard features

    # -------------------------------------------------
    # stage -1: pseudo-field warmup
    # -------------------------------------------------
    if bool(getattr(cfg.training, "use_init_field_warmup", True)):
        flow_nodes_init, _, _ = make_flow_nodes(float(cfg.training.nu_schedule[0]))  # nodes for init stage

        init_constraint = build_pseudo_init_constraint(
            flow_nodes=flow_nodes_init,
            cfg=cfg,
            x_i=x_i,
            y_i=y_i,
            d_w_i=d_w_i,
            s_in_i=s_in_i,
            s_out_i=s_out_i,
            init_inside_fields=init_inside_fields,
            batch_size_key="flow_init_batch_size",
            scale=float(getattr(cfg.training, "init_field_warmup_scale", 1.0)),
        )  # pseudo-label warmup constraint

        domain_init = Domain()  # init domain
        domain_init.add_constraint(init_constraint, "init_field_fit")  # add init fit

        cfg_init = make_stage_cfg(
            cfg,
            "stage_m1_init_guess_warmup",
            int(getattr(cfg.training, "k_flow_init", 2000)),
            init_dir="",
        )  # init stage cfg
        slv_init = Solver(cfg_init, domain_init)  # init solver
        _disable_recording(slv_init)  # disable logging
        slv_init.solve()  # warmup to pseudo-field
        prev_dir = str(Path(cfg_init.network_dir))  # checkpoint dir after init
    else:
        prev_dir = ""  # no init checkpoint

    # -------------------------------------------------
    # stage 0: BC warmup only
    # -------------------------------------------------
    flow_nodes0, geo_nodes0, wall_guard_nodes0 = make_flow_nodes(float(cfg.training.nu_schedule[0]))  # stage0 nodes

    pde_c0, wall_c0, inlet_c0, outlet_c0, inlet_p_c0 = build_primary_constraints(
        flow_nodes=flow_nodes0,
        cfg=cfg,
        x_w=x_w,
        y_w=y_w,
        d_w_w=d_w_w,
        s_in_w=s_in_w,
        s_out_w=s_out_w,
        x_i_sorted=x_i_sorted,
        y_i_sorted=y_i_sorted,
        d_w_i_sorted=d_w_i_sorted,
        s_in_i_sorted=s_in_i_sorted,
        s_out_i_sorted=s_out_i_sorted,
        x_i=x_i,
        y_i=y_i,
        d_w_i=d_w_i,
        s_in_i=s_in_i,
        s_out_i=s_out_i,
        inlet_mask=inlet_mask,
        outlet_mask=outlet_mask,
        weight_scale=float(getattr(cfg.training, "bc_stage_weight_scale", 1.0)),
    )  # primary constraints stage0

    soft_init_bc = build_pseudo_init_constraint(
        flow_nodes=flow_nodes0,
        cfg=cfg,
        x_i=x_i,
        y_i=y_i,
        d_w_i=d_w_i,
        s_in_i=s_in_i,
        s_out_i=s_out_i,
        init_inside_fields=init_inside_fields,
        batch_size_key="flow_soft_init_batch_size",
        scale=float(getattr(cfg.training, "soft_init_bc_scale", 1.0)),
    ) if bool(getattr(cfg.training, "use_soft_init_during_bc", True)) else None  # BC soft init

    geo_dir_bc, geo_parallel_bc, geo_speed_bc = build_geo_guidance_constraints(
        geo_nodes=geo_nodes0,
        cfg=cfg,
        x_i=x_i,
        y_i=y_i,
        d_w_i=d_w_i,
        s_in_i=s_in_i,
        s_out_i=s_out_i,
        init_inside_fields=init_inside_fields,
        inlet_mask=inlet_mask,
        outlet_mask=outlet_mask,
        scale=float(getattr(cfg.training, "geo_guidance_bc_scale", 1.0)),
    ) if bool(getattr(cfg.training, "use_geo_guidance_during_bc", True)) else (None, None, None)  # BC geo guidance

    wall_guard_bc, wall_guard_sep_bc = build_wall_guard_constraints(
        wall_guard_nodes=wall_guard_nodes0,
        cfg=cfg,
        xy_wall_guard=xy_wall_guard,
        xy_wall_guard_sep=xy_wall_guard_sep,
        xy_wall=xy_wall,
        graph_feature_lookup_inside=graph_feature_lookup_inside,
        scale=float(getattr(cfg.training, "wall_guard_scale", 1.0)),
    )  # BC wall guard constraints

    domain_bc = Domain()  # BC domain
    domain_bc.add_constraint(wall_c0, "wall_noslip")  # wall
    domain_bc.add_constraint(inlet_c0, "inlet_vel")  # inlet
    if outlet_c0 is not None:
        domain_bc.add_constraint(outlet_c0, "outlet_p")  # optional outlet
    if inlet_p_c0 is not None:
        domain_bc.add_constraint(inlet_p_c0, "inlet_p_anchor")  # optional inlet p
    if soft_init_bc is not None:
        domain_bc.add_constraint(soft_init_bc, "soft_init_bc")  # keep init alive in BC stage
    if geo_dir_bc is not None:
        domain_bc.add_constraint(geo_dir_bc, "geo_dir_bc")  # BC direction guidance
    if geo_parallel_bc is not None:
        domain_bc.add_constraint(geo_parallel_bc, "geo_parallel_bc")  # BC parallel guidance
    if geo_speed_bc is not None:
        domain_bc.add_constraint(geo_speed_bc, "geo_speed_bc")  # BC speed guidance
    if wall_guard_bc is not None:
        domain_bc.add_constraint(wall_guard_bc, "wall_guard_bc")  # global wall guard
    if wall_guard_sep_bc is not None:
        domain_bc.add_constraint(wall_guard_sep_bc, "wall_guard_sep_bc")  # separator wall guard

    cfg_bc = make_stage_cfg(cfg, "stage_00_bc_warmup", int(cfg.training.k_flow_bc), init_dir=prev_dir)  # stage cfg
    slv_bc = Solver(cfg_bc, domain_bc)  # solver
    _disable_recording(slv_bc)  # disable logs
    slv_bc.solve()  # train warmup
    prev_dir = str(Path(cfg_bc.network_dir))  # previous checkpoint dir

    # -------------------------------------------------
    # continuation stages
    # -------------------------------------------------
    n_stages = len(cfg.training.nu_schedule)  # stage count
    for stage_idx, nu_stage in enumerate(cfg.training.nu_schedule):
        flow_nodes, geo_nodes, wall_guard_nodes = make_flow_nodes(float(nu_stage))  # stage nodes

        pde_weight_scale = float(getattr(cfg.training, "pde_stage_weight_scale", 1.0))  # stage pde weight scale

        flow_pde_constraint, wall_constraint, inlet_constraint, outlet_constraint, inlet_p_constraint = build_primary_constraints(
            flow_nodes=flow_nodes,
            cfg=cfg,
            x_w=x_w,
            y_w=y_w,
            d_w_w=d_w_w,
            s_in_w=s_in_w,
            s_out_w=s_out_w,
            x_i_sorted=x_i_sorted,
            y_i_sorted=y_i_sorted,
            d_w_i_sorted=d_w_i_sorted,
            s_in_i_sorted=s_in_i_sorted,
            s_out_i_sorted=s_out_i_sorted,
            x_i=x_i,
            y_i=y_i,
            d_w_i=d_w_i,
            s_in_i=s_in_i,
            s_out_i=s_out_i,
            inlet_mask=inlet_mask,
            outlet_mask=outlet_mask,
            weight_scale=pde_weight_scale,
        )  # stage primary constraints

        soft_init_scale = _linear_stage_scale(
            stage_idx=stage_idx,
            num_stages=n_stages,
            start_scale=float(getattr(cfg.training, "soft_init_start_scale", 0.5)),
            end_scale=float(getattr(cfg.training, "soft_init_end_scale", 0.05)),
        )  # decay soft init across stages

        geo_guidance_scale = _linear_stage_scale(
            stage_idx=stage_idx,
            num_stages=n_stages,
            start_scale=float(getattr(cfg.training, "geo_guidance_start_scale", 1.0)),
            end_scale=float(getattr(cfg.training, "geo_guidance_end_scale", 0.1)),
        )  # decay geo guidance across stages

        init_soft = build_pseudo_init_constraint(
            flow_nodes=flow_nodes,
            cfg=cfg,
            x_i=x_i,
            y_i=y_i,
            d_w_i=d_w_i,
            s_in_i=s_in_i,
            s_out_i=s_out_i,
            init_inside_fields=init_inside_fields,
            batch_size_key="flow_soft_init_batch_size",
            scale=soft_init_scale,
        ) if bool(getattr(cfg.training, "use_soft_init_constraint", True)) else None  # soft init during PDE stages

        geo_dir_c, geo_parallel_c, geo_speed_c = build_geo_guidance_constraints(
            geo_nodes=geo_nodes,
            cfg=cfg,
            x_i=x_i,
            y_i=y_i,
            d_w_i=d_w_i,
            s_in_i=s_in_i,
            s_out_i=s_out_i,
            init_inside_fields=init_inside_fields,
            inlet_mask=inlet_mask,
            outlet_mask=outlet_mask,
            scale=geo_guidance_scale,
        ) if bool(getattr(cfg.training, "use_geo_direction_guidance", True)) else (None, None, None)  # stage geo guidance

        wall_guard_c, wall_guard_sep_c = build_wall_guard_constraints(
            wall_guard_nodes=wall_guard_nodes,
            cfg=cfg,
            xy_wall_guard=xy_wall_guard,
            xy_wall_guard_sep=xy_wall_guard_sep,
            xy_wall=xy_wall,
            graph_feature_lookup_inside=graph_feature_lookup_inside,
            scale=float(getattr(cfg.training, "wall_guard_scale", 1.0)),
        )  # stage wall guard constraints

        domain_full = Domain()  # stage domain
        domain_full.add_constraint(flow_pde_constraint, "flow_pde")  # pde
        domain_full.add_constraint(wall_constraint, "wall_noslip")  # wall
        domain_full.add_constraint(inlet_constraint, "inlet_vel")  # inlet
        if outlet_constraint is not None:
            domain_full.add_constraint(outlet_constraint, "outlet_p")  # optional outlet

        if inlet_p_constraint is not None:
            if bool(getattr(cfg.bc, "use_inlet_pressure_anchor_last_stage_only", False)):
                if stage_idx == len(cfg.training.nu_schedule) - 1:
                    domain_full.add_constraint(inlet_p_constraint, "inlet_p_anchor")  # only final stage
            else:
                domain_full.add_constraint(inlet_p_constraint, "inlet_p_anchor")  # all stages

        if init_soft is not None:
            domain_full.add_constraint(init_soft, "soft_init")  # keep init alive during PDE stages
        if geo_dir_c is not None:
            domain_full.add_constraint(geo_dir_c, "geo_dir")  # direction guidance
        if geo_parallel_c is not None:
            domain_full.add_constraint(geo_parallel_c, "geo_parallel")  # parallel guidance
        if geo_speed_c is not None:
            domain_full.add_constraint(geo_speed_c, "geo_speed")  # speed guidance
        if wall_guard_c is not None:
            domain_full.add_constraint(wall_guard_c, "wall_guard")  # global wall guard
        if wall_guard_sep_c is not None:
            domain_full.add_constraint(wall_guard_sep_c, "wall_guard_sep")  # separator wall guard

        stage_steps = int(cfg.training.k_flow_per_stage[stage_idx])  # current stage steps
        stage_name = f"stage_{stage_idx + 1:02d}_nu_{nu_stage:.2e}".replace("+", "")  # stage name
        cfg_stage = make_stage_cfg(cfg, stage_name, stage_steps, init_dir=prev_dir)  # stage cfg

        print(
            f"[INFO] training stage {stage_idx + 1}/{len(cfg.training.nu_schedule)} "
            f"with nu={nu_stage:.4e} for {stage_steps} steps | "
            f"soft_init_scale={soft_init_scale:.4f} geo_guidance_scale={geo_guidance_scale:.4f}"
        )  # stage log

        slv = Solver(cfg_stage, domain_full)  # stage solver
        _disable_recording(slv)  # disable logs
        slv.solve()  # train stage

        prev_dir = str(Path(cfg_stage.network_dir))  # update checkpoint dir

    end_time = time.time()  # end timer

    # -------------------------------------------------
    # inference
    # -------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device
    flow_net = flow_net.to(device).eval()  # model eval

    xy_all = np.vstack([xy_wall, xy_inside]).astype(np.float32)  # all points
    point_type = np.concatenate(
        [
            np.full((xy_wall.shape[0],), 1, np.int32),  # wall labels
            np.full((xy_inside.shape[0],), 2, np.int32),  # inside labels
        ],
        axis=0,
    )  # labels

    d_w_all = np.vstack([d_w_w, d_w_i]).astype(np.float32)  # all wall distances
    s_in_all = np.vstack([s_in_w, s_in_i]).astype(np.float32)  # all inlet progress
    s_out_all = np.vstack([s_out_w, s_out_i]).astype(np.float32)  # all outlet progress

    N = xy_all.shape[0]  # point count
    batch = 65536  # batch size
    flow_fields = {
        "u": np.zeros((N,), dtype=np.float32),  # u field
        "v": np.zeros((N,), dtype=np.float32),  # v field
        "p": np.zeros((N,), dtype=np.float32),  # p field
    }  # output arrays

    with torch.no_grad():
        for s0 in range(0, N, batch):
            s1 = min(s0 + batch, N)  # batch end
            out = flow_net(
                {
                    "x": torch.from_numpy(xy_all[s0:s1, 0:1]).to(device),  # x batch
                    "y": torch.from_numpy(xy_all[s0:s1, 1:2]).to(device),  # y batch
                    "dw": torch.from_numpy(d_w_all[s0:s1]).to(device),  # dw batch
                    "sin": torch.from_numpy(s_in_all[s0:s1]).to(device),  # sin batch
                    "sout": torch.from_numpy(s_out_all[s0:s1]).to(device),  # sout batch
                }
            )  # forward
            flow_fields["u"][s0:s1] = out["u"].detach().cpu().numpy().reshape(-1)  # save u
            flow_fields["v"][s0:s1] = out["v"].detach().cpu().numpy().reshape(-1)  # save v
            flow_fields["p"][s0:s1] = out["p"].detach().cpu().numpy().reshape(-1)  # save p

    nw = xy_wall.shape[0]  # num wall points
    flow_fields["u"][:nw] = 0.0  # exact no-slip u
    flow_fields["v"][:nw] = 0.0  # exact no-slip v

    stem = str(Path(geom_path).with_suffix(""))  # output stem
    out_json_path = stem + "_pred_flow_steady.json"  # json output
    _write_multi_field_inference_json(
        out_json_path,
        xy_norm=xy_all,
        fields_dict=flow_fields,
        norm=norm,
        point_type=point_type,
    )  # save json

    if bool(getattr(cfg.inference, "save_plot", False)):
        out_plot_path = stem + "_pred_flow_steady.png"  # png output
        _plot_fields(xy_all, flow_fields, out_plot_path, title_prefix="Steady flow")  # save plot
        print(f"[OK] wrote preview plot to: {out_plot_path}")  # success

    total_time = (end_time - start_time) / 60.0  # minutes
    print(f"[OK] wrote flow results to: {out_json_path}")  # success
    print(f"Total elapsed training time: {total_time:.4f} minutes")  # time


if __name__ == "__main__":
    run()  # entry point
