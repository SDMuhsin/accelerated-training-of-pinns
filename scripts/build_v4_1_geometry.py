"""Build the V4.1 3D extruded point-cloud geometry JSON.

Reads the partner V4 2D JSON (`data/partner_v4/pipe_three_class_fixed.json`
by default), extrudes it to 3D, builds inlet/outlet geodesics, generates
an initial flow guess, and writes a self-contained 3D geometry JSON.

Usage:
    python scripts/build_v4_1_geometry.py                    # full-res
    python scripts/build_v4_1_geometry.py --downsample-xy 4  # smoke
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make the sibling `src/` directory importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent  # project root
_SRC_DIR = _REPO_ROOT / "src"  # src dir
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))  # allow `import partner_v4_1_geometry`

from partner_v4_1_geometry import (  # noqa: E402 (import after sys.path edit)
    load_2d_geometry_json,
    build_3d_point_cloud,
    compute_wall_distance_3d,
    build_inside_graph_3d,
    compute_geodesic_info_3d,
    compute_initial_flow_guess_3d,
    save_geometry_json_3d,
)


def _parse_args() -> argparse.Namespace:
    """CLI parser for the geometry builder."""
    p = argparse.ArgumentParser(
        description="Build the V4.1 3D geometry JSON from the V4 2D JSON.",
    )  # parser
    p.add_argument(
        "--input",
        type=str,
        default="data/partner_v4/pipe_three_class_fixed.json",
        help="Path to the 2D geometry JSON.",
    )  # input
    p.add_argument(
        "--output",
        type=str,
        default="data/partner_v4_1/pipe_three_class_3d.json",
        help="Path to write the 3D geometry JSON.",
    )  # output
    p.add_argument(
        "--z-aspect",
        type=float,
        default=0.10,
        help="Zmax / Lx ratio (default 0.10).",
    )  # z aspect
    p.add_argument(
        "--z-slices",
        type=int,
        default=9,
        help="Total z-levels including both caps (default 9).",
    )  # z slices
    p.add_argument(
        "--downsample-xy",
        type=int,
        default=1,
        help="If >1, keep every Nth 2D point (for smoke tests).",
    )  # xy downsample
    p.add_argument(
        "--inlet-u",
        type=float,
        default=1.0,
        help="Inlet u component (default 1.0).",
    )  # inlet u
    p.add_argument(
        "--inlet-v",
        type=float,
        default=0.0,
        help="Inlet v component (default 0.0).",
    )  # inlet v
    p.add_argument(
        "--inlet-p",
        type=float,
        default=1.0,
        help="Inlet pressure anchor (default 1.0).",
    )  # inlet p
    p.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for downstream sampling (default 1234).",
    )  # seed
    return p.parse_args()  # parsed namespace


def _compute_geodesic_with_fallback(
    pc: dict,
    target_xyz: np.ndarray,
    label: str,
    cov_threshold: float = 0.99,
) -> tuple:
    """Compute geodesic with automatic voxel->kNN fallback if coverage < 99%.

    Returns (geo_info, adj_used, mode_used).
    """

    # First attempt: voxel 6-connectivity
    print(f"[geom] building voxel graph for {label}...")
    t0 = time.time()  # timing
    adj = build_inside_graph_3d(
        pc["xyz_inside"],
        pc["xyz_inside_raw"],
        mode="voxel",
    )  # voxel adjacency
    print(
        f"[geom] voxel graph: nnz={adj.nnz}, built in {time.time() - t0:.2f}s"
    )

    gi = compute_geodesic_info_3d(adj, pc["xyz_inside"], target_xyz)  # geodesic
    N_in = int(pc["xyz_inside"].shape[0])  # interior count
    reached = int(np.sum(gi["predecessors"] != -9999))  # reached count
    coverage = float(reached) / float(max(N_in, 1))  # ratio

    if coverage >= cov_threshold:
        print(f"[geom] {label} voxel coverage: {coverage:.4f}  OK")
        return gi, adj, "voxel"  # done

    print(
        f"[geom] WARNING: {label} voxel coverage {coverage:.4f} < {cov_threshold:.2f}; "
        f"falling back to kNN graph"
    )
    t0 = time.time()  # reset
    adj = build_inside_graph_3d(
        pc["xyz_inside"],
        pc["xyz_inside_raw"],
        mode="knn",
        knn_k=14,
        max_edge_len=0.05,
    )  # kNN fallback graph
    print(
        f"[geom] kNN graph: nnz={adj.nnz}, built in {time.time() - t0:.2f}s"
    )

    gi = compute_geodesic_info_3d(adj, pc["xyz_inside"], target_xyz)  # redo geodesic
    reached = int(np.sum(gi["predecessors"] != -9999))
    coverage = float(reached) / float(max(N_in, 1))
    print(f"[geom] {label} kNN coverage: {coverage:.4f}")
    if coverage < cov_threshold:
        print(
            f"[geom] WARNING: {label} coverage still below {cov_threshold:.2f} "
            f"after kNN fallback"
        )
    return gi, adj, "knn"  # return with fallback adj


def main():
    args = _parse_args()  # parse CLI

    # Reproducibility (we do not currently sample but respect the flag).
    _ = np.random.default_rng(int(args.seed))  # seed rng

    in_path = Path(args.input)  # input path
    out_path = Path(args.output)  # output path
    out_path.parent.mkdir(parents=True, exist_ok=True)  # ensure output dir

    print(f"[geom] input  : {in_path}")  # log
    print(f"[geom] output : {out_path}")
    print(
        f"[geom] params : z_aspect={args.z_aspect}, z_slices={args.z_slices}, "
        f"downsample_xy={args.downsample_xy}, inlet=(u={args.inlet_u}, "
        f"v={args.inlet_v}, p={args.inlet_p}), seed={args.seed}"
    )

    # -------------------
    # Load + optional downsample
    # -------------------
    t0_all = time.time()  # overall timing
    t0 = time.time()
    obj_2d = load_2d_geometry_json(str(in_path))  # parse 2D
    print(f"[geom] loaded 2D JSON in {time.time() - t0:.2f}s, "
          f"points={len(obj_2d['points'])}")

    if int(args.downsample_xy) > 1:
        ds = int(args.downsample_xy)  # factor
        obj_in = dict(obj_2d)  # shallow copy (keeps shared refs)
        obj_in["points"] = obj_2d["points"][::ds]  # slice
        print(
            f"[geom] downsampled 2D points: {len(obj_2d['points'])} -> "
            f"{len(obj_in['points'])} (every {ds}th point)"
        )
    else:
        obj_in = obj_2d  # full resolution

    # -------------------
    # Build 3D point cloud
    # -------------------
    t0 = time.time()
    pc = build_3d_point_cloud(
        obj_in,
        z_aspect=float(args.z_aspect),
        z_slices=int(args.z_slices),
    )  # extrude
    print(f"[geom] built 3D cloud in {time.time() - t0:.2f}s")

    N_in = int(pc["xyz_inside"].shape[0])  # interior count
    N_w = int(pc["xyz_wall"].shape[0])  # wall count
    N_side = int(np.sum(pc["class_wall"] == 0))  # side wall count
    N_bot = int(np.sum(pc["class_wall"] == 1))  # bottom cap count
    N_top = int(np.sum(pc["class_wall"] == 2))  # top cap count
    N_il = int(pc["xyz_inlet"].shape[0])  # inlet patch count
    N_ol = int(pc["xyz_outlet"].shape[0])  # outlet patch count

    print("[geom] point counts:")
    print(f"  interior (fluid) : {N_in}")
    print(f"  wall (total)     : {N_w}")
    print(f"    side           : {N_side}")
    print(f"    bottom cap     : {N_bot}")
    print(f"    top cap        : {N_top}")
    print(f"  inlet patch      : {N_il}")
    print(f"  outlet patch     : {N_ol}")

    # -------------------
    # Wall-normal sanity
    # -------------------
    n_mag = np.linalg.norm(pc["n_wall"], axis=1)  # per-point magnitude
    n_err = float(np.max(np.abs(n_mag - 1.0)))  # unit-norm deviation
    print(f"[geom] wall normals: max |n|-1| = {n_err:.3e}")
    if n_err > 1.0e-5:
        print("[geom] WARNING: wall normals not unit within 1e-5")
    side_mask = pc["class_wall"] == 0  # side mask
    if np.any(side_mask):
        side_nz = float(np.max(np.abs(pc["n_wall"][side_mask, 2])))
        print(f"[geom] side-wall max |nz| = {side_nz:.3e} (should be 0)")
    bot_mask = pc["class_wall"] == 1  # bottom cap
    if np.any(bot_mask):
        bot_nz = float(np.mean(pc["n_wall"][bot_mask, 2]))
        print(f"[geom] bottom-cap mean nz = {bot_nz:.3f} (should be -1)")
    top_mask = pc["class_wall"] == 2  # top cap
    if np.any(top_mask):
        top_nz = float(np.mean(pc["n_wall"][top_mask, 2]))
        print(f"[geom] top-cap mean nz = {top_nz:.3f} (should be +1)")

    # -------------------
    # Wall distance (for sanity + to feed init-guess)
    # -------------------
    t0 = time.time()
    dw = compute_wall_distance_3d(pc["xyz_inside"], pc["xyz_wall"])  # (N_in,1)
    print(
        f"[geom] wall distance: min={float(np.min(dw)):.4f}, "
        f"max={float(np.max(dw)):.4f}, mean={float(np.mean(dw)):.4f} "
        f"(computed in {time.time() - t0:.2f}s)"
    )

    # -------------------
    # Geodesics from inlet and outlet (with auto-fallback)
    # -------------------
    geo_info_in, adj_in, mode_in = _compute_geodesic_with_fallback(
        pc, pc["inlet_center_xyz"], label="inlet"
    )
    # Reuse same adj for outlet if it passed coverage; else rebuild via fallback.
    geo_info_out, adj_out, mode_out = _compute_geodesic_with_fallback(
        pc, pc["outlet_center_xyz"], label="outlet"
    )
    print(f"[geom] geodesic graph modes used: inlet={mode_in}, outlet={mode_out}")

    # -------------------
    # Initial flow guess
    # -------------------
    t0 = time.time()
    init_fields = compute_initial_flow_guess_3d(
        xyz_inside=pc["xyz_inside"],
        xyz_wall=pc["xyz_wall"],
        z_max=float(pc["z_aspect"]),
        inlet_u=float(args.inlet_u),
        inlet_v=float(args.inlet_v),
        inlet_p=float(args.inlet_p),
        geo_info_in=geo_info_in,
        geo_info_out=geo_info_out,
        adj=adj_in,
        velocity_scale=1.0,
        velocity_power=1.0,
        pressure_power=1.0,
        pressure_drop_guess=0.0,
    )  # initial guess
    print(f"[geom] initial flow guess in {time.time() - t0:.2f}s")
    # Summary ranges for sanity
    for k in ("u", "v", "w", "p", "dw", "s_in", "s_out"):
        arr = init_fields[k]  # fetch
        print(
            f"  init_fields[{k}]: shape={tuple(arr.shape)}, "
            f"min={float(np.min(arr)):.4f}, max={float(np.max(arr)):.4f}, "
            f"mean={float(np.mean(arr)):.4f}"
        )

    # -------------------
    # Write JSON
    # -------------------
    t0 = time.time()
    save_geometry_json_3d(
        str(out_path),
        pc,
        init_fields_inside=init_fields,
    )  # write
    size_bytes = out_path.stat().st_size  # output size
    print(
        f"[geom] wrote {out_path} ({size_bytes / 1024.0 / 1024.0:.1f} MB) "
        f"in {time.time() - t0:.2f}s"
    )
    print(f"[geom] total runtime: {time.time() - t0_all:.2f}s")
    print("[geom] done")


if __name__ == "__main__":
    main()
