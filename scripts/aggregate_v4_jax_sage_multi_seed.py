"""Aggregate multi-seed JAX-SAGE task-quality metrics.

Loads each per-seed JAX-SAGE checkpoint (pkl), evaluates task-quality
metrics (NS PDE residual RMSE per component, temp IC T@t=0, final temp
loss) on the same 20 000 interior points used for the single-seed
eval, then reports mean ± std across seeds.

Run after ``scripts/run_v4_jax_sage_multi_seed.sh`` completes::

    source env/bin/activate
    python scripts/aggregate_v4_jax_sage_multi_seed.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse loaders from the 3-way eval.
from evaluate_v4_jax_sage_task_quality import (  # noqa: E402
    FlowMLPPlain, _flax_flow_to_torch, _flax_temp_to_torch,
    _flow_pde_autograd, _temp_pde_autograd, _rmse,
    GEOM_JSON, INLET_T, RHO, NU_LAST_STAGE, D_TEMP, Q_TEMP,
)
from partner_v4_flow import (  # noqa: E402
    _load_points_from_geom, compute_wall_distance_feature,
    build_inside_graph, compute_geodesic_info_on_graph,
)
from partner_v4_temp import TemperatureNet, _forward_temperature  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402


SEEDS = [1234, 2345, 3456, 4567, 5678]


def _seed_dir(seed: int) -> Path:
    """Return the output dir for a given seed."""
    if seed == 1234:
        return ROOT / "results" / "partner_v4_jax_sage"
    return ROOT / "results" / f"partner_v4_jax_sage_seed{seed}"


def _flow_ckpt(seed: int) -> Path:
    return _seed_dir(seed) / "flow" / "stage_03_nu_1.00e-03" / "flow_network.pkl"


def _temp_ckpt(seed: int) -> Path:
    return _seed_dir(seed) / "temp" / "temperature_net.pkl"


def _load_flow(seed: int, device) -> Optional[torch.nn.Module]:
    ckpt = _flow_ckpt(seed)
    if not ckpt.exists():
        return None
    with open(ckpt, "rb") as f:
        params = pickle.load(f)
    return _flax_flow_to_torch(params, device)


def _load_temp(seed: int, device) -> Optional[torch.nn.Module]:
    ckpt = _temp_ckpt(seed)
    if not ckpt.exists():
        return None
    with open(ckpt, "rb") as f:
        params = pickle.load(f)
    return _flax_temp_to_torch(params, device)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[multi-seed-agg] Device: {device}")
    print(f"[multi-seed-agg] Seeds to aggregate: {SEEDS}")

    # --- Geometry + interior points (fixed across seeds) ---
    (x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm,
     inside_raw_xy, _wall_raw_xy, _inlet_raw_obj, _outlet_raw_obj, _geom_obj
     ) = _load_points_from_geom(str(GEOM_JSON))
    xmin, xmax, ymin, ymax = norm
    Lx = max(float(xmax - xmin), 1e-12)
    Ly = max(float(ymax - ymin), 1e-12)
    inv_Lx = 1.0 / Lx; inv_Ly = 1.0 / Ly

    xy_inside = np.concatenate([x_i, y_i], axis=1).astype(np.float32)
    xy_wall = np.concatenate([x_w, y_w], axis=1).astype(np.float32)
    d_w_i = compute_wall_distance_feature(xy_inside, xy_wall)

    spacing_tree = NearestNeighbors(n_neighbors=min(2, xy_inside.shape[0]), algorithm="ball_tree")
    spacing_tree.fit(xy_inside)
    spacing_d, _ = spacing_tree.kneighbors(xy_inside[:min(512, xy_inside.shape[0])])
    spacing = float(np.median(spacing_d[:, -1])) if spacing_d.shape[1] > 1 else 1.0e-3
    graph = build_inside_graph(xy_inside=xy_inside, inside_raw_xy=inside_raw_xy,
                                norm=norm, mode="pixel", knn_k=8,
                                max_edge_len=2.0 * spacing,
                                pixel_connectivity=8)
    geo_in = compute_geodesic_info_on_graph(graph, xy_inside, inlet_xy)
    geo_out = compute_geodesic_info_on_graph(graph, xy_inside, outlet_xy)
    s_in_i = geo_in["s_geo"].astype(np.float32)
    s_out_i = geo_out["s_geo"].astype(np.float32)

    rng = np.random.default_rng(42)
    N_sample = min(20000, xy_inside.shape[0])
    idx = rng.choice(xy_inside.shape[0], size=N_sample, replace=False)
    x = torch.from_numpy(xy_inside[idx, 0:1]).to(device)
    y = torch.from_numpy(xy_inside[idx, 1:2]).to(device)
    dw = torch.from_numpy(d_w_i[idx]).to(device)
    s_in = torch.from_numpy(s_in_i[idx]).to(device)
    s_out = torch.from_numpy(s_out_i[idx]).to(device)

    # --- Per-seed eval ---
    per_seed = {}
    for s in SEEDS:
        flow_net = _load_flow(s, device)
        if flow_net is None:
            print(f"[multi-seed-agg] seed {s}: flow ckpt MISSING, skipping")
            continue
        res = _flow_pde_autograd(flow_net, x, y, dw, s_in, s_out,
                                  nu=NU_LAST_STAGE, rho=RHO,
                                  inv_Lx=inv_Lx, inv_Ly=inv_Ly)
        row = {
            "cont": _rmse(res["cont"]),
            "mom_x": _rmse(res["mom_x"]),
            "mom_y": _rmse(res["mom_y"]),
            "u_rms": float(torch.sqrt(torch.mean(res["u"] ** 2)).item()),
            "u_max": float(torch.abs(res["u"]).max().item()),
            "p_rms": float(torch.sqrt(torch.mean(res["p"] ** 2)).item()),
        }

        temp_net = _load_temp(s, device)
        if temp_net is not None:
            t0 = torch.zeros_like(x)
            u_flow = res["u"]; v_flow = res["v"]
            with torch.no_grad():
                T_ic = _forward_temperature(temp_net, x, y, t0, u_flow, v_flow)
            row["temp_ic_mean"] = float(T_ic.mean().item())
            # Temp residual at t=5, 20, 35
            for tval in (5.0, 20.0, 35.0):
                t = torch.full_like(x, tval)
                tr = _temp_pde_autograd(temp_net, x, y, t, u_flow, v_flow, D_TEMP, Q_TEMP)
                row[f"temp_resid_t{int(tval)}"] = _rmse(tr["resid"])

        per_seed[s] = row
        print(f"[multi-seed-agg] seed {s}: cont={row['cont']:.3e} "
              f"mom_x={row['mom_x']:.3e} mom_y={row['mom_y']:.3e} "
              f"temp_ic={row.get('temp_ic_mean', 'N/A')}")

    # --- Aggregation ---
    print("\n" + "=" * 80)
    print("JAX-SAGE multi-seed aggregate (mean ± std across completed seeds)")
    print("=" * 80)
    metrics = ["cont", "mom_x", "mom_y", "u_rms", "u_max", "p_rms",
               "temp_ic_mean", "temp_resid_t5", "temp_resid_t20", "temp_resid_t35"]
    for k in metrics:
        vals = [row[k] for row in per_seed.values() if k in row]
        if not vals:
            continue
        vals = np.asarray(vals)
        mean = float(vals.mean()); std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {k:<22} mean={mean:.4e}  std={std:.4e}  (n={len(vals)})")

    # --- Reference values for comparison ---
    print("\nReference (baseline / PyTorch SAGE, seed 1234 only):")
    print("  cont:      baseline 2.459e-04 | PyTorch SAGE 2.275e-04")
    print("  mom_x:     baseline 3.022e-05 | PyTorch SAGE 2.431e-05")
    print("  mom_y:     baseline 4.783e-06 | PyTorch SAGE 3.865e-06")
    print("  temp IC:   baseline 59.52      | PyTorch SAGE 60.39")

    # Save JSON for downstream use.
    out_json = ROOT / "results" / "partner_v4_jax_sage" / "multi_seed_aggregate.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds_completed": list(per_seed.keys()),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "n_interior_points": N_sample,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[multi-seed-agg] wrote {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
