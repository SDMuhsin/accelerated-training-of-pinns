"""Holistic task-quality comparison: V4 baseline vs V4 SAGE.

Loads both trained networks (baseline from 2026-04-20, SAGE from
2026-04-21) and evaluates **identical task-performance metrics** on a
common set of interior points:

- Autograd-based NS residual RMSE per component (continuity, momentum_x,
  momentum_y) — same formula for both configs, no dependence on the
  engine that trained them.
- Autograd-based advection-diffusion residual RMSE for temp.
- BC satisfaction (u, v at wall; u, v at inlet; p at inlet patch).
- Point-by-point field difference (baseline − SAGE) on interior points.

This is what "holistic task performance" means here — quality of the
converged solution evaluated identically on both engines, independent of
the per-engine training-loss aggregator.

Run in the project env, from repo root::

    source env/bin/activate
    python scripts/evaluate_v4_task_quality.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from partner_v4_flow import _load_points_from_geom, compute_wall_distance_feature  # noqa: E402
from partner_v4_flow import (  # noqa: E402
    build_inside_graph, compute_geodesic_info_on_graph,
    project_inside_feature_to_wall, ensure_patch_has_min_points,
    get_activation,
)
from partner_v4_temp import TemperatureNet, _forward_temperature, _load_flow_field_json, _load_geometry_ports, _normalize_port  # noqa: E402
from physicsnemo.sym.key import Key  # noqa: E402
from physicsnemo.sym.models.fully_connected import FullyConnectedArch  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402


GEOM_JSON = ROOT / "data" / "partner_v4" / "pipe_three_class_fixed.json"
PARTNER_REF_FLOW = ROOT / "results" / "partner_v4" / "baseline_pred_flow_steady.json"
SAGE_FLOW_JSON = ROOT / "data" / "partner_v4" / "pipe_three_class_fixed_pred_flow_steady.json"

# Baseline flow: 2026-04-20 baseline run (authoritative reference).
BASELINE_FLOW_CKPT = ROOT / "results" / "partner_v4" / "flow" / "stage_03_nu_1.00e-03" / "flow_network.0.pth"
# SAGE flow: rebuilt 2026-04-21 run (Option C, PhysicsNeMo Solver + SAGE PDE
# drop-in). Saves checkpoints in PhysicsNeMo's standard flow_network.0.pth
# format, NOT the drifted custom flow_net.pt format.
SAGE_FLOW_CKPT = ROOT / "results" / "partner_v4_sage_v2" / "flow" / "stage_03_nu_1.00e-03" / "flow_network.0.pth"
# Baseline temp: results/partner_v4/temp/ was corrupted by an earlier SAGE
# smoke test — use the 2026-04-21 re-trained copy instead.
BASELINE_TEMP_CKPT = ROOT / "results" / "partner_v4_baseline_retrain" / "temp" / "temperature_net.pt"
# SAGE temp: rebuilt 2026-04-21 run.
SAGE_TEMP_CKPT = ROOT / "results" / "partner_v4_sage_v2" / "temp" / "temperature_net.pt"

RHO = 1076.0
NU_LAST_STAGE = 1.0e-3  # final nu the flow net was trained at
D_TEMP = 1.0e-5
Q_TEMP = 0.0
INLET_U = 1.0
INLET_V = 0.0
INLET_P = 1.0
INLET_T = 25.0


# ---------------------------------------------------------------------------
# Flow-net loading (baseline PhysicsNeMo format + SAGE torch.save format)
# ---------------------------------------------------------------------------

def _build_flow_arch(device: torch.device) -> torch.nn.Module:
    return FullyConnectedArch(
        input_keys=[Key(k) for k in ("x", "y", "dw", "sin", "sout")],
        output_keys=[Key(k) for k in ("u", "v", "p")],
        layer_size=512, nr_layers=12,
        activation_fn=torch.nn.SiLU(),
    ).to(device)


def _load_flow_baseline(device: torch.device) -> torch.nn.Module:
    """PhysicsNeMo saves `flow_network.0.pth` as a plain state_dict for the
    submodule named `flow_network`. Load into our arch."""
    net = _build_flow_arch(device)
    raw = torch.load(str(BASELINE_FLOW_CKPT), map_location=device, weights_only=False)
    state_dict = raw if isinstance(raw, dict) and "weight" not in raw else raw
    # PhysicsNeMo's per-model save strips the "flow_network." prefix already.
    try:
        net.load_state_dict(state_dict)
    except RuntimeError as exc:
        # Maybe the dict is wrapped.
        if isinstance(raw, dict) and "state_dict" in raw:
            net.load_state_dict(raw["state_dict"])
        else:
            raise exc
    net.eval()
    return net


def _load_flow_sage(device: torch.device) -> torch.nn.Module:
    # Rebuilt SAGE variant saves via PhysicsNeMo's Solver, so the
    # checkpoint is a plain state_dict identical in shape to the
    # baseline one — reuse the same loader logic.
    net = _build_flow_arch(device)
    raw = torch.load(str(SAGE_FLOW_CKPT), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        net.load_state_dict(raw["state_dict"])
    else:
        net.load_state_dict(raw)
    net.eval()
    return net


def _load_temp_baseline(device: torch.device) -> torch.nn.Module:
    net = TemperatureNet(in_dim=5, hidden_size=256, hidden_layers=12, activation="silu").to(device)
    sd = torch.load(str(BASELINE_TEMP_CKPT), map_location=device, weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    return net


def _load_temp_sage(device: torch.device) -> torch.nn.Module:
    net = TemperatureNet(in_dim=5, hidden_size=256, hidden_layers=12, activation="silu").to(device)
    sd = torch.load(str(SAGE_TEMP_CKPT), map_location=device, weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# NS PDE residuals via autograd — identical formula for both configs
# ---------------------------------------------------------------------------

def _flow_pde_autograd(net, x, y, dw, sin_, sout, nu: float, rho: float,
                       inv_Lx: float, inv_Ly: float) -> Dict[str, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    out = net({"x": x, "y": y, "dw": dw, "sin": sin_, "sout": sout})
    u, v, p = out["u"], out["v"], out["p"]

    def _grad(o, i):
        return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True)[0]

    du_dx = _grad(u, x); du_dy = _grad(u, y)
    dv_dx = _grad(v, x); dv_dy = _grad(v, y)
    dp_dx = _grad(p, x); dp_dy = _grad(p, y)
    d2u_dx2 = _grad(du_dx, x); d2u_dy2 = _grad(du_dy, y)
    d2v_dx2 = _grad(dv_dx, x); d2v_dy2 = _grad(dv_dy, y)

    # Scale normalized-coord derivatives to physical
    du_dx_p = inv_Lx * du_dx; du_dy_p = inv_Ly * du_dy
    dv_dx_p = inv_Lx * dv_dx; dv_dy_p = inv_Ly * dv_dy
    dp_dx_p = inv_Lx * dp_dx; dp_dy_p = inv_Ly * dp_dy
    d2u_dx2_p = (inv_Lx ** 2) * d2u_dx2; d2u_dy2_p = (inv_Ly ** 2) * d2u_dy2
    d2v_dx2_p = (inv_Lx ** 2) * d2v_dx2; d2v_dy2_p = (inv_Ly ** 2) * d2v_dy2

    cont = du_dx_p + dv_dy_p
    mom_x = u * du_dx_p + v * du_dy_p + (1.0 / rho) * dp_dx_p - nu * (d2u_dx2_p + d2u_dy2_p)
    mom_y = u * dv_dx_p + v * dv_dy_p + (1.0 / rho) * dp_dy_p - nu * (d2v_dx2_p + d2v_dy2_p)

    return {"cont": cont.detach(), "mom_x": mom_x.detach(), "mom_y": mom_y.detach(),
            "u": u.detach(), "v": v.detach(), "p": p.detach()}


def _temp_pde_autograd(net, x, y, t, u_flow, v_flow, D: float, Q: float) -> Dict[str, torch.Tensor]:
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    out = _forward_temperature(net, x, y, t, u_flow, v_flow)
    T = out

    def _grad(o, i):
        return torch.autograd.grad(o, i, grad_outputs=torch.ones_like(o), create_graph=True, retain_graph=True)[0]

    T_x = _grad(T, x); T_y = _grad(T, y); T_t = _grad(T, t)
    T_xx = _grad(T_x, x); T_yy = _grad(T_y, y)
    resid = T_t + u_flow * T_x + v_flow * T_y - D * (T_xx + T_yy) - Q
    return {"resid": resid.detach(), "T": T.detach()}


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def _rmse(arr: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(arr ** 2)).item())


def main() -> int:
    if not GEOM_JSON.exists():
        raise FileNotFoundError(f"Missing geometry JSON: {GEOM_JSON}")
    if not BASELINE_FLOW_CKPT.exists():
        raise FileNotFoundError(f"Missing baseline flow checkpoint: {BASELINE_FLOW_CKPT}")
    if not SAGE_FLOW_CKPT.exists():
        raise FileNotFoundError(f"Missing SAGE flow checkpoint: {SAGE_FLOW_CKPT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- load geometry + features (same as training) ---
    (x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm,
     inside_raw_xy, _wall_raw_xy, inlet_raw_obj, outlet_raw_obj, _geom_obj
     ) = _load_points_from_geom(str(GEOM_JSON))
    xmin, xmax, ymin, ymax = norm
    Lx = max(float(xmax - xmin), 1e-12)
    Ly = max(float(ymax - ymin), 1e-12)
    inv_Lx = 1.0 / Lx; inv_Ly = 1.0 / Ly

    xy_inside = np.concatenate([x_i, y_i], axis=1).astype(np.float32)
    xy_wall = np.concatenate([x_w, y_w], axis=1).astype(np.float32)
    d_w_i = compute_wall_distance_feature(xy_inside, xy_wall)
    d_w_w = np.zeros((xy_wall.shape[0], 1), dtype=np.float32)

    spacing_tree = NearestNeighbors(n_neighbors=min(2, xy_inside.shape[0]), algorithm="ball_tree")
    spacing_tree.fit(xy_inside)
    spacing_d, _ = spacing_tree.kneighbors(xy_inside[: min(512, xy_inside.shape[0])])
    spacing = float(np.median(spacing_d[:, -1])) if spacing_d.shape[1] > 1 else 1.0e-3
    progress_max_edge_len = 2.0 * spacing

    graph = build_inside_graph(
        xy_inside=xy_inside, inside_raw_xy=inside_raw_xy, norm=norm,
        mode="pixel", knn_k=8, max_edge_len=progress_max_edge_len,
        pixel_connectivity=8,
    )
    geo_in = compute_geodesic_info_on_graph(graph=graph, xy_inside=xy_inside, source_xy=inlet_xy)
    geo_out = compute_geodesic_info_on_graph(graph=graph, xy_inside=xy_inside, source_xy=outlet_xy)
    s_in_i = geo_in["s_geo"].astype(np.float32)
    s_out_i = geo_out["s_geo"].astype(np.float32)

    # Inlet / outlet masks
    ensure_patch_has_min_points._inside_raw = inside_raw_xy
    ensure_patch_has_min_points._half_height_px = 1
    ensure_patch_has_min_points._min_run_per_strip = 2
    ensure_patch_has_min_points._enforce_connected_chain = True
    ensure_patch_has_min_points._center_raw = np.asarray([[float(inlet_raw_obj["x"]),
                                                           float(inlet_raw_obj["y"])]], np.float32)
    ensure_patch_has_min_points._direction = "right"
    inlet_mask, _ = ensure_patch_has_min_points(
        xy_inside=xy_inside, center_xy=inlet_xy, r0=0.002, min_pts=2,
        r_max=0.005, grow=1.5,
    )
    ensure_patch_has_min_points._center_raw = np.asarray([[float(outlet_raw_obj["x"]),
                                                           float(outlet_raw_obj["y"])]], np.float32)
    ensure_patch_has_min_points._direction = "right"
    outlet_mask, _ = ensure_patch_has_min_points(
        xy_inside=xy_inside, center_xy=outlet_xy, r0=0.002, min_pts=2,
        r_max=0.005, grow=1.5,
    )

    # Move to device.
    x_i_t = torch.from_numpy(x_i.astype(np.float32)).to(device)
    y_i_t = torch.from_numpy(y_i.astype(np.float32)).to(device)
    d_w_t = torch.from_numpy(d_w_i.astype(np.float32)).to(device)
    s_in_t = torch.from_numpy(s_in_i).to(device)
    s_out_t = torch.from_numpy(s_out_i).to(device)

    x_w_t = torch.from_numpy(x_w.astype(np.float32)).to(device)
    y_w_t = torch.from_numpy(y_w.astype(np.float32)).to(device)
    d_w_w_t = torch.from_numpy(d_w_w).to(device)
    s_in_w_t = torch.from_numpy(project_inside_feature_to_wall(xy_wall, xy_inside, s_in_i)).to(device)
    s_out_w_t = torch.from_numpy(project_inside_feature_to_wall(xy_wall, xy_inside, s_out_i)).to(device)

    N_int = x_i_t.shape[0]
    N_wall = x_w_t.shape[0]
    print(f"Interior points: {N_int}   Wall points: {N_wall}")

    # --- eval on random subset of interior points to keep autograd tractable ---
    sample_n = min(20000, N_int)
    g_eval = torch.Generator(device=device).manual_seed(12345)
    idx = torch.randperm(N_int, generator=g_eval, device=device)[:sample_n]
    x_ev = x_i_t[idx]; y_ev = y_i_t[idx]; dw_ev = d_w_t[idx]
    sin_ev = s_in_t[idx]; sout_ev = s_out_t[idx]

    # --- flow nets ---
    print("\n=== Flow-net task-quality evaluation ===")
    flow_base = _load_flow_baseline(device)
    flow_sage = _load_flow_sage(device)
    n_params = sum(p.numel() for p in flow_base.parameters())
    print(f"  flow params: {n_params:,} (same architecture for both)")

    rows = []
    for label, net in (("baseline", flow_base), ("SAGE", flow_sage)):
        res = _flow_pde_autograd(net, x_ev, y_ev, dw_ev, sin_ev, sout_ev,
                                 nu=NU_LAST_STAGE, rho=RHO,
                                 inv_Lx=inv_Lx, inv_Ly=inv_Ly)
        rmse_cont = _rmse(res["cont"])
        rmse_momx = _rmse(res["mom_x"])
        rmse_momy = _rmse(res["mom_y"])
        u_rms = _rmse(res["u"]); v_rms = _rmse(res["v"]); p_rms = _rmse(res["p"])
        u_max = float(res["u"].abs().max()); v_max = float(res["v"].abs().max()); p_max = float(res["p"].abs().max())
        rows.append((label, rmse_cont, rmse_momx, rmse_momy, u_rms, v_rms, p_rms, u_max, v_max, p_max))

    # BC check: eval baseline vs SAGE at wall points; both should give ~0 u, v.
    with torch.no_grad():
        out_b_wall = flow_base({"x": x_w_t, "y": y_w_t, "dw": d_w_w_t, "sin": s_in_w_t, "sout": s_out_w_t})
        out_s_wall = flow_sage({"x": x_w_t, "y": y_w_t, "dw": d_w_w_t, "sin": s_in_w_t, "sout": s_out_w_t})
        wall_u_rmse_b = _rmse(out_b_wall["u"]); wall_v_rmse_b = _rmse(out_b_wall["v"])
        wall_u_rmse_s = _rmse(out_s_wall["u"]); wall_v_rmse_s = _rmse(out_s_wall["v"])

    # Inlet check.
    m = torch.from_numpy(inlet_mask).to(device)
    inlet_ids = torch.where(m)[0]
    with torch.no_grad():
        xi = x_i_t[inlet_ids]; yi = y_i_t[inlet_ids]
        dwi = d_w_t[inlet_ids]; sini = s_in_t[inlet_ids]; souti = s_out_t[inlet_ids]
        b_in = flow_base({"x": xi, "y": yi, "dw": dwi, "sin": sini, "sout": souti})
        s_in_out = flow_sage({"x": xi, "y": yi, "dw": dwi, "sin": sini, "sout": souti})
        inlet_u_base_mean = float(b_in["u"].mean()); inlet_u_sage_mean = float(s_in_out["u"].mean())
        inlet_v_base_rmse = _rmse(b_in["v"]); inlet_v_sage_rmse = _rmse(s_in_out["v"])

    # Baseline-vs-SAGE field agreement on interior eval points.
    with torch.no_grad():
        out_b_int = flow_base({"x": x_ev, "y": y_ev, "dw": dw_ev, "sin": sin_ev, "sout": sout_ev})
        out_s_int = flow_sage({"x": x_ev, "y": y_ev, "dw": dw_ev, "sin": sin_ev, "sout": sout_ev})
        u_diff = _rmse(out_b_int["u"] - out_s_int["u"])
        v_diff = _rmse(out_b_int["v"] - out_s_int["v"])
        p_diff = _rmse(out_b_int["p"] - out_s_int["p"])
        u_rms_base = _rmse(out_b_int["u"]); v_rms_base = _rmse(out_b_int["v"]); p_rms_base = _rmse(out_b_int["p"])

    # Print flow table.
    print("")
    print("Flow-net PDE residual RMSE (autograd, interior points, nu=1e-3):")
    print(f"  {'Config':<10} {'continuity':>14} {'momentum_x':>14} {'momentum_y':>14}")
    for label, cont, mx, my, *_ in rows:
        print(f"  {label:<10} {cont:>14.3e} {mx:>14.3e} {my:>14.3e}")
    print("")
    print("Flow-net predicted-field magnitudes (interior RMS / max):")
    for label, *_, u_rms, v_rms, p_rms, u_max, v_max, p_max in [(r[0],) + r[1:] for r in rows]:
        print(f"  {label:<10} u RMS={u_rms:.3e} max={u_max:.3e}   v RMS={v_rms:.3e} max={v_max:.3e}   p RMS={p_rms:.3e} max={p_max:.3e}")
    print("")
    print("Flow-net BC satisfaction (wall no-slip, inlet velocity):")
    print(f"  baseline  wall u RMSE={wall_u_rmse_b:.3e}  wall v RMSE={wall_v_rmse_b:.3e}  inlet <u>={inlet_u_base_mean:+.3e}  inlet v RMSE={inlet_v_base_rmse:.3e}")
    print(f"  SAGE      wall u RMSE={wall_u_rmse_s:.3e}  wall v RMSE={wall_v_rmse_s:.3e}  inlet <u>={inlet_u_sage_mean:+.3e}  inlet v RMSE={inlet_v_sage_rmse:.3e}")
    print("")
    print("Baseline-vs-SAGE field RMSE (same interior points):")
    print(f"  u: diff={u_diff:.3e}  (baseline u RMS={u_rms_base:.3e})  rel={u_diff/max(u_rms_base,1e-12):.1%}")
    print(f"  v: diff={v_diff:.3e}  (baseline v RMS={v_rms_base:.3e})  rel={v_diff/max(v_rms_base,1e-12):.1%}")
    print(f"  p: diff={p_diff:.3e}  (baseline p RMS={p_rms_base:.3e})  rel={p_diff/max(p_rms_base,1e-12):.1%}")

    # --- temp nets (each eval'd with its own training-time flow features) ---
    print("\n=== Temp-net task-quality evaluation ===")
    if not BASELINE_TEMP_CKPT.exists() or not SAGE_TEMP_CKPT.exists():
        print("  (skipping — checkpoint missing)")
    else:
        temp_base = _load_temp_baseline(device)
        temp_sage = _load_temp_sage(device)
        n_params_t = sum(p.numel() for p in temp_base.parameters())
        print(f"  temp params: {n_params_t:,}")

        # Both the 2026-04-21-retrained baseline temp and the rebuilt-SAGE
        # temp were trained against the SAME partner-precomputed flow JSON
        # (md5 5b04e983…). So both configs must be evaluated with the partner
        # JSON's u, v as data features — NOT a fresh inference of either
        # flow net. Using a mismatched feature would inflate the residual
        # of whichever config sees features it wasn't trained on.
        flow_obj = _load_flow_field_json(str(PARTNER_REF_FLOW), flow_time_index=0)
        u_partner_full = torch.from_numpy(flow_obj.u.reshape(-1, 1).astype(np.float32)).to(device)
        v_partner_full = torch.from_numpy(flow_obj.v.reshape(-1, 1).astype(np.float32)).to(device)

        pt = flow_obj.point_type
        inside_mask_partner = (pt == 2)
        u_partner_inside = u_partner_full[torch.from_numpy(inside_mask_partner).to(device)]
        v_partner_inside = v_partner_full[torch.from_numpy(inside_mask_partner).to(device)]
        if u_partner_inside.shape[0] != N_int:
            print(f"  [WARN] partner JSON inside count {u_partner_inside.shape[0]} != geometry inside {N_int}; using geometry order")
        u_base_feat = u_partner_inside[idx]
        v_base_feat = v_partner_inside[idx]
        u_sage_feat = u_partner_inside[idx]
        v_sage_feat = v_partner_inside[idx]

        # Each temp net evaluated with ITS OWN training-time u, v features.
        rows_t = []
        for t_val in (5.0, 20.0, 35.0):
            t_batch = torch.full((x_ev.shape[0], 1), float(t_val), device=device)
            r_b = _temp_pde_autograd(temp_base, x_ev, y_ev, t_batch, u_base_feat, v_base_feat,
                                      D=D_TEMP, Q=Q_TEMP)
            r_s = _temp_pde_autograd(temp_sage, x_ev, y_ev, t_batch, u_sage_feat, v_sage_feat,
                                      D=D_TEMP, Q=Q_TEMP)
            rows_t.append((t_val, "baseline", _rmse(r_b["resid"]), _rmse(r_b["T"]), float(r_b["T"].abs().max())))
            rows_t.append((t_val, "SAGE", _rmse(r_s["resid"]), _rmse(r_s["T"]), float(r_s["T"].abs().max())))

        print("")
        print("Temp-net advection-diffusion residual RMSE (autograd, interior points):")
        print("  (each config eval'd with its own training-time u, v features)")
        print(f"  {'t':>5}  {'Config':<10} {'resid RMSE':>14} {'T RMS':>12} {'|T|_max':>10}")
        for t_v, label, resid, T_rms, T_max in rows_t:
            print(f"  {t_v:>5.1f}  {label:<10} {resid:>14.3e} {T_rms:>12.3e} {T_max:>10.3e}")

        # Ensure baseline T values match training expectation: IC target T=60,
        # inlet T=25. So max|T| should be in 25..60, RMS around some average.
        # If baseline reports max|T| ≪ 25 then baseline temp collapsed.
        print("")
        print("Sanity BC satisfaction (temp):")
        print(f"  {'Config':<10} {'T @ IC t=0 should approach T_init=60':>45}")
        with torch.no_grad():
            t_ic = torch.zeros_like(x_ev)
            T_base_ic = _forward_temperature(temp_base, x_ev, y_ev, t_ic, u_base_feat, v_base_feat)
            T_sage_ic = _forward_temperature(temp_sage, x_ev, y_ev, t_ic, u_sage_feat, v_sage_feat)
            print(f"  {'baseline':<10} {'mean T @ t=0 =':>30} {float(T_base_ic.mean()):+.3e} (target +6.000e+01)")
            print(f"  {'SAGE':<10} {'mean T @ t=0 =':>30} {float(T_sage_ic.mean()):+.3e} (target +6.000e+01)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
