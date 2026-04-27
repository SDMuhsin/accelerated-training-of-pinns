"""Three-way task-quality comparison: V4 baseline vs PyTorch SAGE vs JAX-SAGE.

Extends ``evaluate_v4_task_quality.py`` with a JAX-SAGE column. Loads
the Flax params (pickled pytree) from the JAX-SAGE run, transfers them
to an equivalently-shaped PyTorch ``FullyConnectedArch`` / raw
``TemperatureNet``, then reuses the baseline's autograd-based residual
formula so all three columns are evaluated identically on the same
20 000 interior points.

Run after all three training configs have produced checkpoints::

    source env/bin/activate
    python scripts/evaluate_v4_jax_sage_task_quality.py

If JAX-SAGE checkpoints are missing (training not yet complete), the
script reports what it can and skips the JAX-SAGE column.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the existing evaluator's loading / eval code.
from evaluate_v4_task_quality import (  # noqa: E402
    _build_flow_arch, _load_flow_baseline, _load_flow_sage,
    _load_temp_baseline, _load_temp_sage,
    _flow_pde_autograd, _temp_pde_autograd,
    _rmse,
    GEOM_JSON, INLET_U, INLET_V, INLET_P, INLET_T,
    RHO, NU_LAST_STAGE, D_TEMP, Q_TEMP,
)
from partner_v4_temp import (  # noqa: E402
    TemperatureNet, _forward_temperature, _load_flow_field_json,
    _load_geometry_ports, _normalize_port,
)
from partner_v4_flow import (  # noqa: E402
    _load_points_from_geom, compute_wall_distance_feature,
    build_inside_graph, compute_geodesic_info_on_graph,
    project_inside_feature_to_wall, ensure_patch_has_min_points,
    get_activation,
)
from sklearn.neighbors import NearestNeighbors  # noqa: E402


JAX_SAGE_FLOW_CKPT = ROOT / "results" / "partner_v4_jax_sage" / "flow" / "stage_03_nu_1.00e-03" / "flow_network.pkl"
JAX_SAGE_TEMP_CKPT = ROOT / "results" / "partner_v4_jax_sage" / "temp" / "temperature_net.pkl"


# ---------------------------------------------------------------------------
# Plain PyTorch MLP that mirrors FlowNetFlax exactly
# (so we can load Flax params + reuse the baseline's autograd residual eval
# without touching PhysicsNeMo's WeightNormLinear).
# ---------------------------------------------------------------------------
class FlowMLPPlain(torch.nn.Module):
    """Plain (no weight-norm) 12×512 SiLU MLP; inputs (x,y,dw,sin,sout),
    outputs (u,v,p). Structurally identical to ``FlowNetFlax`` so Flax
    Dense_i.kernel.T → Linear_i.weight, Dense_i.bias → Linear_i.bias
    is a 1:1 transfer."""

    def __init__(self, hidden_layers: int = 12, hidden_size: int = 512):
        super().__init__()
        layers = [torch.nn.Linear(5, hidden_size), torch.nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.SiLU())
        layers.append(torch.nn.Linear(hidden_size, 3))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, var):
        """Accept the same dict-in/dict-out contract baseline's
        ``_flow_pde_autograd`` uses."""
        x = var["x"]; y = var["y"]
        dw = var["dw"]; sin_ = var["sin"]; sout_ = var["sout"]
        inp = torch.cat([x, y, dw, sin_, sout_], dim=1)
        out = self.net(inp)
        return {"u": out[:, 0:1], "v": out[:, 1:2], "p": out[:, 2:3]}


def _flax_flow_to_torch(params_pytree, device: torch.device) -> torch.nn.Module:
    """Transfer Flax ``FlowNetFlax`` params into our plain ``FlowMLPPlain``.

    Flax stores the 13 Dense layers as ``params['params']['Dense_i']`` with
    `kernel` (in, out) and `bias` (out,). Our FlowMLPPlain has 13
    `torch.nn.Linear` layers in ``self.net`` (indices 0, 2, 4, ..., 24).
    """
    net = FlowMLPPlain(hidden_layers=12, hidden_size=512).to(device)
    p = params_pytree.get("params", params_pytree)
    dense_keys = sorted([k for k in p.keys() if k.startswith("Dense_")],
                        key=lambda k: int(k.split("_")[1]))
    assert len(dense_keys) == 13, f"Expected 13 Dense layers, got {len(dense_keys)}"
    torch_linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert len(torch_linears) == 13
    with torch.no_grad():
        for torch_lin, flax_key in zip(torch_linears, dense_keys):
            kernel = np.asarray(p[flax_key]["kernel"], dtype=np.float32)
            bias = np.asarray(p[flax_key]["bias"], dtype=np.float32)
            torch_lin.weight.copy_(torch.from_numpy(kernel.T).to(device))
            torch_lin.bias.copy_(torch.from_numpy(bias).to(device))
    net.eval()
    return net


def _flax_temp_to_torch(params_pytree, device: torch.device) -> torch.nn.Module:
    """Transfer Flax ``TempNetFlax`` params into the PyTorch ``TemperatureNet``."""
    net = TemperatureNet(in_dim=5, hidden_size=256, hidden_layers=12,
                         activation="silu").to(device)
    p = params_pytree.get("params", params_pytree)
    dense_keys = sorted([k for k in p.keys() if k.startswith("Dense_")],
                        key=lambda k: int(k.split("_")[1]))
    # TemperatureNet has 12 hidden (Linear + SiLU pairs) + 1 final Linear.
    # Its layers are `net.net[0, 2, 4, ..., 24]`. Let's just enumerate Linears.
    torch_linears: List[torch.nn.Linear] = [m for m in net.modules()
                                             if isinstance(m, torch.nn.Linear)]
    assert len(torch_linears) == 13
    assert len(dense_keys) == 13
    with torch.no_grad():
        for torch_lin, flax_key in zip(torch_linears, dense_keys):
            kernel = np.asarray(p[flax_key]["kernel"], dtype=np.float32)
            bias = np.asarray(p[flax_key]["bias"], dtype=np.float32)
            torch_lin.weight.copy_(torch.from_numpy(kernel.T).to(device))
            torch_lin.bias.copy_(torch.from_numpy(bias).to(device))
    net.eval()
    return net


def _load_flow_jax_sage(device: torch.device) -> Optional[torch.nn.Module]:
    if not JAX_SAGE_FLOW_CKPT.exists():
        print(f"[WARN] JAX-SAGE flow ckpt not found: {JAX_SAGE_FLOW_CKPT}")
        return None
    with open(JAX_SAGE_FLOW_CKPT, "rb") as f:
        params = pickle.load(f)
    return _flax_flow_to_torch(params, device)


def _load_temp_jax_sage(device: torch.device) -> Optional[torch.nn.Module]:
    if not JAX_SAGE_TEMP_CKPT.exists():
        print(f"[WARN] JAX-SAGE temp ckpt not found: {JAX_SAGE_TEMP_CKPT}")
        return None
    with open(JAX_SAGE_TEMP_CKPT, "rb") as f:
        params = pickle.load(f)
    return _flax_temp_to_torch(params, device)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-3way] Device: {device}")

    # --- geometry (same as training) ---
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

    spacing_tree = NearestNeighbors(n_neighbors=min(2, xy_inside.shape[0]), algorithm="ball_tree")
    spacing_tree.fit(xy_inside)
    spacing_d, _ = spacing_tree.kneighbors(xy_inside[:min(512, xy_inside.shape[0])])
    spacing = float(np.median(spacing_d[:, -1])) if spacing_d.shape[1] > 1 else 1.0e-3
    progress_max_edge_len = 2.0 * spacing

    graph = build_inside_graph(xy_inside=xy_inside, inside_raw_xy=inside_raw_xy,
                                norm=norm, mode="pixel", knn_k=8,
                                max_edge_len=progress_max_edge_len,
                                pixel_connectivity=8)
    geo_in = compute_geodesic_info_on_graph(graph, xy_inside, inlet_xy)
    geo_out = compute_geodesic_info_on_graph(graph, xy_inside, outlet_xy)
    s_in_i = geo_in["s_geo"].astype(np.float32)
    s_out_i = geo_out["s_geo"].astype(np.float32)

    # --- Sample 20 000 random interior points (fixed seed) ---
    rng = np.random.default_rng(42)
    N_sample = min(20000, xy_inside.shape[0])
    idx = rng.choice(xy_inside.shape[0], size=N_sample, replace=False)

    x = torch.from_numpy(xy_inside[idx, 0:1]).to(device)
    y = torch.from_numpy(xy_inside[idx, 1:2]).to(device)
    dw = torch.from_numpy(d_w_i[idx]).to(device)
    s_in = torch.from_numpy(s_in_i[idx]).to(device)
    s_out = torch.from_numpy(s_out_i[idx]).to(device)

    # --- Load all three flow nets ---
    print("[eval-3way] Loading baseline flow ...")
    net_base = _load_flow_baseline(device)
    print("[eval-3way] Loading PyTorch SAGE flow ...")
    net_sage = _load_flow_sage(device)
    print("[eval-3way] Loading JAX-SAGE flow ...")
    net_jax_sage = _load_flow_jax_sage(device)

    # --- Evaluate NS residuals ---
    def _eval_flow(net, label):
        res = _flow_pde_autograd(net, x, y, dw, s_in, s_out,
                                  nu=NU_LAST_STAGE, rho=RHO,
                                  inv_Lx=inv_Lx, inv_Ly=inv_Ly)
        return {
            "cont": _rmse(res["cont"]),
            "mom_x": _rmse(res["mom_x"]),
            "mom_y": _rmse(res["mom_y"]),
            "u_rms": float(torch.sqrt(torch.mean(res["u"] ** 2)).item()),
            "u_max": float(torch.abs(res["u"]).max().item()),
            "v_rms": float(torch.sqrt(torch.mean(res["v"] ** 2)).item()),
            "v_max": float(torch.abs(res["v"]).max().item()),
            "p_rms": float(torch.sqrt(torch.mean(res["p"] ** 2)).item()),
            "p_max": float(torch.abs(res["p"]).max().item()),
            "label": label,
        }

    cols = []
    cols.append(_eval_flow(net_base, "baseline"))
    cols.append(_eval_flow(net_sage, "PyTorch SAGE"))
    if net_jax_sage is not None:
        cols.append(_eval_flow(net_jax_sage, "JAX-SAGE"))

    # --- Print three-way comparison ---
    labels = [c["label"] for c in cols]
    print("\n" + "=" * 80)
    print("Flow NS PDE residual RMSE (autograd, final nu=1e-3, N=20k interior)")
    print("=" * 80)
    print(f"{'Metric':<20}" + "".join(f"{l:>20}" for l in labels))
    for k in ["cont", "mom_x", "mom_y"]:
        print(f"{k:<20}" + "".join(f"{c[k]:>20.3e}" for c in cols))
    print(f"{'':<20}" + "-" * (20 * len(cols)))
    for k in ["u_rms", "u_max", "v_rms", "v_max", "p_rms", "p_max"]:
        print(f"{k:<20}" + "".join(f"{c[k]:>20.3e}" for c in cols))

    # --- Temp evaluation ---
    print("\n[eval-3way] Loading baseline temp ...")
    temp_base = _load_temp_baseline(device)
    print("[eval-3way] Loading PyTorch SAGE temp ...")
    temp_sage = _load_temp_sage(device)
    print("[eval-3way] Loading JAX-SAGE temp ...")
    temp_jax_sage = _load_temp_jax_sage(device)

    # --- IC: T at t=0 ---
    t0 = torch.zeros_like(x)
    u_flow_base = _flow_pde_autograd(net_base, x, y, dw, s_in, s_out, NU_LAST_STAGE, RHO, inv_Lx, inv_Ly)
    u_flow = u_flow_base["u"]; v_flow = u_flow_base["v"]

    def _eval_temp_ic(net):
        if net is None:
            return None
        with torch.no_grad():
            T = _forward_temperature(net, x, y, t0, u_flow, v_flow)
        return float(T.mean().item())

    temp_ics = [
        ("baseline", _eval_temp_ic(temp_base)),
        ("PyTorch SAGE", _eval_temp_ic(temp_sage)),
    ]
    if temp_jax_sage is not None:
        temp_ics.append(("JAX-SAGE", _eval_temp_ic(temp_jax_sage)))

    # --- Temp residual at t=5, 20, 35 ---
    def _eval_temp_resid(net):
        if net is None:
            return {}
        out = {}
        for tval in (5.0, 20.0, 35.0):
            t = torch.full_like(x, tval)
            res = _temp_pde_autograd(net, x, y, t, u_flow, v_flow, D_TEMP, Q_TEMP)
            out[tval] = {"rmse": _rmse(res["resid"]),
                         "T_rms": float(torch.sqrt(torch.mean(res["T"] ** 2)).item()),
                         "T_max": float(torch.abs(res["T"]).max().item())}
        return out

    temp_resids = [
        ("baseline", _eval_temp_resid(temp_base)),
        ("PyTorch SAGE", _eval_temp_resid(temp_sage)),
    ]
    if temp_jax_sage is not None:
        temp_resids.append(("JAX-SAGE", _eval_temp_resid(temp_jax_sage)))

    print("\n" + "=" * 80)
    print("Temp advection-diffusion residual RMSE (autograd)")
    print("=" * 80)
    labels_t = [l for l, _ in temp_resids]
    print(f"{'Metric':<20}" + "".join(f"{l:>20}" for l in labels_t))
    for tval in (5.0, 20.0, 35.0):
        for k_label, k in (("rmse", "rmse"), ("T_rms", "T_rms"), ("T_max", "T_max")):
            row = f"t={tval} {k_label:<10}"
            print(f"{row:<20}" + "".join(
                f"{r[1][tval][k]:>20.3e}" if r[1] else f"{'N/A':>20}"
                for r in temp_resids))

    print("\nTemp IC mean T @ t=0 (target 60):")
    for label, val in temp_ics:
        if val is not None:
            print(f"  {label}: {val:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
