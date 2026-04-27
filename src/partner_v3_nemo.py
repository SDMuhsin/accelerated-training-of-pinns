"""
Partner V3 PINN — PhysicsNeMo Sym reproduction.

Faithful port of from_partner_team/partner_code_v3/train_and_infer_stable_v2.py.
Three-stage training: flow-only -> temp-only (frozen flow) -> joint.

Usage:
    source env/bin/activate
    python src/partner_v3_nemo.py                           # full run with defaults
    python src/partner_v3_nemo.py training.k_flow=100 ...   # override via Hydra
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from physicsnemo.sym.key import Key
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.domain.constraint import PointwiseConstraint

from sympy import Symbol, Function, Number
from sympy import Derivative as D
from physicsnemo.sym.eq.pde import PDE

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# GPU init
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")


# ---------------------------------------------------------------------------
# Geometry + IO helpers  (identical to partner V3)
# ---------------------------------------------------------------------------
def points_in_radius(xy: np.ndarray, center_xy: np.ndarray, r: float) -> np.ndarray:
    dx = xy[:, 0] - center_xy[0, 0]
    dy = xy[:, 1] - center_xy[0, 1]
    return (dx * dx + dy * dy) <= (r * r)


def _normalize_xy(
    x_raw: np.ndarray, y_raw: np.ndarray,
    xmin: float, xmax: float, ymin: float, ymax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xden = (xmax - xmin) if (xmax > xmin) else 1.0
    yden = (ymax - ymin) if (ymax > ymin) else 1.0
    x = (x_raw - xmin) / xden
    y = (y_raw - ymin) / yden
    return x, y


def _denormalize_xy(
    x: np.ndarray, y: np.ndarray,
    xmin: float, xmax: float, ymin: float, ymax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    xden = (xmax - xmin) if (xmax > xmin) else 1.0
    yden = (ymax - ymin) if (ymax > ymin) else 1.0
    xr = x * xden + xmin
    yr = y * yden + ymin
    return xr, yr


def _load_points_from_geom_new(obj):
    """
    JSON semantics:
      points: [x, y, type]  — type=0 background, type=1 pipe_wall, type=2 pipe_inside
      inlet/outlet are dicts with x,y in original coordinates.
    Normalization from ALL non-background points (types 1 & 2).
    """
    inlet = obj.get("inlet", None)
    outlet = obj.get("outlet", None)
    pts = obj["points"]

    wall_x, wall_y = [], []
    inside_x, inside_y = [], []
    all_x, all_y = [], []

    for p in pts:
        xr, yr, typ = float(p[0]), float(p[1]), int(p[2])
        if typ == 0:
            continue
        if typ == 1:
            wall_x.append(xr)
            wall_y.append(yr)
        elif typ == 2:
            inside_x.append(xr)
            inside_y.append(yr)
        else:
            raise ValueError(f"Unknown point type {typ}. Expected 0/1/2.")
        all_x.append(xr)
        all_y.append(yr)

    if len(all_x) == 0:
        raise ValueError("No non-background points found (types 1 or 2).")

    xmin, xmax = float(min(all_x)), float(max(all_x))
    ymin, ymax = float(min(all_y)), float(max(all_y))

    wall_x = np.asarray(wall_x, np.float32).reshape(-1, 1)
    wall_y = np.asarray(wall_y, np.float32).reshape(-1, 1)
    x_w, y_w = _normalize_xy(wall_x, wall_y, xmin, xmax, ymin, ymax)

    inside_x = np.asarray(inside_x, np.float32).reshape(-1, 1)
    inside_y = np.asarray(inside_y, np.float32).reshape(-1, 1)
    x_i, y_i = _normalize_xy(inside_x, inside_y, xmin, xmax, ymin, ymax)

    def norm_xy(d: Optional[dict]):
        if d is None:
            return None
        xr = np.asarray([[float(d["x"])]], np.float32)
        yr = np.asarray([[float(d["y"])]], np.float32)
        xn, yn = _normalize_xy(xr, yr, xmin, xmax, ymin, ymax)
        return np.asarray([[float(xn[0, 0]), float(yn[0, 0])]], np.float32)

    inlet_xy = norm_xy(inlet)
    outlet_xy = norm_xy(outlet)

    norm = (xmin, xmax, ymin, ymax)
    return x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm


def _write_inference_json(
    out_path: str,
    xy_norm: np.ndarray,
    times: np.ndarray,
    temps: np.ndarray,
    norm: Tuple[float, float, float, float],
    point_type: np.ndarray,
):
    xmin, xmax, ymin, ymax = norm
    xr, yr = _denormalize_xy(xy_norm[:, 0:1], xy_norm[:, 1:2], xmin, xmax, ymin, ymax)
    xr = xr.reshape(-1)
    yr = yr.reshape(-1)

    records: List[Dict[str, float]] = []
    N = xy_norm.shape[0]
    Tn = times.shape[0]
    for ti in range(Tn):
        tval = float(times[ti])
        for i in range(N):
            records.append(
                {
                    "x": float(xr[i]),
                    "y": float(yr[i]),
                    "x_norm": float(xy_norm[i, 0]),
                    "y_norm": float(xy_norm[i, 1]),
                    "time": tval,
                    "temperature": float(temps[ti, i]),
                    "type": int(point_type[i]),
                }
            )
    Path(out_path).write_text(json.dumps(records, indent=2))


# ---------------------------------------------------------------------------
# PDEs  (identical to partner V3)
# ---------------------------------------------------------------------------
class NavierStokes2D(PDE):
    """
    Incompressible Navier-Stokes (2D, time-dependent):
      continuity: u_x + v_y = 0
      momentum x: u_t + u*u_x + v*u_y + (1/rho)*p_x - nu*(u_xx + u_yy) = 0
      momentum y: v_t + u*v_x + v*v_y + (1/rho)*p_y - nu*(v_xx + v_yy) = 0
    """

    name = "NavierStokes2D"

    def __init__(self, rho=1.0, nu=1.0e-3):
        super().__init__()
        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")
        rhoN = Number(float(rho))
        nuN = Number(float(nu))

        u = Function("u")(x, y, t)
        v = Function("v")(x, y, t)
        p = Function("p")(x, y, t)

        self.equations = {
            "continuity": D(u, x) + D(v, y),
            "momentum_x": (
                D(u, t)
                + u * D(u, x)
                + v * D(u, y)
                + (Number(1.0) / rhoN) * D(p, x)
                - nuN * (D(u, x, 2) + D(u, y, 2))
            ),
            "momentum_y": (
                D(v, t)
                + u * D(v, x)
                + v * D(v, y)
                + (Number(1.0) / rhoN) * D(p, y)
                - nuN * (D(v, x, 2) + D(v, y, 2))
            ),
        }


class AdvectionDiffusion2D_Coupled(PDE):
    """
    Coupled advection-diffusion:
      T_t + u*T_x + v*T_y - alpha*(T_xx + T_yy) = Q

    u and v are learned functions (outputs of flow_net).
    The temperature network explicitly takes (u,v) as inputs.
    """

    name = "AdvectionDiffusion2D_Coupled"

    def __init__(self, T="T", alpha=1e-5, Q=0.0):
        super().__init__()
        x = Symbol("x")
        y = Symbol("y")
        t = Symbol("t")

        Tfun = Function(T)(x, y, t)
        ufun = Function("u")(x, y, t)
        vfun = Function("v")(x, y, t)

        aN = Number(float(alpha))
        qN = Number(float(Q))

        self.equations = {
            "advection_diffusion_T": (
                D(Tfun, t)
                + ufun * D(Tfun, x)
                + vfun * D(Tfun, y)
                - aN * (D(Tfun, x, 2) + D(Tfun, y, 2))
                - qN
            )
        }


# ---------------------------------------------------------------------------
# Training helpers  (identical to partner V3)
# ---------------------------------------------------------------------------
def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def _disable_recording(slv: Solver):
    slv.record_constraints = lambda *args, **kwargs: None
    slv.record_validators = lambda *args, **kwargs: None
    slv.record_inferencers = lambda *args, **kwargs: None
    slv.record_monitors = lambda *args, **kwargs: None


def get_activation(act_name: str):
    act_name = str(act_name).lower()
    if act_name in ("silu", "swish"):
        return torch.nn.SiLU()
    if act_name == "tanh":
        return torch.nn.Tanh()
    if act_name == "relu":
        return torch.nn.ReLU()
    if act_name == "gelu":
        return torch.nn.GELU()
    raise ValueError(f"Unknown activation: {act_name}")


class FlowThenTempWrapper(torch.nn.Module):
    """
    Inference wrapper: (x,y,t) -> flow_net -> (u,v,p) -> temp_net(x,y,t,u,v) -> T
    """

    def __init__(self, flow_net: torch.nn.Module, temp_net: torch.nn.Module):
        super().__init__()
        self.flow_net = flow_net
        self.temp_net = temp_net

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outF = self.flow_net({"x": invar["x"], "y": invar["y"], "t": invar["t"]})
        outT = self.temp_net(
            {
                "x": invar["x"],
                "y": invar["y"],
                "t": invar["t"],
                "u": outF["u"],
                "v": outF["v"],
            }
        )
        return {"T": outT["T"], "u": outF["u"], "v": outF["v"], "p": outF["p"]}


# ---------------------------------------------------------------------------
# Geodesic distance  (identical to partner V3)
# ---------------------------------------------------------------------------
def compute_geodesic_distance_from_inlet(
    xy_inside: np.ndarray,
    inlet_xy: np.ndarray,
    k: int = 8,
    max_edge_len: float = None,
) -> Tuple[np.ndarray, csr_matrix, int]:
    """
    Geodesic distance from inlet to every inside point via k-NN graph.
    Wall/background points excluded so paths cannot pass through walls.
    """
    N = xy_inside.shape[0]

    dx = xy_inside[:, 0] - inlet_xy[0, 0]
    dy = xy_inside[:, 1] - inlet_xy[0, 1]
    source_idx = int(np.argmin(dx**2 + dy**2))

    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    knn.fit(xy_inside)
    distances, indices = knn.kneighbors(xy_inside)

    rows, cols, vals = [], [], []
    for i in range(N):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            d = distances[i, j_idx]

            if max_edge_len is not None and d > max_edge_len:
                continue

            rows.append(i)
            cols.append(j)
            vals.append(d)
            rows.append(j)
            cols.append(i)
            vals.append(d)

    graph = csr_matrix((vals, (rows, cols)), shape=(N, N))

    dist_matrix = shortest_path(
        graph, method="D", indices=source_idx, directed=False,
    )

    dist = dist_matrix.astype(np.float32)

    n_inf = np.sum(np.isinf(dist))
    if n_inf > 0:
        print(
            f"[WARN] {n_inf} inside points unreachable from inlet in k-NN graph. "
            f"Consider increasing k (currently {k})."
        )
        euc = np.sqrt(dx**2 + dy**2).astype(np.float32)
        dist[np.isinf(dist)] = euc[np.isinf(dist)]

    return dist, graph, source_idx


def estimate_point_spacing(xy_inside: np.ndarray, sample_n: int = 500) -> float:
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(xy_inside)
    idx = np.random.choice(len(xy_inside), min(sample_n, len(xy_inside)), replace=False)
    dists, _ = knn.kneighbors(xy_inside[idx])
    return float(np.median(dists[:, 1]))


# ---------------------------------------------------------------------------
# Main  (faithful reproduction of partner V3 run() function)
# ---------------------------------------------------------------------------
def run(cfg: DictConfig) -> None:
    # Hydra changes CWD; go back to project root for relative paths
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    outdir = project_root / "results" / "partner_v3"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Partner V3 PINN — PhysicsNeMo Sym Reproduction")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Output dir:   {outdir}")

    # -----------------------------------------------------------------
    # 1) Load geometry
    # -----------------------------------------------------------------
    geom_json_path = str(project_root / cfg.problem.geom_json_path)
    geom_step_path = str(project_root / cfg.problem.geom_step_path)

    if os.path.exists(geom_json_path):
        print(f"[geom] Loading pre-converted JSON: {geom_json_path}")
        with open(geom_json_path, "r", encoding="utf-8") as f:
            jsonObj = json.load(f)
    else:
        print(f"[geom] JSON not found, converting from STEP: {geom_step_path}")
        sys.path.insert(0, str(project_root / "src"))
        from point_cloud_sampler import PointCloudSampler

        pcs = PointCloudSampler(cad_path=geom_step_path)
        jsonObj = pcs.convert_to_json(geom_json_path)

    x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm = _load_points_from_geom_new(
        jsonObj
    )

    if x_i.shape[0] == 0:
        raise ValueError("No pipe_inside points found (type=2).")
    if x_w.shape[0] == 0:
        raise ValueError("No pipe_wall points found (type=1).")

    print(f"[geom] Wall points: {x_w.shape[0]}, Inside points: {x_i.shape[0]}")
    print(f"[geom] Norm bounds: x=[{norm[0]:.1f}, {norm[1]:.1f}], y=[{norm[2]:.1f}, {norm[3]:.1f}]")

    xy_inside = np.concatenate([x_i, y_i], axis=1).astype(np.float32)

    spacing = estimate_point_spacing(xy_inside)
    max_edge_len = 2 * spacing
    print(f"[geom] Point spacing: {spacing:.6f}, max_edge_len: {max_edge_len:.6f}")

    dist_inside, graph, source_idx = compute_geodesic_distance_from_inlet(
        xy_inside, inlet_xy, k=8, max_edge_len=max_edge_len
    )

    # Save geodesic check plot (non-blocking, replaces plt.show())
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        xy_wall = np.concatenate([x_w, y_w], axis=1)
        ax.scatter(xy_wall[:, 0], xy_wall[:, 1], c="gray", s=1, alpha=0.3)
        finite_mask = np.isfinite(dist_inside)
        sc = ax.scatter(
            xy_inside[finite_mask, 0], xy_inside[finite_mask, 1],
            c=dist_inside[finite_mask], cmap="plasma", s=2,
        )
        if inlet_xy is not None:
            ax.scatter(*inlet_xy[0], c="cyan", s=80, marker="*", zorder=5)
        ax.set_aspect("equal")
        ax.set_title("Geodesic distance from inlet")
        plt.colorbar(sc, ax=ax, fraction=0.03)
        plt.savefig(str(outdir / "geodesic_check.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"[geom] Saved geodesic plot: {outdir / 'geodesic_check.png'}")
    except Exception as e:
        print(f"[WARN] Could not save geodesic plot: {e}")

    mean_flow_speed = max(float(cfg.bc.inlet_u), 1e-6)
    t_arrive_inside = (dist_inside / mean_flow_speed).astype(np.float32)
    t_arrive_inside = np.clip(
        t_arrive_inside, float(cfg.problem.t_min), float(cfg.problem.t_max),
    )

    if inlet_xy is None:
        raise ValueError("Need inlet point in JSON to apply inlet BCs.")
    if outlet_xy is None:
        print("[WARN] outlet missing: outlet BCs will be skipped.")

    inlet_mask = (
        points_in_radius(xy_inside, inlet_xy, float(cfg.bc.inlet_radius_norm))
        if inlet_xy is not None
        else None
    )
    outlet_mask = (
        points_in_radius(xy_inside, outlet_xy, float(cfg.bc.outlet_radius_norm))
        if outlet_xy is not None
        else None
    )
    print(f"[geom] Inlet points: {sum(inlet_mask) if inlet_mask is not None else 0}")
    print(f"[geom] Outlet points: {sum(outlet_mask) if outlet_mask is not None else 0}")

    # -----------------------------------------------------------------
    # 2) Build TWO networks  (identical architecture to partner V3)
    # -----------------------------------------------------------------
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        layer_size=int(cfg.flow_model.hidden_size),
        nr_layers=int(cfg.flow_model.hidden_layers),
        activation_fn=get_activation(cfg.flow_model.activation),
    )

    temp_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t"), Key("u"), Key("v")],
        output_keys=[Key("T")],
        layer_size=int(cfg.model.hidden_size),
        nr_layers=int(cfg.model.hidden_layers),
        activation_fn=get_activation(cfg.model.activation),
    )

    flow_params = sum(p.numel() for p in flow_net.parameters())
    temp_params = sum(p.numel() for p in temp_net.parameters())
    print(f"[model] Flow net: {flow_params:,} params ({cfg.flow_model.hidden_layers} layers x {cfg.flow_model.hidden_size})")
    print(f"[model] Temp net: {temp_params:,} params ({cfg.model.hidden_layers} layers x {cfg.model.hidden_size})")

    ns_pde = NavierStokes2D(rho=float(cfg.physics.rho), nu=float(cfg.physics.nu))
    T_pde = AdvectionDiffusion2D_Coupled(
        T="T", alpha=float(cfg.physics.D), Q=float(cfg.physics.Q)
    )

    flow_nodes = ns_pde.make_nodes() + [flow_net.make_node(name="flow_network")]
    temp_nodes = T_pde.make_nodes() + [
        flow_net.make_node(name="flow_network"),
        temp_net.make_node(name="temperature_network_uv"),
    ]

    # -----------------------------------------------------------------
    # 3) Constraints  (identical to partner V3)
    # -----------------------------------------------------------------
    def sample_time(n: int) -> np.ndarray:
        u = np.random.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
        t = float(cfg.problem.t_min) + (
            float(cfg.problem.t_max) - float(cfg.problem.t_min)
        ) * (u**2)
        return t

    time_samples = int(getattr(cfg.problem, "time_samples", 1))

    def expand_in_time(
        xp: np.ndarray, yp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if time_samples <= 1:
            return xp, yp, sample_time(xp.shape[0])
        xp2 = np.repeat(xp, time_samples, axis=0)
        yp2 = np.repeat(yp, time_samples, axis=0)
        tp2 = sample_time(xp2.shape[0])
        return xp2, yp2, tp2

    # ---- FLOW constraints ----
    x_it, y_it, t_it = expand_in_time(x_i, y_i)
    flow_interior_invar = {"x": x_it, "y": y_it, "t": t_it}
    flow_eq_names = list(ns_pde.equations.keys())
    flow_pde_outvar = {
        name: np.zeros((x_it.shape[0], 1), np.float32) for name in flow_eq_names
    }

    flow_pde_constraint = PointwiseConstraint.from_numpy(
        nodes=flow_nodes,
        invar=flow_interior_invar,
        outvar=flow_pde_outvar,
        batch_size=int(getattr(cfg.training, "flow_pde_batch_size", 64)),
        shuffle=True,
    )

    # No-slip wall
    x_wt, y_wt, t_wt = expand_in_time(x_w, y_w)
    wall_invar = {"x": x_wt, "y": y_wt, "t": t_wt}
    wall_outvar = {
        "u": np.zeros((x_wt.shape[0], 1), np.float32),
        "v": np.zeros((x_wt.shape[0], 1), np.float32),
    }
    wall_noslip = PointwiseConstraint.from_numpy(
        nodes=flow_nodes,
        invar=wall_invar,
        outvar=wall_outvar,
        batch_size=int(getattr(cfg.training, "flow_bc_batch_size", 16)),
        shuffle=True,
    )

    # Inlet velocity
    inlet_flow_constraint = None
    inlet_u = float(getattr(cfg.bc, "inlet_u", 1.0))
    inlet_v = float(getattr(cfg.bc, "inlet_v", 0.0))

    if inlet_mask is not None and np.any(inlet_mask):
        x_in = x_i[inlet_mask].astype(np.float32)
        y_in = y_i[inlet_mask].astype(np.float32)
        x_in_t, y_in_t, t_in_t = expand_in_time(x_in, y_in)
        inlet_flow_invar = {"x": x_in_t, "y": y_in_t, "t": t_in_t}
        inlet_flow_outvar = {
            "u": np.full((x_in_t.shape[0], 1), inlet_u, np.float32),
            "v": np.full((x_in_t.shape[0], 1), inlet_v, np.float32),
        }
        inlet_flow_constraint = PointwiseConstraint.from_numpy(
            nodes=flow_nodes,
            invar=inlet_flow_invar,
            outvar=inlet_flow_outvar,
            batch_size=min(
                int(getattr(cfg.training, "flow_bc_batch_size", 16)),
                x_in_t.shape[0],
            ),
            shuffle=True,
        )
    else:
        print("[WARN] inlet_mask empty: inlet flow BC skipped.")

    # Outlet pressure
    outlet_p_constraint = None
    if outlet_mask is not None and np.any(outlet_mask):
        x_out = x_i[outlet_mask].astype(np.float32)
        y_out = y_i[outlet_mask].astype(np.float32)
        x_ot, y_ot, t_ot = expand_in_time(x_out, y_out)
        outlet_invar = {"x": x_ot, "y": y_ot, "t": t_ot}
        outlet_outvar = {"p": np.zeros((x_ot.shape[0], 1), np.float32)}
        outlet_p_constraint = PointwiseConstraint.from_numpy(
            nodes=flow_nodes,
            invar=outlet_invar,
            outvar=outlet_outvar,
            batch_size=min(
                int(getattr(cfg.training, "flow_bc_batch_size", 16)),
                x_ot.shape[0],
            ),
            shuffle=True,
        )
    else:
        if outlet_xy is not None:
            print("[WARN] outlet_mask empty: outlet p=0 BC skipped.")

    # ---- TEMPERATURE constraints ----
    x_it2, y_it2, t_it2 = expand_in_time(x_i, y_i)
    temp_interior_invar = {"x": x_it2, "y": y_it2, "t": t_it2}
    temp_eq_names = list(T_pde.equations.keys())
    temp_pde_outvar = {
        name: np.zeros((x_it2.shape[0], 1), np.float32) for name in temp_eq_names
    }

    temp_pde_constraint = PointwiseConstraint.from_numpy(
        nodes=temp_nodes,
        invar=temp_interior_invar,
        outvar=temp_pde_outvar,
        batch_size=int(cfg.training.pde_batch_size),
        shuffle=True,
    )

    # IC at t=0
    x0 = np.vstack([x_w, x_i]).astype(np.float32)
    y0 = np.vstack([y_w, y_i]).astype(np.float32)
    T0 = np.full((x0.shape[0], 1), float(cfg.problem.T_init), np.float32)

    temp_ic_invar = {
        "x": x0,
        "y": y0,
        "t": np.zeros((x0.shape[0], 1), np.float32),
    }
    temp_ic_outvar = {"T": T0}

    temp_ic_constraint = PointwiseConstraint.from_numpy(
        nodes=temp_nodes,
        invar=temp_ic_invar,
        outvar=temp_ic_outvar,
        batch_size=int(cfg.training.bc_batch_size),
        shuffle=True,
    )

    # Arrival-time Dirichlet
    arrival_invar = {
        "x": x_i,
        "y": y_i,
        "t": t_arrive_inside.reshape(-1, 1),
    }
    arrival_outvar = {
        "T": np.full((x_i.shape[0], 1), float(cfg.bc.inlet_T), np.float32)
    }

    arrival_constraint = PointwiseConstraint.from_numpy(
        nodes=temp_nodes,
        invar=arrival_invar,
        outvar=arrival_outvar,
        batch_size=int(cfg.training.bc_batch_size),
        shuffle=True,
    )

    # Before-arrival constraint
    n_pre = x_i.shape[0]
    frac = np.random.uniform(0.0, 0.95, size=(n_pre,)).astype(np.float32)
    t_pre = (frac * t_arrive_inside).reshape(-1, 1)
    t_pre = np.clip(t_pre, float(cfg.problem.t_min), None)

    pre_arrival_invar = {"x": x_i, "y": y_i, "t": t_pre}
    pre_arrival_outvar = {
        "T": np.full((n_pre, 1), float(cfg.problem.T_init), np.float32)
    }

    pre_arrival_constraint = PointwiseConstraint.from_numpy(
        nodes=temp_nodes,
        invar=pre_arrival_invar,
        outvar=pre_arrival_outvar,
        batch_size=int(cfg.training.bc_batch_size),
        shuffle=True,
    )

    # Inlet temperature Dirichlet
    inlet_T_constraint = None
    if inlet_mask is not None and np.any(inlet_mask):
        x_in = x_i[inlet_mask].astype(np.float32)
        y_in = y_i[inlet_mask].astype(np.float32)
        x_in_t, y_in_t, t_in_t = expand_in_time(x_in, y_in)
        inlet_T_invar = {"x": x_in_t, "y": y_in_t, "t": t_in_t}
        inlet_T_outvar = {
            "T": np.full((x_in_t.shape[0], 1), float(cfg.bc.inlet_T), np.float32)
        }
        inlet_T_constraint = PointwiseConstraint.from_numpy(
            nodes=temp_nodes,
            invar=inlet_T_invar,
            outvar=inlet_T_outvar,
            batch_size=min(int(cfg.training.bc_batch_size), x_in_t.shape[0]),
            shuffle=True,
        )

    # Outlet Neumann: T__x = 0
    outlet_T_constraint = None
    if outlet_mask is not None and np.any(outlet_mask):
        x_out = x_i[outlet_mask].astype(np.float32)
        y_out = y_i[outlet_mask].astype(np.float32)
        x_ot, y_ot, t_ot = expand_in_time(x_out, y_out)
        outlet_T_invar = {"x": x_ot, "y": y_ot, "t": t_ot}
        outlet_T_outvar = {"T__x": np.zeros((x_ot.shape[0], 1), np.float32)}
        outlet_T_constraint = PointwiseConstraint.from_numpy(
            nodes=temp_nodes,
            invar=outlet_T_invar,
            outvar=outlet_T_outvar,
            batch_size=min(int(cfg.training.bc_batch_size), x_ot.shape[0]),
            shuffle=True,
        )

    # -----------------------------------------------------------------
    # 4) Three-stage training  (identical to partner V3)
    # -----------------------------------------------------------------
    k1 = int(getattr(cfg.training, "k_flow", 1000))
    k2 = int(getattr(cfg.training, "k_temp", 1000))
    k3 = int(getattr(cfg.training, "k_joint", 1000))
    print(f"\n[train] Three-stage training: flow={k1}, temp={k2}, joint={k3}")

    def make_stage_cfg(
        base_cfg: DictConfig, stage_name: str, max_steps: int
    ) -> DictConfig:
        c = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        c.training.max_steps = int(max_steps)
        c.network_dir = str(outdir / "checkpoints" / stage_name)
        return c

    start_time = time.time()

    # ---- Stage 1: train FLOW only ----
    print(f"\n{'='*70}")
    print(f"STAGE 1: Flow only ({k1} steps)")
    print(f"{'='*70}")
    set_requires_grad(flow_net, True)
    set_requires_grad(temp_net, False)

    domain1 = Domain()
    domain1.add_constraint(flow_pde_constraint, "flow_pde_inside")
    domain1.add_constraint(wall_noslip, "wall_noslip")
    if inlet_flow_constraint is not None:
        domain1.add_constraint(inlet_flow_constraint, "inlet_flow")
    if outlet_p_constraint is not None:
        domain1.add_constraint(outlet_p_constraint, "outlet_p0")

    cfg1 = make_stage_cfg(cfg, "stage1_flow", k1)
    slv1 = Solver(cfg1, domain1)
    _disable_recording(slv1)
    slv1.solve()

    stage1_time = time.time() - start_time
    print(f"[stage1] Flow training done in {stage1_time / 60:.2f} min")

    # ---- Stage 2: freeze FLOW, train TEMP only ----
    print(f"\n{'='*70}")
    print(f"STAGE 2: Temperature only, frozen flow ({k2} steps)")
    print(f"{'='*70}")
    set_requires_grad(flow_net, False)
    set_requires_grad(temp_net, True)

    domain2 = Domain()
    domain2.add_constraint(temp_pde_constraint, "temp_pde_inside")
    domain2.add_constraint(temp_ic_constraint, "temp_ic_t0")
    domain2.add_constraint(arrival_constraint, "temp_arrival_front")
    domain2.add_constraint(pre_arrival_constraint, "temp_pre_arrival")
    if inlet_T_constraint is not None:
        domain2.add_constraint(inlet_T_constraint, "inlet_T")
    if outlet_T_constraint is not None:
        domain2.add_constraint(outlet_T_constraint, "outlet_T_neumann")

    cfg2 = make_stage_cfg(cfg, "stage2_temp_frozen_flow", k2)
    slv2 = Solver(cfg2, domain2)
    _disable_recording(slv2)
    slv2.solve()

    stage2_time = time.time() - start_time - stage1_time
    print(f"[stage2] Temp training done in {stage2_time / 60:.2f} min")

    # ---- Stage 3: unfreeze FLOW, train both jointly ----
    print(f"\n{'='*70}")
    print(f"STAGE 3: Joint training ({k3} steps)")
    print(f"{'='*70}")
    set_requires_grad(flow_net, True)
    set_requires_grad(temp_net, True)

    domain3 = Domain()
    domain3.add_constraint(flow_pde_constraint, "flow_pde_inside")
    domain3.add_constraint(wall_noslip, "wall_noslip")
    if inlet_flow_constraint is not None:
        domain3.add_constraint(inlet_flow_constraint, "inlet_flow")
    if outlet_p_constraint is not None:
        domain3.add_constraint(outlet_p_constraint, "outlet_p0")
    domain3.add_constraint(temp_pde_constraint, "temp_pde_inside")
    domain3.add_constraint(temp_ic_constraint, "temp_ic_t0")
    domain3.add_constraint(arrival_constraint, "temp_arrival_front")
    domain3.add_constraint(pre_arrival_constraint, "temp_pre_arrival")
    if inlet_T_constraint is not None:
        domain3.add_constraint(inlet_T_constraint, "inlet_T")
    if outlet_T_constraint is not None:
        domain3.add_constraint(outlet_T_constraint, "outlet_T_neumann")

    cfg3 = make_stage_cfg(cfg, "stage3_joint", k3)
    slv3 = Solver(cfg3, domain3)
    _disable_recording(slv3)
    slv3.solve()

    end_time = time.time()
    total_time = end_time - start_time
    stage3_time = total_time - stage1_time - stage2_time

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"  Stage 1 (flow):  {stage1_time / 60:.2f} min")
    print(f"  Stage 2 (temp):  {stage2_time / 60:.2f} min")
    print(f"  Stage 3 (joint): {stage3_time / 60:.2f} min")
    print(f"  Total:           {total_time / 60:.2f} min")
    print(f"{'='*70}")

    # -----------------------------------------------------------------
    # 5) Inference  (identical to partner V3)
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_net = flow_net.to(device).eval()
    temp_net = temp_net.to(device).eval()
    infer_model = FlowThenTempWrapper(flow_net, temp_net).to(device).eval()

    xy = np.vstack(
        [np.concatenate([x_w, y_w], 1), np.concatenate([x_i, y_i], 1)]
    ).astype(np.float32)

    point_type = np.concatenate(
        [
            np.full((x_w.shape[0],), 1, np.int32),
            np.full((x_i.shape[0],), 2, np.int32),
        ],
        axis=0,
    )

    t0 = float(cfg.problem.infer_t_start)
    t1 = float(cfg.problem.infer_t_end)
    dt = float(cfg.problem.infer_dt)
    times = np.arange(t0, t1 + 1e-9, dt, dtype=np.float32)
    temps = np.zeros((times.shape[0], xy.shape[0]), dtype=np.float32)

    print(f"\n[infer] Running inference on {xy.shape[0]} points x {len(times)} timesteps...")
    with torch.no_grad():
        N = xy.shape[0]
        batch = 65536
        for ti, tval in enumerate(times):
            for s in range(0, N, batch):
                e = min(s + batch, N)
                x_tensor = torch.from_numpy(xy[s:e, 0:1]).to(device)
                y_tensor = torch.from_numpy(xy[s:e, 1:2]).to(device)
                t_tensor = torch.full_like(x_tensor, float(tval))

                out = infer_model({"x": x_tensor, "y": y_tensor, "t": t_tensor})
                temps[ti, s:e] = out["T"].detach().cpu().numpy().reshape(-1)

    out_json_path = str(outdir / "pred_T.json")
    _write_inference_json(out_json_path, xy, times, temps, norm=norm, point_type=point_type)
    print(f"[infer] Wrote inference results to: {out_json_path}")

    # Save timing summary
    summary = {
        "stage1_flow_min": stage1_time / 60,
        "stage2_temp_min": stage2_time / 60,
        "stage3_joint_min": stage3_time / 60,
        "total_min": total_time / 60,
        "flow_params": flow_params,
        "temp_params": temp_params,
        "wall_points": int(x_w.shape[0]),
        "inside_points": int(x_i.shape[0]),
        "k_flow": k1,
        "k_temp": k2,
        "k_joint": k3,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    summary_path = str(outdir / "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {summary_path}")
    print(f"Total elapsed training time: {total_time / 60:.2f} minutes")


def run_sage(cfg: DictConfig) -> None:
    """SAGE-enhanced training: RBF-FD matrices + hand-derived backward."""
    from physicsnemo.sym.eq.sage_pde import (
        build_rkpm_matrices, SAGENSLoss, SAGEAdvDiffLoss,
        compute_pde_rmse,
    )

    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    outdir = project_root / "results" / "partner_v3_sage"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Partner V3 PINN — SAGE-Enhanced Training")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1) Load geometry (same as baseline)
    # -----------------------------------------------------------------
    geom_json_path = str(project_root / cfg.problem.geom_json_path)
    with open(geom_json_path, "r", encoding="utf-8") as f:
        jsonObj = json.load(f)

    x_w, y_w, x_i, y_i, inlet_xy, outlet_xy, norm = _load_points_from_geom_new(jsonObj)
    print(f"[geom] Wall: {x_w.shape[0]}, Inside: {x_i.shape[0]}")

    xy_inside = np.concatenate([x_i, y_i], axis=1).astype(np.float32)

    # Geodesic distance for arrival-time constraints
    spacing = estimate_point_spacing(xy_inside)
    max_edge_len = 2 * spacing
    dist_inside, _, _ = compute_geodesic_distance_from_inlet(
        xy_inside, inlet_xy, k=8, max_edge_len=max_edge_len
    )
    mean_flow_speed = max(float(cfg.bc.inlet_u), 1e-6)
    t_arrive_inside = np.clip(
        (dist_inside / mean_flow_speed).astype(np.float32),
        float(cfg.problem.t_min), float(cfg.problem.t_max),
    )

    inlet_mask = (
        points_in_radius(xy_inside, inlet_xy, float(cfg.bc.inlet_radius_norm))
        if inlet_xy is not None else None
    )
    outlet_mask = (
        points_in_radius(xy_inside, outlet_xy, float(cfg.bc.outlet_radius_norm))
        if outlet_xy is not None else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # 2) Build RBF-FD matrices
    # -----------------------------------------------------------------
    rbffd_cache = project_root / "data" / "partner_v3" / "rbffd_matrices.pt"
    if rbffd_cache.exists():
        print(f"[sage] Loading cached RBF-FD matrices: {rbffd_cache}")
        rkpm = torch.load(str(rbffd_cache), map_location=device, weights_only=False)
    else:
        rkpm = build_rkpm_matrices(
            xy_inside, device,
            stencil_size=int(getattr(cfg.training, "sage_stencil", 25)),
        )
        torch.save(rkpm, str(rbffd_cache))
        print(f"[sage] Saved RBF-FD matrices to {rbffd_cache}")

    rkpm_tuple = (
        rkpm["Dx"], rkpm["Dy"], rkpm["Dxx"], rkpm["Dyy"],
        rkpm["DxT"], rkpm["DyT"], rkpm["DxxT"], rkpm["DyyT"],
    )

    # -----------------------------------------------------------------
    # 3) Build networks (same as baseline)
    # -----------------------------------------------------------------
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        layer_size=int(cfg.flow_model.hidden_size),
        nr_layers=int(cfg.flow_model.hidden_layers),
        activation_fn=get_activation(cfg.flow_model.activation),
    ).to(device)

    temp_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("t"), Key("u"), Key("v")],
        output_keys=[Key("T")],
        layer_size=int(cfg.model.hidden_size),
        nr_layers=int(cfg.model.hidden_layers),
        activation_fn=get_activation(cfg.model.activation),
    ).to(device)

    flow_params = sum(p.numel() for p in flow_net.parameters())
    temp_params = sum(p.numel() for p in temp_net.parameters())
    print(f"[model] Flow net: {flow_params:,} params, Temp net: {temp_params:,} params")

    # -----------------------------------------------------------------
    # 4) Prepare spatial tensors (all points, fixed)
    # -----------------------------------------------------------------
    N_inside = x_i.shape[0]
    x_all = torch.tensor(x_i, dtype=torch.float32, device=device)  # (N, 1)
    y_all = torch.tensor(y_i, dtype=torch.float32, device=device)

    # Wall points
    x_wall = torch.tensor(x_w, dtype=torch.float32, device=device)
    y_wall = torch.tensor(y_w, dtype=torch.float32, device=device)
    N_wall = x_w.shape[0]

    # Inlet/outlet masks → point tensors
    inlet_x = torch.tensor(x_i[inlet_mask], dtype=torch.float32, device=device) if inlet_mask is not None and np.any(inlet_mask) else None
    inlet_y = torch.tensor(y_i[inlet_mask], dtype=torch.float32, device=device) if inlet_mask is not None and np.any(inlet_mask) else None
    outlet_x = torch.tensor(x_i[outlet_mask], dtype=torch.float32, device=device) if outlet_mask is not None and np.any(outlet_mask) else None
    outlet_y = torch.tensor(y_i[outlet_mask], dtype=torch.float32, device=device) if outlet_mask is not None and np.any(outlet_mask) else None

    # IC points (wall + inside at t=0)
    x_ic = torch.cat([x_wall, x_all], dim=0)
    y_ic = torch.cat([y_wall, y_all], dim=0)
    N_ic = x_ic.shape[0]

    # Arrival time constraints
    t_arrive_tensor = torch.tensor(t_arrive_inside.reshape(-1, 1), dtype=torch.float32, device=device)

    nu = float(cfg.physics.nu)
    rho = float(cfg.physics.rho)
    inv_rho = 1.0 / rho
    alpha = float(cfg.physics.D)
    T_init = float(cfg.problem.T_init)
    inlet_u = float(cfg.bc.inlet_u)
    inlet_T = float(cfg.bc.inlet_T)
    t_min = float(cfg.problem.t_min)
    t_max = float(cfg.problem.t_max)

    # -----------------------------------------------------------------
    # 5) Time sampling helper
    # -----------------------------------------------------------------
    def sample_time():
        u = np.random.uniform(0, 1)
        return t_min + (t_max - t_min) * (u ** 2)

    # -----------------------------------------------------------------
    # 6) Training step functions
    # -----------------------------------------------------------------
    chunk_size = int(getattr(cfg.training, "sage_chunk", 16384))
    bwd_batch = int(getattr(cfg.training, "sage_bwd_batch", 4096))

    def ns_pde_loss(flow_net_, t_val):
        """NS PDE loss via SAGE: full-batch spatial derivs, sub-batch backward.

        1. Forward no_grad on ALL 98K points → field values
        2. RKPM matrices → ALL spatial derivatives
        3. FD time derivatives (3 forward passes, no_grad) → ALL u_t, v_t
        4. PDE residuals (ALL points) + SAGE backward (ALL points)
        5. Sub-sample bwd_batch points for network backward (with autograd)
        """
        Dx, Dy, Dxx, Dyy, DxT, DyT, DxxT, DyyT = rkpm_tuple
        N = N_inside
        dt_fd = 0.005  # finite difference step for time derivative

        # Step 1: Forward no_grad at t, t+dt, t-dt
        def fwd_nograd(t_v):
            parts_u, parts_v, parts_p = [], [], []
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                t_c = torch.full((e - s, 1), t_v, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = flow_net_({"x": x_all[s:e], "y": y_all[s:e], "t": t_c})
                parts_u.append(out["u"])
                parts_v.append(out["v"])
                parts_p.append(out["p"])
            return torch.cat(parts_u), torch.cat(parts_v), torch.cat(parts_p)

        u_d, v_d, p_d = fwd_nograd(t_val)
        u_plus, v_plus, _ = fwd_nograd(t_val + dt_fd)
        u_minus, v_minus, _ = fwd_nograd(t_val - dt_fd)

        # Step 2: Spatial derivatives
        u_x = torch.sparse.mm(Dx, u_d)
        u_y = torch.sparse.mm(Dy, u_d)
        v_x = torch.sparse.mm(Dx, v_d)
        v_y = torch.sparse.mm(Dy, v_d)
        p_x = torch.sparse.mm(Dx, p_d)
        p_y = torch.sparse.mm(Dy, p_d)
        u_xx = torch.sparse.mm(Dxx, u_d)
        u_yy = torch.sparse.mm(Dyy, u_d)
        v_xx = torch.sparse.mm(Dxx, v_d)
        v_yy = torch.sparse.mm(Dyy, v_d)

        # Step 3: FD time derivatives
        u_t_d = (u_plus - u_minus) / (2.0 * dt_fd)
        v_t_d = (v_plus - v_minus) / (2.0 * dt_fd)

        # Step 4: PDE residuals (all points)
        cont = u_x + v_y
        mom_x = u_t_d + u_d * u_x + v_d * u_y + inv_rho * p_x - nu * (u_xx + u_yy)
        mom_y = v_t_d + u_d * v_x + v_d * v_y + inv_rho * p_y - nu * (v_xx + v_yy)

        loss_val = (cont.pow(2).sum() + mom_x.pow(2).sum() + mom_y.pow(2).sum()) / N
        scale = 2.0 / N

        # SAGE backward (all points)
        dc = cont * scale
        dmu = mom_x * scale
        dmv = mom_y * scale

        # Adjoint for network output at time t
        adj_u_t_val = (
            torch.sparse.mm(DxT, dc) + dmu * u_x
            + torch.sparse.mm(DxT, dmu * u_d) + torch.sparse.mm(DyT, dmu * v_d)
            - nu * torch.sparse.mm(DxxT, dmu) - nu * torch.sparse.mm(DyyT, dmu)
            + dmv * v_x
        )
        adj_v_t_val = (
            torch.sparse.mm(DyT, dc) + dmu * u_y + dmv * v_y
            + torch.sparse.mm(DxT, dmv * u_d) + torch.sparse.mm(DyT, dmv * v_d)
            - nu * torch.sparse.mm(DxxT, dmv) - nu * torch.sparse.mm(DyyT, dmv)
        )
        adj_p_t_val = inv_rho * (torch.sparse.mm(DxT, dmu) + torch.sparse.mm(DyT, dmv))

        # Adjoint for network output at t+dt (from FD time derivative)
        adj_u_t_plus = dmu / (2.0 * dt_fd)
        adj_v_t_plus = dmv / (2.0 * dt_fd)

        # Adjoint for network output at t-dt
        adj_u_t_minus = -dmu / (2.0 * dt_fd)
        adj_v_t_minus = -dmv / (2.0 * dt_fd)

        # Step 5: Network backward via sub-sampled pred.backward()
        # Sample random subset for backward
        idx = torch.randperm(N, device=device)[:bwd_batch]

        # Backward at time t
        t_c = torch.full((bwd_batch, 1), t_val, dtype=torch.float32, device=device)
        out_t = flow_net_({"x": x_all[idx], "y": y_all[idx], "t": t_c})
        upstream_t = torch.cat([
            adj_u_t_val[idx].detach(),
            adj_v_t_val[idx].detach(),
            adj_p_t_val[idx].detach(),
        ], dim=1) * (N / bwd_batch)  # scale to account for sub-sampling
        pred_t = torch.cat([out_t["u"], out_t["v"], out_t["p"]], dim=1)
        pred_t.backward(gradient=upstream_t)

        # Backward at t+dt
        t_plus_c = torch.full((bwd_batch, 1), t_val + dt_fd, dtype=torch.float32, device=device)
        out_plus = flow_net_({"x": x_all[idx], "y": y_all[idx], "t": t_plus_c})
        upstream_plus = torch.cat([
            adj_u_t_plus[idx].detach(),
            adj_v_t_plus[idx].detach(),
            torch.zeros(bwd_batch, 1, device=device),  # no p contribution from FD
        ], dim=1) * (N / bwd_batch)
        pred_plus = torch.cat([out_plus["u"], out_plus["v"], out_plus["p"]], dim=1)
        pred_plus.backward(gradient=upstream_plus)

        # Backward at t-dt
        t_minus_c = torch.full((bwd_batch, 1), t_val - dt_fd, dtype=torch.float32, device=device)
        out_minus = flow_net_({"x": x_all[idx], "y": y_all[idx], "t": t_minus_c})
        upstream_minus = torch.cat([
            adj_u_t_minus[idx].detach(),
            adj_v_t_minus[idx].detach(),
            torch.zeros(bwd_batch, 1, device=device),
        ], dim=1) * (N / bwd_batch)
        pred_minus = torch.cat([out_minus["u"], out_minus["v"], out_minus["p"]], dim=1)
        pred_minus.backward(gradient=upstream_minus)

        return loss_val.detach()

    def advdiff_pde_loss(flow_net_, temp_net_, t_val, flow_frozen):
        """Advection-diffusion PDE loss via SAGE: full-batch spatial, sub-batch backward."""
        Dx, Dy, Dxx, Dyy, DxT, DyT, DxxT, DyyT = rkpm_tuple
        N = N_inside
        dt_fd = 0.005

        # Step 1: Get flow field ONCE (reuse for all FD steps — valid since dt_fd is tiny)
        uf_parts, vf_parts = [], []
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            t_c = torch.full((e - s, 1), t_val, dtype=torch.float32, device=device)
            with torch.no_grad():
                out_f = flow_net_({"x": x_all[s:e], "y": y_all[s:e], "t": t_c})
            uf_parts.append(out_f["u"])
            vf_parts.append(out_f["v"])
        u_flow_d = torch.cat(uf_parts)
        v_flow_d = torch.cat(vf_parts)

        # Evaluate temp_net at t, t+dt, t-dt using CACHED flow field
        def fwd_temp_nograd(t_v):
            T_parts = []
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                t_c = torch.full((e - s, 1), t_v, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out_t = temp_net_({"x": x_all[s:e], "y": y_all[s:e], "t": t_c,
                                      "u": u_flow_d[s:e], "v": v_flow_d[s:e]})
                T_parts.append(out_t["T"])
            return torch.cat(T_parts)

        T_d = fwd_temp_nograd(t_val)
        T_plus = fwd_temp_nograd(t_val + dt_fd)
        T_minus = fwd_temp_nograd(t_val - dt_fd)
        T_t_d = (T_plus - T_minus) / (2.0 * dt_fd)

        # Step 2: Spatial derivatives
        T_x = torch.sparse.mm(Dx, T_d)
        T_y = torch.sparse.mm(Dy, T_d)
        T_xx = torch.sparse.mm(Dxx, T_d)
        T_yy = torch.sparse.mm(Dyy, T_d)

        # Step 3: PDE residual
        residual = T_t_d + u_flow_d * T_x + v_flow_d * T_y - alpha * (T_xx + T_yy)
        loss_val = residual.pow(2).mean()
        scale = 2.0 / N
        dr = residual * scale

        # Step 4: SAGE backward
        adj_T_spatial = (
            torch.sparse.mm(DxT, dr * u_flow_d) + torch.sparse.mm(DyT, dr * v_flow_d)
            - alpha * torch.sparse.mm(DxxT, dr) - alpha * torch.sparse.mm(DyyT, dr)
        )
        adj_T_t_plus = dr / (2.0 * dt_fd)
        adj_T_t_minus = -dr / (2.0 * dt_fd)
        adj_u_coupling = dr * T_x
        adj_v_coupling = dr * T_y

        # Step 5: Sub-batch backward (use cached flow field for all)
        idx = torch.randperm(N, device=device)[:bwd_batch]
        ratio = N / bwd_batch
        uf_sub = u_flow_d[idx].detach()
        vf_sub = v_flow_d[idx].detach()

        # Backward temp_net at t (spatial contribution)
        t_c = torch.full((bwd_batch, 1), t_val, dtype=torch.float32, device=device)
        out_t = temp_net_({"x": x_all[idx], "y": y_all[idx], "t": t_c,
                          "u": uf_sub, "v": vf_sub})
        upstream_T = adj_T_spatial[idx].detach() * ratio
        out_t["T"].backward(gradient=upstream_T)

        # Backward temp_net at t+dt and t-dt (FD time contribution)
        for t_v, adj_T_fd in [(t_val + dt_fd, adj_T_t_plus), (t_val - dt_fd, adj_T_t_minus)]:
            t_fd = torch.full((bwd_batch, 1), t_v, dtype=torch.float32, device=device)
            out_t_fd = temp_net_({"x": x_all[idx], "y": y_all[idx], "t": t_fd,
                                 "u": uf_sub, "v": vf_sub})
            out_t_fd["T"].backward(gradient=adj_T_fd[idx].detach() * ratio)

        # Backward flow coupling (Stage 3 only)
        if not flow_frozen:
            out_f2 = flow_net_({"x": x_all[idx], "y": y_all[idx], "t": t_c})
            upstream_uv = torch.cat([
                adj_u_coupling[idx].detach() * ratio,
                adj_v_coupling[idx].detach() * ratio,
                torch.zeros(bwd_batch, 1, device=device),
            ], dim=1)
            pred_f2 = torch.cat([out_f2["u"], out_f2["v"], out_f2["p"]], dim=1)
            pred_f2.backward(gradient=upstream_uv)

        return loss_val.detach()

    # -----------------------------------------------------------------
    # FD mini-batch SAGE: same 512-point sampling as baseline
    # -----------------------------------------------------------------
    fd_dx = float(getattr(cfg.training, "sage_fd_dx", 1e-3))
    pde_batch = int(cfg.training.pde_batch_size)  # 512, same as baseline

    def ns_pde_loss_fd(flow_net_, t_val):
        """NS PDE loss via FD-SAGE on mini-batch (same sampling as baseline).

        7 forward passes on 512 random points (no autograd graph),
        then SAGE backward + 7 network backward passes.
        """
        N = N_inside
        # Sample random mini-batch (same as baseline)
        idx = torch.randperm(N, device=device)[:pde_batch]
        x_b = x_all[idx]  # (B, 1)
        y_b = y_all[idx]
        dx_s = fd_dx
        dy_s = fd_dx
        dt_s = 0.005

        # 7 forward passes (no_grad) at stencil points
        def fwd(xp, yp, tp):
            with torch.no_grad():
                out = flow_net_({"x": xp, "y": yp, "t": tp})
            return out["u"], out["v"], out["p"]

        B = pde_batch
        t_b = torch.full((B, 1), t_val, dtype=torch.float32, device=device)

        u0, v0, p0 = fwd(x_b, y_b, t_b)                      # center
        u_xp, v_xp, p_xp = fwd(x_b + dx_s, y_b, t_b)        # x+dx
        u_xm, v_xm, p_xm = fwd(x_b - dx_s, y_b, t_b)        # x-dx
        u_yp, v_yp, p_yp = fwd(x_b, y_b + dy_s, t_b)        # y+dy
        u_ym, v_ym, p_ym = fwd(x_b, y_b - dy_s, t_b)        # y-dy
        u_tp, v_tp, _    = fwd(x_b, y_b, t_b + dt_s)         # t+dt
        u_tm, v_tm, _    = fwd(x_b, y_b, t_b - dt_s)         # t-dt

        # FD derivatives
        u_x = (u_xp - u_xm) / (2 * dx_s)
        u_y = (u_yp - u_ym) / (2 * dy_s)
        u_t = (u_tp - u_tm) / (2 * dt_s)
        u_xx = (u_xp - 2 * u0 + u_xm) / (dx_s ** 2)
        u_yy = (u_yp - 2 * u0 + u_ym) / (dy_s ** 2)

        v_x = (v_xp - v_xm) / (2 * dx_s)
        v_y = (v_yp - v_ym) / (2 * dy_s)
        v_t = (v_tp - v_tm) / (2 * dt_s)
        v_xx = (v_xp - 2 * v0 + v_xm) / (dx_s ** 2)
        v_yy = (v_yp - 2 * v0 + v_ym) / (dy_s ** 2)

        p_x = (p_xp - p_xm) / (2 * dx_s)
        p_y = (p_yp - p_ym) / (2 * dy_s)

        # PDE residuals
        cont = u_x + v_y
        mom_x = u_t + u0 * u_x + v0 * u_y + inv_rho * p_x - nu * (u_xx + u_yy)
        mom_y = v_t + u0 * v_x + v0 * v_y + inv_rho * p_y - nu * (v_xx + v_yy)

        loss_val = (cont.pow(2).sum() + mom_x.pow(2).sum() + mom_y.pow(2).sum()) / B
        scale = 2.0 / B
        dc = cont * scale
        dmu = mom_x * scale
        dmv = mom_y * scale

        # SAGE backward: adjoint for network output at each stencil point
        # Center (x, y, t): contributions from direct u,v,p terms in PDE
        adj_u0 = dmu * u_x + dmv * v_x  # d(mom_x)/d(u) = u_x, d(mom_y)/d(u) = v_x
        adj_v0 = dmu * u_y + dmv * v_y
        adj_p0 = torch.zeros_like(p0)
        # Also second-derivative FD contributions to center: -2/dx^2 and -2/dy^2
        adj_u0 = adj_u0 + (-nu * dmu) * (-2.0 / dx_s**2) + (-nu * dmu) * (-2.0 / dy_s**2)
        adj_v0 = adj_v0 + (-nu * dmv) * (-2.0 / dx_s**2) + (-nu * dmv) * (-2.0 / dy_s**2)

        # x+dx stencil: u_x contrib + u_xx contrib + p_x contrib + cont contrib
        c_fx = 1.0 / (2 * dx_s)  # FD coeff for first deriv
        c_sx = 1.0 / dx_s**2      # FD coeff for second deriv
        adj_u_xp = dc * c_fx + dmu * u0 * c_fx + dmu * (-nu) * c_sx
        adj_v_xp = dmv * u0 * c_fx + dmv * (-nu) * c_sx
        adj_p_xp = dmu * inv_rho * c_fx

        # x-dx stencil
        adj_u_xm = dc * (-c_fx) + dmu * u0 * (-c_fx) + dmu * (-nu) * c_sx
        adj_v_xm = dmv * u0 * (-c_fx) + dmv * (-nu) * c_sx
        adj_p_xm = dmu * inv_rho * (-c_fx)

        # y+dy stencil
        c_fy = 1.0 / (2 * dy_s)
        c_sy = 1.0 / dy_s**2
        adj_u_yp = dmu * v0 * c_fy + dmu * (-nu) * c_sy
        adj_v_yp = dc * c_fy + dmv * v0 * c_fy + dmv * (-nu) * c_sy
        adj_p_yp = dmv * inv_rho * c_fy

        # y-dy stencil
        adj_u_ym = dmu * v0 * (-c_fy) + dmu * (-nu) * c_sy
        adj_v_ym = dc * (-c_fy) + dmv * v0 * (-c_fy) + dmv * (-nu) * c_sy
        adj_p_ym = dmv * inv_rho * (-c_fy)

        # t+dt, t-dt stencils (time derivative)
        c_ft = 1.0 / (2 * dt_s)
        adj_u_tp = dmu * c_ft
        adj_v_tp = dmv * c_ft
        adj_u_tm = dmu * (-c_ft)
        adj_v_tm = dmv * (-c_ft)

        # 7 network backward passes (each on B=512 points, with grad)
        stencil_points = [
            (x_b, y_b, t_b, adj_u0, adj_v0, adj_p0),
            (x_b + dx_s, y_b, t_b, adj_u_xp, adj_v_xp, adj_p_xp),
            (x_b - dx_s, y_b, t_b, adj_u_xm, adj_v_xm, adj_p_xm),
            (x_b, y_b + dy_s, t_b, adj_u_yp, adj_v_yp, adj_p_yp),
            (x_b, y_b - dy_s, t_b, adj_u_ym, adj_v_ym, adj_p_ym),
            (x_b, y_b, t_b + dt_s, adj_u_tp, adj_v_tp, torch.zeros_like(p0)),
            (x_b, y_b, t_b - dt_s, adj_u_tm, adj_v_tm, torch.zeros_like(p0)),
        ]

        for xp, yp, tp, au, av, ap in stencil_points:
            out = flow_net_({"x": xp, "y": yp, "t": tp})
            upstream = torch.cat([au.detach(), av.detach(), ap.detach()], dim=1)
            pred = torch.cat([out["u"], out["v"], out["p"]], dim=1)
            pred.backward(gradient=upstream)

        return loss_val.detach()

    def advdiff_pde_loss_fd(flow_net_, temp_net_, t_val, flow_frozen):
        """AdvDiff PDE loss via FD-SAGE on mini-batch."""
        N = N_inside
        idx = torch.randperm(N, device=device)[:pde_batch]
        x_b, y_b = x_all[idx], y_all[idx]
        B = pde_batch
        dx_s, dy_s, dt_s = fd_dx, fd_dx, 0.005
        t_b = torch.full((B, 1), t_val, dtype=torch.float32, device=device)

        def fwd_temp(xp, yp, tp):
            with torch.no_grad():
                of = flow_net_({"x": xp, "y": yp, "t": tp})
                ot = temp_net_({"x": xp, "y": yp, "t": tp,
                               "u": of["u"], "v": of["v"]})
            return ot["T"], of["u"], of["v"]

        T0, uf0, vf0 = fwd_temp(x_b, y_b, t_b)
        T_xp, _, _ = fwd_temp(x_b + dx_s, y_b, t_b)
        T_xm, _, _ = fwd_temp(x_b - dx_s, y_b, t_b)
        T_yp, _, _ = fwd_temp(x_b, y_b + dy_s, t_b)
        T_ym, _, _ = fwd_temp(x_b, y_b - dy_s, t_b)
        T_tp, _, _ = fwd_temp(x_b, y_b, t_b + dt_s)
        T_tm, _, _ = fwd_temp(x_b, y_b, t_b - dt_s)

        T_x = (T_xp - T_xm) / (2 * dx_s)
        T_y = (T_yp - T_ym) / (2 * dy_s)
        T_t = (T_tp - T_tm) / (2 * dt_s)
        T_xx = (T_xp - 2 * T0 + T_xm) / dx_s**2
        T_yy = (T_yp - 2 * T0 + T_ym) / dy_s**2

        residual = T_t + uf0 * T_x + vf0 * T_y - alpha * (T_xx + T_yy)
        loss_val = residual.pow(2).mean()
        scale = 2.0 / B
        dr = residual * scale

        # SAGE backward for temp_net at each stencil point
        c_fx, c_sx = 1.0 / (2 * dx_s), 1.0 / dx_s**2
        c_fy, c_sy = 1.0 / (2 * dy_s), 1.0 / dy_s**2
        c_ft = 1.0 / (2 * dt_s)

        adj_T0 = dr * (-alpha) * (-2.0 / dx_s**2 + -2.0 / dy_s**2)
        adj_T_xp = dr * (uf0 * c_fx - alpha * c_sx)
        adj_T_xm = dr * (uf0 * (-c_fx) - alpha * c_sx)  # note: -alpha*c_sx is +, since (T_xm coeff in T_xx is +1/dx^2)
        adj_T_yp = dr * (vf0 * c_fy - alpha * c_sy)
        adj_T_ym = dr * (vf0 * (-c_fy) - alpha * c_sy)
        adj_T_tp = dr * c_ft
        adj_T_tm = dr * (-c_ft)

        stencil = [
            (x_b, y_b, t_b, adj_T0),
            (x_b + dx_s, y_b, t_b, adj_T_xp),
            (x_b - dx_s, y_b, t_b, adj_T_xm),
            (x_b, y_b + dy_s, t_b, adj_T_yp),
            (x_b, y_b - dy_s, t_b, adj_T_ym),
            (x_b, y_b, t_b + dt_s, adj_T_tp),
            (x_b, y_b, t_b - dt_s, adj_T_tm),
        ]

        for xp, yp, tp, adj_T in stencil:
            with torch.no_grad() if flow_frozen else torch.enable_grad():
                of = flow_net_({"x": xp, "y": yp, "t": tp})
            ot = temp_net_({"x": xp, "y": yp, "t": tp,
                           "u": of["u"].detach() if flow_frozen else of["u"],
                           "v": of["v"].detach() if flow_frozen else of["v"]})
            ot["T"].backward(gradient=adj_T.detach())
            # Flow coupling: dR/du = T_x, dR/dv = T_y
            if not flow_frozen:
                of2 = flow_net_({"x": xp, "y": yp, "t": tp})
                adj_uv = torch.cat([dr.detach() * T_x.detach(),
                                    dr.detach() * T_y.detach(),
                                    torch.zeros_like(p0 if 'p0' in dir() else of2["p"])], dim=1)
                pred_f = torch.cat([of2["u"], of2["v"], of2["p"]], dim=1)
                # Only do this once (center point), not all stencil points
                if xp is x_b and tp is t_b:
                    pred_f.backward(gradient=adj_uv)

        return loss_val.detach()

    def bc_loss_flow(flow_net_, t_val):
        """Flow BC losses: wall no-slip + inlet + outlet."""
        losses = {}

        # Wall no-slip
        t_w = torch.full((N_wall, 1), t_val, dtype=torch.float32, device=device)
        out_w = flow_net_({"x": x_wall, "y": y_wall, "t": t_w})
        losses["wall_u"] = out_w["u"].pow(2).mean()
        losses["wall_v"] = out_w["v"].pow(2).mean()

        # Inlet
        if inlet_x is not None:
            t_in = torch.full_like(inlet_x, t_val)
            out_in = flow_net_({"x": inlet_x, "y": inlet_y, "t": t_in})
            losses["inlet_u"] = (out_in["u"] - inlet_u).pow(2).mean()
            losses["inlet_v"] = out_in["v"].pow(2).mean()

        # Outlet p=0
        if outlet_x is not None:
            t_out = torch.full_like(outlet_x, t_val)
            out_out = flow_net_({"x": outlet_x, "y": outlet_y, "t": t_out})
            losses["outlet_p"] = out_out["p"].pow(2).mean()

        return sum(losses.values())

    bc_batch = int(cfg.training.bc_batch_size)  # 128, same as baseline

    def bc_loss_temp(flow_net_, temp_net_, t_val, flow_frozen):
        """Temperature BC/IC losses (mini-batched, same as baseline)."""
        losses = {}

        # IC at t=0: T = T_init (sub-sample)
        ic_idx = torch.randperm(N_ic, device=device)[:bc_batch]
        t_0 = torch.zeros((bc_batch, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            out_f0 = flow_net_({"x": x_ic[ic_idx], "y": y_ic[ic_idx], "t": t_0})
        out_t0 = temp_net_({"x": x_ic[ic_idx], "y": y_ic[ic_idx], "t": t_0,
                           "u": out_f0["u"], "v": out_f0["v"]})
        losses["ic_T"] = (out_t0["T"] - T_init).pow(2).mean()

        # Arrival-time Dirichlet: T = inlet_T at t_arrive (sub-sample)
        arr_idx = torch.randperm(N_inside, device=device)[:bc_batch]
        with torch.no_grad():
            out_fa = flow_net_({"x": x_all[arr_idx], "y": y_all[arr_idx],
                               "t": t_arrive_tensor[arr_idx]})
        out_ta = temp_net_({"x": x_all[arr_idx], "y": y_all[arr_idx],
                           "t": t_arrive_tensor[arr_idx],
                           "u": out_fa["u"], "v": out_fa["v"]})
        losses["arrival_T"] = (out_ta["T"] - inlet_T).pow(2).mean()

        # Pre-arrival: T = T_init for t < t_arrive (sub-sample)
        pre_idx = torch.randperm(N_inside, device=device)[:bc_batch]
        frac = torch.rand(bc_batch, 1, device=device) * 0.95
        t_pre = frac * t_arrive_tensor[pre_idx]
        t_pre = t_pre.clamp(min=t_min)
        with torch.no_grad():
            out_fp = flow_net_({"x": x_all[pre_idx], "y": y_all[pre_idx], "t": t_pre})
        out_tp = temp_net_({"x": x_all[pre_idx], "y": y_all[pre_idx], "t": t_pre,
                           "u": out_fp["u"], "v": out_fp["v"]})
        losses["pre_arrival_T"] = (out_tp["T"] - T_init).pow(2).mean()

        # Inlet T
        if inlet_x is not None:
            t_in = torch.full_like(inlet_x, t_val)
            with torch.no_grad():
                out_fi = flow_net_({"x": inlet_x, "y": inlet_y, "t": t_in})
            out_ti = temp_net_({"x": inlet_x, "y": inlet_y, "t": t_in,
                               "u": out_fi["u"], "v": out_fi["v"]})
            losses["inlet_T"] = (out_ti["T"] - inlet_T).pow(2).mean()

        return sum(losses.values())

    # -----------------------------------------------------------------
    # 7) Training loop
    # -----------------------------------------------------------------
    k1 = int(getattr(cfg.training, "k_flow", 5000))
    k2 = int(getattr(cfg.training, "k_temp", 5000))
    k3 = int(getattr(cfg.training, "k_joint", 10000))
    lr = float(cfg.training.lr)
    grad_clip = float(getattr(cfg.training, "grad_clip_norm", 1.0))

    print(f"\n[sage] Three-stage training: flow={k1}, temp={k2}, joint={k3}")

    start_time = time.time()

    def train_stage(stage_name, n_steps, trainable_nets, loss_fn):
        """Generic training stage.

        loss_fn() returns a detached scalar (PDE loss handles its own backward
        via chunked processing). BC losses return autograd tensors that need .backward().
        """
        params = []
        for net in trainable_nets:
            params.extend(net.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg.training.lr_decay_steps),
            gamma=float(cfg.training.lr_decay_rate),
        )

        t0 = time.time()
        for step in range(1, n_steps + 1):
            optimizer.zero_grad()
            loss_val = loss_fn()
            # PDE loss handles its own backward via chunked processing
            # BC losses may still need backward (returned as autograd tensors)
            if loss_val.requires_grad:
                loss_val.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0 or step == 1:
                elapsed = time.time() - t0
                ms_step = 1000 * elapsed / step
                lv = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
                print(f"  [{stage_name}] step {step}/{n_steps}: "
                      f"loss={lv:.6f}, {ms_step:.1f} ms/step")

        elapsed = time.time() - t0
        ms_step = 1000 * elapsed / n_steps
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  [{stage_name}] done: {elapsed/60:.2f} min, {ms_step:.1f} ms/step, "
              f"peak GPU={peak_mem:.2f} GB")
        return elapsed

    # Select PDE loss function based on mode
    sage_mode = str(getattr(cfg.training, "sage_mode", "fd"))
    if sage_mode == "fd":
        _ns_pde = ns_pde_loss_fd
        _ad_pde = advdiff_pde_loss_fd
        print(f"[sage] Using FD mini-batch mode (batch={pde_batch}, dx={fd_dx})")
    else:
        _ns_pde = ns_pde_loss
        _ad_pde = advdiff_pde_loss
        print(f"[sage] Using full-batch RKPM mode")

    # Stage 1: Flow only
    print(f"\n{'='*70}\nSTAGE 1: Flow only ({k1} steps)\n{'='*70}")
    set_requires_grad(flow_net, True)
    set_requires_grad(temp_net, False)
    torch.cuda.reset_peak_memory_stats()

    def stage1_loss():
        t_val = sample_time()
        pde_val = _ns_pde(flow_net, t_val)
        bc = bc_loss_flow(flow_net, t_val)
        bc.backward()
        return pde_val

    stage1_time = train_stage("stage1", k1, [flow_net], stage1_loss)

    # Stage 2: Temperature only (frozen flow)
    print(f"\n{'='*70}\nSTAGE 2: Temperature, frozen flow ({k2} steps)\n{'='*70}")
    set_requires_grad(flow_net, False)
    set_requires_grad(temp_net, True)

    def stage2_loss():
        t_val = sample_time()
        pde_val = _ad_pde(flow_net, temp_net, t_val, flow_frozen=True)
        bc = bc_loss_temp(flow_net, temp_net, t_val, flow_frozen=True)
        bc.backward()
        return pde_val

    stage2_time = train_stage("stage2", k2, [temp_net], stage2_loss)

    # Stage 3: Joint
    print(f"\n{'='*70}\nSTAGE 3: Joint training ({k3} steps)\n{'='*70}")
    set_requires_grad(flow_net, True)
    set_requires_grad(temp_net, True)

    def stage3_loss():
        t_val = sample_time()
        pde_ns_val = _ns_pde(flow_net, t_val)
        pde_ad_val = _ad_pde(flow_net, temp_net, t_val, flow_frozen=False)
        bc = bc_loss_flow(flow_net, t_val) + bc_loss_temp(flow_net, temp_net, t_val, flow_frozen=False)
        bc.backward()
        return pde_ns_val + pde_ad_val

    stage3_time = train_stage("stage3", k3, [flow_net, temp_net], stage3_loss)

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"SAGE TRAINING COMPLETE")
    print(f"  Stage 1 (flow):  {stage1_time / 60:.2f} min")
    print(f"  Stage 2 (temp):  {stage2_time / 60:.2f} min")
    print(f"  Stage 3 (joint): {stage3_time / 60:.2f} min")
    print(f"  Total:           {total_time / 60:.2f} min")
    print(f"{'='*70}")

    # -----------------------------------------------------------------
    # 8) PDE RMSE evaluation
    # -----------------------------------------------------------------
    print("\n[eval] Computing PDE RMSE...")
    xy_tensor = torch.tensor(xy_inside, dtype=torch.float32, device=device)
    t_eval = [0.0, 2.5, 5.0, 7.5, 10.0]
    physics = {"nu": nu, "rho": rho, "alpha": alpha}
    rmse = compute_pde_rmse(flow_net, temp_net, xy_tensor, rkpm, physics, t_eval, device)
    for k, v in rmse.items():
        print(f"  {k}: {v:.6f}")

    # -----------------------------------------------------------------
    # 9) Inference (same as baseline)
    # -----------------------------------------------------------------
    flow_net.eval()
    temp_net.eval()
    infer_model = FlowThenTempWrapper(flow_net, temp_net).to(device).eval()

    xy = np.vstack([
        np.concatenate([x_w, y_w], 1),
        np.concatenate([x_i, y_i], 1),
    ]).astype(np.float32)
    point_type = np.concatenate([
        np.full((x_w.shape[0],), 1, np.int32),
        np.full((x_i.shape[0],), 2, np.int32),
    ])

    t0_inf = float(cfg.problem.infer_t_start)
    t1_inf = float(cfg.problem.infer_t_end)
    dt_inf = float(cfg.problem.infer_dt)
    times = np.arange(t0_inf, t1_inf + 1e-9, dt_inf, dtype=np.float32)
    temps = np.zeros((times.shape[0], xy.shape[0]), dtype=np.float32)

    print(f"\n[infer] Running inference on {xy.shape[0]} points x {len(times)} timesteps...")
    with torch.no_grad():
        N = xy.shape[0]
        batch = 65536
        for ti, tval in enumerate(times):
            for s in range(0, N, batch):
                e = min(s + batch, N)
                xt = torch.from_numpy(xy[s:e, 0:1]).to(device)
                yt = torch.from_numpy(xy[s:e, 1:2]).to(device)
                tt = torch.full_like(xt, float(tval))
                out = infer_model({"x": xt, "y": yt, "t": tt})
                temps[ti, s:e] = out["T"].cpu().numpy().reshape(-1)

    out_json_path = str(outdir / "pred_T.json")
    _write_inference_json(out_json_path, xy, times, temps, norm=norm, point_type=point_type)
    print(f"[infer] Wrote: {out_json_path}")

    # Save summary
    summary = {
        "mode": "sage",
        "stage1_flow_min": stage1_time / 60,
        "stage2_temp_min": stage2_time / 60,
        "stage3_joint_min": stage3_time / 60,
        "total_min": total_time / 60,
        "flow_params": flow_params,
        "temp_params": temp_params,
        "inside_points": int(x_i.shape[0]),
        "wall_points": int(x_w.shape[0]),
        "k_flow": k1, "k_temp": k2, "k_joint": k3,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "pde_rmse": rmse,
    }
    summary_path = str(outdir / "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {summary_path}")


@hydra.main(version_base=None, config_path="conf", config_name="partner_v3_config")
def run_main(cfg: DictConfig) -> None:
    use_sage = bool(getattr(cfg.training, "use_sage", False))
    if use_sage:
        run_sage(cfg)
    else:
        run(cfg)


if __name__ == "__main__":
    run_main()
