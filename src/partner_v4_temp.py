import json
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import h5py
import hydra
import numpy as np
import torch
import sys
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

from pathlib import Path
from pcs_runtime import cad_to_geometry_json, derive_flow_json_path, resolve_cad_path

if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")


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


def _load_geometry_ports(
    json_path: str,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Tuple[float, float, float, float]]:
    obj = json.loads(Path(json_path).read_text())
    pts = obj["points"]
    inlet = obj.get("inlet", None)
    outlet = obj.get("outlet", None)

    all_x, all_y = [], []
    for p in pts:
        xr, yr, typ = float(p[0]), float(p[1]), int(p[2])
        if typ == 0:
            continue
        all_x.append(xr)
        all_y.append(yr)

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
        raise ValueError(
            f"flow_time_index={flow_time_index} is out of range for field with shape {arr.shape}"
        )
    return arr[flow_time_index]


def _load_flow_field_json(json_path: str, flow_time_index: int = 0) -> FlowFieldData:
    obj = json.loads(Path(json_path).read_text())

    if "fields" not in obj:
        raise ValueError("Flow JSON must contain a top-level 'fields' object.")
    if "u" not in obj["fields"] or "v" not in obj["fields"]:
        raise ValueError("Flow JSON fields must contain both 'u' and 'v'.")

    xy_norm = np.asarray(obj["xy"], dtype=np.float32)
    point_type = np.asarray(obj["point_type"], dtype=np.int32)
    norm = tuple(float(v) for v in obj["norm"])

    if xy_norm.ndim != 2 or xy_norm.shape[1] != 2:
        raise ValueError(f"Expected xy to have shape (N,2), got {xy_norm.shape}")
    if point_type.shape[0] != xy_norm.shape[0]:
        raise ValueError("point_type length must match number of xy points.")

    u = _select_flow_snapshot(obj["fields"]["u"], flow_time_index)
    v = _select_flow_snapshot(obj["fields"]["v"], flow_time_index)
    p = None
    if "p" in obj["fields"]:
        p = _select_flow_snapshot(obj["fields"]["p"], flow_time_index)

    if u.shape[0] != xy_norm.shape[0] or v.shape[0] != xy_norm.shape[0]:
        raise ValueError("Flow field lengths do not match xy point count.")

    if "xy_raw" in obj:
        xy_raw = np.asarray(obj["xy_raw"], dtype=np.float32)
    else:
        xr, yr = _denormalize_xy(
            xy_norm[:, 0:1], xy_norm[:, 1:2], norm[0], norm[1], norm[2], norm[3]
        )
        xy_raw = np.concatenate([xr, yr], axis=1).astype(np.float32)

    flow_times = None
    if "times" in obj:
        flow_times = np.asarray(obj["times"], dtype=np.float32)

    return FlowFieldData(
        xy_norm=xy_norm,
        xy_raw=xy_raw,
        point_type=point_type,
        u=np.asarray(u, dtype=np.float32),
        v=np.asarray(v, dtype=np.float32),
        p=None if p is None else np.asarray(p, dtype=np.float32),
        norm=norm,
        flow_times=flow_times,
    )


def _normalize_port(
    port: Optional[Dict[str, float]], norm: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    if port is None:
        return None
    xr = np.asarray([[float(port["x"])]], dtype=np.float32)
    yr = np.asarray([[float(port["y"])]], dtype=np.float32)
    xn, yn = _normalize_xy(xr, yr, norm[0], norm[1], norm[2], norm[3])
    return np.asarray([[float(xn[0, 0]), float(yn[0, 0])]], dtype=np.float32)


def compute_geodesic_distance_from_inlet(
    xy_inside: np.ndarray,
    inlet_xy: np.ndarray,
    k: int = 8,
    max_edge_len: Optional[float] = None,
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
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([d, d])

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


class TemperatureNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, hidden_layers: int, activation: str):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden_size), get_activation(activation)]
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _sample_indices(size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if size <= 0:
        raise ValueError("Cannot sample from an empty set.")
    return torch.randint(0, size, (min(size, batch_size),), device=device)


def _make_time_samples(n: int, t_min: float, t_max: float, device: torch.device) -> torch.Tensor:
    u = torch.rand((n, 1), device=device)
    return t_min + (t_max - t_min) * (u ** 2)


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def _forward_temperature(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    inp = torch.cat([x, y, t, u, v], dim=1)
    return model(inp)


def _save_h5(
    path: str,
    xy_norm: np.ndarray,
    xy_raw: np.ndarray,
    point_type: np.ndarray,
    times: np.ndarray,
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    norm: Tuple[float, float, float, float],
    flow_json_path: str,
    geom_json_path: str,
) -> None:
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("xy_norm", data=xy_norm)
        h5f.create_dataset("xy_raw", data=xy_raw)
        h5f.create_dataset("point_type", data=point_type)
        h5f.create_dataset("times", data=times)
        h5f.create_dataset("temperature", data=temperature)
        h5f.create_dataset("u", data=u)
        h5f.create_dataset("v", data=v)
        h5f.attrs["xmin"] = float(norm[0])
        h5f.attrs["xmax"] = float(norm[1])
        h5f.attrs["ymin"] = float(norm[2])
        h5f.attrs["ymax"] = float(norm[3])
        h5f.attrs["flow_json_path"] = flow_json_path
        h5f.attrs["geom_json_path"] = geom_json_path


def _run_visualization(geom_path: str, pred_path: str, outdir: Path) -> None:
    viz_script = Path(__file__).resolve().parent / "visualize_partner_v4.py"
    if not viz_script.exists():
        print(f"[WARN] visualize script not found at: {viz_script}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(viz_script),
        "--geom",
        geom_path,
        "--pred",
        pred_path,
        "--outdir",
        str(outdir),
        "--field",
        "temperature",
        "--gif",
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] visualization artifacts written to: {outdir}")
    except Exception as exc:
        print(f"[WARN] visualization step failed: {exc}")


@hydra.main(version_base=None, config_path="conf", config_name="partner_v4_config")
def run(cfg: DictConfig) -> None:
    cfg = cfg.temp if "temp" in cfg else cfg
    torch.manual_seed(int(cfg.training.seed))
    np.random.seed(int(cfg.training.seed))

    cad_path = resolve_cad_path()  # required CAD source
    geom_json_path = cad_to_geometry_json(
        cad_path=cad_path,
        output_dir=Path.cwd(),
        res=int(getattr(cfg.problem, "pcs_res", 512)),
        strip_w=int(getattr(cfg.problem, "pcs_strip_w", 10)),
        white_thr=int(getattr(cfg.problem, "pcs_white_thr", 250)),
    )  # convert CAD to geometry json via PCS
    flow_json = derive_flow_json_path(geom_json_path)  # expected flow output json
    if not flow_json.exists():
        raise FileNotFoundError(
            f"Flow JSON not found: {flow_json}. Run train_and_infer_flow_net.py with the same CAD path first."
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
        xy_inside,
        inlet_xy,
        k=int(cfg.problem.geodesic_knn_k),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemperatureNet(
        in_dim=5,
        hidden_size=int(cfg.model.hidden_size),
        hidden_layers=int(cfg.model.hidden_layers),
        activation=str(cfg.model.activation),
    ).to(device)

    ckpt_path = network_dir / "temperature_net.pt"

    if str(cfg.run_mode).lower() == "eval":
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for eval mode: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        x_all_t = torch.from_numpy(xy[:, 0:1]).to(device)
        y_all_t = torch.from_numpy(xy[:, 1:2]).to(device)
        u_all_t = torch.from_numpy(u_all).to(device)
        v_all_t = torch.from_numpy(v_all).to(device)

        x_inside_t = torch.from_numpy(xy_inside[:, 0:1]).to(device)
        y_inside_t = torch.from_numpy(xy_inside[:, 1:2]).to(device)
        u_inside_t = torch.from_numpy(u_inside).to(device)
        v_inside_t = torch.from_numpy(v_inside).to(device)
        t_arrive_t = torch.from_numpy(t_arrive_inside).to(device)

        inlet_idx_t = None
        if np.any(inlet_mask_inside):
            inlet_idx_t = torch.from_numpy(np.where(inlet_mask_inside)[0].astype(np.int64)).to(device)

        outlet_idx_t = None
        if outlet_mask_inside is not None and np.any(outlet_mask_inside):
            outlet_idx_t = torch.from_numpy(np.where(outlet_mask_inside)[0].astype(np.int64)).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.training.lr),
            betas=tuple(float(v) for v in cfg.optimizer.betas),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.training.lr_decay_steps),
            gamma=float(cfg.training.lr_decay_rate),
        )

        loss_weights = cfg.training.loss_weights
        t_min = float(cfg.problem.t_min)
        t_max = float(cfg.problem.t_max)
        t_init_val = float(cfg.problem.T_init)
        inlet_t_val = float(cfg.bc.inlet_T)
        grad_clip = float(cfg.training.grad_clip_max_norm)

        for step in range(1, int(cfg.training.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)

            pde_idx = _sample_indices(x_inside_t.shape[0], int(cfg.training.pde_batch_size), device)
            x_pde = x_inside_t[pde_idx].clone().detach().requires_grad_(True)
            y_pde = y_inside_t[pde_idx].clone().detach().requires_grad_(True)
            t_pde = _make_time_samples(x_pde.shape[0], t_min, t_max, device).requires_grad_(True)
            u_pde = u_inside_t[pde_idx]
            v_pde = v_inside_t[pde_idx]

            T_pde = _forward_temperature(model, x_pde, y_pde, t_pde, u_pde, v_pde)
            T_x = _grad(T_pde, x_pde)
            T_y = _grad(T_pde, y_pde)
            T_t = _grad(T_pde, t_pde)
            T_xx = _grad(T_x, x_pde)
            T_yy = _grad(T_y, y_pde)

            residual = (
                T_t
                + u_pde * T_x
                + v_pde * T_y
                - float(cfg.physics.D) * (T_xx + T_yy)
                - float(cfg.physics.Q)
            )
            loss_pde = torch.mean(residual ** 2)

            ic_idx = _sample_indices(x_all_t.shape[0], int(cfg.training.bc_batch_size), device)
            T_ic = _forward_temperature(
                model,
                x_all_t[ic_idx],
                y_all_t[ic_idx],
                torch.zeros((ic_idx.shape[0], 1), device=device),
                u_all_t[ic_idx],
                v_all_t[ic_idx],
            )
            loss_ic = torch.mean((T_ic - t_init_val) ** 2)

            arr_idx = _sample_indices(x_inside_t.shape[0], int(cfg.training.bc_batch_size), device)
            T_arrive = _forward_temperature(
                model,
                x_inside_t[arr_idx],
                y_inside_t[arr_idx],
                t_arrive_t[arr_idx],
                u_inside_t[arr_idx],
                v_inside_t[arr_idx],
            )
            loss_arrival = torch.mean((T_arrive - inlet_t_val) ** 2)

            pre_idx = _sample_indices(x_inside_t.shape[0], int(cfg.training.bc_batch_size), device)
            frac = 0.95 * torch.rand((pre_idx.shape[0], 1), device=device)
            t_pre = torch.clamp(frac * t_arrive_t[pre_idx], min=t_min)
            T_pre = _forward_temperature(
                model,
                x_inside_t[pre_idx],
                y_inside_t[pre_idx],
                t_pre,
                u_inside_t[pre_idx],
                v_inside_t[pre_idx],
            )
            loss_pre = torch.mean((T_pre - t_init_val) ** 2)

            if inlet_idx_t is not None and inlet_idx_t.numel() > 0:
                inlet_pick = inlet_idx_t[_sample_indices(inlet_idx_t.shape[0], int(cfg.training.bc_batch_size), device)]
                T_inlet = _forward_temperature(
                    model,
                    x_inside_t[inlet_pick],
                    y_inside_t[inlet_pick],
                    _make_time_samples(inlet_pick.shape[0], t_min, t_max, device),
                    u_inside_t[inlet_pick],
                    v_inside_t[inlet_pick],
                )
                loss_inlet = torch.mean((T_inlet - inlet_t_val) ** 2)
            else:
                loss_inlet = torch.zeros((), device=device)

            if outlet_idx_t is not None and outlet_idx_t.numel() > 0:
                outlet_pick = outlet_idx_t[_sample_indices(outlet_idx_t.shape[0], int(cfg.training.bc_batch_size), device)]
                x_out = x_inside_t[outlet_pick].clone().detach().requires_grad_(True)
                y_out = y_inside_t[outlet_pick].clone().detach().requires_grad_(True)
                t_out = _make_time_samples(outlet_pick.shape[0], t_min, t_max, device).requires_grad_(True)
                T_out = _forward_temperature(
                    model,
                    x_out,
                    y_out,
                    t_out,
                    u_inside_t[outlet_pick],
                    v_inside_t[outlet_pick],
                )
                T_out_x = _grad(T_out, x_out)
                loss_outlet = torch.mean(T_out_x ** 2)
            else:
                loss_outlet = torch.zeros((), device=device)

            total_loss = (
                float(loss_weights.pde) * loss_pde
                + float(loss_weights.ic) * loss_ic
                + float(loss_weights.arrival) * loss_arrival
                + float(loss_weights.pre_arrival) * loss_pre
                + float(loss_weights.inlet) * loss_inlet
                + float(loss_weights.outlet) * loss_outlet
            )
            total_loss.backward()

            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            scheduler.step()

            if step % int(cfg.training.print_stats_freq) == 0 or step == 1:
                print(
                    f"[step {step:06d}] "
                    f"loss={total_loss.item():.6e} "
                    f"pde={loss_pde.item():.6e} "
                    f"ic={loss_ic.item():.6e} "
                    f"arrival={loss_arrival.item():.6e} "
                    f"pre={loss_pre.item():.6e} "
                    f"inlet={loss_inlet.item():.6e} "
                    f"outlet={loss_outlet.item():.6e}"
                )

            if step % int(cfg.training.save_network_freq) == 0 or step == int(cfg.training.max_steps):
                torch.save(model.state_dict(), ckpt_path)
                print(f"[OK] saved checkpoint to {ckpt_path}")

    model.eval()

    times = np.arange(
        float(cfg.problem.infer_t_start),
        float(cfg.problem.infer_t_end) + 1e-9,
        float(cfg.problem.infer_dt),
        dtype=np.float32,
    )
    temperature = np.zeros((times.shape[0], xy.shape[0]), dtype=np.float32)
    batch = int(cfg.inference.batch_size)

    with torch.no_grad():
        x_all_t = torch.from_numpy(xy[:, 0:1]).to(device)
        y_all_t = torch.from_numpy(xy[:, 1:2]).to(device)
        u_all_t = torch.from_numpy(u_all).to(device)
        v_all_t = torch.from_numpy(v_all).to(device)

        for ti, tval in enumerate(times):
            for start in range(0, xy.shape[0], batch):
                end = min(start + batch, xy.shape[0])
                t_batch = torch.full((end - start, 1), float(tval), device=device)
                pred = _forward_temperature(
                    model,
                    x_all_t[start:end],
                    y_all_t[start:end],
                    t_batch,
                    u_all_t[start:end],
                    v_all_t[start:end],
                )
                temperature[ti, start:end] = pred.squeeze(1).cpu().numpy()

    output_path = network_dir / str(cfg.inference.output_filename)
    _save_h5(
        str(output_path),
        xy_norm=xy,
        xy_raw=xy_raw,
        point_type=point_type,
        times=times,
        temperature=temperature,
        u=flow.u.astype(np.float32),
        v=flow.v.astype(np.float32),
        norm=norm,
        flow_json_path=flow_json_path,
        geom_json_path=geom_path,
    )
    print(f"[OK] wrote temperature predictions to {output_path}")

    if str(cfg.run_mode).lower() != "eval":
        _run_visualization(
            geom_path=geom_path,
            pred_path=str(output_path),
            outdir=network_dir / "visualizations",
        )


if __name__ == "__main__":
    run()
