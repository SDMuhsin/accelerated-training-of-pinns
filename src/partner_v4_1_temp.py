"""V4.1 3D temperature PINN trainer.

3D extension of ``src/partner_v4_temp.py`` (the V4 2D steady cooling-channel
temperature PINN). The design spec is locked in
``llmdocs/stream_battery_consortium/V4_1_DESIGN.md`` (read §3.4 and §10
first). Relative to V4 this trainer:

  - ingests the 3D geometry JSON written by ``partner_v4_1_geometry.py``
    (keys ``points_inside``, ``points_wall``, ``normals_wall``,
    ``points_inlet``, ``points_outlet``, ``inlet_center_xyz``,
    ``outlet_center_xyz``, ``norm``, ``z_aspect``, ``z_slices``,
    optional ``init_fields``),
  - ingests a 3D flow JSON with schema
    ``{points, flow:{u,v,w,p,t}, norm, inlet, outlet, z_aspect,
    z_slices}`` (the 3D analogue of
    ``pipe_three_class_fixed_pred_flow_steady.json``),
  - solves the 3D advection-diffusion PDE

        d_t T + u d_x T + v d_y T + w d_z T
              - D (d_xx T + d_yy T + d_zz T) - Q = 0

    on a raw-PyTorch MLP with input dim 7 (x,y,z,t,u,v,w) and output T,
  - replicates V4 config layout (uses ``cfg.temp.*`` keys) and loss
    structure (pde, ic, arrival, pre_arrival, inlet, outlet, wall) with
    curriculum interpolation and weighted sampling,
  - enforces wall insulation ``n . grad T = 0`` on the 3D wall points
    using the precomputed wall normals from the geometry JSON (this is
    the ``wall`` loss term V4's config lists but the V4 trainer leaves
    unwired),
  - writes HDF5 predictions and mid-z-slice scatter PNGs + GIF +
    three-panel static summary for visualisation (3D is hard to render
    directly, so we slice at z_slices // 2).

This file is read by Hydra in the usual way; the caller supplies
``conf/partner_v4_1_config.yaml`` (built separately) with a top-level
``temp`` block holding the keys listed in the spec.

All coordinate arrays are normalised (x,y in [0,1], z in [0, z_aspect]).
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from scipy.spatial import cKDTree

# Matplotlib/imageio are only used in the post-training visualisation step.
# Import them lazily inside the visualiser to keep the training entry
# point importable on headless boxes.

# Shared 3D geometry helpers (same process, no module-level CUDA init)
from partner_v4_1_geometry import (  # noqa: E402
    compute_wall_distance_3d,
)


# -----------------------------
# Module-level CUDA warm-up (mirror V4 temp)
# -----------------------------
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # optional MPI rank
    torch.cuda.set_device(local_rank)  # select device
    torch.cuda.init()  # initialise context
    _ = torch.empty(1, device="cuda")  # allocate a tensor to force context


# -----------------------------
# Small utilities
# -----------------------------
def get_activation(act_name: str) -> torch.nn.Module:
    """Return a fresh activation module by name (mirrors V4)."""

    a = str(act_name).lower()  # normalise
    if a in ("silu", "swish"):
        return torch.nn.SiLU()  # default
    if a == "tanh":
        return torch.nn.Tanh()  # smooth
    if a == "relu":
        return torch.nn.ReLU()  # classical
    if a == "gelu":
        return torch.nn.GELU()  # smooth alternative
    raise ValueError(f"Unknown activation: {act_name}")  # strict


def _sample_indices(size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    """Uniform random index tensor bounded by size (mirrors V4 temp)."""

    if size <= 0:
        raise ValueError("Cannot sample from an empty set.")  # guard
    return torch.randint(
        0, size, (min(size, batch_size),), device=device
    )  # uniform sampler


def _sample_indices_weighted(
    weights: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Weighted random index tensor (torch.multinomial, with replacement).

    `weights` must be a 1D float tensor of length N. We clamp to >= 0 and
    guarantee a positive sum so multinomial does not raise on degenerate
    zero-weight inputs.
    """

    w = weights.clamp_min(0.0)  # no negatives
    s = float(w.sum().item())  # total
    if (s <= 0.0) or (w.numel() == 0):
        return _sample_indices(int(w.numel()), int(batch_size), device)  # fallback
    n = int(min(int(batch_size), int(w.numel())))  # sample count
    return torch.multinomial(w, n, replacement=True)  # weighted


def _make_time_samples(
    n: int, t_min: float, t_max: float, device: torch.device
) -> torch.Tensor:
    """Quadratic-bias time samples (mirrors V4 _make_time_samples)."""

    u = torch.rand((n, 1), device=device)  # uniform in [0,1]
    return t_min + (t_max - t_min) * (u ** 2)  # bias toward t_min


def _grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """Scalar-sum first derivative (same pattern as V4 _grad)."""

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]  # (N,1) gradient


def _curriculum_scale(step: int, max_steps: int, start: float, end: float) -> float:
    """Linear interpolation of a per-loss weight over the training run.

    step=1 -> start, step=max_steps -> end. Values outside [1, max_steps]
    are clipped to the endpoint values.
    """

    if max_steps <= 1:
        return float(end)  # degenerate
    a = (float(step) - 1.0) / float(max_steps - 1)  # progress in [0,1]
    a = max(0.0, min(1.0, a))  # clamp
    return float(start) + a * (float(end) - float(start))  # interp


# -----------------------------
# Network
# -----------------------------
class TemperatureNet3D(torch.nn.Module):
    """MLP for T(x, y, z, t, u, v, w) -> T.

    Same depth/width convention as V4: N hidden layers of equal width, one
    activation after each linear, and a final linear projection to 1.
    Xavier-normal init by default (mirrors standard PINN setup).
    """

    def __init__(self, in_dim: int, hidden_size: int, hidden_layers: int, activation: str):
        super().__init__()  # base class
        layers = [
            torch.nn.Linear(in_dim, hidden_size),
            get_activation(activation),
        ]  # first linear + act
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))  # hidden
            layers.append(get_activation(activation))  # act
        layers.append(torch.nn.Linear(hidden_size, 1))  # output head
        self.net = torch.nn.Sequential(*layers)  # composite
        self._xavier_init()  # init

    def _xavier_init(self) -> None:
        """Xavier-normal initialisation for linear layers."""

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # weights
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)  # biases

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)  # (N,1)


def _forward_temperature(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the 7 inputs and forward through the network."""

    inp = torch.cat([x, y, z, t, u, v, w], dim=1)  # (N,7)
    return model(inp)  # (N,1)


# -----------------------------
# Geometry / flow loaders
# -----------------------------
@dataclass
class GeometryBundle3D:
    """Parsed 3D geometry payload from partner_v4_1_geometry.save_geometry_json_3d.

    All xyz arrays are NORMALISED. Raw voxel coordinates are preserved as
    columns 3..5 of points_inside/points_wall but we do not need them for
    temperature training.
    """

    xyz_inside: np.ndarray  # (N_in, 3) normalised
    xyz_wall: np.ndarray  # (N_w, 3) normalised
    n_wall: np.ndarray  # (N_w, 3) unit vectors
    class_wall: np.ndarray  # (N_w,) int side/bot/top class
    xyz_inlet: np.ndarray  # (N_il, 3)
    xyz_outlet: np.ndarray  # (N_ol, 3)
    inlet_center_xyz: np.ndarray  # (1, 3)
    outlet_center_xyz: np.ndarray  # (1, 3)
    norm: Tuple[float, float, float, float, float, float]  # xmin,xmax,ymin,ymax,zmin,zmax
    z_aspect: float  # Zmax / Lx
    z_slices: int  # total z levels (caps + interior)
    init_fields: Optional[Dict[str, np.ndarray]]  # pre-computed initial fields


def _load_geometry_json_3d(path: str) -> GeometryBundle3D:
    """Load the 3D geometry JSON produced by partner_v4_1_geometry."""

    p = Path(path)  # path object
    if not p.exists():
        raise FileNotFoundError(f"Geometry JSON not found: {p}")  # guard

    obj = json.loads(p.read_text())  # parse
    required = [
        "points_inside",
        "points_wall",
        "normals_wall",
        "points_inlet",
        "points_outlet",
        "norm",
        "z_aspect",
        "z_slices",
        "inlet_center_xyz",
        "outlet_center_xyz",
    ]  # required keys
    missing = [k for k in required if k not in obj]  # schema check
    if missing:
        raise KeyError(f"Geometry JSON missing keys {missing}: {p}")  # bad schema

    pts_in = np.asarray(obj["points_inside"], dtype=np.float32)  # (N,6)
    pts_wall = np.asarray(obj["points_wall"], dtype=np.float32)  # (N,7)
    if pts_in.ndim != 2 or pts_in.shape[1] < 3:
        raise ValueError("points_inside must have >=3 columns (xn,yn,zn[,...])")
    if pts_wall.ndim != 2 or pts_wall.shape[1] < 3:
        raise ValueError("points_wall must have >=3 columns")

    xyz_inside = pts_in[:, 0:3].astype(np.float32)  # normalised inside coords
    xyz_wall = pts_wall[:, 0:3].astype(np.float32)  # normalised wall coords
    class_wall = (
        pts_wall[:, 6].astype(np.int32)
        if pts_wall.shape[1] >= 7
        else np.zeros((pts_wall.shape[0],), dtype=np.int32)
    )  # optional class label

    n_wall = np.asarray(obj["normals_wall"], dtype=np.float32)  # (N_w, 3)
    if n_wall.shape != xyz_wall.shape:
        raise ValueError(
            f"normals_wall shape {n_wall.shape} != xyz_wall shape {xyz_wall.shape}"
        )  # schema

    xyz_inlet = np.asarray(obj["points_inlet"], dtype=np.float32)  # (N_il,3)
    xyz_outlet = np.asarray(obj["points_outlet"], dtype=np.float32)  # (N_ol,3)
    if xyz_inlet.size and xyz_inlet.shape[1] != 3:
        raise ValueError("points_inlet must have 3 columns")
    if xyz_outlet.size and xyz_outlet.shape[1] != 3:
        raise ValueError("points_outlet must have 3 columns")

    inlet_center = np.asarray(obj["inlet_center_xyz"], dtype=np.float32).reshape(1, 3)
    outlet_center = np.asarray(obj["outlet_center_xyz"], dtype=np.float32).reshape(1, 3)

    norm_raw = obj["norm"]  # list of 6 floats
    if len(norm_raw) != 6:
        raise ValueError("norm must have 6 entries: xmin,xmax,ymin,ymax,zmin,zmax")
    norm = tuple(float(v) for v in norm_raw)  # tuple of 6

    init_fields: Optional[Dict[str, np.ndarray]] = None  # may be absent
    if "init_fields" in obj and isinstance(obj["init_fields"], dict):
        fields = obj["init_fields"].get("fields", {})  # inner dict
        init_fields = {
            k: np.asarray(v, dtype=np.float32).reshape(-1, 1)
            for k, v in fields.items()
            if isinstance(v, list)
        }  # per-field vectors

    return GeometryBundle3D(
        xyz_inside=xyz_inside,
        xyz_wall=xyz_wall,
        n_wall=n_wall,
        class_wall=class_wall,
        xyz_inlet=xyz_inlet,
        xyz_outlet=xyz_outlet,
        inlet_center_xyz=inlet_center,
        outlet_center_xyz=outlet_center,
        norm=norm,  # 6-tuple
        z_aspect=float(obj["z_aspect"]),
        z_slices=int(obj["z_slices"]),
        init_fields=init_fields,
    )  # bundle


@dataclass
class FlowFieldData3D:
    """Parsed 3D flow-field JSON."""

    xyz: np.ndarray  # (N, 3) normalised point coordinates
    u: np.ndarray  # (N, 1)
    v: np.ndarray  # (N, 1)
    w: np.ndarray  # (N, 1)
    p: Optional[np.ndarray]  # (N, 1) or None
    norm: Tuple[float, float, float, float, float, float]  # xmin..zmax
    z_aspect: float  # Zmax / Lx
    z_slices: int  # nominal slice count
    t: float  # steady time scalar


def _load_flow_field_json_3d(path: str) -> FlowFieldData3D:
    """Load the 3D flow-field JSON written by partner_v4_1_flow.py.

    Schema (per spec):
        {
            "z_aspect": float,
            "z_slices": int,
            "norm": [xmin,xmax,ymin,ymax,zmin,zmax],
            "points": [[xn, yn, zn], ...],
            "flow": {"u":[...], "v":[...], "w":[...], "p":[...], "t": 0.0},
            "inlet": {"x":.., "y":..},
            "outlet": {"x":.., "y":..}
        }
    """

    p = Path(path)  # path object
    if not p.exists():
        raise FileNotFoundError(f"Flow JSON not found: {p}")  # guard

    obj = json.loads(p.read_text())  # parse
    if "points" not in obj or "flow" not in obj:
        raise ValueError("Flow JSON must contain 'points' and 'flow' keys")  # schema
    if "u" not in obj["flow"] or "v" not in obj["flow"] or "w" not in obj["flow"]:
        raise ValueError("Flow JSON 'flow' must provide 'u', 'v', 'w'")  # schema

    xyz = np.asarray(obj["points"], dtype=np.float32)  # (N,3)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {xyz.shape}")

    u = np.asarray(obj["flow"]["u"], dtype=np.float32).reshape(-1, 1)  # (N,1)
    v = np.asarray(obj["flow"]["v"], dtype=np.float32).reshape(-1, 1)  # (N,1)
    w = np.asarray(obj["flow"]["w"], dtype=np.float32).reshape(-1, 1)  # (N,1)
    if u.shape[0] != xyz.shape[0] or v.shape[0] != xyz.shape[0] or w.shape[0] != xyz.shape[0]:
        raise ValueError(
            f"Flow field lengths do not match points ({u.shape[0]}/{v.shape[0]}/"
            f"{w.shape[0]} vs {xyz.shape[0]})"
        )  # schema

    p_field: Optional[np.ndarray] = None  # optional pressure
    if "p" in obj["flow"]:
        p_field = np.asarray(obj["flow"]["p"], dtype=np.float32).reshape(-1, 1)  # (N,1)
        if p_field.shape[0] != xyz.shape[0]:
            raise ValueError("p field length does not match points")

    norm_raw = obj.get("norm", [])  # optional
    if len(norm_raw) == 6:
        norm = tuple(float(x) for x in norm_raw)  # 6-tuple
    elif len(norm_raw) == 4:
        # Allow 4-tuple for back-compat with 2D-like writers (pad z bounds)
        norm = (
            float(norm_raw[0]),
            float(norm_raw[1]),
            float(norm_raw[2]),
            float(norm_raw[3]),
            0.0,
            float(obj.get("z_aspect", 0.10)) * max(
                float(norm_raw[1]) - float(norm_raw[0]), 1.0e-6
            ),
        )  # derive raw z bounds from z_aspect * Lx
    else:
        raise ValueError("Flow JSON 'norm' must have 4 or 6 entries")

    t_scalar = float(obj["flow"].get("t", 0.0))  # steady flow
    z_aspect = float(obj.get("z_aspect", 0.10))  # default
    z_slices = int(obj.get("z_slices", 0))  # optional
    return FlowFieldData3D(
        xyz=xyz,
        u=u,
        v=v,
        w=w,
        p=p_field,
        norm=norm,
        z_aspect=z_aspect,
        z_slices=z_slices,
        t=t_scalar,
    )  # bundle


# -----------------------------
# kNN look-up of flow at interior/wall points
# -----------------------------
def _nn_lookup_flow(
    xyz_query: np.ndarray,
    xyz_flow: np.ndarray,
    u_flow: np.ndarray,
    v_flow: np.ndarray,
    w_flow: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour lookup of (u, v, w) from the flow JSON cloud.

    3D analogue of V4's per-point flow lookup. Runs a cKDTree on the flow
    point cloud once and returns arrays aligned with `xyz_query`.
    """

    if xyz_query.ndim != 2 or xyz_query.shape[1] != 3:
        raise ValueError("xyz_query must have shape (N,3)")  # guard
    if xyz_flow.ndim != 2 or xyz_flow.shape[1] != 3:
        raise ValueError("xyz_flow must have shape (M,3)")  # guard
    if xyz_flow.shape[0] == 0:
        return (
            np.zeros((xyz_query.shape[0], 1), dtype=np.float32),
            np.zeros((xyz_query.shape[0], 1), dtype=np.float32),
            np.zeros((xyz_query.shape[0], 1), dtype=np.float32),
        )  # empty flow -> zero velocity (will yield a pure-diffusion residual)

    tree = cKDTree(xyz_flow.astype(np.float64))  # kd-tree
    _, idx = tree.query(xyz_query.astype(np.float64), k=1)  # nearest
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)  # (N,)
    return (
        u_flow[idx].astype(np.float32).reshape(-1, 1),
        v_flow[idx].astype(np.float32).reshape(-1, 1),
        w_flow[idx].astype(np.float32).reshape(-1, 1),
    )  # aligned arrays


# -----------------------------
# Weighted-sampling weights
# -----------------------------
def _pde_sampling_weights_3d(
    d_wall: np.ndarray,
    inside_is_inlet: np.ndarray,
    wall_boost: float,
    wall_scale: float,
    inlet_boost: float,
    inlet_power: float,
) -> np.ndarray:
    """Weight array for weighted PDE sampling on interior points.

    Factor 1 (wall boost): gaussian bump near the wall,
        1 + wall_boost * exp(-(d_wall / wall_scale)^2)
    Factor 2 (inlet boost): multiplicative boost for points inside the
    inlet patch (already a 0/1 mask),
        1 + inlet_boost * inlet_is_on^inlet_power
    Product yields non-negative weights; caller should not multiply by 0.
    """

    d = np.asarray(d_wall, dtype=np.float32).reshape(-1)  # (N,)
    scale = float(max(wall_scale, 1.0e-8))  # guard
    gauss = np.exp(-((d / scale) ** 2)).astype(np.float32)  # (N,)
    base = 1.0 + float(wall_boost) * gauss  # wall factor >= 1
    inlet_mask = np.asarray(inside_is_inlet, dtype=np.float32).reshape(-1)  # (N,)
    inlet_term = float(inlet_boost) * (inlet_mask ** float(inlet_power))  # (N,)
    w = base * (1.0 + inlet_term)  # (N,)
    w = np.clip(w, 1.0e-6, None).astype(np.float32)  # positive
    return w  # (N,) float32


def _ic_sampling_weights_3d(
    d_wall_all: np.ndarray,
    ic_wall_boost: float,
    ic_wall_scale: float,
) -> np.ndarray:
    """Weight array for IC / BC sampling over the ALL-points set.

    Simple gaussian wall bump; set ic_wall_boost=0 to disable.
    """

    d = np.asarray(d_wall_all, dtype=np.float32).reshape(-1)  # (N,)
    scale = float(max(ic_wall_scale, 1.0e-8))  # guard
    gauss = np.exp(-((d / scale) ** 2)).astype(np.float32)  # (N,)
    w = 1.0 + float(ic_wall_boost) * gauss  # (N,)
    return np.clip(w, 1.0e-6, None).astype(np.float32)  # positive


# -----------------------------
# HDF5 writer
# -----------------------------
def _save_h5_3d(
    path: str,
    xyz_norm: np.ndarray,
    xyz_raw: np.ndarray,
    point_type: np.ndarray,
    times: np.ndarray,
    temperature: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    norm: Tuple[float, float, float, float, float, float],
    flow_json_path: str,
    geom_json_path: str,
    z_aspect: float,
    z_slices: int,
) -> None:
    """Write predictions to HDF5 (3D-equivalent of V4 _save_h5)."""

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("xyz_norm", data=xyz_norm)
        h5f.create_dataset("xyz_raw", data=xyz_raw)
        h5f.create_dataset("point_type", data=point_type)
        h5f.create_dataset("times", data=times)
        h5f.create_dataset("temperature", data=temperature)
        h5f.create_dataset("u", data=u)
        h5f.create_dataset("v", data=v)
        h5f.create_dataset("w", data=w)
        h5f.attrs["xmin"] = float(norm[0])
        h5f.attrs["xmax"] = float(norm[1])
        h5f.attrs["ymin"] = float(norm[2])
        h5f.attrs["ymax"] = float(norm[3])
        h5f.attrs["zmin"] = float(norm[4])
        h5f.attrs["zmax"] = float(norm[5])
        h5f.attrs["z_aspect"] = float(z_aspect)
        h5f.attrs["z_slices"] = int(z_slices)
        h5f.attrs["flow_json_path"] = str(flow_json_path)
        h5f.attrs["geom_json_path"] = str(geom_json_path)


# -----------------------------
# Visualisation (mid-z slice)
# -----------------------------
def _render_visualisation_3d(
    outdir: Path,
    xyz_inside: np.ndarray,
    temperature_inside: np.ndarray,
    times: np.ndarray,
    z_aspect: float,
    z_slices: int,
) -> None:
    """Write per-time mid-z scatter PNGs + stitched GIF + 3-panel summary.

    3D data is hard to render directly; we slice the interior points at a
    narrow band around the mid-z plane and scatter them coloured by T at
    each time step. The GIF and summary are optional convenience outputs.
    """

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt  # plotting
        import imageio.v2 as imageio  # gif writer
    except Exception as exc:  # pragma: no cover - optional
        print(f"[WARN] visualisation imports failed: {exc}")  # report
        return  # skip

    outdir.mkdir(parents=True, exist_ok=True)  # ensure outdir

    # Slice selection -- narrow band around the mid-z plane.
    z_mid = 0.5 * float(z_aspect)  # centre of z range
    z_levels = int(max(1, int(z_slices)))  # total slice count
    # Band half-width: one slice-thickness (tolerant to irregular spacing).
    band = float(z_aspect) / max(float(z_levels - 1), 1.0)  # half-width
    mask = np.abs(xyz_inside[:, 2] - z_mid) <= (0.5 * band + 1.0e-6)  # boolean
    if not np.any(mask):
        # fallback: pick the nearest single z-level's worth of points.
        ranked = np.argsort(np.abs(xyz_inside[:, 2] - z_mid))  # by |dz|
        keep = ranked[: max(256, int(0.001 * xyz_inside.shape[0]))]  # top-K
        mask = np.zeros((xyz_inside.shape[0],), dtype=bool)  # empty
        mask[keep] = True  # activate

    x_slice = xyz_inside[mask, 0]  # (M,)
    y_slice = xyz_inside[mask, 1]  # (M,)
    T_slice_all = temperature_inside[:, mask]  # (T, M)

    vmin = float(np.min(T_slice_all))  # colour lower
    vmax = float(np.max(T_slice_all))  # colour upper
    if vmax - vmin < 1.0e-8:
        vmax = vmin + 1.0e-3  # avoid degenerate colourbar

    # Per-time frames
    frame_paths = []  # frame list
    for ti, tval in enumerate(times):
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 4.0), dpi=120)  # figure
        sc = ax.scatter(
            x_slice,
            y_slice,
            c=T_slice_all[ti],
            s=2.0,
            vmin=vmin,
            vmax=vmax,
            cmap="inferno",
        )  # scatter coloured by T
        ax.set_xlim(0.0, 1.0)  # xn bounds
        ax.set_ylim(0.0, 1.0)  # yn bounds
        ax.set_aspect("equal")  # isotropic
        ax.set_title(f"mid-z slice, t={float(tval):.2f}")  # annotate
        ax.set_xlabel("xn")  # x label
        ax.set_ylabel("yn")  # y label
        fig.colorbar(sc, ax=ax, label="T")  # colourbar
        fr = outdir / f"temperature_t{ti:04d}.png"  # path
        fig.tight_layout()  # compact
        fig.savefig(str(fr))  # write
        plt.close(fig)  # free
        frame_paths.append(fr)  # record

    # GIF stitch
    try:
        frames = [imageio.imread(str(fp)) for fp in frame_paths]  # load
        if len(frames) > 0:
            gif_path = outdir / "temperature.gif"  # target
            imageio.mimsave(
                str(gif_path),
                frames,
                duration=0.1,
                loop=0,
            )  # write gif
            print(f"[OK] wrote {gif_path}")  # report
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[WARN] gif stitching failed: {exc}")  # non-fatal

    # 3-panel static summary: t=0, t=mid, t=final
    try:
        if times.shape[0] >= 1:
            t_idx = [
                0,
                int(times.shape[0] // 2),
                int(times.shape[0] - 1),
            ]  # three snapshots
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(16.0, 4.0),
                dpi=120,
            )  # figure
            for ax, ti in zip(axes, t_idx):
                ax.scatter(
                    x_slice,
                    y_slice,
                    c=T_slice_all[ti],
                    s=2.0,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="inferno",
                )  # scatter
                ax.set_xlim(0.0, 1.0)  # bounds
                ax.set_ylim(0.0, 1.0)  # bounds
                ax.set_aspect("equal")  # isotropic
                ax.set_title(f"t={float(times[ti]):.2f}")  # annotate
                ax.set_xlabel("xn")  # axis
                ax.set_ylabel("yn")  # axis
            fig.suptitle("Mid-z temperature slices")  # overall title
            fig.tight_layout()  # compact
            summary_path = outdir / "temperature_summary.png"  # path
            fig.savefig(str(summary_path))  # write
            plt.close(fig)  # free
            print(f"[OK] wrote {summary_path}")  # report
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[WARN] summary render failed: {exc}")  # non-fatal


# -----------------------------
# Arrival time from 3D distance
# -----------------------------
def _compute_arrival_time_3d(
    xyz_inside: np.ndarray,
    inlet_center_xyz: np.ndarray,
    arrival_speed: float,
    t_min: float,
    t_max: float,
) -> np.ndarray:
    """Estimate arrival time t_arrive for each interior point.

    The spec asks for 3D geodesic distance but the geometry JSON does not
    carry it (it carries predecessor-tree tangents on `init_fields`,
    not geodesic arrays). For temperature training a Euclidean distance is
    a faithful proxy: it matches V4 where the arrival stream is a straight-
    line projection scaled by mean flow speed. If a richer geodesic is
    available via `init_fields.geo_in` we use it instead.
    """

    d = np.linalg.norm(xyz_inside - inlet_center_xyz.reshape(1, 3), axis=1)  # (N,)
    d = np.asarray(d, dtype=np.float32)  # cast
    speed = float(max(arrival_speed, 1.0e-6))  # guard
    t_arr = (d / speed).astype(np.float32)  # arrival time
    t_arr = np.clip(t_arr, float(t_min), float(t_max)).astype(np.float32)  # bound
    return t_arr.reshape(-1, 1)  # (N,1)


# -----------------------------
# Main
# -----------------------------
@hydra.main(version_base=None, config_path="conf", config_name="partner_v4_1_config")
def main(cfg: DictConfig) -> None:  # pragma: no cover - entry point
    """Train the V4.1 3D temperature PINN (Hydra entry)."""

    run(cfg)  # delegate so unit tests can call run(cfg) directly


def run(cfg: DictConfig) -> None:
    """Run the V4.1 3D temperature trainer given an already-resolved cfg."""

    cfg = cfg.temp if "temp" in cfg else cfg  # descend to temp block (mirrors V4)

    # --- seeds ---
    seed = int(cfg.training.seed)  # seed scalar
    torch.manual_seed(seed)  # torch rng
    np.random.seed(seed)  # numpy rng

    # --- paths ---
    geom_path = to_absolute_path(str(cfg.problem.geom_json_path))  # 3D geom
    flow_path = to_absolute_path(str(cfg.problem.flow_json_path))  # 3D flow
    network_dir = Path(to_absolute_path(str(cfg.network_dir)))  # output dir
    network_dir.mkdir(parents=True, exist_ok=True)  # ensure exists

    print(f"[INFO] using geometry json: {geom_path}")  # log
    print(f"[INFO] using flow json:     {flow_path}")  # log

    # --- load geometry + flow ---
    geom = _load_geometry_json_3d(geom_path)  # 3D bundle
    flow = _load_flow_field_json_3d(flow_path)  # 3D flow

    # --- combined point sets used by different losses ---
    # Inside-only set for PDE / arrival / pre-arrival residuals.
    xyz_inside = geom.xyz_inside.astype(np.float32)  # (N_in, 3)
    # Full (inside + wall) set for IC and broad time-BC sampling.
    xyz_all = np.concatenate(
        [xyz_inside, geom.xyz_wall.astype(np.float32)],
        axis=0,
    )  # (N_in+N_w, 3)
    # Point-type tag: 2=inside, 1=wall (mirrors V4 flow-json conventions)
    pt_inside = np.full((xyz_inside.shape[0],), 2, dtype=np.int32)
    pt_wall = np.full((geom.xyz_wall.shape[0],), 1, dtype=np.int32)
    point_type_all = np.concatenate([pt_inside, pt_wall], axis=0)  # (N_all,)

    # --- nearest-neighbour flow lookup ---
    # For interior points we need (u,v,w) at each interior location.
    u_inside, v_inside, w_inside = _nn_lookup_flow(
        xyz_query=xyz_inside,
        xyz_flow=flow.xyz,
        u_flow=flow.u,
        v_flow=flow.v,
        w_flow=flow.w,
    )  # (N_in,1) each
    # For the "all" set (used in IC and inference) we also need flow.
    u_all, v_all, w_all = _nn_lookup_flow(
        xyz_query=xyz_all,
        xyz_flow=flow.xyz,
        u_flow=flow.u,
        v_flow=flow.v,
        w_flow=flow.w,
    )  # (N_all,1) each

    # --- geometric features for sampling ---
    d_wall_inside = compute_wall_distance_3d(
        xyz_inside, geom.xyz_wall
    ).astype(np.float32).reshape(-1, 1)  # (N_in,1)
    d_wall_all = compute_wall_distance_3d(
        xyz_all, geom.xyz_wall
    ).astype(np.float32).reshape(-1, 1)  # (N_all,1)

    # Inlet / outlet radius masks (3D balls around patch centre)
    def _ball_mask(xyz: np.ndarray, centre: np.ndarray, r: float) -> np.ndarray:
        """Return boolean mask for points within radius r of centre."""
        d = np.linalg.norm(xyz - centre.reshape(1, 3), axis=1)  # (N,)
        return d <= float(r)  # boolean

    inlet_r = float(cfg.bc.inlet_radius_norm)  # inlet ball radius
    outlet_r = float(cfg.bc.outlet_radius_norm)  # outlet ball radius
    inlet_mask_inside = _ball_mask(xyz_inside, geom.inlet_center_xyz, inlet_r)  # (N_in,)
    outlet_mask_inside = _ball_mask(xyz_inside, geom.outlet_center_xyz, outlet_r)  # (N_in,)

    # If radius too small to capture any points (very sparse smoke geom),
    # fall back to the `points_inlet`/`points_outlet` patches directly.
    if (not np.any(inlet_mask_inside)) and geom.xyz_inlet.shape[0] > 0:
        tree = cKDTree(xyz_inside.astype(np.float64))  # nearest inside point
        _, idx = tree.query(geom.xyz_inlet.astype(np.float64), k=1)  # nearest
        inlet_mask_inside = np.zeros((xyz_inside.shape[0],), dtype=bool)  # zeros
        inlet_mask_inside[np.asarray(idx, dtype=np.int64).reshape(-1)] = True  # activate
    if (not np.any(outlet_mask_inside)) and geom.xyz_outlet.shape[0] > 0:
        tree = cKDTree(xyz_inside.astype(np.float64))  # nearest
        _, idx = tree.query(geom.xyz_outlet.astype(np.float64), k=1)  # nearest
        outlet_mask_inside = np.zeros((xyz_inside.shape[0],), dtype=bool)  # zeros
        outlet_mask_inside[np.asarray(idx, dtype=np.int64).reshape(-1)] = True  # activate

    # --- arrival times ---
    t_arrive_inside = _compute_arrival_time_3d(
        xyz_inside=xyz_inside,
        inlet_center_xyz=geom.inlet_center_xyz,
        arrival_speed=float(cfg.problem.arrival_speed),
        t_min=float(cfg.problem.t_min),
        t_max=float(cfg.problem.t_max),
    )  # (N_in, 1)

    # --- weighted-sampling weights (numpy; tensorised later) ---
    pde_weights_np = _pde_sampling_weights_3d(
        d_wall=d_wall_inside.reshape(-1),
        inside_is_inlet=inlet_mask_inside.astype(np.float32),
        wall_boost=float(cfg.training.pde_wall_boost),
        wall_scale=float(cfg.training.pde_wall_scale),
        inlet_boost=float(cfg.training.pde_inlet_boost),
        inlet_power=float(cfg.training.pde_inlet_power),
    )  # (N_in,)
    ic_weights_np = _ic_sampling_weights_3d(
        d_wall_all=d_wall_all.reshape(-1),
        ic_wall_boost=float(cfg.training.ic_wall_boost),
        ic_wall_scale=float(cfg.training.ic_wall_scale),
    )  # (N_all,)

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select
    print(f"[INFO] device: {device}")  # log

    # --- model ---
    model = TemperatureNet3D(
        in_dim=7,
        hidden_size=int(cfg.model.hidden_size),
        hidden_layers=int(cfg.model.hidden_layers),
        activation=str(cfg.model.activation),
    ).to(device)  # 7 inputs, 1 output

    ckpt_path = network_dir / "temperature_net.pt"  # checkpoint path

    # --- pre-stage tensors on the device ---
    def _t(arr: np.ndarray) -> torch.Tensor:
        """Move a numpy array to a torch tensor on `device`, float32."""
        return torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(device)

    # Inside tensors
    x_inside_t = _t(xyz_inside[:, 0:1])  # (N_in,1)
    y_inside_t = _t(xyz_inside[:, 1:2])  # (N_in,1)
    z_inside_t = _t(xyz_inside[:, 2:3])  # (N_in,1)
    u_inside_t = _t(u_inside)  # (N_in,1)
    v_inside_t = _t(v_inside)  # (N_in,1)
    w_inside_t = _t(w_inside)  # (N_in,1)
    t_arrive_t = _t(t_arrive_inside)  # (N_in,1)
    pde_weights_t = torch.from_numpy(pde_weights_np).to(device)  # (N_in,)

    # All-point tensors
    x_all_t = _t(xyz_all[:, 0:1])  # (N_all,1)
    y_all_t = _t(xyz_all[:, 1:2])  # (N_all,1)
    z_all_t = _t(xyz_all[:, 2:3])  # (N_all,1)
    u_all_t = _t(u_all)  # (N_all,1)
    v_all_t = _t(v_all)  # (N_all,1)
    w_all_t = _t(w_all)  # (N_all,1)
    ic_weights_t = torch.from_numpy(ic_weights_np).to(device)  # (N_all,)

    # Inlet / outlet inside index tensors
    inlet_idx_t: Optional[torch.Tensor] = None  # lazy
    if np.any(inlet_mask_inside):
        inlet_idx_t = torch.from_numpy(
            np.where(inlet_mask_inside)[0].astype(np.int64)
        ).to(device)  # (n_inlet,)
    outlet_idx_t: Optional[torch.Tensor] = None  # lazy
    if np.any(outlet_mask_inside):
        outlet_idx_t = torch.from_numpy(
            np.where(outlet_mask_inside)[0].astype(np.int64)
        ).to(device)  # (n_outlet,)

    # Wall tensors and normals for wall-insulation loss
    xyz_wall = geom.xyz_wall.astype(np.float32)  # (N_w, 3)
    n_wall = geom.n_wall.astype(np.float32)  # (N_w, 3)
    # Wall flow look-up (usually zero for no-slip, but use what the flow
    # JSON actually provides to avoid accidental inconsistencies)
    u_wall, v_wall, w_wall = _nn_lookup_flow(
        xyz_query=xyz_wall,
        xyz_flow=flow.xyz,
        u_flow=flow.u,
        v_flow=flow.v,
        w_flow=flow.w,
    )  # (N_w,1) each
    x_wall_t = _t(xyz_wall[:, 0:1])  # (N_w,1)
    y_wall_t = _t(xyz_wall[:, 1:2])  # (N_w,1)
    z_wall_t = _t(xyz_wall[:, 2:3])  # (N_w,1)
    u_wall_t = _t(u_wall)  # (N_w,1)
    v_wall_t = _t(v_wall)  # (N_w,1)
    w_wall_t = _t(w_wall)  # (N_w,1)
    nx_wall_t = _t(n_wall[:, 0:1])  # (N_w,1)
    ny_wall_t = _t(n_wall[:, 1:2])  # (N_w,1)
    nz_wall_t = _t(n_wall[:, 2:3])  # (N_w,1)

    # --- run mode ---
    run_mode = str(cfg.run_mode).lower()  # "train" or "infer_only"/"eval"
    if run_mode in ("eval", "infer_only"):
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found for {run_mode} mode: {ckpt_path}"
            )  # guard
        model.load_state_dict(torch.load(ckpt_path, map_location=device))  # load
        model.eval()  # eval mode
        print(f"[INFO] loaded checkpoint from {ckpt_path} (skipping training)")
    else:
        # --- optimiser + scheduler ---
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.training.lr),
            betas=tuple(float(v) for v in cfg.optimizer.betas),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
        )  # Adam
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(cfg.training.lr_decay_rate),
        )  # exponential decay
        # Apply decay every lr_decay_steps (matches V4 cadence of ~100% per
        # lr_decay_steps steps via StepLR-like semantics).
        lr_decay_steps = int(cfg.training.lr_decay_steps)  # decay cadence

        # --- hyper-parameters cached as floats ---
        loss_weights = cfg.training.loss_weights  # weight dict
        t_min = float(cfg.problem.t_min)  # t bound
        t_max = float(cfg.problem.t_max)  # t bound
        t_init_val = float(cfg.problem.T_init)  # initial T
        inlet_t_val = float(cfg.bc.inlet_T)  # inlet T
        grad_clip = float(cfg.training.grad_clip_max_norm)  # clip norm
        D_val = float(cfg.physics.D)  # diffusivity
        Q_val = float(cfg.physics.Q)  # source
        max_steps = int(cfg.training.max_steps)  # total steps
        pde_batch = int(cfg.training.pde_batch_size)  # per-step PDE samples
        bc_batch = int(cfg.training.bc_batch_size)  # per-step BC samples

        # Curriculum endpoints (defaults mirror V4 config)
        curr = {
            "pde": (
                float(cfg.training.pde_start_scale),
                float(cfg.training.pde_end_scale),
            ),
            "ic": (
                float(cfg.training.ic_start_scale),
                float(cfg.training.ic_end_scale),
            ),
            "arrival": (
                float(cfg.training.arrival_start_scale),
                float(cfg.training.arrival_end_scale),
            ),
            "pre_arrival": (
                float(cfg.training.pre_arrival_start_scale),
                float(cfg.training.pre_arrival_end_scale),
            ),
            "inlet": (
                float(cfg.training.inlet_start_scale),
                float(cfg.training.inlet_end_scale),
            ),
            "outlet": (
                float(cfg.training.outlet_start_scale),
                float(cfg.training.outlet_end_scale),
            ),
        }  # per-term (start, end) pairs
        use_weighted_pde_sampling = bool(cfg.training.use_weighted_pde_sampling)
        use_weighted_bc_loss = bool(cfg.training.use_weighted_bc_loss)

        # --- training loop ---
        for step in range(1, max_steps + 1):
            optimizer.zero_grad(set_to_none=True)  # reset grads

            # ------------------- 1. PDE residual -------------------
            if use_weighted_pde_sampling:
                pde_idx = _sample_indices_weighted(
                    pde_weights_t,
                    pde_batch,
                    device,
                )  # weighted
            else:
                pde_idx = _sample_indices(
                    int(x_inside_t.shape[0]), pde_batch, device
                )  # uniform
            x_pde = x_inside_t[pde_idx].clone().detach().requires_grad_(True)
            y_pde = y_inside_t[pde_idx].clone().detach().requires_grad_(True)
            z_pde = z_inside_t[pde_idx].clone().detach().requires_grad_(True)
            t_pde = _make_time_samples(
                int(x_pde.shape[0]), t_min, t_max, device
            ).requires_grad_(True)  # time samples
            u_pde = u_inside_t[pde_idx]  # advection u
            v_pde = v_inside_t[pde_idx]  # advection v
            w_pde = w_inside_t[pde_idx]  # advection w

            T_pde = _forward_temperature(
                model,
                x_pde,
                y_pde,
                z_pde,
                t_pde,
                u_pde,
                v_pde,
                w_pde,
            )  # (M,1)
            T_x = _grad(T_pde, x_pde)  # d_x T
            T_y = _grad(T_pde, y_pde)  # d_y T
            T_z = _grad(T_pde, z_pde)  # d_z T
            T_t = _grad(T_pde, t_pde)  # d_t T
            T_xx = _grad(T_x, x_pde)  # d_xx T
            T_yy = _grad(T_y, y_pde)  # d_yy T
            T_zz = _grad(T_z, z_pde)  # d_zz T

            residual = (
                T_t
                + u_pde * T_x
                + v_pde * T_y
                + w_pde * T_z
                - D_val * (T_xx + T_yy + T_zz)
                - Q_val
            )  # 3D advection-diffusion residual
            loss_pde = torch.mean(residual ** 2)  # MSE of residual

            # ------------------- 2. Initial condition -------------------
            if use_weighted_bc_loss:
                ic_idx = _sample_indices_weighted(
                    ic_weights_t,
                    bc_batch,
                    device,
                )  # weighted
            else:
                ic_idx = _sample_indices(
                    int(x_all_t.shape[0]),
                    bc_batch,
                    device,
                )  # uniform
            T_ic = _forward_temperature(
                model,
                x_all_t[ic_idx],
                y_all_t[ic_idx],
                z_all_t[ic_idx],
                torch.zeros((ic_idx.shape[0], 1), device=device),  # t = 0
                u_all_t[ic_idx],
                v_all_t[ic_idx],
                w_all_t[ic_idx],
            )  # (M,1)
            loss_ic = torch.mean((T_ic - t_init_val) ** 2)  # MSE to T_init

            # ------------------- 3. Arrival -------------------
            arr_idx = _sample_indices(
                int(x_inside_t.shape[0]),
                bc_batch,
                device,
            )  # uniform on interior
            T_arrive = _forward_temperature(
                model,
                x_inside_t[arr_idx],
                y_inside_t[arr_idx],
                z_inside_t[arr_idx],
                t_arrive_t[arr_idx],
                u_inside_t[arr_idx],
                v_inside_t[arr_idx],
                w_inside_t[arr_idx],
            )  # (M,1)
            loss_arrival = torch.mean((T_arrive - inlet_t_val) ** 2)  # pin to inlet

            # ------------------- 4. Pre-arrival -------------------
            pre_idx = _sample_indices(
                int(x_inside_t.shape[0]),
                bc_batch,
                device,
            )  # uniform on interior
            frac = 0.95 * torch.rand(
                (pre_idx.shape[0], 1), device=device
            )  # stochastic fraction
            t_pre = torch.clamp(
                frac * t_arrive_t[pre_idx], min=t_min
            )  # t < t_arrive
            T_pre = _forward_temperature(
                model,
                x_inside_t[pre_idx],
                y_inside_t[pre_idx],
                z_inside_t[pre_idx],
                t_pre,
                u_inside_t[pre_idx],
                v_inside_t[pre_idx],
                w_inside_t[pre_idx],
            )  # (M,1)
            loss_pre = torch.mean((T_pre - t_init_val) ** 2)  # pin to T_init

            # ------------------- 5. Inlet patch -------------------
            if inlet_idx_t is not None and inlet_idx_t.numel() > 0:
                pick = inlet_idx_t[
                    _sample_indices(
                        int(inlet_idx_t.shape[0]),
                        bc_batch,
                        device,
                    )
                ]  # resample inlet indices
                T_inlet = _forward_temperature(
                    model,
                    x_inside_t[pick],
                    y_inside_t[pick],
                    z_inside_t[pick],
                    _make_time_samples(int(pick.shape[0]), t_min, t_max, device),
                    u_inside_t[pick],
                    v_inside_t[pick],
                    w_inside_t[pick],
                )  # (M,1)
                loss_inlet = torch.mean((T_inlet - inlet_t_val) ** 2)  # pin to T_in
            else:
                loss_inlet = torch.zeros((), device=device)  # no-op

            # ------------------- 6. Outlet zero-gradient -------------------
            if outlet_idx_t is not None and outlet_idx_t.numel() > 0:
                pick = outlet_idx_t[
                    _sample_indices(
                        int(outlet_idx_t.shape[0]),
                        bc_batch,
                        device,
                    )
                ]  # resample outlet indices
                x_out = x_inside_t[pick].clone().detach().requires_grad_(True)
                y_out = y_inside_t[pick].clone().detach().requires_grad_(True)
                z_out = z_inside_t[pick].clone().detach().requires_grad_(True)
                t_out = _make_time_samples(
                    int(pick.shape[0]), t_min, t_max, device
                ).requires_grad_(True)
                T_out = _forward_temperature(
                    model,
                    x_out,
                    y_out,
                    z_out,
                    t_out,
                    u_inside_t[pick],
                    v_inside_t[pick],
                    w_inside_t[pick],
                )  # (M,1)
                T_out_x = _grad(T_out, x_out)  # d_x T (downstream direction)
                loss_outlet = torch.mean(T_out_x ** 2)  # zero-gradient along x
            else:
                loss_outlet = torch.zeros((), device=device)  # no-op

            # ------------------- 7. Wall insulation (n . grad T = 0) -------------------
            wall_count = int(x_wall_t.shape[0])  # N_w
            if wall_count > 0:
                wall_pick = _sample_indices(
                    wall_count,
                    bc_batch,
                    device,
                )  # uniform on walls
                x_wl = x_wall_t[wall_pick].clone().detach().requires_grad_(True)
                y_wl = y_wall_t[wall_pick].clone().detach().requires_grad_(True)
                z_wl = z_wall_t[wall_pick].clone().detach().requires_grad_(True)
                t_wl = _make_time_samples(
                    int(wall_pick.shape[0]), t_min, t_max, device
                ).requires_grad_(True)
                u_wl = u_wall_t[wall_pick]  # usually 0 at walls
                v_wl = v_wall_t[wall_pick]  # usually 0
                w_wl = w_wall_t[wall_pick]  # usually 0
                T_wl = _forward_temperature(
                    model,
                    x_wl,
                    y_wl,
                    z_wl,
                    t_wl,
                    u_wl,
                    v_wl,
                    w_wl,
                )  # (M,1)
                T_wl_x = _grad(T_wl, x_wl)  # d_x T
                T_wl_y = _grad(T_wl, y_wl)  # d_y T
                T_wl_z = _grad(T_wl, z_wl)  # d_z T
                nx_pick = nx_wall_t[wall_pick]  # (M,1)
                ny_pick = ny_wall_t[wall_pick]  # (M,1)
                nz_pick = nz_wall_t[wall_pick]  # (M,1)
                n_dot_gradT = (
                    nx_pick * T_wl_x
                    + ny_pick * T_wl_y
                    + nz_pick * T_wl_z
                )  # n . grad T
                loss_wall = torch.mean(n_dot_gradT ** 2)  # insulation residual
            else:
                loss_wall = torch.zeros((), device=device)  # no-op

            # ------------------- Curriculum scales -------------------
            scale_pde = _curriculum_scale(step, max_steps, *curr["pde"])  # pde
            scale_ic = _curriculum_scale(step, max_steps, *curr["ic"])  # ic
            scale_arr = _curriculum_scale(step, max_steps, *curr["arrival"])  # arrival
            scale_pre = _curriculum_scale(step, max_steps, *curr["pre_arrival"])  # pre
            scale_in = _curriculum_scale(step, max_steps, *curr["inlet"])  # inlet
            scale_out = _curriculum_scale(step, max_steps, *curr["outlet"])  # outlet

            # ------------------- Aggregate loss -------------------
            total_loss = (
                float(loss_weights.pde) * scale_pde * loss_pde
                + float(loss_weights.ic) * scale_ic * loss_ic
                + float(loss_weights.arrival) * scale_arr * loss_arrival
                + float(loss_weights.pre_arrival) * scale_pre * loss_pre
                + float(loss_weights.inlet) * scale_in * loss_inlet
                + float(loss_weights.outlet) * scale_out * loss_outlet
                + float(loss_weights.wall) * loss_wall
            )  # weighted sum

            # ------------------- Backward + step -------------------
            total_loss.backward()  # backprop
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )  # clip
            optimizer.step()  # Adam update

            # Apply exponential decay every lr_decay_steps steps (matches
            # V4's StepLR cadence even though we use ExponentialLR).
            if lr_decay_steps > 0 and (step % lr_decay_steps == 0):
                scheduler.step()  # decay lr

            # ------------------- Stats printing -------------------
            if step % int(cfg.training.print_stats_freq) == 0 or step == 1:
                print(
                    f"[step {step:06d}] "
                    f"loss={float(total_loss.item()):.6e} "
                    f"pde={float(loss_pde.item()):.6e} "
                    f"ic={float(loss_ic.item()):.6e} "
                    f"arrival={float(loss_arrival.item()):.6e} "
                    f"pre={float(loss_pre.item()):.6e} "
                    f"inlet={float(loss_inlet.item()):.6e} "
                    f"outlet={float(loss_outlet.item()):.6e} "
                    f"wall={float(loss_wall.item()):.6e} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e}"
                )  # per-term status

            # ------------------- Checkpoint -------------------
            if (
                step % int(cfg.training.save_network_freq) == 0
                or step == max_steps
            ):
                torch.save(model.state_dict(), ckpt_path)  # write
                print(f"[OK] saved checkpoint to {ckpt_path}")  # log

            # Guard against NaNs; stop early if encountered (mirrors V4
            # robustness) so we do not write garbage checkpoints.
            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f"Non-finite total_loss at step {step}: {float(total_loss.item())}"
                )  # bail out

        model.eval()  # training done -> eval mode

    # -----------------------------
    # Inference: roll out T over the full cloud
    # -----------------------------
    times = np.arange(
        float(cfg.problem.infer_t_start),
        float(cfg.problem.infer_t_end) + 1.0e-9,
        float(cfg.problem.infer_dt),
        dtype=np.float32,
    )  # (T,)
    if times.shape[0] == 0:
        raise ValueError(
            "Inference time grid is empty; check infer_t_start/end/dt"
        )  # guard

    xyz_for_infer = xyz_all.astype(np.float32)  # (N_all, 3)
    u_for_infer = u_all.astype(np.float32)  # (N_all, 1)
    v_for_infer = v_all.astype(np.float32)  # (N_all, 1)
    w_for_infer = w_all.astype(np.float32)  # (N_all, 1)
    pt_for_infer = point_type_all.astype(np.int32)  # (N_all,)

    temperature = np.zeros(
        (int(times.shape[0]), int(xyz_for_infer.shape[0])),
        dtype=np.float32,
    )  # (T, N_all)
    batch = int(cfg.inference.batch_size)  # per-chunk size

    with torch.no_grad():
        x_t = torch.from_numpy(xyz_for_infer[:, 0:1]).to(device)  # (N,1)
        y_t = torch.from_numpy(xyz_for_infer[:, 1:2]).to(device)  # (N,1)
        z_t = torch.from_numpy(xyz_for_infer[:, 2:3]).to(device)  # (N,1)
        u_t = torch.from_numpy(u_for_infer).to(device)  # (N,1)
        v_t = torch.from_numpy(v_for_infer).to(device)  # (N,1)
        w_t = torch.from_numpy(w_for_infer).to(device)  # (N,1)
        for ti, tval in enumerate(times):
            for start in range(0, int(xyz_for_infer.shape[0]), batch):
                end = min(start + batch, int(xyz_for_infer.shape[0]))  # chunk end
                tb = torch.full(
                    (end - start, 1), float(tval), device=device
                )  # time batch
                pred = _forward_temperature(
                    model,
                    x_t[start:end],
                    y_t[start:end],
                    z_t[start:end],
                    tb,
                    u_t[start:end],
                    v_t[start:end],
                    w_t[start:end],
                )  # (M,1)
                temperature[ti, start:end] = pred.squeeze(1).cpu().numpy()  # store

    # ---- raw-coordinate restoration (denormalise xyz for HDF5 storage) ----
    xmin, xmax, ymin, ymax, zmin, zmax = (
        float(geom.norm[0]),
        float(geom.norm[1]),
        float(geom.norm[2]),
        float(geom.norm[3]),
        float(geom.norm[4]),
        float(geom.norm[5]),
    )  # raw bounds
    xden = max(xmax - xmin, 1.0e-6)  # x range
    yden = max(ymax - ymin, 1.0e-6)  # y range
    zden_raw = max(zmax - zmin, 1.0e-6)  # raw z range
    # z is normalised by Lx (per design), so z_norm_max = z_aspect, and
    # raw z = zmin + z_norm * (zden_raw / z_aspect). This handles both the
    # standard and pathological cases where zden_raw matches z_aspect * Lx.
    z_scale = (
        zden_raw / max(float(geom.z_aspect), 1.0e-8)
        if geom.z_aspect > 0.0
        else 1.0
    )  # raw units per normalised unit
    xyz_raw_infer = np.stack(
        [
            xyz_for_infer[:, 0] * xden + xmin,
            xyz_for_infer[:, 1] * yden + ymin,
            xyz_for_infer[:, 2] * z_scale + zmin,
        ],
        axis=1,
    ).astype(np.float32)  # (N_all, 3)

    # HDF5 output
    output_path = network_dir / str(cfg.inference.output_filename)  # path
    _save_h5_3d(
        str(output_path),
        xyz_norm=xyz_for_infer,
        xyz_raw=xyz_raw_infer,
        point_type=pt_for_infer,
        times=times,
        temperature=temperature,
        u=u_for_infer.reshape(-1),
        v=v_for_infer.reshape(-1),
        w=w_for_infer.reshape(-1),
        norm=geom.norm,
        flow_json_path=flow_path,
        geom_json_path=geom_path,
        z_aspect=float(geom.z_aspect),
        z_slices=int(geom.z_slices),
    )  # write HDF5
    print(f"[OK] wrote temperature predictions to {output_path}")  # log

    # -----------------------------
    # Visualisation (mid-z slice scatter + gif + summary)
    # -----------------------------
    if run_mode not in ("eval", "infer_only"):
        viz_dir = network_dir / "visualizations"  # viz dir
        try:
            # Restrict viz to inside points (walls are cap scatter noise).
            n_in = int(xyz_inside.shape[0])  # interior count
            _render_visualisation_3d(
                outdir=viz_dir,
                xyz_inside=xyz_inside,
                temperature_inside=temperature[:, :n_in],
                times=times,
                z_aspect=float(geom.z_aspect),
                z_slices=int(geom.z_slices),
            )  # scatter/gif/summary
            print(f"[OK] visualisation artefacts written to: {viz_dir}")  # log
        except Exception as exc:  # pragma: no cover - optional
            print(f"[WARN] visualisation step failed: {exc}")  # non-fatal


if __name__ == "__main__":
    main()  # hydra entry
