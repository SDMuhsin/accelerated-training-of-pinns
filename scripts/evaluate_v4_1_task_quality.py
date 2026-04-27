"""V4.1 3D task-quality evaluator.

Runs AFTER training completes. Loads a trained V4.1 flow checkpoint and
(optionally) a V4.1 temperature checkpoint, then reports held-out residuals
and field statistics using **baseline torch.autograd** (not SAGE) so this
evaluator can serve as a reference that a future SAGE V4.1 variant compares
against.

Outputs:
    - JSON: results/partner_v4_1/task_quality.json (by default)
    - Log: results/partner_v4_1/task_quality.log (human-readable)
    - Progress prints to stdout.

CLI:
    python scripts/evaluate_v4_1_task_quality.py [--flow-checkpoint PATH] ...

All flags are optional with sensible defaults. See ``_parse_args``.

Constraints:
    - Do NOT modify V4.1 source files.
    - Do NOT use SAGE / symbolic_vjp / JAX. Baseline torch autograd only.
    - Missing optional inputs (temp checkpoint etc.) are logged and skipped
      gracefully (never crash).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# PhysicsNeMo imports (for flow arch). Deferred: only import when needed so
# the script fails with a clear message if the env is not activated.

# ----------------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------------

DEFAULT_GEOM_JSON = ROOT / "data" / "partner_v4_1" / "pipe_three_class_3d.json"
DEFAULT_TEMP_CKPT = ROOT / "results" / "partner_v4_1" / "temp" / "temperature_net.pt"
DEFAULT_TEMP_PREDICTIONS = ROOT / "results" / "partner_v4_1" / "temp" / "temperature_predictions.h5"
DEFAULT_CONFIG = ROOT / "src" / "conf" / "partner_v4_1_config.yaml"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "partner_v4_1" / "task_quality.json"
DEFAULT_OUTPUT_LOG = ROOT / "results" / "partner_v4_1" / "task_quality.log"

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V4.1 3D PINN task-quality evaluator (baseline autograd)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--geom-json", type=str, default=str(DEFAULT_GEOM_JSON),
                   help="Path to the V4.1 3D geometry JSON (source of "
                        "points_inside, points_wall, points_inlet, init_fields).")
    p.add_argument("--flow-checkpoint", type=str, default=None,
                   help="Path to flow_network.0.pth (PhysicsNeMo state_dict). "
                        "If unset, we search results/partner_v4_1/flow/ for the "
                        "last-stage checkpoint.")
    p.add_argument("--temp-checkpoint", type=str, default=str(DEFAULT_TEMP_CKPT),
                   help="Path to temperature_net.pt (raw PyTorch state_dict). "
                        "Missing -> temp section skipped gracefully.")
    p.add_argument("--temp-predictions", type=str, default=str(DEFAULT_TEMP_PREDICTIONS),
                   help="Path to temperature_predictions.h5 written by the "
                        "temp trainer. Missing -> rollout summary skipped.")
    p.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG),
                   help="Path to partner_v4_1_config.yaml. Used to pull rho, "
                        "nu_final, D, Q, T_init, inlet_T, model widths/depths.")
    p.add_argument("--n-eval-points", type=int, default=20000,
                   help="Number of interior points to sample for held-out "
                        "residual/field evaluation.")
    p.add_argument("--seed", type=int, default=1234,
                   help="Seed for the evaluation random subsample.")
    p.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON),
                   help="Where to write the JSON summary.")
    p.add_argument("--output-log", type=str, default=str(DEFAULT_OUTPUT_LOG),
                   help="Where to write the human-readable log.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (cuda/cpu). If unset, picks cuda if "
                        "available else cpu.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

class TeeLogger:
    """Dual-writer: stdout + a log file, with a small queued-write buffer."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "w", encoding="utf-8")
        self._lines: List[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self._lines.append(msg)

    def close(self) -> None:
        self._f.write("\n".join(self._lines))
        self._f.write("\n")
        self._f.close()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(cli_device: Optional[str]) -> torch.device:
    if cli_device is not None:
        return torch.device(cli_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rmse(arr: torch.Tensor) -> float:
    """Root-mean-square of a tensor (handles empty tensors)."""
    if arr.numel() == 0:
        return float("nan")
    return float(torch.sqrt(torch.mean(arr.to(torch.float32) ** 2)).item())


def _absmax(arr: torch.Tensor) -> float:
    if arr.numel() == 0:
        return float("nan")
    return float(arr.to(torch.float32).abs().max().item())


def _mean(arr: torch.Tensor) -> float:
    if arr.numel() == 0:
        return float("nan")
    return float(arr.to(torch.float32).mean().item())


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        ax = abs(value)
        if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
            return f"{value:+.4e}"
        return f"{value:+.4f}"
    return str(value)


# ----------------------------------------------------------------------------
# Config loader
# ----------------------------------------------------------------------------

def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load and flatten the V4.1 YAML config into a plain dict.

    Returns a dict with scalar physics/model values we actually use. Any
    missing key falls back to a V4.1 config default (matches the default
    yaml at the time the evaluator was written).
    """

    defaults = {
        "rho": 1076.0,
        "nu_final": 1.0e-3,
        "inlet_u": 1.0,
        "inlet_v": 0.0,
        "inlet_p": 1.0,
        "outlet_p": 1.0,
        "D": 1.0e-5,
        "Q": 0.0,
        "T_init": 60.0,
        "inlet_T": 25.0,
        "flow_hidden_size": 512,
        "flow_hidden_layers": 12,
        "flow_activation": "silu",
        "temp_hidden_size": 256,
        "temp_hidden_layers": 12,
        "temp_activation": "silu",
    }
    if not config_path.exists():
        return defaults

    # Use OmegaConf if present (it's in requirements through Hydra). Fall
    # back to PyYAML.
    cfg_raw = None
    try:
        from omegaconf import OmegaConf  # type: ignore
        cfg_raw = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    except Exception:
        try:
            import yaml  # type: ignore
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_raw = yaml.safe_load(f)
        except Exception:
            return defaults

    if not isinstance(cfg_raw, dict):
        return defaults

    out = dict(defaults)
    flow = cfg_raw.get("flow", {}) or {}
    temp = cfg_raw.get("temp", {}) or {}

    # Physics
    phys_f = flow.get("physics", {}) or {}
    out["rho"] = float(phys_f.get("rho", defaults["rho"]))

    # nu_final = last element of nu_schedule
    nu_sched = ((flow.get("training", {}) or {}).get("nu_schedule") or [])
    if nu_sched:
        out["nu_final"] = float(nu_sched[-1])

    # BC
    bc_f = flow.get("bc", {}) or {}
    out["inlet_u"] = float(bc_f.get("inlet_u", defaults["inlet_u"]))
    out["inlet_v"] = float(bc_f.get("inlet_v", defaults["inlet_v"]))
    out["inlet_p"] = float(bc_f.get("inlet_p", defaults["inlet_p"]))
    out["outlet_p"] = float(bc_f.get("outlet_p", defaults["outlet_p"]))

    # Temp physics + bc
    phys_t = temp.get("physics", {}) or {}
    out["D"] = float(phys_t.get("D", defaults["D"]))
    out["Q"] = float(phys_t.get("Q", defaults["Q"]))
    bc_t = temp.get("bc", {}) or {}
    out["inlet_T"] = float(bc_t.get("inlet_T", defaults["inlet_T"]))
    prob_t = temp.get("problem", {}) or {}
    out["T_init"] = float(prob_t.get("T_init", defaults["T_init"]))

    # Flow model
    fm = flow.get("flow_model", {}) or {}
    out["flow_hidden_size"] = int(fm.get("hidden_size", defaults["flow_hidden_size"]))
    out["flow_hidden_layers"] = int(fm.get("hidden_layers", defaults["flow_hidden_layers"]))
    out["flow_activation"] = str(fm.get("activation", defaults["flow_activation"]))

    # Temp model
    tm = temp.get("model", {}) or {}
    out["temp_hidden_size"] = int(tm.get("hidden_size", defaults["temp_hidden_size"]))
    out["temp_hidden_layers"] = int(tm.get("hidden_layers", defaults["temp_hidden_layers"]))
    out["temp_activation"] = str(tm.get("activation", defaults["temp_activation"]))

    return out


# ----------------------------------------------------------------------------
# Flow-checkpoint path resolution
# ----------------------------------------------------------------------------

def _resolve_flow_checkpoint(cli_path: Optional[str]) -> Path:
    """Locate the trained flow checkpoint.

    Resolution order (first match wins):
    1. The explicit --flow-checkpoint CLI flag.
    2. The named last stage: results/partner_v4_1/flow/stage_03_nu_1.00e-03/flow_network.0.pth
    3. Glob: results/partner_v4_1/flow/stage_03_*/flow_network.0.pth
    4. Glob: results/partner_v4_1/flow/stage_0*/flow_network.0.pth (last stage wins).

    Raises FileNotFoundError with the searched paths enumerated.
    """

    if cli_path:
        p = Path(cli_path)
        if not p.exists():
            raise FileNotFoundError(f"--flow-checkpoint does not exist: {p}")
        return p

    candidates: List[Path] = [
        ROOT / "results" / "partner_v4_1" / "flow" / "stage_03_nu_1.00e-03" / "flow_network.0.pth",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Glob for stage_03_*
    stage3 = sorted(glob.glob(str(ROOT / "results" / "partner_v4_1" / "flow" / "stage_03_*" / "flow_network.0.pth")))
    if stage3:
        return Path(stage3[-1])
    # Any stage_0*
    stage0 = sorted(glob.glob(str(ROOT / "results" / "partner_v4_1" / "flow" / "stage_0*" / "flow_network.0.pth")))
    if stage0:
        return Path(stage0[-1])

    search_paths = [
        str(ROOT / "results" / "partner_v4_1" / "flow" / "stage_03_nu_1.00e-03" / "flow_network.0.pth"),
        str(ROOT / "results" / "partner_v4_1" / "flow" / "stage_03_*" / "flow_network.0.pth"),
        str(ROOT / "results" / "partner_v4_1" / "flow" / "stage_0*" / "flow_network.0.pth"),
    ]
    raise FileNotFoundError(
        "Could not locate flow checkpoint. Searched:\n  "
        + "\n  ".join(search_paths)
        + "\nPass --flow-checkpoint PATH to override."
    )


# ----------------------------------------------------------------------------
# Network loaders
# ----------------------------------------------------------------------------

def _build_flow_arch(hidden_size: int, hidden_layers: int, activation: str,
                     device: torch.device) -> torch.nn.Module:
    from physicsnemo.sym.key import Key  # type: ignore
    from physicsnemo.sym.models.fully_connected import FullyConnectedArch  # type: ignore

    act_name = str(activation).strip().lower()
    act_module: torch.nn.Module
    if act_name == "silu":
        act_module = torch.nn.SiLU()
    elif act_name == "tanh":
        act_module = torch.nn.Tanh()
    elif act_name == "relu":
        act_module = torch.nn.ReLU()
    elif act_name == "gelu":
        act_module = torch.nn.GELU()
    else:
        raise ValueError(f"Unsupported flow activation: {activation}")

    net = FullyConnectedArch(
        input_keys=[Key(k) for k in ("x", "y", "z", "dw", "sin", "sout")],
        output_keys=[Key(k) for k in ("u", "v", "w", "p")],
        layer_size=int(hidden_size),
        nr_layers=int(hidden_layers),
        activation_fn=act_module,
    ).to(device)
    return net


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """If every key starts with `prefix`, strip it; otherwise return as-is."""
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _load_flow_net(ckpt_path: Path, hidden_size: int, hidden_layers: int,
                   activation: str, device: torch.device) -> torch.nn.Module:
    """Load a PhysicsNeMo flow checkpoint into a FullyConnectedArch.

    PhysicsNeMo saves the raw state_dict of the submodule (keys look like
    `fc_layers.0.linear.weight`). If the dict is wrapped in {'model': ...}
    or {'state_dict': ...}, unwrap it. If the keys are prefixed with
    `arch.` (or `flow_network.`), strip the prefix and retry.
    """

    net = _build_flow_arch(hidden_size, hidden_layers, activation, device)
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    state_dict: Dict[str, torch.Tensor]
    if isinstance(raw, dict):
        if "state_dict" in raw and isinstance(raw["state_dict"], dict):
            state_dict = dict(raw["state_dict"])
        elif "model" in raw and isinstance(raw["model"], dict):
            state_dict = dict(raw["model"])
        else:
            state_dict = dict(raw)
    else:
        state_dict = dict(raw)

    # Try loading as-is; if that fails with a key mismatch, strip common
    # prefixes and retry.
    tried: List[str] = []
    last_err: Optional[Exception] = None
    for prefix in ("", "arch.", "flow_network.", "module."):
        sd = _strip_prefix(state_dict, prefix) if prefix else state_dict
        tried.append(repr(prefix))
        try:
            net.load_state_dict(sd)
            last_err = None
            break
        except RuntimeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise RuntimeError(
            f"Failed to load flow checkpoint {ckpt_path}. Tried prefix strips: "
            f"{tried}. Last error: {last_err}"
        )

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


class _TempNet(torch.nn.Module):
    """Rebuild of partner_v4_1_temp.TemperatureNet3D compatible with
    state_dicts saved by that trainer.

    The trainer wraps a ``Sequential`` under ``self.net`` and saves
    ``model.state_dict()`` directly, so saved keys look like
    ``net.0.weight`` / ``net.0.bias``.
    """

    def __init__(self, in_dim: int, hidden_size: int, hidden_layers: int,
                 activation: str):
        super().__init__()
        act_name = str(activation).strip().lower()

        def _act() -> torch.nn.Module:
            if act_name == "silu":
                return torch.nn.SiLU()
            if act_name == "tanh":
                return torch.nn.Tanh()
            if act_name == "relu":
                return torch.nn.ReLU()
            if act_name == "gelu":
                return torch.nn.GELU()
            raise ValueError(f"Unsupported temp activation: {activation}")

        layers: List[torch.nn.Module] = [
            torch.nn.Linear(in_dim, hidden_size),
            _act(),
        ]
        for _ in range(hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(_act())
        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _build_temp_net(hidden_size: int, hidden_layers: int, activation: str,
                    device: torch.device) -> torch.nn.Module:
    """Factory for the V4.1 temp net (input dim = 7)."""
    return _TempNet(
        in_dim=7,
        hidden_size=int(hidden_size),
        hidden_layers=int(hidden_layers),
        activation=str(activation),
    ).to(device)


def _load_temp_net(ckpt_path: Path, hidden_size: int, hidden_layers: int,
                   activation: str, device: torch.device) -> torch.nn.Module:
    """Load TemperatureNet3D state_dict.

    The V4.1 temp trainer saves ``model.state_dict()`` directly, so the
    keys are of the form ``net.0.weight`` (matching our
    ``module.net = Sequential(...)`` wrapper). We also handle the case
    where the save was unwrapped (e.g. ``0.weight``) by adding a
    ``net.`` prefix, and the reverse case where the save was double-
    wrapped.
    """

    net = _build_temp_net(hidden_size, hidden_layers, activation, device)
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
    else:
        state_dict = raw

    tried: List[str] = []
    last_err: Optional[Exception] = None

    def _candidates(sd: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Yield several plausible reshapes of the saved state_dict."""
        cands: List[Dict[str, torch.Tensor]] = []
        # 1) as-is
        cands.append(sd)
        # 2) strip common outer prefix -> as-is
        for pre in ("module.", "model."):
            if sd and all(k.startswith(pre) for k in sd.keys()):
                cands.append({k[len(pre):]: v for k, v in sd.items()})
        # 3) if keys don't already start with "net.", add it (Sequential-only save)
        if sd and not any(k.startswith("net.") for k in sd.keys()):
            cands.append({f"net.{k}": v for k, v in sd.items()})
        # 4) if keys start with "net.net." collapse one level
        if sd and all(k.startswith("net.net.") for k in sd.keys()):
            cands.append({k[len("net."):]: v for k, v in sd.items()})
        return cands

    for cand in _candidates(dict(state_dict)):
        tried.append(",".join(sorted(cand.keys())[:2]))
        try:
            net.load_state_dict(cand, strict=True)
            last_err = None
            break
        except RuntimeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise RuntimeError(
            f"Failed to load temp checkpoint {ckpt_path}. "
            f"Tried first-key samples: {tried}. Last error: {last_err}"
        )

    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


# ----------------------------------------------------------------------------
# Geometry loader (3D JSON -> numpy bundles)
# ----------------------------------------------------------------------------

@dataclass
class Geometry3D:
    """Parsed 3D geometry bundle."""

    xyz_inside: np.ndarray  # (N_in, 3) normalised
    xyz_wall: np.ndarray  # (N_w, 3) normalised
    xyz_inlet: np.ndarray  # (N_il, 3)
    xyz_outlet: np.ndarray  # (N_ol, 3)
    dw_inside: np.ndarray  # (N_in, 1)
    sin_inside: np.ndarray  # (N_in, 1)
    sout_inside: np.ndarray  # (N_in, 1)
    z_aspect: float
    z_slices: int
    norm: Tuple[float, float, float, float, float, float]
    # Optional pre-computed features for wall / inlet (nearest-interior projection
    # at evaluation time; we build them locally here).
    # Left here for future extension; we build from_init_fields on the fly.


def _load_geometry(geom_json: Path) -> Geometry3D:
    """Parse the V4.1 3D geometry JSON for evaluation.

    Implementation note: we use a standalone parser here rather than
    reaching into src/partner_v4_1_flow.py to keep the evaluator
    decoupled from the trainer's CUDA-initialising imports.
    """

    if not geom_json.exists():
        raise FileNotFoundError(f"Geometry JSON not found: {geom_json}")

    with open(geom_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    required = [
        "z_aspect", "z_slices", "norm",
        "points_inside", "points_wall", "points_inlet", "points_outlet",
    ]
    missing = [k for k in required if k not in obj]
    if missing:
        raise KeyError(f"geometry JSON missing required keys: {missing}")

    pts_in = np.asarray(obj["points_inside"], dtype=np.float32)
    pts_w = np.asarray(obj["points_wall"], dtype=np.float32)
    pts_il = np.asarray(obj["points_inlet"], dtype=np.float32)
    pts_ol = np.asarray(obj["points_outlet"], dtype=np.float32)

    if pts_in.ndim != 2 or pts_in.shape[1] < 3:
        raise ValueError(f"points_inside must have >=3 cols, got {pts_in.shape}")
    if pts_w.ndim != 2 or pts_w.shape[1] < 3:
        raise ValueError(f"points_wall must have >=3 cols, got {pts_w.shape}")

    xyz_in = pts_in[:, 0:3].astype(np.float32)
    xyz_w = pts_w[:, 0:3].astype(np.float32)
    xyz_il = pts_il[:, 0:3].astype(np.float32) if pts_il.size > 0 else np.zeros((0, 3), dtype=np.float32)
    xyz_ol = pts_ol[:, 0:3].astype(np.float32) if pts_ol.size > 0 else np.zeros((0, 3), dtype=np.float32)

    z_aspect = float(obj["z_aspect"])
    z_slices = int(obj["z_slices"])
    norm_list = obj["norm"]
    if len(norm_list) != 6:
        raise ValueError(f"norm must have 6 entries, got {norm_list}")
    norm = tuple(float(v) for v in norm_list)  # type: ignore

    # Interior features: prefer init_fields for parity with training; if not
    # present, recompute via a cKDTree wall distance. (geodesic s_in/s_out
    # are expensive; for evaluation we fall back to zeros and warn.)
    init = obj.get("init_fields")
    if init is not None and "fields" in init:
        f = init["fields"]
        dw = np.asarray(f.get("dw", []), dtype=np.float32).reshape(-1, 1)
        s_in = np.asarray(f.get("s_in", []), dtype=np.float32).reshape(-1, 1)
        s_out = np.asarray(f.get("s_out", []), dtype=np.float32).reshape(-1, 1)
        if dw.shape[0] != xyz_in.shape[0]:
            dw = _compute_wall_distance_3d(xyz_in, xyz_w)
        if s_in.shape[0] != xyz_in.shape[0]:
            s_in = np.zeros((xyz_in.shape[0], 1), dtype=np.float32)
        if s_out.shape[0] != xyz_in.shape[0]:
            s_out = np.zeros((xyz_in.shape[0], 1), dtype=np.float32)
    else:
        dw = _compute_wall_distance_3d(xyz_in, xyz_w)
        s_in = np.zeros((xyz_in.shape[0], 1), dtype=np.float32)
        s_out = np.zeros((xyz_in.shape[0], 1), dtype=np.float32)

    return Geometry3D(
        xyz_inside=xyz_in, xyz_wall=xyz_w,
        xyz_inlet=xyz_il, xyz_outlet=xyz_ol,
        dw_inside=dw, sin_inside=s_in, sout_inside=s_out,
        z_aspect=z_aspect, z_slices=z_slices, norm=norm,
    )


def _compute_wall_distance_3d(xyz_query: np.ndarray, xyz_wall: np.ndarray) -> np.ndarray:
    """3D Euclidean nearest-wall distance via cKDTree (matches
    partner_v4_1_geometry.compute_wall_distance_3d)."""
    from scipy.spatial import cKDTree  # type: ignore
    if xyz_wall.shape[0] == 0 or xyz_query.shape[0] == 0:
        return np.zeros((xyz_query.shape[0], 1), dtype=np.float32)
    tree = cKDTree(xyz_wall.astype(np.float32))
    d, _ = tree.query(xyz_query.astype(np.float32), k=1)
    return d.astype(np.float32).reshape(-1, 1)


def _project_inside_to_wall(xyz_wall: np.ndarray, xyz_inside: np.ndarray,
                            feat_inside: np.ndarray) -> np.ndarray:
    """Nearest-interior-neighbour projection of an interior feature onto
    wall points (used for s_in/s_out on wall-point queries).
    """
    from scipy.spatial import cKDTree  # type: ignore
    if xyz_wall.shape[0] == 0 or xyz_inside.shape[0] == 0:
        return np.zeros((xyz_wall.shape[0], feat_inside.shape[1]), dtype=np.float32)
    tree = cKDTree(xyz_inside.astype(np.float32))
    _, idx = tree.query(xyz_wall.astype(np.float32), k=1)
    return feat_inside[idx].astype(np.float32).reshape(xyz_wall.shape[0], -1)


# ----------------------------------------------------------------------------
# Autograd helpers
# ----------------------------------------------------------------------------

def _flow_forward(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                  z: torch.Tensor, dw: torch.Tensor, sin_: torch.Tensor,
                  sout: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Call the PhysicsNeMo FullyConnectedArch. All inputs are (N,1)."""
    return net({"x": x, "y": y, "z": z, "dw": dw, "sin": sin_, "sout": sout})


def _grad(o: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        o, i, grad_outputs=torch.ones_like(o),
        create_graph=True, retain_graph=True,
    )[0]


def _flow_residuals(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                    z: torch.Tensor, dw: torch.Tensor, sin_: torch.Tensor,
                    sout: torch.Tensor, rho: float, nu: float
                    ) -> Dict[str, torch.Tensor]:
    """Compute 3D NS residuals via baseline torch autograd.

    Returns dict: cont, momx, momy, momz, u, v, w, p (all detached).
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    out = _flow_forward(net, x, y, z, dw, sin_, sout)
    u, v, w, p = out["u"], out["v"], out["w"], out["p"]

    u_x = _grad(u, x); u_y = _grad(u, y); u_z = _grad(u, z)
    v_x = _grad(v, x); v_y = _grad(v, y); v_z = _grad(v, z)
    w_x = _grad(w, x); w_y = _grad(w, y); w_z = _grad(w, z)
    p_x = _grad(p, x); p_y = _grad(p, y); p_z = _grad(p, z)
    u_xx = _grad(u_x, x); u_yy = _grad(u_y, y); u_zz = _grad(u_z, z)
    v_xx = _grad(v_x, x); v_yy = _grad(v_y, y); v_zz = _grad(v_z, z)
    w_xx = _grad(w_x, x); w_yy = _grad(w_y, y); w_zz = _grad(w_z, z)

    cont = u_x + v_y + w_z
    mom_x = u * u_x + v * u_y + w * u_z + p_x / rho - nu * (u_xx + u_yy + u_zz)
    mom_y = u * v_x + v * v_y + w * v_z + p_y / rho - nu * (v_xx + v_yy + v_zz)
    mom_z = u * w_x + v * w_y + w * w_z + p_z / rho - nu * (w_xx + w_yy + w_zz)

    return {
        "cont": cont.detach(),
        "momx": mom_x.detach(),
        "momy": mom_y.detach(),
        "momz": mom_z.detach(),
        "u": u.detach(),
        "v": v.detach(),
        "w": w.detach(),
        "p": p.detach(),
    }


def _temp_residual_3d(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                       z: torch.Tensor, t: torch.Tensor, u: torch.Tensor,
                       v: torch.Tensor, w: torch.Tensor, D: float, Q: float
                       ) -> Dict[str, torch.Tensor]:
    """Autograd 3D advection-diffusion residual.

    R = T_t + u T_x + v T_y + w T_z - D (T_xx + T_yy + T_zz) - Q
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    inp = torch.cat([x, y, z, t, u, v, w], dim=1)
    T = net(inp)  # (N,1)

    T_x = _grad(T, x); T_y = _grad(T, y); T_z = _grad(T, z); T_t = _grad(T, t)
    T_xx = _grad(T_x, x); T_yy = _grad(T_y, y); T_zz = _grad(T_z, z)

    R = T_t + u * T_x + v * T_y + w * T_z - D * (T_xx + T_yy + T_zz) - Q
    return {"resid": R.detach(), "T": T.detach()}


# ----------------------------------------------------------------------------
# Batched autograd over large point counts
# ----------------------------------------------------------------------------

def _chunked_flow_residuals(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                            z: torch.Tensor, dw: torch.Tensor, sin_: torch.Tensor,
                            sout: torch.Tensor, rho: float, nu: float,
                            chunk: int = 8192) -> Dict[str, torch.Tensor]:
    """Run ``_flow_residuals`` in chunks and concatenate."""
    n = int(x.shape[0])
    if n == 0:
        empty = torch.zeros((0, 1), device=x.device, dtype=x.dtype)
        return {k: empty.clone() for k in ("cont", "momx", "momy", "momz",
                                            "u", "v", "w", "p")}
    out_parts: Dict[str, List[torch.Tensor]] = {k: [] for k in
                                                  ("cont", "momx", "momy", "momz",
                                                   "u", "v", "w", "p")}
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        res = _flow_residuals(
            net, x[s:e], y[s:e], z[s:e],
            dw[s:e], sin_[s:e], sout[s:e],
            rho=rho, nu=nu,
        )
        for k in out_parts:
            out_parts[k].append(res[k])
    return {k: torch.cat(v, dim=0) for k, v in out_parts.items()}


def _chunked_temp_residual(net: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                            z: torch.Tensor, t: torch.Tensor, u: torch.Tensor,
                            v: torch.Tensor, w: torch.Tensor, D: float, Q: float,
                            chunk: int = 8192) -> Dict[str, torch.Tensor]:
    n = int(x.shape[0])
    if n == 0:
        empty = torch.zeros((0, 1), device=x.device, dtype=x.dtype)
        return {"resid": empty.clone(), "T": empty.clone()}
    R_parts: List[torch.Tensor] = []
    T_parts: List[torch.Tensor] = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        res = _temp_residual_3d(
            net, x[s:e], y[s:e], z[s:e], t[s:e],
            u[s:e], v[s:e], w[s:e], D=D, Q=Q,
        )
        R_parts.append(res["resid"])
        T_parts.append(res["T"])
    return {"resid": torch.cat(R_parts, dim=0), "T": torch.cat(T_parts, dim=0)}


# ----------------------------------------------------------------------------
# Flow-field lookup for temp evaluation (uses the saved pred_flow_steady.json
# if available; otherwise runs the flow net to produce u,v,w on interior pts)
# ----------------------------------------------------------------------------

def _load_flow_field_for_temp(flow_json_path: Path, xyz_inside: np.ndarray
                               ) -> Optional[Dict[str, np.ndarray]]:
    """Load (u, v, w) evaluated on interior points from the flow JSON, if
    present. Returns a dict with keys 'u', 'v', 'w', each shape (N_in, 1),
    or None if the JSON is missing.
    """
    if not flow_json_path.exists():
        return None
    try:
        with open(flow_json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return None
    if "points" not in obj or "flow" not in obj:
        return None
    xyz_all = np.asarray(obj["points"], dtype=np.float32)
    u_all = np.asarray(obj["flow"].get("u", []), dtype=np.float32).reshape(-1, 1)
    v_all = np.asarray(obj["flow"].get("v", []), dtype=np.float32).reshape(-1, 1)
    w_all = np.asarray(obj["flow"].get("w", []), dtype=np.float32).reshape(-1, 1)
    if u_all.shape[0] != xyz_all.shape[0]:
        return None
    # Interpolate/gather onto xyz_inside via nearest-neighbour.
    from scipy.spatial import cKDTree  # type: ignore
    tree = cKDTree(xyz_all)
    _, idx = tree.query(xyz_inside.astype(np.float32), k=1)
    return {"u": u_all[idx].reshape(-1, 1),
            "v": v_all[idx].reshape(-1, 1),
            "w": w_all[idx].reshape(-1, 1)}


# ----------------------------------------------------------------------------
# Section drivers
# ----------------------------------------------------------------------------

def _evaluate_flow_residuals(log: TeeLogger, net: torch.nn.Module,
                              geom: Geometry3D, idx: np.ndarray,
                              cfg: Dict[str, Any], device: torch.device
                              ) -> Tuple[Dict[str, float], Dict[str, float],
                                         Dict[str, float]]:
    """Section A + B: held-out residuals + field stats on the same sample."""

    log("")
    log("=== Section A: held-out NS residuals (baseline autograd, nu=%.3e) ===" %
        cfg["nu_final"])
    t0 = time.time()
    xyz = geom.xyz_inside[idx]
    dw = geom.dw_inside[idx]
    sin_ = geom.sin_inside[idx]
    sout = geom.sout_inside[idx]

    x = torch.from_numpy(xyz[:, 0:1]).to(device)
    y = torch.from_numpy(xyz[:, 1:2]).to(device)
    z = torch.from_numpy(xyz[:, 2:3]).to(device)
    dw_t = torch.from_numpy(dw).to(device)
    sin_t = torch.from_numpy(sin_).to(device)
    sout_t = torch.from_numpy(sout).to(device)

    res = _chunked_flow_residuals(
        net, x, y, z, dw_t, sin_t, sout_t,
        rho=float(cfg["rho"]), nu=float(cfg["nu_final"]),
        chunk=4096,
    )
    elapsed = time.time() - t0
    log(f"  computed {x.shape[0]} residuals in {elapsed:.2f}s")

    resid = {
        "continuity_rmse": _rmse(res["cont"]),
        "continuity_absmax": _absmax(res["cont"]),
        "momentum_x_rmse": _rmse(res["momx"]),
        "momentum_x_absmax": _absmax(res["momx"]),
        "momentum_y_rmse": _rmse(res["momy"]),
        "momentum_y_absmax": _absmax(res["momy"]),
        "momentum_z_rmse": _rmse(res["momz"]),
        "momentum_z_absmax": _absmax(res["momz"]),
    }

    log(f"  continuity RMSE={resid['continuity_rmse']:.4e}  "
        f"absmax={resid['continuity_absmax']:.4e}")
    log(f"  momentum_x RMSE={resid['momentum_x_rmse']:.4e}  "
        f"absmax={resid['momentum_x_absmax']:.4e}")
    log(f"  momentum_y RMSE={resid['momentum_y_rmse']:.4e}  "
        f"absmax={resid['momentum_y_absmax']:.4e}")
    log(f"  momentum_z RMSE={resid['momentum_z_rmse']:.4e}  "
        f"absmax={resid['momentum_z_absmax']:.4e}")

    # --- Section B: field statistics on the same sample ---
    log("")
    log("=== Section B: flow-field statistics on the same sample ===")
    u, v, w, p = res["u"], res["v"], res["w"], res["p"]
    speed = torch.sqrt(u ** 2 + v ** 2 + w ** 2 + 1.0e-20)
    ratio_wspd = torch.abs(w) / speed

    stats = {
        "u_rms": _rmse(u), "u_min": float(u.min().item()), "u_max": float(u.max().item()),
        "v_rms": _rmse(v), "v_min": float(v.min().item()), "v_max": float(v.max().item()),
        "w_rms": _rmse(w), "w_min": float(w.min().item()), "w_max": float(w.max().item()),
        "p_rms": _rmse(p), "p_min": float(p.min().item()), "p_max": float(p.max().item()),
        "p_mean": _mean(p),
        "speed_rms": _rmse(speed), "speed_max": float(speed.max().item()),
        "speed_mean": _mean(speed),
        "abs_w_over_speed_mean": _mean(ratio_wspd),
        "abs_w_over_speed_max": float(ratio_wspd.max().item()),
    }
    log(f"  u RMS={stats['u_rms']:.4e}  min={stats['u_min']:.4e}  max={stats['u_max']:.4e}")
    log(f"  v RMS={stats['v_rms']:.4e}  min={stats['v_min']:.4e}  max={stats['v_max']:.4e}")
    log(f"  w RMS={stats['w_rms']:.4e}  min={stats['w_min']:.4e}  max={stats['w_max']:.4e}")
    log(f"  p RMS={stats['p_rms']:.4e}  min={stats['p_min']:.4e}  max={stats['p_max']:.4e}  "
        f"mean={stats['p_mean']:.4e}")
    log(f"  speed RMS={stats['speed_rms']:.4e}  mean={stats['speed_mean']:.4e}  "
        f"max={stats['speed_max']:.4e}")
    log(f"  |w|/speed  mean={stats['abs_w_over_speed_mean']:.4e}  "
        f"max={stats['abs_w_over_speed_max']:.4e}")

    # --- Section C placeholder dict is returned separately by caller ---
    return resid, stats, {"n_eval_points": int(x.shape[0])}


def _evaluate_wall_noslip(log: TeeLogger, net: torch.nn.Module,
                           geom: Geometry3D, device: torch.device,
                           max_pts: int = 5000, seed: int = 1234
                           ) -> Dict[str, Any]:
    log("")
    log("=== Section C: wall no-slip exactness ===")
    if geom.xyz_wall.shape[0] == 0:
        log("  [SKIP] no wall points in geometry")
        return {"u_absmax": float("nan"), "v_absmax": float("nan"),
                "w_absmax": float("nan"), "n_wall_sampled": 0,
                "note": "no wall points"}

    rng = np.random.RandomState(seed)
    n = min(int(max_pts), int(geom.xyz_wall.shape[0]))
    idx = rng.choice(geom.xyz_wall.shape[0], size=n, replace=False)
    xyz_w = geom.xyz_wall[idx]
    # wall dw = 0 exactly by construction
    dw_w = np.zeros((n, 1), dtype=np.float32)
    # s_in/s_out projected from interior via nearest neighbour
    s_in_w = _project_inside_to_wall(xyz_w, geom.xyz_inside, geom.sin_inside)
    s_out_w = _project_inside_to_wall(xyz_w, geom.xyz_inside, geom.sout_inside)

    x_t = torch.from_numpy(xyz_w[:, 0:1]).to(device)
    y_t = torch.from_numpy(xyz_w[:, 1:2]).to(device)
    z_t = torch.from_numpy(xyz_w[:, 2:3]).to(device)
    dw_t = torch.from_numpy(dw_w).to(device)
    sin_t = torch.from_numpy(s_in_w).to(device)
    sout_t = torch.from_numpy(s_out_w).to(device)

    with torch.no_grad():
        out = _flow_forward(net, x_t, y_t, z_t, dw_t, sin_t, sout_t)

    result = {
        "u_absmax": _absmax(out["u"]),
        "v_absmax": _absmax(out["v"]),
        "w_absmax": _absmax(out["w"]),
        "u_rmse": _rmse(out["u"]),
        "v_rmse": _rmse(out["v"]),
        "w_rmse": _rmse(out["w"]),
        "n_wall_sampled": int(n),
    }
    log(f"  sampled n={n}")
    log(f"  u absmax={result['u_absmax']:.4e}  rmse={result['u_rmse']:.4e}")
    log(f"  v absmax={result['v_absmax']:.4e}  rmse={result['v_rmse']:.4e}")
    log(f"  w absmax={result['w_absmax']:.4e}  rmse={result['w_rmse']:.4e}")
    return result


def _evaluate_inlet_bc(log: TeeLogger, net: torch.nn.Module,
                        geom: Geometry3D, cfg: Dict[str, Any],
                        device: torch.device) -> Dict[str, Any]:
    log("")
    log("=== Section D: inlet BC (parabolic Poiseuille profile) ===")
    if geom.xyz_inlet.shape[0] == 0:
        log("  [SKIP] no inlet points in geometry")
        return {"u_mae": float("nan"), "v_mae": float("nan"),
                "w_mae": float("nan"), "n_inlet_sampled": 0,
                "note": "no inlet points"}

    xyz_il = geom.xyz_inlet
    n = int(xyz_il.shape[0])

    # Project interior dw / s_in / s_out onto inlet patch via nearest-neighbour
    dw_il = _project_inside_to_wall(xyz_il, geom.xyz_inside, geom.dw_inside)
    s_in_il = _project_inside_to_wall(xyz_il, geom.xyz_inside, geom.sin_inside)
    s_out_il = _project_inside_to_wall(xyz_il, geom.xyz_inside, geom.sout_inside)

    z_max = max(float(geom.z_aspect), 1.0e-8)
    z_il = xyz_il[:, 2:3]
    phi = 4.0 * (z_il / z_max) * (1.0 - z_il / z_max)
    phi = np.clip(phi, 0.0, 1.0).astype(np.float32)

    u_tgt = float(cfg["inlet_u"]) * phi
    v_tgt = float(cfg["inlet_v"]) * phi
    w_tgt = np.zeros_like(phi)

    x_t = torch.from_numpy(xyz_il[:, 0:1].astype(np.float32)).to(device)
    y_t = torch.from_numpy(xyz_il[:, 1:2].astype(np.float32)).to(device)
    z_t = torch.from_numpy(z_il.astype(np.float32)).to(device)
    dw_t = torch.from_numpy(dw_il).to(device)
    sin_t = torch.from_numpy(s_in_il).to(device)
    sout_t = torch.from_numpy(s_out_il).to(device)

    with torch.no_grad():
        out = _flow_forward(net, x_t, y_t, z_t, dw_t, sin_t, sout_t)

    u_tgt_t = torch.from_numpy(u_tgt).to(device)
    v_tgt_t = torch.from_numpy(v_tgt).to(device)
    w_tgt_t = torch.from_numpy(w_tgt).to(device)

    mae_u = float((out["u"] - u_tgt_t).abs().mean().item())
    mae_v = float((out["v"] - v_tgt_t).abs().mean().item())
    mae_w = float((out["w"] - w_tgt_t).abs().mean().item())

    result = {
        "u_mae": mae_u,
        "v_mae": mae_v,
        "w_mae": mae_w,
        "u_target_mean": float(u_tgt_t.mean().item()),
        "v_target_mean": float(v_tgt_t.mean().item()),
        "u_pred_mean": float(out["u"].mean().item()),
        "v_pred_mean": float(out["v"].mean().item()),
        "w_pred_mean": float(out["w"].mean().item()),
        "n_inlet_sampled": int(n),
    }
    log(f"  n={n}  z_max={z_max:.4f}")
    log(f"  u MAE={mae_u:.4e}  (target mean={result['u_target_mean']:.4e}, "
        f"pred mean={result['u_pred_mean']:.4e})")
    log(f"  v MAE={mae_v:.4e}  (target mean={result['v_target_mean']:.4e}, "
        f"pred mean={result['v_pred_mean']:.4e})")
    log(f"  w MAE={mae_w:.4e}  (target = 0, pred mean={result['w_pred_mean']:.4e})")
    return result


def _evaluate_temp(log: TeeLogger, temp_ckpt: Path, temp_pred_h5: Path,
                    flow_pred_json: Path, flow_net: Optional[torch.nn.Module],
                    geom: Geometry3D, idx: np.ndarray, cfg: Dict[str, Any],
                    device: torch.device) -> Tuple[Dict[str, Any],
                                                     Dict[str, Any],
                                                     Dict[str, Any]]:
    """Section E: temperature residuals + IC + rollout summary.

    Returns (temp_residuals, temp_ic, temp_rollout_summary).
    """

    log("")
    log("=== Section E: temperature evaluation ===")

    if not temp_ckpt.exists():
        log(f"  [SKIP] temp checkpoint missing: {temp_ckpt}")
        nan_temp_res = {f"rmse_at_t_{t:g}".replace(".", "_"): float("nan")
                         for t in (0.0, 5.0, 10.0, 20.0, 40.0)}
        return (dict(nan_temp_res, skipped=True, reason="ckpt_missing"),
                {"mean_T_t0": float("nan"), "abs_err_t0": float("nan"),
                 "target": float(cfg["T_init"]), "skipped": True,
                 "reason": "ckpt_missing"},
                {"times": [], "T_min": [], "T_max": [], "T_mean": [],
                 "skipped": True, "reason": "ckpt_missing"})

    temp_net = _load_temp_net(
        temp_ckpt,
        hidden_size=int(cfg["temp_hidden_size"]),
        hidden_layers=int(cfg["temp_hidden_layers"]),
        activation=str(cfg["temp_activation"]),
        device=device,
    )
    log(f"  loaded temp net from {temp_ckpt}")
    log(f"  params: {sum(p.numel() for p in temp_net.parameters()):,}")

    # --- Fetch u,v,w at interior sample points ---
    uvw = _load_flow_field_for_temp(flow_pred_json, geom.xyz_inside)
    if uvw is not None:
        log(f"  using saved flow JSON at {flow_pred_json}")
        u_full = uvw["u"]; v_full = uvw["v"]; w_full = uvw["w"]
    elif flow_net is not None:
        log(f"  saved flow JSON not found, running flow net on interior points")
        x_all = torch.from_numpy(geom.xyz_inside[:, 0:1].astype(np.float32)).to(device)
        y_all = torch.from_numpy(geom.xyz_inside[:, 1:2].astype(np.float32)).to(device)
        z_all = torch.from_numpy(geom.xyz_inside[:, 2:3].astype(np.float32)).to(device)
        dw_all = torch.from_numpy(geom.dw_inside.astype(np.float32)).to(device)
        sin_all = torch.from_numpy(geom.sin_inside.astype(np.float32)).to(device)
        sout_all = torch.from_numpy(geom.sout_inside.astype(np.float32)).to(device)

        u_chunks: List[np.ndarray] = []
        v_chunks: List[np.ndarray] = []
        w_chunks: List[np.ndarray] = []
        chunk = 32768
        with torch.no_grad():
            for s in range(0, x_all.shape[0], chunk):
                e = min(s + chunk, x_all.shape[0])
                o = _flow_forward(
                    flow_net, x_all[s:e], y_all[s:e], z_all[s:e],
                    dw_all[s:e], sin_all[s:e], sout_all[s:e],
                )
                u_chunks.append(o["u"].cpu().numpy())
                v_chunks.append(o["v"].cpu().numpy())
                w_chunks.append(o["w"].cpu().numpy())
        u_full = np.concatenate(u_chunks, axis=0)
        v_full = np.concatenate(v_chunks, axis=0)
        w_full = np.concatenate(w_chunks, axis=0)
    else:
        log("  [SKIP] no flow-field source available for temp residual")
        nan_temp_res = {f"rmse_at_t_{t:g}".replace(".", "_"): float("nan")
                         for t in (0.0, 5.0, 10.0, 20.0, 40.0)}
        return (dict(nan_temp_res, skipped=True, reason="no_flow_uvw"),
                {"mean_T_t0": float("nan"), "abs_err_t0": float("nan"),
                 "target": float(cfg["T_init"]), "skipped": True,
                 "reason": "no_flow_uvw"},
                {"times": [], "T_min": [], "T_max": [], "T_mean": [],
                 "skipped": True, "reason": "no_flow_uvw"})

    u_s = u_full[idx].astype(np.float32)
    v_s = v_full[idx].astype(np.float32)
    w_s = w_full[idx].astype(np.float32)

    xyz_s = geom.xyz_inside[idx]
    x_t = torch.from_numpy(xyz_s[:, 0:1].astype(np.float32)).to(device)
    y_t = torch.from_numpy(xyz_s[:, 1:2].astype(np.float32)).to(device)
    z_t = torch.from_numpy(xyz_s[:, 2:3].astype(np.float32)).to(device)
    u_t = torch.from_numpy(u_s).to(device)
    v_t = torch.from_numpy(v_s).to(device)
    w_t = torch.from_numpy(w_s).to(device)

    eval_times = (0.0, 5.0, 10.0, 20.0, 40.0)
    temp_residuals: Dict[str, Any] = {}
    log("  computing temp residuals at eval times:")
    for t_val in eval_times:
        t_tensor = torch.full((x_t.shape[0], 1), float(t_val), device=device)
        res = _chunked_temp_residual(
            temp_net, x_t, y_t, z_t, t_tensor, u_t, v_t, w_t,
            D=float(cfg["D"]), Q=float(cfg["Q"]), chunk=2048,
        )
        key = f"rmse_at_t_{t_val:g}".replace(".", "_")
        temp_residuals[key] = _rmse(res["resid"])
        log(f"    t={t_val:5.1f}  resid RMSE={temp_residuals[key]:.4e}  "
            f"T RMS={_rmse(res['T']):.4e}  T abs-max={_absmax(res['T']):.4e}")

    # --- IC (t=0) ---
    with torch.no_grad():
        t0 = torch.zeros_like(x_t)
        inp0 = torch.cat([x_t, y_t, z_t, t0, u_t, v_t, w_t], dim=1)
        T0 = temp_net(inp0)
    mean_T_t0 = float(T0.mean().item())
    target = float(cfg["T_init"])
    temp_ic = {
        "mean_T_t0": mean_T_t0,
        "abs_err_t0": abs(mean_T_t0 - target),
        "target": target,
        "std_T_t0": float(T0.std().item()),
    }
    log(f"  IC @ t=0: mean T = {mean_T_t0:+.4e}  (target {target:+.4e}, "
        f"abs err = {temp_ic['abs_err_t0']:.4e}, "
        f"std T = {temp_ic['std_T_t0']:.4e})")

    # --- Rollout summary from the HDF5 (free) ---
    rollout: Dict[str, Any] = {"times": [], "T_min": [], "T_max": [], "T_mean": []}
    if temp_pred_h5.exists():
        try:
            import h5py  # type: ignore
            with h5py.File(str(temp_pred_h5), "r") as hf:
                # V4.1 temp trainer schema (partner_v4_1_temp.py):
                # top-level datasets `temperature` (T, N), `times` (T,),
                # `xyz_norm`, `xyz_raw`, `u`, `v`, `w`, `point_type`.
                T_arr: Optional[np.ndarray] = None
                times_arr: Optional[np.ndarray] = None
                if "temperature" in hf and "times" in hf:
                    T_arr = np.asarray(hf["temperature"])
                    times_arr = np.asarray(hf["times"]).reshape(-1)
                elif "T" in hf and "times" in hf:
                    T_arr = np.asarray(hf["T"])
                    times_arr = np.asarray(hf["times"]).reshape(-1)
                if T_arr is not None and times_arr is not None:
                    if T_arr.ndim == 2 and T_arr.shape[0] == times_arr.shape[0]:
                        for i, t in enumerate(times_arr):
                            rollout["times"].append(float(t))
                            Ti = T_arr[i].astype(np.float32)
                            rollout["T_min"].append(float(np.min(Ti)))
                            rollout["T_max"].append(float(np.max(Ti)))
                            rollout["T_mean"].append(float(np.mean(Ti)))
                    elif T_arr.ndim == 2 and T_arr.shape[1] == times_arr.shape[0]:
                        # transposed layout: (N, T)
                        for i, t in enumerate(times_arr):
                            rollout["times"].append(float(t))
                            Ti = T_arr[:, i].astype(np.float32)
                            rollout["T_min"].append(float(np.min(Ti)))
                            rollout["T_max"].append(float(np.max(Ti)))
                            rollout["T_mean"].append(float(np.mean(Ti)))
                else:
                    # group-per-time fallback
                    for k in sorted(hf.keys()):
                        g = hf[k]
                        if not hasattr(g, "keys"):
                            continue
                        t_val = float(g.attrs.get("t", k.split("_")[-1])) \
                                 if hasattr(g, "attrs") else float("nan")
                        T_vals: Optional[np.ndarray] = None
                        for tkey in ("T", "temperature"):
                            if tkey in g.keys():
                                T_vals = np.asarray(g[tkey])
                                break
                        if T_vals is None:
                            continue
                        rollout["times"].append(t_val)
                        rollout["T_min"].append(float(np.min(T_vals)))
                        rollout["T_max"].append(float(np.max(T_vals)))
                        rollout["T_mean"].append(float(np.mean(T_vals)))
            log(f"  rollout HDF5 loaded: {len(rollout['times'])} snapshots")
            if rollout["times"]:
                log(f"    t range: [{min(rollout['times']):.2f}, "
                    f"{max(rollout['times']):.2f}]  "
                    f"T_mean range: [{min(rollout['T_mean']):.3e}, "
                    f"{max(rollout['T_mean']):.3e}]")
        except Exception as exc:
            rollout = {"times": [], "T_min": [], "T_max": [], "T_mean": [],
                       "skipped": True, "reason": f"h5py_error: {exc}"}
            log(f"  [WARN] failed to read HDF5 {temp_pred_h5}: {exc}")
    else:
        rollout = {"times": [], "T_min": [], "T_max": [], "T_mean": [],
                   "skipped": True, "reason": "h5_missing"}
        log(f"  [SKIP] rollout HDF5 missing: {temp_pred_h5}")

    return temp_residuals, temp_ic, rollout


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    _seed_everything(int(args.seed))

    geom_json = Path(args.geom_json)
    temp_ckpt = Path(args.temp_checkpoint)
    temp_pred_h5 = Path(args.temp_predictions)
    config_path = Path(args.config_path)
    out_json = Path(args.output_json)
    out_log = Path(args.output_log)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    log = TeeLogger(out_log)

    try:
        log("V4.1 3D PINN task-quality evaluator")
        log(f"  geometry JSON:     {geom_json}")
        log(f"  config path:       {config_path}")
        log(f"  temp checkpoint:   {temp_ckpt}")
        log(f"  temp predictions:  {temp_pred_h5}")
        log(f"  n_eval_points:     {args.n_eval_points}")
        log(f"  seed:              {args.seed}")
        log(f"  device:            {device}")

        # --- Config ---
        cfg = _load_config(config_path)
        log("  --- config values ---")
        log(f"  rho={cfg['rho']:.3f}  nu_final={cfg['nu_final']:.3e}  "
            f"D={cfg['D']:.3e}  Q={cfg['Q']:.3e}")
        log(f"  T_init={cfg['T_init']:.3f}  inlet_T={cfg['inlet_T']:.3f}  "
            f"inlet_u={cfg['inlet_u']:.3f}  inlet_v={cfg['inlet_v']:.3f}")
        log(f"  flow_net: {cfg['flow_hidden_layers']} x {cfg['flow_hidden_size']} "
            f"({cfg['flow_activation']})")
        log(f"  temp_net: {cfg['temp_hidden_layers']} x {cfg['temp_hidden_size']} "
            f"({cfg['temp_activation']})")

        # --- Flow checkpoint path ---
        flow_ckpt = _resolve_flow_checkpoint(args.flow_checkpoint)
        log(f"  flow checkpoint:   {flow_ckpt}")

        # --- Load geometry ---
        log("")
        log("Loading 3D geometry JSON...")
        t0 = time.time()
        geom = _load_geometry(geom_json)
        log(f"  inside={geom.xyz_inside.shape[0]}  wall={geom.xyz_wall.shape[0]}  "
            f"inlet={geom.xyz_inlet.shape[0]}  outlet={geom.xyz_outlet.shape[0]}")
        log(f"  z_aspect={geom.z_aspect:.4f}  z_slices={geom.z_slices}  "
            f"norm={geom.norm}")
        log(f"  (loaded in {time.time() - t0:.2f}s)")

        # --- Load flow net ---
        log("")
        log("Loading flow net...")
        t0 = time.time()
        flow_net = _load_flow_net(
            flow_ckpt,
            hidden_size=int(cfg["flow_hidden_size"]),
            hidden_layers=int(cfg["flow_hidden_layers"]),
            activation=str(cfg["flow_activation"]),
            device=device,
        )
        log(f"  flow net loaded in {time.time() - t0:.2f}s, "
            f"params={sum(p.numel() for p in flow_net.parameters()):,}")

        # --- Sample indices ---
        rng = np.random.RandomState(int(args.seed))
        N_int = int(geom.xyz_inside.shape[0])
        n_eval = int(min(args.n_eval_points, N_int))
        idx = rng.choice(N_int, size=n_eval, replace=False)
        log(f"  sampled {n_eval} / {N_int} interior points for eval")

        # --- Sections A + B ---
        flow_resid, flow_stats, misc = _evaluate_flow_residuals(
            log, flow_net, geom, idx, cfg, device,
        )

        # --- Section C: wall ---
        wall_res = _evaluate_wall_noslip(
            log, flow_net, geom, device, max_pts=5000,
            seed=int(args.seed),
        )

        # --- Section D: inlet ---
        inlet_res = _evaluate_inlet_bc(log, flow_net, geom, cfg, device)

        # --- Section E: temperature ---
        # Flow-field JSON sits next to the geometry JSON (e.g.
        # pipe_three_class_3d_pred_flow_steady.json).
        flow_json_path = geom_json.parent / (geom_json.stem + "_pred_flow_steady.json")
        temp_residuals, temp_ic, temp_rollout = _evaluate_temp(
            log, temp_ckpt, temp_pred_h5, flow_json_path, flow_net,
            geom, idx, cfg, device,
        )

        # --- Build final JSON ---
        summary = {
            "config": {
                "rho": cfg["rho"],
                "nu_final": cfg["nu_final"],
                "D": cfg["D"],
                "Q": cfg["Q"],
                "T_init": cfg["T_init"],
                "inlet_T": cfg["inlet_T"],
                "inlet_u": cfg["inlet_u"],
                "inlet_v": cfg["inlet_v"],
                "flow_hidden_size": cfg["flow_hidden_size"],
                "flow_hidden_layers": cfg["flow_hidden_layers"],
                "temp_hidden_size": cfg["temp_hidden_size"],
                "temp_hidden_layers": cfg["temp_hidden_layers"],
                "n_eval_points": int(n_eval),
                "seed": int(args.seed),
                "device": str(device),
                "geom_json": str(geom_json),
                "flow_checkpoint": str(flow_ckpt),
                "temp_checkpoint": str(temp_ckpt),
                "temp_predictions": str(temp_pred_h5),
            },
            "flow_residuals": flow_resid,
            "flow_field_stats": flow_stats,
            "wall_violation": wall_res,
            "inlet_violation": inlet_res,
            "temp_residuals": temp_residuals,
            "temp_ic": temp_ic,
            "temp_rollout_summary": temp_rollout,
        }

        # --- Write JSON ---
        log("")
        log("Writing JSON summary...")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=False, default=float)
        log(f"  wrote {out_json}")
        log(f"  log  {out_log}")
        log("")
        log("DONE.")
    finally:
        log.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
