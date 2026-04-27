"""
V4 port of from_partner_team/partner_code_v4_full/all_files 3/pcs_runtime.py

Additive change vs partner source: `cad_to_geometry_json` honours the
`PCS_GEOM_JSON_PATH` env var and returns that path verbatim (skipping
CAD→JSON regeneration). This preserves the partner's code path when the
env var is unset, and lets us honour the stream_battery_consortium
standing rule `pipe_three_class_fixed.json md5 = bd17961ae... ; do not
regenerate`. No other behaviour is changed.
"""

import os
from pathlib import Path
from typing import Optional

from point_cloud_sampler import PointCloudSampler  # V3 port of PCS.py (reused per CONTEXT)


CAD_PATH_ENV = "PCS_CAD_PATH"
GEOM_JSON_OVERRIDE_ENV = "PCS_GEOM_JSON_PATH"
VALID_CAD_SUFFIXES = {".step", ".stp", ".stl"}


def resolve_cad_path(prompt: str = "Enter CAD file path (.step/.stp/.stl): ") -> Path:
    raw_path = os.environ.get(CAD_PATH_ENV, "").strip()
    if not raw_path:
        raw_path = input(prompt).strip()
    if not raw_path:
        raise ValueError("CAD path is required.")

    cad_path = Path(raw_path).expanduser()
    if not cad_path.is_absolute():
        cad_path = (Path.cwd() / cad_path).resolve()
    else:
        cad_path = cad_path.resolve()

    if not cad_path.exists():
        raise FileNotFoundError(f"CAD file not found: {cad_path}")

    suffix = cad_path.suffix.lower()
    if suffix not in VALID_CAD_SUFFIXES:
        valid_str = ", ".join(sorted(VALID_CAD_SUFFIXES))
        raise ValueError(
            f"Expected CAD file with one of [{valid_str}], got: {cad_path.name}"
        )

    return cad_path


def cad_to_geometry_json(
    cad_path: Path,
    output_dir: Optional[Path] = None,
    res: int = 512,
    strip_w: int = 10,
    white_thr: int = 250,
) -> Path:
    override = os.environ.get(GEOM_JSON_OVERRIDE_ENV, "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"{GEOM_JSON_OVERRIDE_ENV} points to non-existent file: {p}"
            )
        print(f"[INFO] pcs_runtime: using {GEOM_JSON_OVERRIDE_ENV} override: {p}")
        return p

    out_dir = Path.cwd() if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = (out_dir / f"{cad_path.stem}_converted.json").resolve()
    # V3-port PointCloudSampler is UI-stripped and does not accept `use_ui`;
    # partner source passed `use_ui=False` which is the only behaviour here.
    pcs = PointCloudSampler(cad_path=str(cad_path))
    pcs.convert_to_json(str(out_json), res=res, strip_w=strip_w, white_thr=white_thr)
    return out_json


def derive_flow_json_path(geom_json_path: Path) -> Path:
    return Path(str(geom_json_path.with_suffix("")) + "_pred_flow_steady.json")
