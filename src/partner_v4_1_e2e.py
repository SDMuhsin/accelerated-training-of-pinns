"""
V4.1 end-to-end orchestrator: 3D flow training then 3D temperature
training + inference + visualisation.

This mirrors ``partner_v4_e2e.py`` but wires to the V4.1 (3D) scripts
and V4.1 config. The 3D geometry is precomputed via
``scripts/build_v4_1_geometry.py`` and lives at
``data/partner_v4_1/pipe_three_class_3d.json``; V4.1 does not need a
CAD step (unlike V4), so PCS_CAD_PATH is not required. The
``--build-geometry`` flag lets this orchestrator (re)build the 3D
geometry JSON in a leading step when needed.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _run_step(name: str, cmd: List[str], cwd: Path, env: dict) -> None:
    print(f"[RUN] {name}: {' '.join(cmd)} (cwd={cwd})", flush=True)  # log cmd
    completed = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)  # run
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")  # bail on failure
    print(f"[OK] {name} completed", flush=True)  # log success


def main() -> None:
    src_dir = Path(__file__).resolve().parent  # src/
    repo_root = src_dir.parent  # repo root

    ap = argparse.ArgumentParser(
        description=(
            "Run full V4.1 E2E (3D): "
            "[optional geometry build] -> flow training -> delay -> temperature training/inference + visualisation"
        )
    )
    ap.add_argument(
        "--flow-script",
        default=str(src_dir / "partner_v4_1_flow.py"),
        help="Path to 3D flow training script.",
    )
    ap.add_argument(
        "--temp-script",
        default=str(src_dir / "partner_v4_1_temp.py"),
        help="Path to 3D temperature training/inference script.",
    )
    ap.add_argument(
        "--geometry-script",
        default=str(repo_root / "scripts" / "build_v4_1_geometry.py"),
        help="Path to 3D geometry builder.",
    )
    ap.add_argument(
        "--delay-seconds",
        type=float,
        default=5.0,
        help="Delay after flow step to allow artifacts to flush.",
    )
    ap.add_argument(
        "--flow-overrides",
        nargs="*",
        default=[],
        help="Optional Hydra overrides for flow script.",
    )
    ap.add_argument(
        "--temp-overrides",
        nargs="*",
        default=[],
        help="Optional Hydra overrides for temperature script.",
    )
    ap.add_argument(
        "--geometry-overrides",
        nargs="*",
        default=[],
        help="Optional CLI overrides for the geometry builder (see build_v4_1_geometry.py).",
    )
    ap.add_argument(
        "--build-geometry",
        action="store_true",
        help="Run the 3D geometry builder before flow training.",
    )
    ap.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip the flow training subprocess (useful when precomputed flow JSON is in place).",
    )
    ap.add_argument(
        "--skip-temp",
        action="store_true",
        help="Skip the temperature subprocess (flow-only runs).",
    )

    args = ap.parse_args()

    flow_script = Path(args.flow_script).resolve()  # absolute
    temp_script = Path(args.temp_script).resolve()  # absolute
    geometry_script = Path(args.geometry_script).resolve()  # absolute

    if not flow_script.exists():
        raise FileNotFoundError(f"Flow script not found: {flow_script}")  # missing
    if not temp_script.exists():
        raise FileNotFoundError(f"Temperature script not found: {temp_script}")  # missing
    if args.build_geometry and not geometry_script.exists():
        raise FileNotFoundError(f"Geometry script not found: {geometry_script}")  # missing

    run_env = os.environ.copy()  # forward env
    # V4.1 reads its own 3D geometry JSON; PCS_CAD_PATH is intentionally NOT required.
    # Retain optional forwarding for convenience if the user sets it anyway.
    if os.environ.get("PCS_GEOM_JSON_PATH"):
        run_env["PCS_GEOM_JSON_PATH"] = os.environ["PCS_GEOM_JSON_PATH"]  # pass-through

    hydra_common = ["hydra.job.chdir=False"]  # keep relative paths anchored to repo root

    if args.build_geometry:
        geom_cmd = [
            sys.executable,
            str(geometry_script),
            *args.geometry_overrides,
        ]
        _run_step("build_v4_1_geometry", geom_cmd, cwd=repo_root, env=run_env)  # 3D geometry

        delay = max(float(args.delay_seconds), 0.0)
        if delay > 0.0:
            print(f"[INFO] waiting {delay:.1f}s after geometry build...", flush=True)  # throttle
            time.sleep(delay)  # pause

    if not args.skip_flow:
        flow_cmd = [
            sys.executable,
            str(flow_script),
            *hydra_common,
            *args.flow_overrides,
        ]
        _run_step("flow_train_and_infer_3d", flow_cmd, cwd=repo_root, env=run_env)  # 3D flow

        delay = max(float(args.delay_seconds), 0.0)
        if delay > 0.0:
            print(f"[INFO] waiting {delay:.1f}s for flow artefacts to flush...", flush=True)  # throttle
            time.sleep(delay)  # pause
    else:
        print("[INFO] --skip-flow set, skipping flow training subprocess", flush=True)  # notify

    if not args.skip_temp:
        temp_cmd = [
            sys.executable,
            str(temp_script),
            *hydra_common,
            *args.temp_overrides,
        ]
        _run_step("temp_train_infer_and_visualize_3d", temp_cmd, cwd=repo_root, env=run_env)  # 3D temp
    else:
        print("[INFO] --skip-temp set, skipping temperature subprocess", flush=True)  # notify

    print("[OK] Full V4.1 E2E (3D) pipeline complete", flush=True)  # done


if __name__ == "__main__":
    main()
