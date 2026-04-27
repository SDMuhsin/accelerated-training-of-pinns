"""
Port of from_partner_team/partner_code_v4_full/all_files 3/full_e2e_with_vis.py.

Changes vs partner source (allowed port adjustments only):
- `input()` prompt replaced with env-var contract (`PCS_CAD_PATH`); the
  partner's pcs_runtime.resolve_cad_path already prefers env var, so we
  make env-var mandatory and drop the interactive prompt.
- Flow/temp script filenames updated to our ported copies:
  `partner_v4_flow.py` and `partner_v4_temp.py` (sitting in this same
  `src/` directory).
- Hydra `job.chdir=False` added so relative paths in the config resolve
  against the project root instead of Hydra's timestamped output dir.
- Additional env-var forward (`PCS_GEOM_JSON_PATH`) supports the
  standing rule to reuse `pipe_three_class_fixed.json` verbatim rather
  than regenerating it from CAD.

Everything else (subprocess orchestration, 5 s sleep, Hydra overrides
pass-through, exit code handling) preserved verbatim.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _run_step(name: str, cmd: List[str], cwd: Path, env: dict) -> None:
    print(f"[RUN] {name}: {' '.join(cmd)} (cwd={cwd})", flush=True)
    completed = subprocess.run(cmd, cwd=str(cwd), check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    print(f"[OK] {name} completed", flush=True)


def main() -> None:
    src_dir = Path(__file__).resolve().parent
    repo_root = src_dir.parent

    ap = argparse.ArgumentParser(
        description="Run full V4 E2E: flow training -> delay -> temperature training/inference + visualization"
    )
    ap.add_argument(
        "--flow-script",
        default=str(src_dir / "partner_v4_flow.py"),
        help="Path to flow training script.",
    )
    ap.add_argument(
        "--temp-script",
        default=str(src_dir / "partner_v4_temp.py"),
        help="Path to temperature training/inference script with visualization hook.",
    )
    ap.add_argument(
        "--delay-seconds",
        type=float,
        default=5.0,
        help="Delay after flow step to allow artifacts to flush/save.",
    )
    ap.add_argument(
        "--flow-overrides",
        nargs="*",
        default=[],
        help="Optional Hydra overrides for flow script, e.g. run_mode=train.",
    )
    ap.add_argument(
        "--temp-overrides",
        nargs="*",
        default=[],
        help="Optional Hydra overrides for temperature script, e.g. run_mode=train.",
    )
    ap.add_argument(
        "--skip-flow",
        action="store_true",
        help="Skip the flow training subprocess (useful when PCS_GEOM_JSON_PATH / precomputed flow JSON is already in place).",
    )

    args = ap.parse_args()

    flow_script = Path(args.flow_script).resolve()
    temp_script = Path(args.temp_script).resolve()

    if not flow_script.exists():
        raise FileNotFoundError(f"Flow script not found: {flow_script}")
    if not temp_script.exists():
        raise FileNotFoundError(f"Temperature script not found: {temp_script}")

    cad_path = os.environ.get("PCS_CAD_PATH", "").strip()
    if not cad_path:
        raise ValueError(
            "PCS_CAD_PATH env var is required (set it to a .step/.stp/.stl file path). "
            "Example: PCS_CAD_PATH=./data/partner_v4/designs/Study_Model_B_1st_4p3T.step"
        )

    run_env = os.environ.copy()
    run_env["PCS_CAD_PATH"] = cad_path
    if os.environ.get("PCS_GEOM_JSON_PATH"):
        run_env["PCS_GEOM_JSON_PATH"] = os.environ["PCS_GEOM_JSON_PATH"]

    hydra_common = ["hydra.job.chdir=False"]

    if not args.skip_flow:
        flow_cmd = [
            sys.executable,
            str(flow_script),
            *hydra_common,
            *args.flow_overrides,
        ]
        _run_step("flow_train_and_infer", flow_cmd, cwd=repo_root, env=run_env)

        delay = max(float(args.delay_seconds), 0.0)
        if delay > 0.0:
            print(f"[INFO] waiting {delay:.1f}s for flow artifacts to finish saving...", flush=True)
            time.sleep(delay)
    else:
        print("[INFO] --skip-flow set, skipping flow training subprocess", flush=True)

    temp_cmd = [
        sys.executable,
        str(temp_script),
        *hydra_common,
        *args.temp_overrides,
    ]
    _run_step("temp_train_infer_and_visualize", temp_cmd, cwd=repo_root, env=run_env)

    print("[OK] Full V4 E2E pipeline complete", flush=True)


if __name__ == "__main__":
    main()
