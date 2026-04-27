"""SAGE variant of ``partner_v4_e2e.py``.

Mirrors the baseline orchestrator exactly, except that the default flow
and temp scripts point at the SAGE variants (``partner_v4_flow_sage.py``
and ``partner_v4_temp_sage.py``). Every other knob — PCS_CAD_PATH /
PCS_GEOM_JSON_PATH env-var contract, 5 s delay after flow, Hydra
override pass-through, ``--skip-flow`` for temp-only iteration — is
preserved. The comparison harness can invoke this interchangeably with
the baseline orchestrator."""

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
        description="Run full V4 E2E with SAGE gradient engine: flow (SAGE) -> delay -> temperature (SAGE) + inference + visualization"
    )
    ap.add_argument(
        "--flow-script",
        default=str(src_dir / "partner_v4_flow_sage.py"),
        help="Path to flow training script (defaults to SAGE variant).",
    )
    ap.add_argument(
        "--temp-script",
        default=str(src_dir / "partner_v4_temp_sage.py"),
        help="Path to temperature training/inference script (defaults to SAGE variant).",
    )
    ap.add_argument("--delay-seconds", type=float, default=5.0)
    ap.add_argument("--flow-overrides", nargs="*", default=[])
    ap.add_argument("--temp-overrides", nargs="*", default=[])
    ap.add_argument("--skip-flow", action="store_true")

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
            "PCS_CAD_PATH env var is required (set it to a .step/.stp/.stl file path)."
        )

    run_env = os.environ.copy()
    run_env["PCS_CAD_PATH"] = cad_path
    if os.environ.get("PCS_GEOM_JSON_PATH"):
        run_env["PCS_GEOM_JSON_PATH"] = os.environ["PCS_GEOM_JSON_PATH"]

    hydra_common = ["hydra.job.chdir=False"]

    if not args.skip_flow:
        flow_cmd = [sys.executable, str(flow_script), *hydra_common, *args.flow_overrides]
        _run_step("flow_train_and_infer (SAGE)", flow_cmd, cwd=repo_root, env=run_env)

        delay = max(float(args.delay_seconds), 0.0)
        if delay > 0.0:
            print(f"[INFO] waiting {delay:.1f}s for flow artifacts to finish saving...", flush=True)
            time.sleep(delay)
    else:
        print("[INFO] --skip-flow set, skipping flow training subprocess", flush=True)

    temp_cmd = [sys.executable, str(temp_script), *hydra_common, *args.temp_overrides]
    _run_step("temp_train_infer_and_visualize (SAGE)", temp_cmd, cwd=repo_root, env=run_env)

    print("[OK] Full V4 E2E (SAGE) pipeline complete", flush=True)


if __name__ == "__main__":
    main()
