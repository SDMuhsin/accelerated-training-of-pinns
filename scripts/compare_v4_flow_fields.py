"""Compare V4 baseline vs SAGE inferred flow fields.

After both trainers complete, they each write a
``pipe_three_class_fixed_pred_flow_steady.json`` file containing per-point
``(u, v, p)`` predictions. This script loads the baseline backup
(kept at ``results/partner_v4/baseline_pred_flow_steady.json``) and the
SAGE output (default path at
``data/partner_v4/pipe_three_class_fixed_pred_flow_steady.json``
after running ``partner_v4_flow_sage.py``) and reports:

- RMSE and max |diff| of (u, v, p) across interior points
- Relative RMSE = RMSE / RMS(baseline)

These numbers say how close SAGE converges to the baseline's solution at
the field level, independent of the loss-scale difference between the
two training engines.

Usage::

    source env/bin/activate
    python scripts/compare_v4_flow_fields.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BASELINE_JSON = ROOT / "results" / "partner_v4" / "baseline_pred_flow_steady.json"
SAGE_JSON = ROOT / "data" / "partner_v4" / "pipe_three_class_fixed_pred_flow_steady.json"


def _load(path: Path) -> Dict:
    return json.loads(path.read_text())


def _arr(obj: Dict, field: str) -> np.ndarray:
    v = obj["fields"][field]
    arr = np.asarray(v, dtype=np.float32)
    return arr.reshape(-1) if arr.ndim > 1 else arr


def _compare(name: str, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
    am = a[mask]
    bm = b[mask]
    diff = am - bm
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    rms_ref = float(np.sqrt(np.mean(am ** 2))) + 1e-30
    rel = rmse / rms_ref
    return rmse, max_abs, rel


def main() -> int:
    if not BASELINE_JSON.exists():
        raise FileNotFoundError(
            f"Baseline backup not found: {BASELINE_JSON}. "
            "Run the V4 baseline first, then copy the output to this path."
        )
    if not SAGE_JSON.exists():
        raise FileNotFoundError(
            f"SAGE flow JSON not found: {SAGE_JSON}. "
            "Run partner_v4_flow_sage.py first."
        )

    base = _load(BASELINE_JSON)
    sage = _load(SAGE_JSON)

    # Geometry layout should match — baseline and SAGE both use the same
    # geometry JSON and write the wall + inside points in the same order.
    pt_base = np.asarray(base["point_type"], dtype=np.int32)
    pt_sage = np.asarray(sage["point_type"], dtype=np.int32)
    if pt_base.shape != pt_sage.shape:
        raise RuntimeError(
            f"Point counts differ: baseline {pt_base.shape} vs SAGE {pt_sage.shape}"
        )
    if not np.array_equal(pt_base, pt_sage):
        # If the ordering differs the fields aren't comparable pointwise.
        print("[WARN] point_type arrays differ — field comparison may be misleading.")

    interior = (pt_base == 2)
    print(f"Interior points: {int(interior.sum())} / {pt_base.size}")

    lines = []
    for field in ("u", "v", "p"):
        a = _arr(base, field)
        b = _arr(sage, field)
        rmse, mx, rel = _compare(field, a, b, interior)
        lines.append(f"  {field}: RMSE = {rmse:.3e}   Max|diff| = {mx:.3e}   rel RMSE = {rel:.3%}")

    print("Field differences (interior only):")
    for ln in lines:
        print(ln)

    # Also report magnitudes of each field for context.
    print()
    print("Field RMS magnitudes (baseline):")
    for field in ("u", "v", "p"):
        a = _arr(base, field)[interior]
        print(f"  {field}: RMS = {float(np.sqrt(np.mean(a ** 2))):.3e}   "
              f"max|val| = {float(np.max(np.abs(a))):.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
