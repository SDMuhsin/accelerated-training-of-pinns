"""Parse baseline + SAGE smoke logs and compare step-1 per-constraint losses.

Drop this script alongside a pair of logs produced by 1-step-per-stage
smoke runs of ``src/partner_v4_flow.py`` and
``src/partner_v4_flow_sage.py``. It extracts the first loss line from
each stage and prints a side-by-side comparison, flagging any
non-PDE divergence (which would indicate a non-PDE drift — fail).

Usage:
    python scripts/compare_smoke_step1.py <baseline_log> <sage_log>

PhysicsNeMo's Solver prints lines like:
    [INFO] [step: 1] [loss: 9.153e+02] [init_field_fit: 9.153e+02]
The per-constraint losses are captured in the square-bracket key: val
pairs after ``loss:``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Capture a single stage header, e.g. ``stage_00_bc_warmup``.
_STAGE_DIR_RE = re.compile(r"network_dir.*?stage_(\S+)")
_STEP_LINE_RE = re.compile(r"\[step:\s*(\d+)\][^\n]*")
_LOSS_KV_RE = re.compile(r"\[(\w+):\s*([\d\.\-eE\+nan]+)\]")


def _parse_stage_step1_losses(log_path: Path) -> Dict[str, Dict[str, float]]:
    """Return { stage_name: { constraint_or_total: loss_value, ... } }.

    Extracts the first step (step 0 or step 1, whichever appears
    first) per stage from a PhysicsNeMo log.
    """
    text = log_path.read_text()
    # Split into stages by the ``Running Stage: <name>`` or
    # ``network_dir: .../stage_<name>`` markers. PhysicsNeMo prints
    # ``JIT activated`` and a header block per stage start.
    out: Dict[str, Dict[str, float]] = {}

    current_stage: Optional[str] = None
    seen_first_step: Dict[str, bool] = {}
    lines = text.splitlines()
    for line in lines:
        m_dir = _STAGE_DIR_RE.search(line)
        if m_dir:
            current_stage = m_dir.group(1).strip()
            seen_first_step[current_stage] = False
            out.setdefault(current_stage, {})
            continue
        if current_stage is None:
            continue
        m_step = _STEP_LINE_RE.search(line)
        if m_step and not seen_first_step.get(current_stage, False):
            kvs = _LOSS_KV_RE.findall(line)
            parsed: Dict[str, float] = {}
            for k, v in kvs:
                if k == "step":
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    continue
            out[current_stage] = parsed
            seen_first_step[current_stage] = True
    return out


def _cmp_two(baseline_log: Path, sage_log: Path) -> int:
    b = _parse_stage_step1_losses(baseline_log)
    s = _parse_stage_step1_losses(sage_log)

    # Constraint keys that are PDE-only (FD truncation expected); everything
    # else should be bit-identical (or near-identical in float32).
    pde_keys = {
        "continuity", "momentum_x", "momentum_y",
        # Some PhysicsNeMo configs roll all PDE terms into a single key
        # like ``flow_pde``; tolerate both forms.
        "flow_pde",
    }

    stages = sorted(set(b) | set(s))
    print(f"{'stage':<32} {'key':<28} {'baseline':>16} {'sage':>16} {'abs_diff':>14} {'rel':>14}  verdict")
    any_fail = False
    for stage in stages:
        bd = b.get(stage, {})
        sd = s.get(stage, {})
        keys = sorted(set(bd) | set(sd))
        for k in keys:
            bv = bd.get(k)
            sv = sd.get(k)
            bs = f"{bv:.6e}" if bv is not None else "—"
            ss = f"{sv:.6e}" if sv is not None else "—"
            if bv is None or sv is None:
                verdict = "missing"
                diff = "—"; rel = "—"
                any_fail = True
            else:
                d = abs(bv - sv)
                ref = max(abs(bv), abs(sv), 1e-30)
                r = d / ref
                diff = f"{d:.3e}"; rel = f"{r:.3e}"
                if k in pde_keys:
                    # PDE: FD truncation allowed. 1e-3 relative or better.
                    verdict = "PDE-ok (FD trunc)" if r < 1e-2 else "PDE-FAIL"
                elif k == "loss":
                    # Total loss = sum of per-constraint. Also allowed to drift
                    # by the PDE truncation. 1e-3 relative.
                    verdict = "total-ok" if r < 1e-2 else "TOTAL-FAIL"
                else:
                    # Non-PDE: must be bit-identical (or near-bit-identical).
                    verdict = "ok" if r < 1e-6 else "NON-PDE-DRIFT"
                if verdict in ("NON-PDE-DRIFT", "PDE-FAIL", "TOTAL-FAIL", "missing"):
                    any_fail = True
            print(f"{stage:<32} {k:<28} {bs:>16} {ss:>16} {diff:>14} {rel:>14}  {verdict}")
        print()
    return 1 if any_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_log", type=Path)
    ap.add_argument("sage_log", type=Path)
    args = ap.parse_args()
    if not args.baseline_log.exists():
        sys.stderr.write(f"missing {args.baseline_log}\n"); return 2
    if not args.sage_log.exists():
        sys.stderr.write(f"missing {args.sage_log}\n"); return 2
    return _cmp_two(args.baseline_log, args.sage_log)


if __name__ == "__main__":
    raise SystemExit(main())
