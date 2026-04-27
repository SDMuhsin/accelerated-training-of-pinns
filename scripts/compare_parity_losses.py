"""Extract and compare per-stage, per-step losses from two PhysicsNeMo
training logs. Used to evaluate step-1 / 100-step parity between the
baseline V4 flow trainer and the SAGE drop-in.

Usage:
    python scripts/compare_parity_losses.py <baseline_log> <sage_log>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_STAGE_RE = re.compile(r"stage_(\S+)")
_STEP_LOSS_RE = re.compile(
    r"\[step:\s*(\d+)\]\s*loss:\s*([\d\.eE+\-]+)"
)


def _extract(log_path: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Return {stage_name: [(step, loss), ...]} for every logged step."""
    text = log_path.read_text()
    out: Dict[str, List[Tuple[int, float]]] = {}
    current_stage = None
    for line in text.splitlines():
        if "network_dir" in line or "restore from" in line or "saved checkpoint to" in line:
            m = _STAGE_RE.search(line)
            if m:
                current_stage = m.group(1).strip().rstrip(":")
                out.setdefault(current_stage, [])
        m_loss = _STEP_LOSS_RE.search(line)
        if m_loss and current_stage is not None:
            step = int(m_loss.group(1))
            loss = float(m_loss.group(2))
            # Dedup by (step,) per stage — a step sometimes logs twice
            # (once at save, once in loss line). Keep last.
            existing = [t for t in out[current_stage] if t[0] == step]
            for e in existing:
                out[current_stage].remove(e)
            out[current_stage].append((step, loss))
    for k in out:
        out[k] = sorted(out[k])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_log", type=Path)
    ap.add_argument("sage_log", type=Path)
    args = ap.parse_args()

    if not args.baseline_log.exists():
        sys.stderr.write(f"missing {args.baseline_log}\n"); return 2
    if not args.sage_log.exists():
        sys.stderr.write(f"missing {args.sage_log}\n"); return 2

    b = _extract(args.baseline_log)
    s = _extract(args.sage_log)
    stages = sorted(set(b) | set(s))

    worst = {"rel": 0.0, "where": ""}
    print(f"{'stage':<30} {'step':>5} {'baseline':>14} {'sage':>14} {'rel':>10}")
    for stage in stages:
        bs = b.get(stage, [])
        ss = s.get(stage, [])
        # Index by step
        bmap = dict(bs)
        smap = dict(ss)
        steps = sorted(set(bmap) | set(smap))
        for step in steps:
            bv = bmap.get(step); sv = smap.get(step)
            if bv is None or sv is None:
                print(f"{stage:<30} {step:>5} {bv!s:>14} {sv!s:>14} {'missing':>10}")
                continue
            rel = abs(bv - sv) / max(abs(bv), abs(sv), 1e-30)
            tag = ""
            if rel > worst["rel"]:
                worst = {"rel": rel, "where": f"{stage}@step{step}"}
            if rel > 5e-2:
                tag = "  ***"
            print(f"{stage:<30} {step:>5} {bv:>14.4e} {sv:>14.4e} {rel:>10.2e}{tag}")
        print()
    print(f"\nworst rel divergence: {worst['rel']:.2e} at {worst['where']}")
    print("verdict:", "PASS (<5%)" if worst["rel"] < 5e-2 else "FAIL (>=5%)")
    return 0 if worst["rel"] < 5e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
