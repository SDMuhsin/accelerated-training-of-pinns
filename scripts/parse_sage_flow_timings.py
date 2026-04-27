"""Parse a V4 SAGE (or baseline) flow log into a per-stage timing table.

Extracts:
- Stage name + step count
- Start timestamp (from first log line containing the stage's checkpoint
  dir after a "training stage" marker)
- End timestamp (from the "reached maximum training steps, finished
  training!" line)
- First and last logged loss per stage
- ms/step (step 100 and step ≈end-of-stage values when available)

Emits a markdown table suitable for pasting into
``V4_SAGE_INTEGRATION.md``.

Usage:
    python scripts/parse_sage_flow_timings.py <flow_log>
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}\]")
_STAGE_HEADER_RE = re.compile(
    r"training stage (\S+)\s+(?:with )?(?:nu=([\deE\.+-]+)\s+)?for\s+(\d+)\s+steps"
)
# Only switch ``current`` on lines that unambiguously mark a new stage.
# "saved checkpoint to .../stage_XX" is printed at step 0 of every stage
# and only references the ACTIVE stage.  "Success loading model: .../stage_YY"
# during a warm-start references the PREVIOUS stage — must NOT switch.
_SAVED_CKPT_STAGE_RE = re.compile(
    r"saved checkpoint to\s+\S*?stage_(\S+?)(/|$|\")"
)
_RESTORE_FROM_STAGE_RE = re.compile(
    r"attempting to restore from\s*:\s*\S*?stage_(\S+?)(/|$|\")"
)
_STEP_LOSS_RE = re.compile(
    r"\[step:\s*(\d+)\]\s*loss:\s*([\d\.eE+\-]+)(?:.*time/iteration:\s*([\d\.eE+\-]+)\s*ms)?"
)
_MAX_RE = re.compile(r"\[step:\s*(\d+)\]\s*reached maximum training steps")


@dataclass
class StageStats:
    name: str
    step_count: int
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    first_loss: Optional[float] = None
    last_loss: Optional[float] = None
    ms_per_step_samples: List[float] = field(default_factory=list)


def _parse_ts(line: str) -> Optional[datetime]:
    m = _TS_RE.search(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse(log_path: Path) -> List[StageStats]:
    stages: Dict[str, StageStats] = {}
    order: List[str] = []
    current: Optional[StageStats] = None

    pending_step_count: Optional[int] = None
    lines = log_path.read_text().splitlines()
    for line in lines:
        # ``training stage X/Y ... for N steps`` is printed JUST BEFORE
        # the ``attempting to restore from: .../stage_NN`` line for that
        # same stage. We stash N here and apply it to the next newly
        # created stage so we don't mis-attribute to the previous one.
        m_hdr = _STAGE_HEADER_RE.search(line)
        if m_hdr:
            pending_step_count = int(m_hdr.group(3))
            continue
        # ``attempting to restore from: .../stage_XX`` is the earliest
        # stage-scoped log line — unambiguously marks the BEGINNING of
        # a stage (before any checkpoint save). Use as primary signal.
        m_rstr = _RESTORE_FROM_STAGE_RE.search(line)
        if m_rstr:
            name = m_rstr.group(1).strip(":/\"")
            newly_created = name not in stages
            if newly_created:
                stages[name] = StageStats(name=name, step_count=0)
                order.append(name)
            current = stages[name]
            if newly_created and pending_step_count is not None:
                current.step_count = pending_step_count
                pending_step_count = None
            if current.start_ts is None:
                current.start_ts = _parse_ts(line)
            continue
        # ``saved checkpoint to .../stage_XX`` is an additional confirmation
        # but only applies to the CURRENT stage (not a previous one).
        m_saved = _SAVED_CKPT_STAGE_RE.search(line)
        if m_saved:
            name = m_saved.group(1).strip(":/\"")
            newly_created = name not in stages
            if newly_created:
                stages[name] = StageStats(name=name, step_count=0)
                order.append(name)
            current = stages[name]
            if newly_created and pending_step_count is not None:
                current.step_count = pending_step_count
                pending_step_count = None
            if current.start_ts is None:
                current.start_ts = _parse_ts(line)
            continue
        m_loss = _STEP_LOSS_RE.search(line)
        if m_loss and current is not None:
            step = int(m_loss.group(1))
            loss = float(m_loss.group(2))
            if current.first_loss is None:
                current.first_loss = loss
            current.last_loss = loss
            if m_loss.group(3):
                try:
                    current.ms_per_step_samples.append(float(m_loss.group(3)))
                except ValueError:
                    pass
        m_max = _MAX_RE.search(line)
        if m_max and current is not None:
            step = int(m_max.group(1))
            current.step_count = max(current.step_count, step)
            if current.end_ts is None:
                current.end_ts = _parse_ts(line)
    return [stages[name] for name in order]


def _fmt_min(s: StageStats) -> str:
    if s.start_ts is None or s.end_ts is None:
        return "—"
    delta = s.end_ts - s.start_ts
    return f"{delta.total_seconds() / 60.0:.2f}"


def _fmt_ms(s: StageStats) -> str:
    if not s.ms_per_step_samples:
        return "—"
    # Drop first sample (warmup), report median of the rest.
    samples = sorted(s.ms_per_step_samples[1:] if len(s.ms_per_step_samples) > 1 else s.ms_per_step_samples)
    mid = samples[len(samples) // 2]
    return f"{mid:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", type=Path)
    args = ap.parse_args()
    if not args.log_path.exists():
        sys.stderr.write(f"missing {args.log_path}\n"); return 2

    stages = parse(args.log_path)
    if not stages:
        print("no stages found"); return 1

    print(f"{'stage':<36} {'steps':>6} {'min':>8} {'ms/step':>9} {'start_loss':>12} {'end_loss':>12}")
    total_min = 0.0
    for s in stages:
        m = _fmt_min(s)
        if m != "—":
            try:
                total_min += float(m)
            except ValueError:
                pass
        print(f"{s.name:<36} {s.step_count:>6} {m:>8} {_fmt_ms(s):>9} {s.first_loss!s:>12} {s.last_loss!s:>12}")
    print(f"{'TOTAL':<36} {'':>6} {total_min:>8.2f}")

    # Markdown table
    print()
    print("| Stage | Steps | Duration (min) | ms/step | Start loss | End loss |")
    print("|---|---:|---:|---:|---:|---:|")
    for s in stages:
        m = _fmt_min(s)
        ms = _fmt_ms(s)
        sl = f"{s.first_loss:.3e}" if s.first_loss is not None else "—"
        el = f"{s.last_loss:.3e}" if s.last_loss is not None else "—"
        print(f"| {s.name} | {s.step_count} | {m} | {ms} | {sl} | {el} |")
    print(f"| **Total** | | {total_min:.2f} | | | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
