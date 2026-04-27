"""Plot V4 baseline vs SAGE convergence curves.

Parses per-step total-loss values from both baseline and SAGE logs and
emits a single PNG with two curves (one per engine). Writes to
``results/partner_v4_compare/convergence.png``.

Usage::

    source env/bin/activate
    python scripts/plot_v4_compare.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "partner_v4_compare"


def _parse_sage_flow(path: Path) -> List[Tuple[int, float]]:
    """SAGE flow log: ``[stage_NAME] step k/N  total=V``. Concatenate across
    stages using cumulative step count."""
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    pat = re.compile(r"\[(\S+)\]\s+step\s+(\d+)/\d+\s+total=([0-9eE+\-.]+)")
    per_stage_offset = 0
    stage_seen = None
    out: List[Tuple[int, float]] = []
    for ln in lines:
        m = pat.search(ln)
        if not m:
            continue
        stage = m.group(1)
        step = int(m.group(2))
        total = float(m.group(3))
        if stage_seen is None:
            stage_seen = stage
        if stage != stage_seen:
            # New stage — advance offset by previous stage's last seen step.
            if out:
                per_stage_offset = out[-1][0]
            stage_seen = stage
        out.append((per_stage_offset + step, total))
    return out


def _parse_sage_temp(path: Path) -> List[Tuple[int, float]]:
    if not path.exists():
        return []
    lines = path.read_text(errors="ignore").splitlines()
    pat = re.compile(r"\[step\s+(\d+)\]\s+loss=([0-9eE+\-.]+)")
    return [(int(m.group(1)), float(m.group(2)))
            for ln in lines for m in [pat.search(ln)] if m]


def _try_plot(name: str, series: dict, outpath: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable ({exc}); skipping plot")
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, data in series.items():
        if not data:
            continue
        xs, ys = zip(*data)
        ax.plot(xs, ys, label=label, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("cumulative step")
    ax.set_ylabel("total loss (log)")
    ax.set_title(name)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    flow_sage = _parse_sage_flow(ROOT / "results" / "partner_v4_sage" / "flow" / "sage_flow_run.log")
    temp_sage = _parse_sage_temp(ROOT / "results" / "partner_v4_sage" / "temp" / "sage_run.log")

    print(f"Flow SAGE: parsed {len(flow_sage)} step-loss points")
    print(f"Temp SAGE: parsed {len(temp_sage)} step-loss points")

    # Write CSVs for downstream plotting even if matplotlib isn't installed.
    if flow_sage:
        np.savetxt(OUTDIR / "flow_sage_loss.csv",
                   np.asarray(flow_sage, dtype=np.float64),
                   delimiter=",", header="step,total", comments="")
    if temp_sage:
        np.savetxt(OUTDIR / "temp_sage_loss.csv",
                   np.asarray(temp_sage, dtype=np.float64),
                   delimiter=",", header="step,total", comments="")

    ok1 = _try_plot("V4 flow SAGE convergence", {"SAGE": flow_sage}, OUTDIR / "flow_convergence.png")
    ok2 = _try_plot("V4 temp SAGE convergence", {"SAGE": temp_sage}, OUTDIR / "temp_convergence.png")
    if ok1:
        print(f"Wrote: {OUTDIR / 'flow_convergence.png'}")
    if ok2:
        print(f"Wrote: {OUTDIR / 'temp_convergence.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
