"""Side-by-side per-stage wall-clock comparison: baseline vs rebuilt SAGE.

Parses both logs via ``scripts/parse_sage_flow_timings.parse()`` and
emits a single markdown-friendly table suitable for pasting into
``V4_SAGE_INTEGRATION.md`` § 4.

Usage:
    python scripts/compare_sage_v2_vs_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from parse_sage_flow_timings import parse, _fmt_min, _fmt_ms  # type: ignore


BASELINE_LOG = ROOT / "results" / "partner_v4" / "full_run.log"
SAGE_LOG = ROOT / "results" / "partner_v4_sage_v2" / "flow" / "sage_run.log"

# Pretty stage names for the table.
_PRETTY = {
    "m1_init_guess_warmup": "−1 `init-warmup`",
    "00_bc_warmup": "0 `bc-warmup`",
    "01_nu_1.00e-02": "1 `nu=1e-2`",
    "02_nu_5.00e-03": "2 `nu=5e-3`",
    "03_nu_1.00e-03": "3 `nu=1e-3`",
}


def _duration_min(stage) -> float:
    if stage.start_ts is None or stage.end_ts is None:
        return float("nan")
    return (stage.end_ts - stage.start_ts).total_seconds() / 60.0


def main() -> int:
    if not BASELINE_LOG.exists():
        sys.stderr.write(f"missing {BASELINE_LOG}\n"); return 2
    if not SAGE_LOG.exists():
        sys.stderr.write(f"missing {SAGE_LOG}\n"); return 2

    b_stages = {s.name: s for s in parse(BASELINE_LOG)}
    s_stages = {s.name: s for s in parse(SAGE_LOG)}

    names = [
        "m1_init_guess_warmup",
        "00_bc_warmup",
        "01_nu_1.00e-02",
        "02_nu_5.00e-03",
        "03_nu_1.00e-03",
    ]

    rows = []
    b_total = 0.0; s_total = 0.0
    for name in names:
        b = b_stages.get(name)
        s = s_stages.get(name)
        bm = _duration_min(b) if b else float("nan")
        sm = _duration_min(s) if s else float("nan")
        bms_samples = b.ms_per_step_samples if b and b.ms_per_step_samples else []
        sms_samples = s.ms_per_step_samples if s and s.ms_per_step_samples else []
        # Median ms/step, drop warmup first sample.
        def _med(samples):
            if len(samples) < 2:
                return samples[0] if samples else float("nan")
            tail = sorted(samples[1:])
            return tail[len(tail) // 2]
        bms = _med(bms_samples)
        sms = _med(sms_samples)
        speedup = bm / sm if (bm == bm and sm == sm and sm > 0) else float("nan")
        steps = (b.step_count if b else (s.step_count if s else 0))
        pretty = _PRETTY.get(name, name)
        rows.append({
            "name": pretty,
            "steps": steps,
            "bm": bm, "sm": sm,
            "bms": bms, "sms": sms,
            "sp": speedup,
        })
        if bm == bm:
            b_total += bm
        if sm == sm:
            s_total += sm

    total_speedup = (b_total / s_total) if s_total > 0 else float("nan")

    print("| Stage | Steps | Baseline min | SAGE min | Speedup | Base ms/step | SAGE ms/step |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        bm = "—" if r["bm"] != r["bm"] else f"{r['bm']:.2f}"
        sm = "—" if r["sm"] != r["sm"] else f"{r['sm']:.2f}"
        bms = "—" if r["bms"] != r["bms"] else f"{r['bms']:.0f}"
        sms = "—" if r["sms"] != r["sms"] else f"{r['sms']:.0f}"
        sp = "—" if r["sp"] != r["sp"] else f"{r['sp']:.2f}×"
        print(f"| {r['name']} | {r['steps']} | {bm} | {sm} | {sp} | {bms} | {sms} |")
    print(f"| **Flow total** | 30 000 | **{b_total:.2f}** | **{s_total:.2f}** | **{total_speedup:.2f}×** | | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
