#!/usr/bin/env python
"""Cycle 16 Phase 4: L^2 u-error medians for the cross-optimizer §V paragraph.

Reads per-seed tracking CSVs from results/ and computes the median across the
5 canonical seeds {0,1,7,23,42} of u_rms_error at the best-PDE-RMS checkpoint
for the four (problem, arch) cells §V cites: Kov-MLP, Kov-PN, elas-MLP, elas-PN
under both DT-PINN and SAGE methods.

Mirrors the aggregation logic in scripts/aggregate_l2_and_protocol.py but
reports medians (cycle-16 reporting paradigm) rather than means.

Usage:
    python scripts/compute_l2_medians.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

RESULTS_DIR = Path("/workspace/dt-pinn/results")
TAG = "multiseed_20260427"
SEEDS = [0, 1, 7, 23, 42]

# (problem, arch_csv_spelling, paper_label)
CELLS = [
    ("kovasznay",  "mlp",        "Kov-MLP"),
    ("kovasznay",  "pirate-net", "Kov-PN"),
    ("elasticity", "mlp",        "elas-MLP"),
    ("elasticity", "pirate-net", "elas-PN"),
]

METHODS = [
    ("sage",   "SAGE"),
    ("dtpinn", "DT-PINN"),
]


def safe_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def load_tracking(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                if k == "epoch":
                    try:
                        row[k] = int(v)
                    except (ValueError, TypeError):
                        row[k] = None
                elif k in {
                    "train_loss",
                    "pde_rms",
                    "continuity_rms",
                    "momentum_rms",
                    "u_rms_error",
                    "v_rms_error",
                    "p_rms_error",
                }:
                    row[k] = safe_float(v)
                else:
                    row[k] = v
            rows.append(row)
    return rows


def best_pde_rms_row(rows: list[dict]) -> dict | None:
    best = None
    best_v = math.inf
    for r in rows:
        v = r.get("pde_rms")
        if v is None:
            continue
        if v < best_v:
            best_v = v
            best = r
    return best


def median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def file_for(problem: str, method: str, arch: str, seed: int) -> Path:
    # method file naming: dtpinn -> dtpinn, sage -> sage
    return RESULTS_DIR / f"tracking_{problem}_{method}_{arch}_s{seed}_{TAG}.csv"


def main():
    print("=" * 90)
    print("L^2 u-error MEDIAN across 5 canonical seeds at best-PDE-RMS checkpoint")
    print("Cells: Kov-MLP, Kov-PN, elas-MLP, elas-PN; methods: SAGE, DT-PINN")
    print("=" * 90)

    results = {}  # (cell_label, method_label) -> {"u_median": x, "u_mean": x, "per_seed": [...]}

    for problem, arch, cell_label in CELLS:
        print(f"\n## {cell_label}  ({problem} / {arch})")
        for method, method_label in METHODS:
            per_seed = []
            for seed in SEEDS:
                path = file_for(problem, method, arch, seed)
                rows = load_tracking(path)
                if not rows:
                    per_seed.append((seed, None, None))
                    continue
                best = best_pde_rms_row(rows)
                if best is None:
                    per_seed.append((seed, None, None))
                    continue
                per_seed.append((seed, best.get("epoch"), best.get("u_rms_error")))

            u_vals = [u for _, _, u in per_seed if u is not None]
            u_med = median(u_vals)
            u_mean = mean(u_vals)
            results[(cell_label, method_label)] = {
                "u_median": u_med,
                "u_mean": u_mean,
                "per_seed": per_seed,
                "n": len(u_vals),
            }
            print(f"  {method_label}: n={len(u_vals)}/{len(SEEDS)}")
            for seed, epoch, u in per_seed:
                u_str = f"{u:.6f}" if u is not None else "—"
                print(f"    seed={seed:>2}  best_epoch={epoch}  u_rms_error={u_str}")
            print(f"    -> median u = {u_med:.6f}   mean u = {u_mean:.6f}")

    # Verify reviewer's claim on elas-MLP DT-PINN
    print("\n" + "=" * 90)
    print("REVIEWER VERIFICATION:")
    print("  Reviewer claims elas-MLP DT-PINN paper-printed = 0.027 (mean), median = 0.0051")
    print("=" * 90)
    e = results[("elas-MLP", "DT-PINN")]
    print(f"  Computed mean   = {e['u_mean']:.6f}  (paper printed 0.027)")
    print(f"  Computed median = {e['u_median']:.6f}  (reviewer says 0.0051)")
    if abs(e['u_mean'] - 0.027) < 0.005:
        print("  -> mean matches paper printed value (0.027) [REVIEWER CONFIRMED on F-M1]")
    if abs(e['u_median'] - 0.0051) < 0.0005:
        print("  -> median matches reviewer value (0.0051)")

    # Headline summary table
    print("\n" + "=" * 90)
    print("HEADLINE TABLE: §V Cross-optimizer paragraph paired L^2 u-medians")
    print("=" * 90)
    print(f"  {'Cell':<10}  {'SAGE median':<15}  {'DT-PINN median':<18}  {'(SAGE mean)':<15}  {'(DT-PINN mean)':<18}")
    for _, _, cell_label in CELLS:
        sage = results[(cell_label, "SAGE")]
        dtp  = results[(cell_label, "DT-PINN")]
        print(
            f"  {cell_label:<10}  {sage['u_median']:<15.6f}  {dtp['u_median']:<18.6f}  "
            f"{sage['u_mean']:<15.6f}  {dtp['u_mean']:<18.6f}"
        )

    print("\n" + "=" * 90)
    print("FOR PAPER: SAGE/DT-PINN median pairs (rounded to match paper convention)")
    print("=" * 90)
    for _, _, cell_label in CELLS:
        sage = results[(cell_label, "SAGE")]
        dtp  = results[(cell_label, "DT-PINN")]
        # Paper format: 0.0031/0.0026 type rounding -- show 2 sig figs after leading zeros
        def paper_fmt(v):
            if v is None or math.isnan(v):
                return "n/a"
            # Use 2 significant digits (matching paper precision: 0.0031, 0.027 etc.)
            if v >= 0.01:
                return f"{v:.3f}"
            else:
                return f"{v:.4f}"
        print(f"  {cell_label}: SAGE/DT-PINN = {paper_fmt(sage['u_median'])}/{paper_fmt(dtp['u_median'])}")


if __name__ == "__main__":
    main()
