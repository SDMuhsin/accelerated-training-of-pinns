"""Cycle-8 fixer aggregations.

Extends the cycle-7 aggregator to compute:
  - time-to-best per method × cell (F1)
  - ms/epoch CV per method × cell (F2)
  - raw n=5 mean ± std for the three currently-trimmed cells (F3)
  - CAN-PINN time-to-best vs SAGE for headline (F1)
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("/workspace/dt-pinn/results")

# Production tags for main table cells.
CELL_TAGS = {
    ("sage", "multiseed_20260427"),
    ("autodiff", "multiseed_20260427"),
    ("dtpinn", "multiseed_20260427"),
    ("ropinn", "multiseed_20260427"),
    ("sk-pinn", "multiseed_20260427"),
    ("pielm", "multiseed_20260427"),
    ("chebyshev-pinn", "canpinn_hpc_20260428"),
}

# Display name mapping
DISPLAY = {
    "sage": "SAGE",
    "autodiff": "AutoDiff",
    "dtpinn": "DT-PINN",
    "ropinn": "RoPINN",
    "sk-pinn": "SK-PINN",
    "chebyshev-pinn": "CAN-PINN",
    "pielm": "PIELM",
}


def safe_float(s):
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


def safe_int(s):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var), n


def main_csv_rows():
    """Yield rows from lid_benchmark_results.csv, including excluded ones."""
    with (RESULTS_DIR / "lid_benchmark_results.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield r


def cell_data(include_excl=False):
    """Group main-CSV rows by (problem, method, model). Default: drop _excl_ tags."""
    by_cell = defaultdict(list)
    for r in main_csv_rows():
        method = r.get("method")
        tag = r.get("tag", "")
        if (method, tag) in CELL_TAGS:
            by_cell[(r["problem"], method, r["model"])].append(r)
        elif include_excl and tag.startswith("multiseed_20260427_excl_") and method != "chebyshev-pinn":
            by_cell[(r["problem"], method, r["model"])].append(r)
        elif include_excl and tag.startswith("canpinn_hpc_20260428_excl_") and method == "chebyshev-pinn":
            by_cell[(r["problem"], method, r["model"])].append(r)
    return by_cell


# ----------------------------------------------------------------------
# F2 — ms/epoch CV per method × cell
# ----------------------------------------------------------------------
def f2_msepoch_cv():
    print("=" * 84)
    print("F2 — ms/epoch CV per method (across the 9 cells × 5 seeds)")
    print("CV = std/mean across seeds; aggregated by averaging within method")
    print("=" * 84)
    by = cell_data(include_excl=False)
    method_cvs = defaultdict(list)
    print()
    print(f"{'cell':<28}{'method':<12}{'mean':>10}{'std':>10}{'CV(%)':>10}{'n':>5}")
    for (problem, method, model), rows in sorted(by.items()):
        if method == "pielm":
            continue
        ms = [safe_float(r.get("ms_per_epoch")) for r in rows]
        m, s, n = mean_std(ms)
        if m and m > 0:
            cv = s / m * 100
            method_cvs[method].append(cv)
            print(f"{problem + '/' + model:<28}{DISPLAY[method]:<12}{m:>10.2f}{s:>10.2f}{cv:>10.2f}{n:>5}")
    print()
    print("Aggregate by method (mean of CV across cells):")
    for method, cvs in method_cvs.items():
        m, s, n = mean_std(cvs)
        print(f"  {DISPLAY[method]:<12} mean CV = {m:.1f}%  (range {min(cvs):.1f}-{max(cvs):.1f}%)  across {n} cells")


# ----------------------------------------------------------------------
# F3 — raw n=5 mean±std for currently-trimmed cells
# ----------------------------------------------------------------------
def f3_raw_n5_for_trimmed_cells():
    print("\n" + "=" * 84)
    print("F3 — raw n=5 mean±std for the three trimmed cells")
    print("=" * 84)
    targets = [
        ("elasticity", "dtpinn", "mlp"),
        ("elasticity", "dtpinn", "pirate-net"),
        ("kovasznay", "chebyshev-pinn", "mlp"),
    ]
    by_with_excl = cell_data(include_excl=True)
    print()
    for key in targets:
        rows = by_with_excl.get(key, [])
        seeds = [safe_int(r.get("seed")) for r in rows]
        pde = [safe_float(r.get("pde_rms")) for r in rows]
        cont = [safe_float(r.get("continuity_rms")) for r in rows]
        mom = [safe_float(r.get("momentum_rms")) for r in rows]
        time = [safe_float(r.get("train_time_min")) for r in rows]
        msec = [safe_float(r.get("ms_per_epoch")) for r in rows]
        print(f"{key} — seeds: {seeds}")
        print(f"  PDE n=5: {mean_std(pde)[0]:.4f}+-{mean_std(pde)[1]:.4f}")
        print(f"  Cont n=5: {mean_std(cont)[0]:.4f}+-{mean_std(cont)[1]:.4f}")
        print(f"  Mom n=5: {mean_std(mom)[0]:.4f}+-{mean_std(mom)[1]:.4f}")
        print(f"  Time n=5: {mean_std(time)[0]:.3f}+-{mean_std(time)[1]:.3f} min")
        print(f"  ms/epoch n=5: {mean_std(msec)[0]:.2f}+-{mean_std(msec)[1]:.2f}")
        # also AutoDiff time for the same problem-model to compute speedup
        ad_key = (key[0], "autodiff", key[2])
        ad_rows = by_with_excl.get(ad_key, [])
        ad_time = [safe_float(r.get("train_time_min")) for r in ad_rows]
        ad_m, _, _ = mean_std(ad_time)
        own_m = mean_std(time)[0]
        if ad_m and own_m and own_m > 0:
            print(f"  AutoDiff time: {ad_m:.3f} → speedup: {ad_m/own_m:.2f}×")
        print()


# ----------------------------------------------------------------------
# F1 — Time-to-best for the headline CAN-PINN-vs-SAGE comparison
# ----------------------------------------------------------------------
def f1_time_to_best_headline():
    print("=" * 84)
    print("F1 — time-to-best for SAGE vs CAN-PINN (headline isolation)")
    print("ttb = best_epoch * ms_per_epoch / 60000  (min)")
    print("=" * 84)
    by = cell_data(include_excl=False)
    print()
    print(f"{'cell':<28}{'method':<12}{'best_ep_med':>13}{'ttb_med(min)':>14}{'budget(min)':>14}{'ttb%':>8}")
    cells = []
    for (problem, method, model), rows in sorted(by.items()):
        if method not in ("sage", "chebyshev-pinn"):
            continue
        ttbs = []
        budgets = []
        bestepochs = []
        for r in rows:
            be = safe_int(r.get("best_epoch"))
            ms = safe_float(r.get("ms_per_epoch"))
            bud = safe_float(r.get("train_time_min"))
            if be is not None and ms is not None:
                ttbs.append(be * ms / 60000.0)
            if bud is not None:
                budgets.append(bud)
            if be is not None:
                bestepochs.append(be)
        if not ttbs:
            continue
        med_ttb = sorted(ttbs)[len(ttbs)//2]
        med_bud = sorted(budgets)[len(budgets)//2] if budgets else float("nan")
        med_be = sorted(bestepochs)[len(bestepochs)//2] if bestepochs else float("nan")
        print(f"{problem + '/' + model:<28}{DISPLAY[method]:<12}{med_be:>13.0f}{med_ttb:>14.2f}{med_bud:>14.2f}{(med_ttb/med_bud*100 if med_bud else float('nan')):>7.0f}%")
        cells.append(((problem, model), method, med_ttb, med_bud))

    # Compute SAGE/CAN-PINN ratios
    print("\nSAGE vs CAN-PINN ratios:")
    sage = {(p,m): (ttb,bud) for (p,m), method, ttb, bud in cells if method == "sage"}
    can = {(p,m): (ttb,bud) for (p,m), method, ttb, bud in cells if method == "chebyshev-pinn"}
    bud_ratios = []
    ttb_ratios = []
    print(f"{'cell':<28}{'CAN/SAGE budget':>18}{'CAN/SAGE ttb':>18}")
    for k in sorted(sage.keys() & can.keys()):
        s_ttb, s_bud = sage[k]
        c_ttb, c_bud = can[k]
        bud_r = c_bud/s_bud if s_bud > 0 else float("nan")
        ttb_r = c_ttb/s_ttb if s_ttb > 0 else float("nan")
        bud_ratios.append(bud_r)
        ttb_ratios.append(ttb_r)
        print(f"{k[0] + '/' + k[1]:<28}{bud_r:>17.2f}×{ttb_r:>17.2f}×")
    print(f"\nBudget ratios: median={sorted(bud_ratios)[len(bud_ratios)//2]:.2f}× range {min(bud_ratios):.2f}-{max(bud_ratios):.2f}×")
    print(f"TTB ratios:    median={sorted(ttb_ratios)[len(ttb_ratios)//2]:.2f}× range {min(ttb_ratios):.2f}-{max(ttb_ratios):.2f}×")


if __name__ == "__main__":
    f2_msepoch_cv()
    f3_raw_n5_for_trimmed_cells()
    f1_time_to_best_headline()
