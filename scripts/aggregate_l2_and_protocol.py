"""Aggregate L2 solution-error and protocol diagnostics across the multi-seed sweep.

One-off helper for cycle-7 paper revision (TETCI v2). Reads the per-seed tracking
CSVs in /workspace/dt-pinn/results/ and produces:

  1. Per-cell L2 error (u/v/p, mean +- std across n=5 seeds) at the best-PDE-RMS
     checkpoint for kovasznay and elasticity. Cavity has no exact reference, so
     its L2 columns are unpopulated and we skip them.
  2. Per-cell median best_epoch (and per-seed list) across n=5 seeds, used to
     identify cells where the "best" checkpoint is at near-init (epoch <= 200).
  3. Per-cell time-to-best in minutes (best_epoch * ms_per_epoch / 60_000),
     where ms_per_epoch comes from results/lid_benchmark_results.csv. This
     contrasts the budget-time column in Table III.
  4. Sensitivity numbers for the dagger 5x exclusion threshold: shows how the
     DT-PINN elasticity-PirateNet row changes if seed 0 (5.004x median) is
     re-included (n=5 vs n=4).

The script intentionally avoids any new HPC runs. It is a pure aggregation pass
over data already present in results/.

Usage:
    python scripts/aggregate_l2_and_protocol.py

Outputs are printed to stdout in human-readable form so a fixer can copy
the relevant numbers into experimental_setup.tex / results.tex.
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("/workspace/dt-pinn/results")
TAG = "multiseed_20260427"  # the production sweep tag for the main table

PROBLEMS = ["kovasznay", "elasticity", "cavity"]
MODELS_FILE = ["mlp", "tsa-pinn", "pirate-net"]
SEEDS = [0, 1, 7, 23, 42]

# Method names as they appear in tracking-CSV filenames.
# CAN-PINN tracking files use the prefix "chebyshev-pinn".
METHOD_FILE_NAMES = {
    "sage": "sage",
    "autodiff": "autodiff",
    "dtpinn": "dtpinn",
    "ropinn": "ropinn",
    "sk-pinn": "sk-pinn",
    "canpinn": "chebyshev-pinn",
}

# Tag overrides per (problem, method) — CAN-PINN uses canpinn_hpc_20260428 for
# the production cells and canpinn_cycle4_20260427 for the cells re-run on
# cycle 4 only. Use the HPC tag for the main cells.
TAG_OVERRIDES = {
    ("canpinn",): "canpinn_hpc_20260428",
}


def file_for(problem: str, method: str, model: str, seed: int) -> Path:
    """Return the tracking CSV path for a given (problem, method, model, seed)."""
    method_file = METHOD_FILE_NAMES[method]
    tag = TAG_OVERRIDES.get((method,), TAG)
    return RESULTS_DIR / f"tracking_{problem}_{method_file}_{model}_s{seed}_{tag}.csv"


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
    """Load a tracking CSV into a list of dict rows; empty cells become None."""
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                if k in {"epoch"}:
                    row[k] = int(v)
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
    """Return the row with the smallest pde_rms (the best-checkpoint protocol).

    Mirrors the convention used by the production CSV aggregator: minimum over
    the 100-epoch tracking, ignoring None / NaN.
    """
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


def mean_std(xs: list[float]) -> tuple[float, float, int]:
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var), n


def median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def aggregate_l2_and_best_epoch():
    """Aggregate L2 errors at best-PDE-RMS checkpoint and best_epoch."""
    methods_to_aggregate = ["sage", "autodiff", "dtpinn", "canpinn"]

    print("=" * 84)
    print("L2 SOLUTION ERROR AT BEST-PDE-RMS CHECKPOINT")
    print("(mean +- std across n=5 seeds; cavity skipped — no exact solution)")
    print("=" * 84)

    for problem in ["kovasznay", "elasticity"]:
        print(f"\n## {problem.upper()}")
        print(
            f"{'method':<10}{'model':<14}{'u_l2':<22}{'v_l2':<22}"
            f"{'p_l2' if problem == 'kovasznay' else '':<22}{'med best_ep':<14}"
        )
        for model in MODELS_FILE:
            for method in methods_to_aggregate:
                u_errs, v_errs, p_errs, best_epochs = [], [], [], []
                for seed in SEEDS:
                    path = file_for(problem, method, model, seed)
                    rows = load_tracking(path)
                    if not rows:
                        continue
                    best = best_pde_rms_row(rows)
                    if best is None:
                        continue
                    if best.get("u_rms_error") is not None:
                        u_errs.append(best["u_rms_error"])
                    if best.get("v_rms_error") is not None:
                        v_errs.append(best["v_rms_error"])
                    if problem == "kovasznay" and best.get("p_rms_error") is not None:
                        p_errs.append(best["p_rms_error"])
                    best_epochs.append(best["epoch"])

                u_m, u_s, n = mean_std(u_errs)
                v_m, v_s, _ = mean_std(v_errs)
                p_m, p_s, _ = mean_std(p_errs) if problem == "kovasznay" else (
                    float("nan"),
                    float("nan"),
                    0,
                )
                med_be = median(best_epochs)
                u_str = f"{u_m:.4f}+-{u_s:.4f}(n={n})" if n else "—"
                v_str = f"{v_m:.4f}+-{v_s:.4f}" if n else "—"
                p_str = (
                    f"{p_m:.4f}+-{p_s:.4f}"
                    if problem == "kovasznay" and len(p_errs)
                    else ("" if problem == "kovasznay" else "")
                )
                med_str = f"{int(med_be)}" if not math.isnan(med_be) else "—"
                print(
                    f"{method:<10}{model:<14}{u_str:<22}{v_str:<22}{p_str:<22}{med_str:<14}"
                )

    # ------------------------------------------------------------------
    # Cell-level summary used for the paper L2 footnote/table.
    print("\n" + "=" * 84)
    print("PAPER-READY L2 SUMMARY (per cell, SAGE vs DT-PINN at best-PDE-RMS)")
    print("=" * 84)
    for problem in ["kovasznay", "elasticity"]:
        print(f"\n### {problem}")
        for model in MODELS_FILE:
            sage_u, sage_v, sage_p = [], [], []
            dtp_u, dtp_v, dtp_p = [], [], []
            for seed in SEEDS:
                for store_u, store_v, store_p, method in [
                    (sage_u, sage_v, sage_p, "sage"),
                    (dtp_u, dtp_v, dtp_p, "dtpinn"),
                ]:
                    path = file_for(problem, method, model, seed)
                    rows = load_tracking(path)
                    best = best_pde_rms_row(rows) if rows else None
                    if not best:
                        continue
                    if best.get("u_rms_error") is not None:
                        store_u.append(best["u_rms_error"])
                    if best.get("v_rms_error") is not None:
                        store_v.append(best["v_rms_error"])
                    if problem == "kovasznay" and best.get("p_rms_error") is not None:
                        store_p.append(best["p_rms_error"])

            sm_u = mean_std(sage_u)
            sm_v = mean_std(sage_v)
            dm_u = mean_std(dtp_u)
            dm_v = mean_std(dtp_v)
            line = (
                f"  {model:<10} SAGE u={sm_u[0]:.4f}+-{sm_u[1]:.4f} v={sm_v[0]:.4f}+-{sm_v[1]:.4f}"
                f"  | DT-PINN u={dm_u[0]:.4f}+-{dm_u[1]:.4f} v={dm_v[0]:.4f}+-{dm_v[1]:.4f}"
            )
            if problem == "kovasznay":
                sm_p = mean_std(sage_p)
                dm_p = mean_std(dtp_p)
                line += f"  p_sage={sm_p[0]:.4f}+-{sm_p[1]:.4f} p_dt={dm_p[0]:.4f}+-{dm_p[1]:.4f}"
            # Ratio (SAGE / DT-PINN).
            if dm_u[0] > 0 and not math.isnan(sm_u[0]):
                ratio_u = sm_u[0] / dm_u[0]
            else:
                ratio_u = float("nan")
            if dm_v[0] > 0 and not math.isnan(sm_v[0]):
                ratio_v = sm_v[0] / dm_v[0]
            else:
                ratio_v = float("nan")
            line += f"  | ratio_u={ratio_u:.2f} ratio_v={ratio_v:.2f}"
            print(line)


def aggregate_best_epoch_per_cell():
    """List cells where the median best_epoch <= 200 across all gradient methods.

    Useful for F5 — flagging cells that anchor at near-init checkpoints.
    """
    print("\n" + "=" * 84)
    print("BEST_EPOCH PER CELL (median across n=5; cells where median <= 200")
    print("indicate near-init anchoring)")
    print("=" * 84)

    methods = ["sage", "autodiff", "dtpinn", "canpinn"]
    for problem in PROBLEMS:
        print(f"\n## {problem}")
        for model in MODELS_FILE:
            row = f"  {model:<11}"
            for method in methods:
                best_epochs = []
                for seed in SEEDS:
                    rows = load_tracking(file_for(problem, method, model, seed))
                    best = best_pde_rms_row(rows) if rows else None
                    if best:
                        best_epochs.append(best["epoch"])
                med = median(best_epochs)
                row += f"  {method}={int(med) if not math.isnan(med) else '—'}"
            print(row)


def aggregate_time_to_best():
    """Compute time-to-best (min) per cell.

    Reads ms_per_epoch from results/lid_benchmark_results.csv and pairs it with
    the best_epoch from the tracking CSVs. For DT-PINN this represents wall
    time of best L-BFGS step (best_epoch * ms_per_epoch / 60000).
    """
    main_csv = RESULTS_DIR / "lid_benchmark_results.csv"
    if not main_csv.exists():
        print("(skipping time-to-best: lid_benchmark_results.csv not present)")
        return

    # Load main CSV for ms_per_epoch.
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    with main_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            tag = r.get("tag", "")
            problem = r.get("problem")
            method = r.get("method")
            model = r.get("model")
            seed = r.get("seed")
            # Only the production-sweep tags (multiseed, canpinn_hpc).
            if not (tag == TAG or tag == "canpinn_hpc_20260428"):
                continue
            try:
                seed_i = int(seed)
            except (TypeError, ValueError):
                continue
            key = (problem, method, model)
            by_key[key].append({
                "seed": seed_i,
                "ms_per_epoch": safe_float(r.get("ms_per_epoch", "")),
                "train_time_min": safe_float(r.get("train_time_min", "")),
                "best_epoch": (
                    int(r["best_epoch"]) if r.get("best_epoch") and r["best_epoch"].isdigit() else None
                ),
            })

    print("\n" + "=" * 84)
    print("TIME-TO-BEST (min) vs BUDGET TIME (min)")
    print(
        "  ttb = best_epoch * ms_per_epoch / 60_000;  "
        "budget = train_time_min from main CSV"
    )
    print("=" * 84)

    method_display = {
        "sage": "sage",
        "autodiff": "autodiff",
        "dtpinn": "dtpinn",
        "canpinn": "canpinn",
    }

    # Map main-csv method name to file method (CAN-PINN appears as "canpinn" in main csv).
    for problem in PROBLEMS:
        print(f"\n## {problem}")
        for model in MODELS_FILE:
            for method in ["sage", "autodiff", "canpinn", "dtpinn"]:
                key = (problem, method, model)
                rows = by_key.get(key, [])
                if not rows:
                    continue
                ttbs = []
                budgets = []
                for r in rows:
                    if r["ms_per_epoch"] is None or r["best_epoch"] is None:
                        continue
                    ttb = r["best_epoch"] * r["ms_per_epoch"] / 60_000.0
                    ttbs.append(ttb)
                    if r["train_time_min"] is not None:
                        budgets.append(r["train_time_min"])
                ttb_m, ttb_s, n_ttb = mean_std(ttbs)
                bud_m, bud_s, n_bud = mean_std(budgets)
                if n_ttb == 0:
                    continue
                print(
                    f"  {model:<11} {method:<9}"
                    f" ttb={ttb_m:.2f}+-{ttb_s:.2f}min (n={n_ttb})"
                    f"  budget={bud_m:.2f}+-{bud_s:.2f}min"
                    f"  ttb/budget={(ttb_m / bud_m * 100 if bud_m else float('nan')):.0f}%"
                )


def dagger_sensitivity():
    """Recompute DT-PINN elasticity rows with seed 0 included (n=5)."""
    print("\n" + "=" * 84)
    print("DAGGER SENSITIVITY: DT-PINN n=5 (seed 0 RE-INCLUDED) for elasticity")
    print("=" * 84)
    main_csv = RESULTS_DIR / "lid_benchmark_results.csv"
    if not main_csv.exists():
        return
    rows = []
    with main_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    # Find DT-PINN elasticity rows under multiseed_20260427 (any seed, including
    # the one tagged *_excl_*).
    for model in MODELS_FILE:
        seeds_pde = []
        seeds_time = []
        seeds_taken = []
        for r in rows:
            if r.get("problem") != "elasticity":
                continue
            if r.get("method") != "dtpinn":
                continue
            if r.get("model") != model:
                continue
            tag = r.get("tag", "")
            if not (tag.startswith("multiseed_20260427")):
                continue
            try:
                seed_i = int(r["seed"])
            except (TypeError, ValueError, KeyError):
                continue
            pde = safe_float(r.get("pde_rms", ""))
            tm = safe_float(r.get("train_time_min", ""))
            if pde is None:
                continue
            seeds_pde.append(pde)
            if tm is not None:
                seeds_time.append(tm)
            seeds_taken.append(seed_i)
        m, s, n = mean_std(seeds_pde)
        tm_m, tm_s, _ = mean_std(seeds_time)
        print(
            f"  elasticity-{model:<11} DT-PINN n={n} seeds_used={sorted(seeds_taken)}"
            f" PDE_RMS={m:.4f}+-{s:.4f}  time={tm_m:.2f}+-{tm_s:.2f}min"
        )


def main():
    aggregate_l2_and_best_epoch()
    aggregate_best_epoch_per_cell()
    aggregate_time_to_best()
    dagger_sensitivity()


if __name__ == "__main__":
    main()
