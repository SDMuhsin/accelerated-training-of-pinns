#!/usr/bin/env python
# Cycle 16 Phase 1: per-cell medians for the median-of-5 Table III rewrite.
# Reads results/lid_benchmark_results.csv (read-only) and emits a structured
# markdown doc under llmdocs/stream_sage_paper/paper_rewrite/.

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_SEEDS = [0, 1, 7, 23, 42]

# (csv_method, allowed_tag_set, paper_name)
METHOD_TAG_MAP = [
    ("autodiff",          {"multiseed_20260427"},                                                          "AutoDiff"),
    ("dtpinn",            {"multiseed_20260427"},                                                          "DT-PINN"),
    ("ropinn",            {"multiseed_20260427"},                                                          "RoPINN"),
    ("chebyshev-pinn",    {"canpinn_hpc_20260428", "canpinn_hpc_20260428_excl_s7_time_outlier"},          "Spectral-AD"),
    ("sk-pinn",           {"sk_pinn_matched_20260503"},                                                    "SK-PINN"),
    ("can-pinn-faithful", {"can_pinn_faithful_20260503"},                                                  "CAN-PINN"),
    ("sage",              {"multiseed_20260427"},                                                          "SAGE"),
]

# Methods × archs × problems = 7 × 3 × 3 = 63 cells. PIELM is one extra cell
# (cavity / pielm-arch / pielm-method) — handled separately below.
PROBLEMS = ["cavity", "kovasznay", "elasticity"]
ARCHS    = ["mlp", "tsa-pinn", "pirate-net"]   # CSV spelling

PROBLEM_LABEL = {"cavity": "cavity", "kovasznay": "kovasznay", "elasticity": "elasticity"}
ARCH_LABEL    = {"mlp": "MLP", "tsa-pinn": "TSA-PINN", "pirate-net": "PirateNet"}

METRICS = ["pde_rms", "continuity_rms", "momentum_rms", "ms_per_epoch", "train_time_min"]


def explore(df: pd.DataFrame) -> None:
    print("=== CSV exploration ===")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"tag.unique():     {sorted(df.tag.unique())}")
    print(f"method.unique():  {sorted(df.method.unique())}")
    print(f"problem.unique(): {sorted(df.problem.unique())}")
    print(f"model.unique():   {sorted(df.model.unique())}")
    print(f"seed.unique():    {sorted(df.seed.unique())}")
    print(f"status.unique():  {sorted(df.status.unique())}")
    print()


def select_cell(df: pd.DataFrame, *, method: str, problem: str, arch: str, allowed_tags: set) -> pd.DataFrame:
    sub = df[
        (df.method == method)
        & (df.problem == problem)
        & (df.model == arch)
        & (df.tag.isin(allowed_tags))
        & (df.status == "OK")
    ].copy()
    return sub


def median_metrics(rows: pd.DataFrame) -> dict:
    return {m: float(np.median(rows[m].values)) for m in METRICS}


def fmt(value, sig=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.{sig}g}"


def fmt_speedup(ratio):
    if ratio is None:
        return "n/a"
    return f"{ratio:.2f}x"


def compute_all(df: pd.DataFrame):
    """Returns (cells, autodiff_time_lookup, anomalies) where:
       cells: list of dicts with everything needed for the table.
       autodiff_time_lookup: {(problem, arch): median train_time_min}
       anomalies: list of strings.
    """
    anomalies = []
    cells = []

    # First pass: compute AutoDiff median train_time_min per (problem, arch),
    # since speedups are AutoDiff-relative within the same cell.
    autodiff_time_lookup = {}
    autodiff_tag = {"multiseed_20260427"}
    for problem in PROBLEMS:
        for arch in ARCHS:
            sub = select_cell(df, method="autodiff", problem=problem, arch=arch, allowed_tags=autodiff_tag)
            sub = filter_canonical_seeds(sub, anomalies, label=f"AutoDiff/{problem}/{arch}")
            if len(sub) == 5:
                autodiff_time_lookup[(problem, arch)] = float(np.median(sub.train_time_min.values))
            else:
                autodiff_time_lookup[(problem, arch)] = None

    for csv_method, allowed_tags, paper_name in METHOD_TAG_MAP:
        for problem in PROBLEMS:
            for arch in ARCHS:
                sub = select_cell(df, method=csv_method, problem=problem, arch=arch, allowed_tags=allowed_tags)
                sub = filter_canonical_seeds(
                    sub,
                    anomalies,
                    label=f"{paper_name}/{problem}/{arch}",
                )
                cell = {
                    "problem": problem,
                    "arch": arch,
                    "csv_method": csv_method,
                    "paper_name": paper_name,
                    "n_seeds": len(sub),
                    "missing_seeds": sorted(set(CANONICAL_SEEDS) - set(sub.seed.tolist())),
                    "metrics": None,
                    "speedup_vs_autodiff": None,
                }
                if len(sub) == 5:
                    cell["metrics"] = median_metrics(sub)
                    autodiff_time = autodiff_time_lookup.get((problem, arch))
                    if autodiff_time is not None and cell["metrics"]["train_time_min"] > 0:
                        cell["speedup_vs_autodiff"] = autodiff_time / cell["metrics"]["train_time_min"]
                cells.append(cell)

    # PIELM extra cell on cavity-MLP-only (the CSV row has model=='pielm', not 'mlp').
    pielm_sub = df[
        (df.method == "pielm")
        & (df.problem == "cavity")
        & (df.tag == "multiseed_20260427")
        & (df.status == "OK")
    ].copy()
    pielm_sub = filter_canonical_seeds(pielm_sub, anomalies, label="PIELM/cavity/pielm")
    pielm_cell = {
        "problem": "cavity",
        "arch": "mlp",            # paper places PIELM under cavity-MLP block
        "csv_method": "pielm",
        "paper_name": "PIELM",
        "n_seeds": len(pielm_sub),
        "missing_seeds": sorted(set(CANONICAL_SEEDS) - set(pielm_sub.seed.tolist())),
        "metrics": None,
        "speedup_vs_autodiff": None,
        "note": "PIELM CSV row has model='pielm' (not 'mlp'); paper assigns it under the cavity-MLP block.",
    }
    if len(pielm_sub) == 5:
        pielm_cell["metrics"] = median_metrics(pielm_sub)
        ad_time = autodiff_time_lookup.get(("cavity", "mlp"))
        if ad_time is not None and pielm_cell["metrics"]["train_time_min"] > 0:
            pielm_cell["speedup_vs_autodiff"] = ad_time / pielm_cell["metrics"]["train_time_min"]
    cells.append(pielm_cell)

    return cells, autodiff_time_lookup, anomalies


def filter_canonical_seeds(sub: pd.DataFrame, anomalies: list, *, label: str) -> pd.DataFrame:
    seeds_seen = sub.seed.tolist()
    extras = sorted(set(seeds_seen) - set(CANONICAL_SEEDS))
    if extras:
        anomalies.append(f"[extra seeds] {label} had non-canonical seeds {extras}; filtered out.")
    sub = sub[sub.seed.isin(CANONICAL_SEEDS)].copy()
    # Canonical-seed dedupe (safety; should be one row per seed already)
    if sub.seed.duplicated().any():
        anomalies.append(f"[duplicate seed] {label} has duplicate canonical seeds; using first.")
        sub = sub.drop_duplicates(subset=["seed"], keep="first")
    missing = sorted(set(CANONICAL_SEEDS) - set(sub.seed.tolist()))
    if missing:
        anomalies.append(f"[missing seeds] {label} missing {missing}; n_seeds={len(sub)}.")
    return sub


def render_doc(cells, autodiff_time_lookup, anomalies, df_all, *, out_path: Path) -> None:
    lines = []
    lines.append("# Cycle 16 medians (computed 2026-05-04)")
    lines.append("")
    lines.append("**Source:** `results/lid_benchmark_results.csv`")
    lines.append("**Seeds:** {0, 1, 7, 23, 42}")
    lines.append("**Tags consumed:**")
    lines.append("- `multiseed_20260427` — AutoDiff, DT-PINN, RoPINN, SAGE, PIELM. **Legacy SK-PINN-historical rows under this tag are IGNORED** (paper SK-PINN comes from `sk_pinn_matched_20260503`).")
    lines.append("- `canpinn_hpc_20260428` + `canpinn_hpc_20260428_excl_s7_time_outlier` — Spectral-AD. The `_excl_s7_time_outlier` tag is the dagger re-tag for kov-MLP s=7; included here as the s=7 row (same physical run, re-tagged not deleted).")
    lines.append("- `can_pinn_faithful_20260503` — CAN-PINN-faithful matched-protocol.")
    lines.append("- `sk_pinn_matched_20260503` — SK-PINN-matched (paper rows).")
    lines.append("")
    lines.append("**Method labels:** `autodiff`->AutoDiff, `dtpinn`->DT-PINN, `ropinn`->RoPINN, `chebyshev-pinn`->Spectral-AD, `sk-pinn` (under `sk_pinn_matched_20260503` only)->SK-PINN, `can-pinn-faithful`->CAN-PINN, `sage`->SAGE, `pielm`->PIELM.")
    lines.append("")
    lines.append("**Architectures:** `mlp`->MLP, `tsa-pinn`->TSA-PINN, `pirate-net`->PirateNet. (CSV spellings.)")
    lines.append("")
    lines.append("**Problems:** cavity, kovasznay, elasticity. (CSV spellings unchanged.)")
    lines.append("")
    lines.append("**Cells:** 7 methods x 3 archs x 3 problems = 63 + 1 PIELM (cavity-MLP only) = **64 cells**.")
    lines.append("")
    lines.append("**Speedup definition:** `median(AutoDiff train_time_min)` for the same (problem, arch) divided by `median(method train_time_min)`. Reported as `K.KKx`. AutoDiff cells are 1.00x by construction.")
    lines.append("")
    lines.append("**L^2 fields:** `u_rms_error`, `v_rms_error`, `p_rms_error` are NOT in the main CSV. They live in per-seed `results/tracking_*.csv` files (per CONTEXT.md s 7) and need a follow-up aggregation script (`scripts/aggregate_l2_and_protocol.py` is the reference). Deferred to a follow-up task.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Cell summary")
    lines.append("")
    lines.append("Order: by problem, then by arch, then by method (with PIELM appended after cavity-MLP-MLP-block).")
    lines.append("")
    lines.append("| Problem | Arch | Method | n_seeds | pde_rms | continuity_rms | momentum_rms | ms_per_epoch | train_time_min | speedup_vs_autodiff |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    def cell_sort_key(c):
        # PIELM goes right after AutoDiff in the cavity-MLP block
        problem_idx = PROBLEMS.index(c["problem"])
        arch_idx = ARCHS.index(c["arch"])
        method_order = ["AutoDiff", "PIELM", "DT-PINN", "RoPINN", "Spectral-AD", "SK-PINN", "CAN-PINN", "SAGE"]
        method_idx = method_order.index(c["paper_name"])
        return (problem_idx, arch_idx, method_idx)

    for c in sorted(cells, key=cell_sort_key):
        problem = PROBLEM_LABEL[c["problem"]]
        arch = ARCH_LABEL[c["arch"]]
        if c["paper_name"] == "PIELM" and not (c["problem"] == "cavity" and c["arch"] == "mlp"):
            continue
        if c["metrics"] is None:
            row = (
                f"| {problem} | {arch} | {c['paper_name']} | {c['n_seeds']} | "
                "n/a | n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            m = c["metrics"]
            row = (
                f"| {problem} | {arch} | {c['paper_name']} | {c['n_seeds']} | "
                f"{fmt(m['pde_rms'])} | {fmt(m['continuity_rms'])} | {fmt(m['momentum_rms'])} | "
                f"{fmt(m['ms_per_epoch'])} | {fmt(m['train_time_min'])} | "
                f"{fmt_speedup(c['speedup_vs_autodiff'])} |"
            )
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Verification (3 random hand-checked cells)")
    lines.append("")
    rng = np.random.default_rng(20260504)
    pickable = [c for c in cells if c["metrics"] is not None and c["paper_name"] != "PIELM"]
    # Force diversity: try to pick 3 from different (problem, method) tuples.
    rng.shuffle(pickable)
    picked = []
    seen = set()
    for c in pickable:
        key = (c["problem"], c["paper_name"])
        if key in seen:
            continue
        picked.append(c)
        seen.add(key)
        if len(picked) == 3:
            break
    if len(picked) < 3:
        # fallback: top-up with anything
        for c in pickable:
            if c not in picked:
                picked.append(c)
            if len(picked) == 3:
                break

    for c in picked:
        # Re-derive the median pde_rms by hand
        method_for_filter = c["csv_method"]
        allowed_tags = next(t for cm, t, _ in METHOD_TAG_MAP if cm == method_for_filter)
        sub = df_all[
            (df_all.method == method_for_filter)
            & (df_all.problem == c["problem"])
            & (df_all.model == c["arch"])
            & (df_all.tag.isin(allowed_tags))
            & (df_all.seed.isin(CANONICAL_SEEDS))
            & (df_all.status == "OK")
        ].sort_values("seed")
        five = sub.pde_rms.tolist()
        hand = float(np.median(five))
        script = c["metrics"]["pde_rms"]
        match = "MATCH" if np.isclose(hand, script, rtol=0, atol=1e-12) else "MISMATCH"
        lines.append(
            f"### {c['paper_name']} / {PROBLEM_LABEL[c['problem']]} / {ARCH_LABEL[c['arch']]}"
        )
        lines.append("")
        lines.append(f"- Tag(s): `{', '.join(sorted(allowed_tags))}`")
        seed_value_pairs = list(zip(sub.seed.tolist(), five))
        lines.append(f"- Raw 5 pde_rms (by seed): {seed_value_pairs}")
        lines.append(f"- Hand `np.median(...)`: `{hand!r}`")
        lines.append(f"- Script median: `{script!r}`")
        lines.append(f"- **{match}**")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    if anomalies:
        for a in anomalies:
            lines.append(f"- {a}")
    else:
        lines.append("- All 64 cells have exactly 5 canonical seeds; no missing or extra seeds detected.")
    lines.append("")
    lines.append("Additional notes:")
    lines.append("")
    lines.append("- **Legacy SK-PINN rows** under `multiseed_20260427` (45 rows: 3 problems x 3 archs x 5 seeds, grids N=200/150/100) are IGNORED by this script per the task spec; the paper SK-PINN cells use only `sk_pinn_matched_20260503`.")
    lines.append("- **Spectral-AD kov-MLP s=7** lives under tag `canpinn_hpc_20260428_excl_s7_time_outlier` (1 row, dagger re-tag); included as the s=7 row for that cell, exactly as per CONTEXT.md s 0.6.")
    lines.append("- **L^2 metrics** (`u_rms_error`, `v_rms_error`, `p_rms_error`) are not in the main CSV. They live in per-seed `results/tracking_*.csv` files. A follow-up aggregator (model: `scripts/aggregate_l2_and_protocol.py`) is needed to produce per-cell median L^2.")
    lines.append("- **`sage_lbfgs_fp64_cycle3_20260427`** ablation rows (5 elasticity-MLP rows that backed the cycle-12 F1 paper datum 0.0060 +/- 0.0019) were wiped by the 2026-05-03 evening rsync and are NOT in the CSV. The medians here for SAGE/elasticity/MLP come from the standard `multiseed_20260427` run, not the L-BFGS+fp64 cycle-3 ablation. If the paper's §V/conclusion still relies on the cycle-3 datum, that prose needs a separate decision (re-run on A40 ~2h vs cleanly remove); see CONTEXT.md s 0.6.")
    lines.append("- AutoDiff cells show `speedup_vs_autodiff = 1.00x` by construction.")
    lines.append("- Speedup ratios use median train_time_min, NOT median ms_per_epoch (these can disagree slightly when iteration count varies, but for this CSV all paper rows share a common iteration budget per cell).")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/lid_benchmark_results.csv")
    parser.add_argument(
        "--out",
        default="llmdocs/stream_sage_paper/paper_rewrite/cycle_16_medians.md",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    df = pd.read_csv(csv_path)
    explore(df)

    cells, autodiff_time_lookup, anomalies = compute_all(df)
    if anomalies:
        print("=== Anomalies detected ===")
        for a in anomalies:
            print(f"  {a}")
        print()

    render_doc(cells, autodiff_time_lookup, anomalies, df, out_path=out_path)
    n_cells_with_metrics = sum(1 for c in cells if c["metrics"] is not None)
    print(f"Cells with full 5-seed medians: {n_cells_with_metrics} / {len(cells)}")


if __name__ == "__main__":
    main()
