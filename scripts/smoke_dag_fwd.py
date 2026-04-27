"""Smoke test A: measure T_dag-fwd per cell.

Measures ONLY the DAG forward (residual computation from a fixed U tensor)
for each of the 6 benchmark cells. Does NOT call the NN forward, NN backward,
or DAG backward. Reports median ms/step compared to the parity estimates in
04_design.md § 2.4.

Pre-gate kill rule (04_design.md § 2.5, blacklist § 3 meta-lesson):
  - If measured T_dag_fwd on ANY cell exceeds 2 x the parity estimate,
    recompute the worst-cell & geomean speedup at k=2 per the pessimistic
    table in § 2.5.
  - If the recomputed geomean is < 1.5 or worst-cell < 1.2, fail smoke.
"""
import os
import sys
import time
import math

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lid_benchmark import (
    build_grid_data,
    build_grid_data_kovasznay,
    build_grid_data_elasticity,
    compute_pde_terms,
    compute_pde_kovasznay,
    compute_pde_elasticity,
    make_model,
)


def time_dag_fwd(fn, pred, g, n_iters=2000, n_warmup=100):
    """Synchronous timing of the DAG-forward kernel composition.

    Uses CUDA events for accurate GPU timing (avoids Python-side
    perf_counter overhead per sync).

    Returns average ms/call over n_iters calls.
    """
    device = pred.device
    # Warmup
    for _ in range(n_warmup):
        out = fn(pred, g)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    if device.type == 'cuda':
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(n_iters):
            out = fn(pred, g)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / n_iters  # avg ms/call
    else:
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn(pred, g)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / n_iters


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Parity estimates per 04_design.md § 2.4
    parity = {
        ('cavity', 'mlp'): 0.125,
        ('cavity', 'pirate-net'): 0.191,
        ('kovasznay', 'mlp'): 0.044,
        ('kovasznay', 'pirate-net'): 0.122,
        ('elasticity', 'mlp'): 0.044,
        ('elasticity', 'pirate-net'): 0.120,
    }
    # Reference T_ref (mean of 3) per 02_landscape.md
    T_ref = {
        ('cavity', 'mlp'): 1.020,
        ('cavity', 'pirate-net'): 1.557,
        ('kovasznay', 'mlp'): 0.360,
        ('kovasznay', 'pirate-net'): 0.9933,
        ('elasticity', 'mlp'): 0.360,
        ('elasticity', 'pirate-net'): 0.980,
    }
    # Adam update cost per 04_design.md § 2.4 (per-cell)
    T_adam = {'mlp': 0.050, 'pirate-net': 0.080}

    # Build each grid_data once; pred is a fixed random tensor.
    grid_size = 50
    results = {}
    torch.manual_seed(42)
    for problem in ('cavity', 'kovasznay', 'elasticity'):
        if problem == 'cavity':
            g = build_grid_data(grid_size, device)
            fn = compute_pde_terms
            out_dim = 3
        elif problem == 'kovasznay':
            g = build_grid_data_kovasznay(grid_size, device)
            fn = compute_pde_kovasznay
            out_dim = 3
        else:
            g = build_grid_data_elasticity(grid_size, device)
            fn = compute_pde_elasticity
            out_dim = 2
        N_all = g['N_all']
        # Same pred used for both models (DAG-fwd is architecture-independent,
        # but we time under each cell label since T_ref differs).
        pred = torch.randn(N_all, out_dim, device=device, dtype=torch.float32)

        for model_name in ('mlp', 'pirate-net'):
            ms = time_dag_fwd(fn, pred, g, n_iters=500, n_warmup=50)
            results[(problem, model_name)] = ms
            par = parity[(problem, model_name)]
            ratio = ms / par if par > 0 else float('nan')
            exceeds_2x = (ms > 2.0 * par)
            flag = "!! >2x parity" if exceeds_2x else ""
            print(f"  {problem:11s} / {model_name:10s}: measured T_dag_fwd = {ms:.4f} ms "
                  f"(parity = {par:.3f} ms, ratio = {ratio:.2f}x) {flag}")

    # Compute worst-cell and geomean speedup under k=2 using measured T_dag_fwd.
    # T_fcst = DMD(0.010) + T_dag_fwd (measured) + Adam(per model)
    # T_avg(k=2) = 0.5*T_ref + 0.5*T_fcst
    print()
    print("=== HC-1 recomputation with measured T_dag_fwd (k=2) ===")
    print(f"  {'cell':28s}  {'T_ref':>7s}  {'T_fcst':>7s}  {'T_avg':>7s}  {'speedup':>8s}  {'T_req':>7s}  status")
    speedups = []
    pass_t1_worst = True
    pessimistic_hit = False
    for (problem, model_name), ms_dag in results.items():
        tref = T_ref[(problem, model_name)]
        t_fcst = 0.010 + ms_dag + T_adam[model_name]
        t_avg = 0.5 * tref + 0.5 * t_fcst
        speedup = tref / t_avg if t_avg > 0 else float('inf')
        t_req = tref / 1.5
        cell = f"{problem}/{model_name}"
        status = 'PASS' if speedup >= 1.2 else 'FAIL(worst-cell)'
        if speedup < 1.2:
            pass_t1_worst = False
        # Pessimistic if measured > 2x parity
        par = parity[(problem, model_name)]
        if ms_dag > 2.0 * par:
            pessimistic_hit = True
        print(f"  {cell:28s}  {tref:7.3f}  {t_fcst:7.3f}  {t_avg:7.3f}  "
              f"{speedup:8.3f}  {t_req:7.3f}  {status}")
        speedups.append(speedup)
    # Geomean
    log_sum = sum(math.log(s) for s in speedups)
    geomean = math.exp(log_sum / len(speedups))
    print(f"\n  Geomean speedup = {geomean:.4f}x (target >= 1.5x)")
    print(f"  Worst-cell speedup = {min(speedups):.4f}x (target >= 1.2x)")

    # Pre-gate A verdict
    print()
    print("=== PRE-GATE A VERDICT ===")
    fail = False
    reasons = []
    if geomean < 1.5:
        fail = True
        reasons.append(f"geomean speedup {geomean:.4f}x < 1.5x")
    if min(speedups) < 1.2:
        fail = True
        reasons.append(f"worst-cell speedup {min(speedups):.4f}x < 1.2x")
    if fail:
        print(f"  KILL — {'; '.join(reasons)}")
        sys.exit(1)
    else:
        print(f"  PASS (geomean {geomean:.4f}x, worst {min(speedups):.4f}x)")
        if pessimistic_hit:
            print(f"  Note: at least one cell exceeded 2x parity estimate but the "
                  f"recomputed HC-1 still clears both floors.")


if __name__ == "__main__":
    main()
