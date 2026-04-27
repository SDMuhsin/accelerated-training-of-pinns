#!/usr/bin/env python3
"""Analyze Phase 2 landscape v2 results: bfsa vs sage-jax paired comparison."""
import pandas as pd
import numpy as np
import sys

csv_path = "results/lid_benchmark_results.csv"
df = pd.read_csv(csv_path)

# Filter to the two tags we care about
bfsa = df[df["tag"] == "bfsa_landscape"].copy()
sagejax = df[df["tag"] == "sagejax_landscape"].copy()

print(f"bfsa rows: {len(bfsa)}, sage-jax rows: {len(sagejax)}")
if len(bfsa) < 18 or len(sagejax) < 18:
    print("WARNING: Not all runs completed yet!")
    print(f"  bfsa cells: {bfsa.groupby(['problem','model']).size().to_dict()}")
    print(f"  sage-jax cells: {sagejax.groupby(['problem','model']).size().to_dict()}")

problems = ["cavity", "kovasznay", "elasticity"]
models = ["mlp", "pirate-net"]

print("\n" + "="*80)
print("BFSA LANDSCAPE (mean ± std, n=3 seeds)")
print("="*80)
print(f"{'Cell':<28} {'n':>3} {'T (ms/ep)':>18} {'R (PDE RMS)':>22}")
print("-"*75)

bfsa_times = []
for prob in problems:
    for model in models:
        mask = (bfsa["problem"] == prob) & (bfsa["model"] == model)
        sub = bfsa[mask]
        n = len(sub)
        t_mean = sub["ms_per_epoch"].mean()
        t_std = sub["ms_per_epoch"].std()
        r_mean = sub["pde_rms"].mean()
        r_std = sub["pde_rms"].std()
        bfsa_times.append(t_mean)
        cell = f"{prob} / {model}"
        print(f"{cell:<28} {n:>3} {t_mean:>8.3f} ± {t_std:>5.3f}   {r_mean:>10.5f} ± {r_std:>7.5f}")

bfsa_geomean = np.exp(np.mean(np.log(bfsa_times)))
print(f"\nGeomean T (bfsa): {bfsa_geomean:.4f} ms/ep")

print("\n" + "="*80)
print("SAGE-JAX LANDSCAPE (mean ± std, n=3 seeds)")
print("="*80)
print(f"{'Cell':<28} {'n':>3} {'T (ms/ep)':>18} {'R (PDE RMS)':>22}")
print("-"*75)

sagejax_times = []
for prob in problems:
    for model in models:
        mask = (sagejax["problem"] == prob) & (sagejax["model"] == model)
        sub = sagejax[mask]
        n = len(sub)
        t_mean = sub["ms_per_epoch"].mean()
        t_std = sub["ms_per_epoch"].std()
        r_mean = sub["pde_rms"].mean()
        r_std = sub["pde_rms"].std()
        sagejax_times.append(t_mean)
        cell = f"{prob} / {model}"
        print(f"{cell:<28} {n:>3} {t_mean:>8.3f} ± {t_std:>5.3f}   {r_mean:>10.5f} ± {r_std:>7.5f}")

sagejax_geomean = np.exp(np.mean(np.log(sagejax_times)))
print(f"\nGeomean T (sage-jax): {sagejax_geomean:.4f} ms/ep")

print("\n" + "="*80)
print("PAIRED COMPARISON: bfsa vs sage-jax (speedup = T_sagejax / T_bfsa)")
print("="*80)
print(f"{'Cell':<28} {'T_bfsa':>8} {'T_sagejax':>10} {'Speedup':>8} {'R_bfsa':>10} {'R_sagejax':>10} {'R_ratio':>8}")
print("-"*82)

speedups = []
for i, (prob, model) in enumerate([(p,m) for p in problems for m in models]):
    t_b = bfsa_times[i]
    t_s = sagejax_times[i]
    speedup = t_s / t_b

    mask_b = (bfsa["problem"] == prob) & (bfsa["model"] == model)
    mask_s = (sagejax["problem"] == prob) & (sagejax["model"] == model)
    r_b = bfsa[mask_b]["pde_rms"].mean()
    r_s = sagejax[mask_s]["pde_rms"].mean()
    r_ratio = r_b / r_s  # <1 means bfsa is more accurate, >1 means less accurate

    speedups.append(speedup)
    cell = f"{prob} / {model}"
    print(f"{cell:<28} {t_b:>8.3f} {t_s:>10.3f} {speedup:>8.3f}x {r_b:>10.5f} {r_s:>10.5f} {r_ratio:>8.4f}")

geomean_speedup = np.exp(np.mean(np.log(speedups)))
worst_speedup = min(speedups)
print(f"\nGeomean speedup (bfsa over sage-jax): {geomean_speedup:.4f}x")
print(f"Worst-cell speedup: {worst_speedup:.4f}x")

print("\n" + "="*80)
print("GATE CHECK: Is bfsa consistently faster than sage-jax?")
print("="*80)
all_faster = all(s > 1.0 for s in speedups)
print(f"All cells bfsa faster: {'PASS' if all_faster else 'FAIL'}")
for i, (prob, model) in enumerate([(p,m) for p in problems for m in models]):
    status = "OK" if speedups[i] > 1.0 else "FAIL"
    print(f"  {prob}/{model}: {speedups[i]:.3f}x [{status}]")

print("\n" + "="*80)
print("NEW REFERENCE ENVELOPE (bfsa as T_ref)")
print("v2 T1 bar: ≥1.3× geomean, ≥1.1× worst-cell")
print("="*80)
print(f"{'Cell':<28} {'T_ref (bfsa)':>12} {'T_req (≤, 1.1×)':>16} {'R_ref (bfsa)':>12} {'R_req (≤1.111×)':>16}")
print("-"*88)

for i, (prob, model) in enumerate([(p,m) for p in problems for m in models]):
    mask_b = (bfsa["problem"] == prob) & (bfsa["model"] == model)
    t_ref = bfsa_times[i]
    t_req = t_ref / 1.1  # worst-cell bar
    r_ref = bfsa[mask_b]["pde_rms"].mean()
    r_req = r_ref * 1.111
    cell = f"{prob} / {model}"
    print(f"{cell:<28} {t_ref:>12.4f} {t_req:>16.4f} {r_ref:>12.5f} {r_req:>16.5f}")

geomean_t_req = bfsa_geomean / 1.3
print(f"\nGeomean T_ref (bfsa): {bfsa_geomean:.4f} ms/ep")
print(f"Geomean T_req for ≥1.3×: ≤ {geomean_t_req:.4f} ms/ep")

# Also print per-seed raw data for reference
print("\n" + "="*80)
print("PER-SEED RAW DATA (bfsa)")
print("="*80)
for prob in problems:
    for model in models:
        mask = (bfsa["problem"] == prob) & (bfsa["model"] == model)
        sub = bfsa[mask].sort_values("seed")
        cell = f"{prob} / {model}"
        seeds_t = [(int(r["seed"]), r["ms_per_epoch"], r["pde_rms"]) for _, r in sub.iterrows()]
        print(f"  {cell}: " + " | ".join(f"seed={s}: T={t:.3f}, R={r:.5f}" for s,t,r in seeds_t))

print("\n" + "="*80)
print("PER-SEED RAW DATA (sage-jax)")
print("="*80)
for prob in problems:
    for model in models:
        mask = (sagejax["problem"] == prob) & (sagejax["model"] == model)
        sub = sagejax[mask].sort_values("seed")
        cell = f"{prob} / {model}"
        seeds_t = [(int(r["seed"]), r["ms_per_epoch"], r["pde_rms"]) for _, r in sub.iterrows()]
        print(f"  {cell}: " + " | ".join(f"seed={s}: T={t:.3f}, R={r:.5f}" for s,t,r in seeds_t))
