"""
Aggregate priority-1/2/5 multi-seed reproduction results from
results/lid_benchmark_results.csv.

Produces a clean summary grouped by tag/method showing best vs final pde_rms,
train time, and seed variance.
"""
import os
import sys
import pandas as pd
import numpy as np

CSV = 'results/lid_benchmark_results.csv'


def fmt(x, n=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'nan'
    return f'{x:.{n}f}'


def main():
    df = pd.read_csv(CSV)

    print('=' * 80)
    print('PRIORITY 1 — Kovasznay × PirateNet (multi-seed AD vs spectral methods)')
    print('=' * 80)

    # Pull AD multi-seed (p1_ad_repro) + DT-PINN A40 + SAGE A40 + paper seed 42 + a40_rerun
    sub = df[
        (df['problem'] == 'kovasznay') & (df['model'] == 'pirate-net')
        & (
            df['tag'].isin([
                'p1_ad_repro', 'p1_dtpinn_a40', 'p1_sage_a40', 'a40_rerun',
                'landscape_phase2',
            ])
            | df['tag'].isna()
        )
    ].copy()
    sub['tag_str'] = sub['tag'].fillna('paper_h100')
    cols = ['method', 'tag_str', 'seed', 'train_time_s', 'pde_rms', 'final_loss', 'best_epoch']
    print(sub[cols].sort_values(['method', 'tag_str', 'seed']).to_string(index=False))
    print()

    # Per-method best-pde-rms summary across seeds with --track
    print('--- AD Kov × PirateNet best-pde-rms across seeds (p1_ad_repro tag, --track) ---')
    ad = sub[(sub['method'] == 'autodiff') & (sub['tag'] == 'p1_ad_repro')]
    if not ad.empty:
        for _, r in ad.iterrows():
            print(f"  seed {int(r['seed'])}: best_pde_rms={fmt(r['pde_rms'], 5)}  "
                  f"final_loss={fmt(r['final_loss'])}  best_epoch={r['best_epoch']}")
        print(f"  → mean ± std across {len(ad)} seeds: "
              f"{fmt(ad['pde_rms'].mean(), 5)} ± {fmt(ad['pde_rms'].std(), 5)}")
    else:
        print("  (no rows yet)")

    print()
    print('=' * 80)
    print('PRIORITY 2 — Elasticity × TSA-PINN (multi-seed AD)')
    print('=' * 80)
    sub2 = df[
        (df['problem'] == 'elasticity') & (df['model'] == 'tsa-pinn')
        & (df['tag'].isin(['p2_tsa_repro']) | df['tag'].isna())
    ].copy()
    sub2['tag_str'] = sub2['tag'].fillna('paper_h100')
    cols = ['method', 'tag_str', 'seed', 'train_time_s', 'pde_rms', 'final_loss', 'best_epoch']
    print(sub2[cols].sort_values(['method', 'seed']).to_string(index=False))

    print()
    print('=' * 80)
    print('PRIORITY 5 — Elasticity × PirateNet AD vs RoPINN timing')
    print('=' * 80)
    sub5 = df[
        (df['problem'] == 'elasticity') & (df['model'] == 'pirate-net')
        & (df['method'].isin(['autodiff', 'ropinn']))
        & (df['tag'].isin(['a40_rerun', 'landscape_phase2', 'p5_ropinn_repro', 'p5_ad_repro']) | df['tag'].isna())
    ].copy()
    sub5['tag_str'] = sub5['tag'].fillna('paper_h100')
    cols = ['method', 'tag_str', 'seed', 'train_time_s', 'pde_rms', 'final_loss', 'best_epoch']
    print(sub5[cols].sort_values(['method', 'tag_str', 'seed']).to_string(index=False))

    # Compute AD vs RoPINN timing ratio per platform
    ad5 = sub5[sub5['method'] == 'autodiff']
    rop5 = sub5[sub5['method'] == 'ropinn']
    if not ad5.empty and not rop5.empty:
        print()
        print('  AD median train_time:    {:.1f} s ({:d} runs)'.format(
            ad5['train_time_s'].median(), len(ad5)))
        print('  RoPINN median train_time: {:.1f} s ({:d} runs)'.format(
            rop5['train_time_s'].median(), len(rop5)))
        print('  RoPINN/AD ratio: {:.3f}× (>1 = RoPINN slower, expected)'.format(
            rop5['train_time_s'].median() / ad5['train_time_s'].median()))

    print()
    print('=' * 80)
    print('PRIORITY 3 — SK-PINN elasticity diagnostic (if completed)')
    print('=' * 80)
    diag_dir = 'results/p3_sk_pinn_diagnostic'
    if os.path.isdir(diag_dir):
        import json
        for fn in sorted(os.listdir(diag_dir)):
            if not fn.endswith('.json'):
                continue
            with open(os.path.join(diag_dir, fn)) as f:
                d = json.load(f)
            print(f"  {fn}:")
            print(f"    Train RKPM total rms = {d['train_rkpm_total_rms']:.6e}")
            print(f"    Train AD   total rms = {d['train_ad_total_rms']:.6e}")
            print(f"    Eval AD    pde_rms   = {d['eval_pde_rms']:.6e}")
            print(f"    AD/RKPM ratio (train) = {d['ad_to_rkpm_ratio_train']:.2f}×")
            print(f"    Eval/Train AD ratio   = {d['eval_to_train_ad_ratio']:.2f}×")
    else:
        print("  (no diagnostic results yet)")


if __name__ == '__main__':
    main()
