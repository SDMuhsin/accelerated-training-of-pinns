"""Phase-2 orchestrator — runs all (family × method × instance × seed × arch)
cells sequentially in one Python process so the JAX JIT cache is reused.

Default grid: B1 on F1/F2/F3 × 10 instances × 3 seeds × 2 architectures = 180 runs.
B2 grid is identical in shape. B3 requires a 'prime' run per (family, arch, seed).
B4 = alias for B1 (no separate measurement).

Output: one JSON per run in --out_dir, plus aggregated JSONL.

Usage:
    python -m src.phase2_orchestrator --methods B1 --families F1 F2 F3 \\
        --archs mlp pirate-net --seeds 42 0 1 --n_epochs 30000

The orchestrator skips runs whose output JSON already exists (idempotent).
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase2_runner import run_baseline, load_instances


FAMILIES = ['F1_cavity_NS', 'F2_Kovasznay', 'F3_elasticity']
ARCHS = ['mlp', 'pirate-net']
METHODS = ['B1', 'B2', 'B3']  # B4 = alias for B1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['B1'],
                        choices=METHODS)
    parser.add_argument('--families', nargs='+', default=FAMILIES,
                        choices=FAMILIES)
    parser.add_argument('--archs', nargs='+', default=ARCHS,
                        choices=ARCHS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 0, 1])
    parser.add_argument('--instances', nargs='+', type=int, default=None,
                        help="instance indices (default: all 10 per family)")
    parser.add_argument('--n_epochs', type=int, default=30000)
    parser.add_argument('--probe_every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out_dir', default='results/progB_phase2')
    parser.add_argument('--prime_dir', default='results/progB_phase2_primes',
                        help="Where B3 prime params are cached")
    parser.add_argument('--warm_instance', type=int, default=0,
                        help="instance index used to prime B3 warm-start")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.prime_dir, exist_ok=True)

    instances = load_instances()
    inst_indices = args.instances if args.instances is not None else list(range(10))

    # Enumerate runs
    queue = []
    for method in args.methods:
        for family in args.families:
            for arch in args.archs:
                for seed in args.seeds:
                    for inst in inst_indices:
                        queue.append((method, family, arch, seed, inst))

    print(f"[orchestrator] {len(queue)} runs queued")
    start_ts = time.perf_counter()

    # B3 primes: we need params from B1 on (family, arch, seed, warm_instance)
    prime_cache = {}  # (family, arch, seed) -> params pytree

    completed = 0
    skipped = 0
    errored = 0

    for (method, family, arch, seed, inst) in queue:
        run_id = f"{family}_{method}_i{inst}_s{seed}_{arch}"
        out_path = os.path.join(args.out_dir, f"{run_id}.json")
        if os.path.exists(out_path):
            print(f"[skip] {run_id} exists")
            skipped += 1
            continue

        warm_start = None
        if method == 'B3':
            # Need prime params for (family, arch, seed)
            key = (family, arch, seed)
            if key not in prime_cache:
                prime_path = os.path.join(args.prime_dir,
                                          f"{family}_prime_s{seed}_{arch}.pkl")
                if os.path.exists(prime_path):
                    with open(prime_path, 'rb') as f:
                        prime_cache[key] = pickle.load(f)
                else:
                    print(f"[prime-missing] {prime_path} — running B1 on "
                          f"instance {args.warm_instance} to prime")
                    prime_result, prime_params = run_baseline(
                        family=family, method='B1', instance_idx=args.warm_instance,
                        seed=seed, arch=arch, n_epochs=args.n_epochs,
                        probe_every=args.probe_every, lr=args.lr,
                        instances=instances)
                    # Save the prime run's trajectory too
                    prime_trajectory_path = os.path.join(
                        args.out_dir, f"{family}_B1_i{args.warm_instance}_s{seed}_{arch}.json")
                    if not os.path.exists(prime_trajectory_path):
                        prime_result['tag'] = 'progB_phase2'
                        with open(prime_trajectory_path, 'w') as f:
                            json.dump(prime_result, f)
                    with open(prime_path, 'wb') as f:
                        pickle.dump(prime_params, f)
                    prime_cache[key] = prime_params
            warm_start = prime_cache[key]

        print(f"\n[run {completed+skipped+errored+1}/{len(queue)}] {run_id}")
        try:
            result, _ = run_baseline(
                family=family, method=method, instance_idx=inst,
                seed=seed, arch=arch, n_epochs=args.n_epochs,
                probe_every=args.probe_every, lr=args.lr,
                warm_start_params=warm_start, instances=instances)
            result['tag'] = 'progB_phase2'
            with open(out_path, 'w') as f:
                json.dump(result, f)
            completed += 1
            elapsed = time.perf_counter() - start_ts
            remaining = len(queue) - completed - skipped - errored
            avg_per = elapsed / max(completed, 1)
            eta_s = remaining * avg_per
            print(f"[ok] {run_id} rms={result['final_pde_rms']:.4e} "
                  f"t={result['t_total_s']:.1f}s | "
                  f"done={completed}/{len(queue)}, eta={eta_s/60:.1f}min")
        except Exception as e:
            errored += 1
            print(f"[ERROR] {run_id}: {e}")
            traceback.print_exc()
            err_path = out_path.replace('.json', '.err.txt')
            with open(err_path, 'w') as f:
                f.write(f"{e}\n{traceback.format_exc()}")

    total_elapsed = time.perf_counter() - start_ts
    print(f"\n[orchestrator] done: {completed} ok, {skipped} skipped, "
          f"{errored} errored in {total_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
