#!/usr/bin/env bash
# Multi-seed convergence study for V4 JAX-SAGE.
#
# Runs seeds {2345, 3456, 4567, 5678} sequentially on GPU 1 (seed 1234
# was already run via run_v4_jax_sage_full.sh). Each seed ~26 min on
# A40 shared tenancy → total ~1h 45m.
#
# Outputs per seed:
#   results/partner_v4_jax_sage_seed${s}/{flow,temp}/
#   results/partner_v4_jax_sage_seed${s}_run.log
#
# After all seeds finish, re-run the three-way eval with each seed's
# checkpoints to get the mean ± std numbers for the holistic comparison
# doc.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source "$REPO_ROOT/env/bin/activate"
fi

: "${CUDA_VISIBLE_DEVICES:=1}"
export CUDA_VISIBLE_DEVICES
echo "[multi-seed] GPU(s): $CUDA_VISIBLE_DEVICES"

for s in 2345 3456 4567 5678; do
    echo "[multi-seed] Starting seed $s ..."
    SEED=$s OUT_SUFFIX="_seed${s}" \
        bash scripts/run_v4_jax_sage_full.sh \
        > "$REPO_ROOT/results/partner_v4_jax_sage_seed${s}_run.log" 2>&1
    echo "[multi-seed] Seed $s done."
done

echo "[multi-seed] ALL SEEDS COMPLETE"
