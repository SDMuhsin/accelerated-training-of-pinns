#!/bin/bash
# Phase 5 probe: seed-42 slrm-jax on all 6 cells for early Phase 5 assessment.
# Accuracy target (contract T2): R_method <= 1.111 * R_sage_jax_mean on 5/6 cells.
set -e
cd /workspace/dt-pinn
source env/bin/activate
export CUDA_VISIBLE_DEVICES=0

TAG="slrm_probe"
OUTPUT="results/lid_benchmark_results.csv"
EPOCHS=30000
SEED=42

run() {
  local PROBLEM=$1 MODEL=$2
  echo "=== $(date -Is) :: $PROBLEM slrm-jax $MODEL seed=$SEED ==="
  python -u src/lid_benchmark.py \
    --problem "$PROBLEM" \
    --method slrm-jax \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --tag "$TAG" \
    --output-csv "$OUTPUT" 2>&1
  echo "=== $(date -Is) :: done ==="
}

for PROBLEM in cavity kovasznay elasticity; do
  for MODEL in mlp pirate-net; do
    run "$PROBLEM" "$MODEL"
  done
done

echo "=== ALL SLRM PROBE RUNS DONE $(date -Is) ==="
