#!/bin/bash
# Phase 5 stencil-adjoint grid: 6 cells x 3 seeds x 30k epochs
# Writes rows to results/lid_benchmark_results.csv with tag=stencil_phase5
set -e
cd "$(dirname "$0")/.."
source env/bin/activate

export CUDA_VISIBLE_DEVICES=1
TAG=stencil_phase5
EPOCHS=30000
LOG=logs/stencil_phase5.log
mkdir -p logs

echo "=== Phase 5 stencil-adjoint grid START $(date -Iseconds) ===" | tee $LOG

for problem in cavity kovasznay elasticity; do
  for model in mlp pirate-net; do
    for seed in 42 0 1; do
      echo "--- $problem/$model seed=$seed ---" | tee -a $LOG
      python -u src/lid_benchmark.py \
        --method stencil-adjoint \
        --problem $problem \
        --model $model \
        --seed $seed \
        --epochs $EPOCHS \
        --tag $TAG 2>&1 | tee -a $LOG | tail -25
    done
  done
done

echo "=== Phase 5 stencil-adjoint grid DONE $(date -Iseconds) ===" | tee -a $LOG
