#!/bin/bash
# Phase 2: Landscape re-measurement for bfsa (new baseline) and sage-jax (paired comparison)
# 6 cells × 3 seeds × 2 methods = 36 runs, 30K epochs each
# CUDA_VISIBLE_DEVICES=0 to avoid GPU contention

set -e
source env/bin/activate
export CUDA_VISIBLE_DEVICES=0

PROBLEMS="cavity kovasznay elasticity"
MODELS="mlp pirate-net"
SEEDS="42 0 1"

echo "=== Phase 2 Landscape v2: bfsa runs ==="
for problem in $PROBLEMS; do
  for model in $MODELS; do
    for seed in $SEEDS; do
      echo "[$(date '+%H:%M:%S')] bfsa | $problem | $model | seed=$seed"
      python3 src/lid_benchmark.py \
        --method bfsa \
        --problem "$problem" \
        --model "$model" \
        --seed "$seed" \
        --epochs 30000 \
        --tag bfsa_landscape
    done
  done
done

echo ""
echo "=== Phase 2 Landscape v2: sage-jax runs ==="
for problem in $PROBLEMS; do
  for model in $MODELS; do
    for seed in $SEEDS; do
      echo "[$(date '+%H:%M:%S')] sage-jax | $problem | $model | seed=$seed"
      python3 src/lid_benchmark.py \
        --method sage-jax \
        --problem "$problem" \
        --model "$model" \
        --seed "$seed" \
        --epochs 30000 \
        --tag sagejax_landscape
    done
  done
done

echo ""
echo "=== All 36 runs complete ==="
