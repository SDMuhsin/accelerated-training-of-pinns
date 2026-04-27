#!/bin/bash
# Parallel Phase 6: elasticity-only sweep on GPU 1, alongside GPU 0's
# remaining Kovasznay/Elasticity queue. Launched 2026-04-27 once Phase 5 freed GPU 1.
set -e
cd "$(dirname "$0")/.."
source env/bin/activate
TAG="dtpinn_faithful_2026_04_26"
OUTPUT_CSV="results/lid_benchmark_results.csv"
LOG="logs/dtpinn_faithful_gpu1_${TAG}_elas.log"
{
    echo "=== gpu1 elasticity sweep ($(date)) ==="
    for model in mlp tsa-pinn pirate-net; do
        for seed in 0 1 7 23 42; do
            echo
            echo "--- gpu1: elasticity × $model × seed=$seed ---"
            CUDA_VISIBLE_DEVICES=1 python3 -u src/lid_benchmark.py \
                --problem elasticity --method dtpinn --model "$model" \
                --seed "$seed" --epochs 5000 \
                --output-csv "$OUTPUT_CSV" --tag "$TAG" \
                --track --track-interval 100 \
                || echo "RUN FAILED: elasticity $model $seed"
        done
    done
    echo
    echo "=== gpu1 finished ($(date)) ==="
} > "$LOG" 2>&1
