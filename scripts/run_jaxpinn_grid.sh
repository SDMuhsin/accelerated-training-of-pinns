#!/bin/bash
# Run the full JAX+JIT PINN benchmark grid: 3 problems x {mlp, pirate-net} x seed 42 x 30K epochs.
# Appends to results/lid_benchmark_results.csv with method='jaxpinn'.
# Also tags each row with 'a40_jaxpinn' for hardware provenance.
set -e
cd /workspace/dt-pinn
source env/bin/activate

TAG="a40_jaxpinn"
OUTPUT="results/lid_benchmark_results.csv"
SEED=42
EPOCHS=30000

for PROBLEM in cavity kovasznay elasticity; do
  for MODEL in mlp pirate-net; do
    echo "=== $(date -Is) :: $PROBLEM $MODEL ==="
    python -u src/lid_benchmark.py \
      --problem "$PROBLEM" \
      --method jaxpinn \
      --model "$MODEL" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --tag "$TAG" \
      --output-csv "$OUTPUT" 2>&1
    echo "=== $(date -Is) :: done $PROBLEM $MODEL ==="
  done
done
echo "=== ALL DONE ==="
