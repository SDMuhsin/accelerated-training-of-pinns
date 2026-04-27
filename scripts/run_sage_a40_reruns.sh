#!/bin/bash
# Rerun SAGE on A40 only — used to compute internally-consistent
# SAGE-vs-JAX+JIT speedups for the JAX comparison panel. The rest of
# tab:main_results (autodiff, dt-pinn, ropinn, sk-pinn) keeps its existing
# H100 numbers, with a footnote disclosing the mixed hardware.
set -e
cd /workspace/dt-pinn
source env/bin/activate

TAG="a40_rerun"
OUTPUT="results/lid_benchmark_results.csv"
SEED=42
EPOCHS=30000

for PROBLEM in cavity kovasznay elasticity; do
  for MODEL in mlp pirate-net; do
    echo "=== $(date -Is) :: $PROBLEM sage $MODEL ==="
    python -u src/lid_benchmark.py \
      --problem "$PROBLEM" \
      --method sage \
      --model "$MODEL" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --tag "$TAG" \
      --output-csv "$OUTPUT" 2>&1
    echo "=== $(date -Is) :: done $PROBLEM sage $MODEL ==="
  done
done
echo "=== ALL SAGE A40 RERUNS DONE ==="
