#!/bin/bash
# Run the full SAGE-JAX PINN benchmark grid: 3 problems x {mlp, pirate-net} x seed 42 x 30K epochs.
# Appends to results/lid_benchmark_results.csv with method='sage-jax' and tag='a40_sage_jax'.
#
# SAGE-JAX = SAGE-generated explicit PDE backward emitted via symbolic_vjp.py
# with backend='jax', fused with a jax.vjp network backprop inside a single
# @jit training step. See llmdocs/CONTEXT.md section "SAGE-JAX" for the design.
set -e
cd /workspace/dt-pinn
source env/bin/activate

TAG="a40_sage_jax"
OUTPUT="results/lid_benchmark_results.csv"
SEED=42
EPOCHS=30000

for PROBLEM in cavity kovasznay elasticity; do
  for MODEL in mlp pirate-net; do
    echo "=== $(date -Is) :: $PROBLEM $MODEL ==="
    python -u src/lid_benchmark.py \
      --problem "$PROBLEM" \
      --method sage-jax \
      --model "$MODEL" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --tag "$TAG" \
      --output-csv "$OUTPUT" 2>&1
    echo "=== $(date -Is) :: done $PROBLEM $MODEL ==="
  done
done
echo "=== ALL DONE ==="
