#!/bin/bash
# Rerun the PyTorch comparators (autodiff, sage) on A40 for apples-to-apples
# speedup numbers against the JAX+JIT baseline. Tagged with 'a40_rerun' so the
# paper can present a self-consistent A40 row group without disturbing the
# existing H100 MIG rows.
set -e
cd /workspace/dt-pinn
source env/bin/activate

TAG="a40_rerun"
OUTPUT="results/lid_benchmark_results.csv"
SEED=42
EPOCHS=30000

# autodiff first (slowest), then sage
for METHOD in autodiff sage; do
  for PROBLEM in cavity kovasznay elasticity; do
    for MODEL in mlp pirate-net; do
      echo "=== $(date -Is) :: $PROBLEM $METHOD $MODEL ==="
      python -u src/lid_benchmark.py \
        --problem "$PROBLEM" \
        --method "$METHOD" \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --tag "$TAG" \
        --output-csv "$OUTPUT" 2>&1
      echo "=== $(date -Is) :: done $PROBLEM $METHOD $MODEL ==="
    done
  done
done
echo "=== ALL PYTORCH A40 RERUNS DONE ==="
