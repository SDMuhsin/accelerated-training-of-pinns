#!/bin/bash
# Re-run the JAX+JIT PINN baseline grid with jax_default_matmul_precision='highest'
# (full fp32 matmul, not TF32). This matches PyTorch's default matmul precision
# so that JAX+JIT and SAGE-JAX are measured apples-to-apples against sage-torch.
#
# Tag: 'a40_jaxpinn_hp'
set -e
cd /workspace/dt-pinn
source env/bin/activate

TAG="a40_jaxpinn_hp"
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
