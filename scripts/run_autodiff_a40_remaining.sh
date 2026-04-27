#!/bin/bash
# Catch-up: rerun autodiff on A40 for the 5 cells we haven't done yet
# (cavity mlp is already in the CSV). Provides A40 PyTorch eager numbers
# for fair speedup math against jaxpinn A40 + sage A40.
set -e
cd /workspace/dt-pinn
source env/bin/activate

TAG="a40_rerun"
OUTPUT="results/lid_benchmark_results.csv"
SEED=42
EPOCHS=30000

# cavity MLP autodiff already done. Run the remaining 5.
for SPEC in \
  "cavity pirate-net" \
  "kovasznay mlp" \
  "kovasznay pirate-net" \
  "elasticity mlp" \
  "elasticity pirate-net"; do
  PROBLEM=$(echo $SPEC | cut -d' ' -f1)
  MODEL=$(echo $SPEC | cut -d' ' -f2)
  echo "=== $(date -Is) :: $PROBLEM autodiff $MODEL ==="
  python -u src/lid_benchmark.py \
    --problem "$PROBLEM" \
    --method autodiff \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --tag "$TAG" \
    --output-csv "$OUTPUT" 2>&1
  echo "=== $(date -Is) :: done $PROBLEM autodiff $MODEL ==="
done
echo "=== ALL AUTODIFF A40 REMAINING DONE ==="
