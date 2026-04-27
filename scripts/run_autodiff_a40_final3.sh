#!/bin/bash
# Finish the A40 PyTorch-eager autodiff reruns that were interrupted to make
# room for the SAGE-JAX grid. The three cells below were NOT completed in the
# earlier scripts/run_autodiff_a40_remaining.sh run (killed mid-kovasznay-pn).
#
# Already in the CSV under tag='a40_rerun':
#   cavity   autodiff mlp          ✓  (47.98 ms/ep)
#   cavity   autodiff pirate-net   ✓  (150.51 ms/ep)
#   kovasznay autodiff mlp         ✓  (30.91 ms/ep)
#
# Needed:
set -e
cd /workspace/dt-pinn
source env/bin/activate

TAG="a40_rerun"
OUTPUT="results/lid_benchmark_results.csv"
SEED=42
EPOCHS=30000

for CELL in "kovasznay pirate-net" "elasticity mlp" "elasticity pirate-net"; do
  set -- $CELL
  PROBLEM=$1; MODEL=$2
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
echo "=== ALL DONE ==="
