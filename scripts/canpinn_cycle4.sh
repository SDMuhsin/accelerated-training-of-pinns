#!/bin/bash
# Cycle 4 CAN-PINN baseline: dense Chebyshev forward + AD backward
# (the chebyshev-pinn method in lid_benchmark.py). Adds the head-to-head
# row F01 of cycle_04_review.md asks for: SAGE differs from CAN-PINN
# only in routing the backward through symbolic VJP rather than torch.autograd.
#
# Scope: MLP architecture × 3 PDE problems × 5 seeds = 15 runs.
# Tag: canpinn_cycle4_20260427.
# Hardware: dual NVIDIA A40 (both available locally; H100 MIG that produced
# multiseed_20260427 is on a remote cluster -- tagged separately).

set -euo pipefail
cd /workspace/dt-pinn
source env/bin/activate

TAG="canpinn_cycle4_20260427"
SEEDS=(0 1 7 23 42)
LOGDIR="logs/cycle4_canpinn"
mkdir -p "$LOGDIR"

GPU="$1"
shift
PROBLEMS=("$@")

for problem in "${PROBLEMS[@]}"; do
  case "$problem" in
    cavity) GRID=50 ;;
    kovasznay|elasticity) GRID=30 ;;
    *) echo "unknown problem $problem"; exit 1 ;;
  esac
  for seed in "${SEEDS[@]}"; do
    echo "[$(date '+%T')] CUDA_VISIBLE_DEVICES=$GPU $problem seed=$seed grid=$GRID"
    CUDA_VISIBLE_DEVICES="$GPU" python -u src/lid_benchmark.py \
      --problem "$problem" \
      --method chebyshev-pinn \
      --model mlp \
      --seed "$seed" \
      --epochs 30000 \
      --grid-size "$GRID" \
      --tag "$TAG" \
      --output-csv results/lid_benchmark_results.csv \
      --track \
      > "$LOGDIR/${problem}_mlp_seed${seed}.log" 2>&1
    echo "[$(date '+%T')]   done $problem seed=$seed"
  done
done
