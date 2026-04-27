#!/bin/bash
# Phase 2 landscape data collection for the research protocol at
# llmdocs/research/RESEARCH_PROTOCOL.md. Adds seeds 0 and 1 on top of the
# already-recorded seed 42 so that Phase 2 can report mean ± std across
# ≥3 seeds. Fills the missing autodiff elasticity/pirate-net cell that was
# skipped by scripts/run_autodiff_a40_final3.sh.
#
# Tag: 'landscape_phase2' — isolated from existing tags so that the
# Phase 2 aggregation query is unambiguous.
#
# Resource plan: GPU 0 (GPU 1 was 96% foreign-occupied at launch time).
set -e
cd /workspace/dt-pinn
source env/bin/activate
export CUDA_VISIBLE_DEVICES=0

TAG="landscape_phase2"
OUTPUT="results/lid_benchmark_results.csv"
EPOCHS=30000

run() {
  local PROBLEM=$1 METHOD=$2 MODEL=$3 SEED=$4
  echo "=== $(date -Is) :: $PROBLEM $METHOD $MODEL seed=$SEED ==="
  python -u src/lid_benchmark.py \
    --problem "$PROBLEM" \
    --method "$METHOD" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --tag "$TAG" \
    --output-csv "$OUTPUT" 2>&1
  echo "=== $(date -Is) :: done $PROBLEM $METHOD $MODEL seed=$SEED ==="
}

# ---------- sage-jax : add seeds 0, 1 across all 6 cells ----------
for SEED in 0 1; do
  for PROBLEM in cavity kovasznay elasticity; do
    for MODEL in mlp pirate-net; do
      run "$PROBLEM" sage-jax "$MODEL" "$SEED"
    done
  done
done

# ---------- jaxpinn : add seeds 0, 1 across all 6 cells ----------
for SEED in 0 1; do
  for PROBLEM in cavity kovasznay elasticity; do
    for MODEL in mlp pirate-net; do
      run "$PROBLEM" jaxpinn "$MODEL" "$SEED"
    done
  done
done

# ---------- sage (tape-free torch variant) : add seeds 0, 1 ----------
for SEED in 0 1; do
  for PROBLEM in cavity kovasznay elasticity; do
    for MODEL in mlp pirate-net; do
      run "$PROBLEM" sage "$MODEL" "$SEED"
    done
  done
done

# ---------- autodiff : fill the single missing seed-42 cell ----------
# elasticity/pirate-net at seed 42 was never recorded; everything else is
# kept at single-seed (documented as a pragmatic deviation in 02_landscape.md)
run elasticity autodiff pirate-net 42

echo "=== ALL LANDSCAPE RUNS DONE $(date -Is) ==="
