#!/bin/bash
# ============================================================================
# TGV Phase 3 paperscale — head-to-head with Wang/Perdikaris arXiv:2507.08972
# ============================================================================
#
# Submits the 3D Taylor-Green vortex Phase 3 headline run:
#     PirateNet + causal training + SOAP + Smagorinsky LES at Re=1600.
#
# Reference: llmdocs/CONTEXT.md §0.4 (acceptance gates) and
# llmdocs/trackers/taylor_green_phase3_2026-05-07.md.
#
# IMPORTANT: this script is for the user to submit on HPC. Do NOT run from
# the dev box (per the standing rule). Verify the dry-run pipeline locally
# (./scripts/diff or `bash -n run_taylor_green_phase3.sh`) before sbatch.
#
# Usage:
#     sbatch sbatch/run_taylor_green_phase3.sh                # default seed=0
#     sbatch --export=ALL,SEED=42 sbatch/run_taylor_green_phase3.sh
#     bash sbatch/run_taylor_green_phase3.sh --multi          # multi-seed (n=5)
#
# Wall-clock estimate (A40 dev-box): ~23h for 11 win × 30k ep at 250 ms/ep.
# Wall-clock estimate (H100 MIG 2g.20gb): ~10-12h (same-class to TGV Phase 1).
#
# Hardware footnote: record GPU type in the tag (the row's gpu_name column
# is set automatically; the tag here just helps grouping).
# ============================================================================

#SBATCH --job-name=tgv_phase3
#SBATCH --output=logs/tgv_phase3_%j.out
#SBATCH --error=logs/tgv_phase3_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# ----- defaults (override via --export=ALL,VAR=val) -----
SEED="${SEED:-0}"
RE="${RE:-1600}"
NUM_WINDOWS="${NUM_WINDOWS:-11}"
EPOCHS_PER_WINDOW="${EPOCHS_PER_WINDOW:-30000}"
WINDOW_SIZE="${WINDOW_SIZE:-1.0}"
BATCH_INTERIOR="${BATCH_INTERIOR:-2048}"
BATCH_IC="${BATCH_IC:-2048}"
EVAL_GRID="${EVAL_GRID:-32}"
EVAL_TIMES="${EVAL_TIMES:-4}"
LES_CS="${LES_CS:-0.1}"
PIRATE_NUM_LAYERS="${PIRATE_NUM_LAYERS:-3}"
PIRATE_HIDDEN_DIM="${PIRATE_HIDDEN_DIM:-256}"
CAUSAL_EPS="${CAUSAL_EPS:-1.0}"
CAUSAL_CHUNKS="${CAUSAL_CHUNKS:-16}"
SOAP_LR="${SOAP_LR:-1e-3}"
IC_WEIGHT="${IC_WEIGHT:-100}"
DATE_TAG="$(date +%Y%m%d)"
TAG="${TAG:-tgv_phase3_re${RE%.*}_${DATE_TAG}}"
OUTPUT_CSV="${OUTPUT_CSV:-results/tgv_phase3_re${RE%.*}.csv}"

# Optional --multi flag: launch n=5 seeds in a single batch (use sbatch --array
# instead if you want parallelism — this script keeps it sequential).
if [[ "$1" == "--multi" ]]; then
    SEEDS="${SEEDS:-0 1 7 23 42}"
else
    SEEDS="$SEED"
fi

# ----- env (HPC) -----
# On Compute Canada / Digital Research Alliance, modules and venv go here.
# Adjust per cluster:
# module load StdEnv/2023 python/3.10 cuda/12.2
source env/bin/activate

mkdir -p logs results

echo "============================================================"
echo "TGV Phase 3 paperscale — Re=${RE}, ${NUM_WINDOWS} windows × ${EPOCHS_PER_WINDOW} epochs"
echo "Tag: ${TAG}"
echo "Output: ${OUTPUT_CSV}"
echo "Seeds: ${SEEDS}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================================"

for SD in $SEEDS; do
    echo ""
    echo "===== SEED=${SD} starting at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    python -u src/taylor_green_benchmark.py \
        --seed="$SD" \
        --re="$RE" \
        --num-windows="$NUM_WINDOWS" \
        --window-size="$WINDOW_SIZE" \
        --epochs-per-window="$EPOCHS_PER_WINDOW" \
        --batch-interior="$BATCH_INTERIOR" \
        --batch-ic="$BATCH_IC" \
        --eval-grid="$EVAL_GRID" \
        --eval-times-per-window="$EVAL_TIMES" \
        --les-cs="$LES_CS" \
        --model=pirate-net \
        --pirate-num-layers="$PIRATE_NUM_LAYERS" \
        --pirate-hidden-dim="$PIRATE_HIDDEN_DIM" \
        --causal-eps="$CAUSAL_EPS" \
        --causal-chunks="$CAUSAL_CHUNKS" \
        --optimizer=soap \
        --lr="$SOAP_LR" \
        --ic-weight="$IC_WEIGHT" \
        --output-csv="$OUTPUT_CSV" \
        --tag="${TAG}_s${SD}"
    echo "===== SEED=${SD} finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
done

echo ""
echo "============================================================"
echo "All seeds done. CSV at ${OUTPUT_CSV}"
echo "============================================================"
