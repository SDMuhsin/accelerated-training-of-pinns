#!/bin/bash
# ============================================================================
# Phase 6: Faithful DT-PINN multi-seed sweep — local A40 (HPC unavailable)
# ============================================================================
#
# Re-runs all 9 tab:main_results DT-PINN cells (3 problems × 3 architectures)
# with the new RBF-FD + fp64 + L-BFGS implementation. 5 seeds per cell ⇒ 45
# total runs.
#
# Each run uses:
#   --method dtpinn                # the new (Sharma & Shankar 2022) variant
#   --epochs 5000                  # paper-faithful LBFGS budget (default for dtpinn)
#   --rbf-fd-order 4               # paper recommendation
#   --num-nodes (default)          # ≈ grid_size² to match Chebyshev N
#   --track --track-interval 100   # match the rest of the table's methodology
#   --tag dtpinn_faithful_<date>   # so these rows are easy to filter
#
# Round-robin between GPU 0 and GPU 1. Each GPU runs jobs sequentially.
#
# Usage from project root:
#   ./scripts/run_dtpinn_faithful_a40.sh
#   ./scripts/run_dtpinn_faithful_a40.sh --tag custom_tag --output-csv path.csv
#
# Estimated wall-clock: ~11 hours on 2× A40 (extrapolated from ~30 min smoke
# at N=900, p=4). Safety margins are roomy because L-BFGS step counts are
# fixed at 5K and per-step cost is bounded by the sparse-matmul nnz.
# ============================================================================

set -e
cd "$(dirname "$0")/.."

# ---- args ----
TAG_OVERRIDE=""
OUTPUT_CSV_OVERRIDE=""
SEEDS_OVERRIDE=""
EPOCHS_OVERRIDE=""
GPUS="0,1"  # default: use both A40 GPUs in parallel; pass "--gpus 0" to use just one

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)         TAG_OVERRIDE="$2"; shift 2 ;;
        --output-csv)  OUTPUT_CSV_OVERRIDE="$2"; shift 2 ;;
        --seeds)       SEEDS_OVERRIDE="$2"; shift 2 ;;
        --epochs)      EPOCHS_OVERRIDE="$2"; shift 2 ;;
        --gpus)        GPUS="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--tag TAG] [--output-csv PATH] [--seeds 'S1 S2 ...'] [--epochs N] [--gpus '0,1'|'0'|'1']"
            exit 1
            ;;
    esac
done

IFS=',' read -ra gpu_arr <<< "$GPUS"

TAG="${TAG_OVERRIDE:-dtpinn_faithful_$(date +%Y%m%d)}"
OUTPUT_CSV="${OUTPUT_CSV_OVERRIDE:-results/lid_benchmark_results.csv}"
EPOCHS="${EPOCHS_OVERRIDE:-5000}"

if [[ -n "$SEEDS_OVERRIDE" ]]; then
    read -ra seeds <<< "$SEEDS_OVERRIDE"
else
    seeds=(0 1 7 23 42)
fi

problems=(cavity kovasznay elasticity)
models=(mlp tsa-pinn pirate-net)

# ---- pre-create CSV header (concurrent-safe) ----
mkdir -p ./logs ./results
if [[ ! -f "$OUTPUT_CSV" ]] || [[ ! -s "$OUTPUT_CSV" ]]; then
    echo "Pre-creating $OUTPUT_CSV with header row..."
    python3 - <<EOF
import csv, sys, os
sys.path.insert(0, os.path.abspath("."))
from src.lid_benchmark import CSV_COLUMNS
os.makedirs(os.path.dirname("$OUTPUT_CSV") or ".", exist_ok=True)
with open("$OUTPUT_CSV", 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    w.writeheader()
print(f"Wrote header ({len(CSV_COLUMNS)} cols) to $OUTPUT_CSV")
EOF
else
    echo "Reusing existing $OUTPUT_CSV ($(wc -l < "$OUTPUT_CSV") lines)."
fi

source env/bin/activate

# ---- build the job list ----
jobs=()
for problem in "${problems[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            jobs+=("$problem|$model|$seed")
        done
    done
done

echo "==============================================================="
echo "Phase 6: Faithful DT-PINN sweep ($((${#jobs[@]})) runs)"
echo "  problems: ${problems[*]}"
echo "  models:   ${models[*]}"
echo "  seeds:    ${seeds[*]}"
echo "  epochs:   $EPOCHS"
echo "  tag:      $TAG"
echo "  csv:      $OUTPUT_CSV"
echo "==============================================================="

# ---- partition jobs across the requested GPUs (round-robin) ----
declare -A gpu_jobs_str
for gpu in "${gpu_arr[@]}"; do
    gpu_jobs_str[$gpu]=""
done
for i in "${!jobs[@]}"; do
    gpu="${gpu_arr[$((i % ${#gpu_arr[@]}))]}"
    gpu_jobs_str[$gpu]+="${jobs[$i]} "
done

run_gpu_jobs() {
    local gpu="$1"
    shift
    local jobs_arr=("$@")
    local log_path="logs/dtpinn_faithful_gpu${gpu}_${TAG}.log"
    echo "[gpu${gpu}] starting ${#jobs_arr[@]} jobs → $log_path"
    {
        echo "=== gpu${gpu} sweep ($(date)) ==="
        for spec in "${jobs_arr[@]}"; do
            IFS='|' read -r problem model seed <<< "$spec"
            echo
            echo "--- gpu${gpu}: $problem × $model × seed=$seed ---"
            CUDA_VISIBLE_DEVICES=$gpu python3 -u src/lid_benchmark.py \
                --problem "$problem" \
                --method dtpinn \
                --model "$model" \
                --seed "$seed" \
                --epochs "$EPOCHS" \
                --output-csv "$OUTPUT_CSV" \
                --tag "$TAG" \
                --track --track-interval 100 \
                || echo "RUN FAILED: $problem $model $seed"
        done
        echo
        echo "=== gpu${gpu} finished ($(date)) ==="
    } > "$log_path" 2>&1
}

echo "Launching ${#gpu_arr[@]} GPU(s) in parallel:"
declare -A pids
for gpu in "${gpu_arr[@]}"; do
    read -ra spec_arr <<< "${gpu_jobs_str[$gpu]}"
    echo "  gpu $gpu: ${#spec_arr[@]} jobs"
    run_gpu_jobs "$gpu" "${spec_arr[@]}" &
    pids[$gpu]=$!
done
for gpu in "${gpu_arr[@]}"; do
    wait "${pids[$gpu]}"
done
echo "==============================================================="
echo "Phase 6 sweep complete. CSV: $OUTPUT_CSV"
echo "==============================================================="
