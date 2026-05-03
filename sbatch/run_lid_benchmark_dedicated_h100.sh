#!/bin/bash
# ============================================================================
# EXP-10: Dedicated-H100 wall-clock sanity rerun
# ============================================================================
#
# Purpose: re-measure the SAGE / Spectral-AD / AutoDiff matched-protocol
# wall-clock on a dedicated (non-partitioned) H100 80GB. The headline ratios
# in the paper are collected on H100 MIG 2g.20gb (~2/7 of a full H100); this
# run checks whether those ratios survive on a dedicated GPU.
#
# Tracker entry: paper_rewrite/deferred_experiments.md, EXP-10 (cycle 12 F4).
# Body prose does not reference this run.
#
# Scope:
#   - Problems:    cavity, kovasznay, elasticity            (3)
#   - Models:      mlp, pirate-net                          (2)
#   - Methods:     sage, chebyshev-pinn (Spectral-AD),
#                  autodiff                                 (3)
#   - Seeds:       0 1 7 23 42                              (5)
#   - Total jobs:  3 x 2 x 3 x 5 = 90 GPU jobs.
#
# Hardware:
#   - SLURM GPU type: dedicated H100 80GB, NO MIG partition. The exact
#     SLURM resource name varies by cluster. Edit GPU_TYPE below to match
#     your site's GRES name. Common patterns:
#       nvidia_h100_80gb_hbm3:1          (NVHPC-style)
#       h100:1                            (Compute Canada DRAC)
#       h100_80gb:1
#     The default `nvidia_h100_80gb_hbm3:1` mirrors run_lid_benchmark.sh
#     with the `_2g.20gb` MIG suffix removed.
#
# Output CSV:
#   - Default: results/lid_benchmark_results.csv (same as the main sweep,
#     distinguishable by tag). Override with --output-csv.
#
# Tag:
#   - Default: dedicated_h100_<date>. Override with --tag.
#
# Aggregation:
#   - After the runs land in the CSV, the headline-ratio comparison is
#     a CSV-level pass: filter rows by tag = dedicated_h100_<date>, group
#     by (problem, model, method), compute mean train_time_s, then compute
#     SAGE-vs-Spectral-AD and SAGE-vs-AutoDiff ratios per cell. Compare
#     to the same ratios at tag = multiseed_<date> / canpinn_hpc_<date>.
#
# Usage (run on the SLURM login node):
#   ./sbatch/run_lid_benchmark_dedicated_h100.sh
#   ./sbatch/run_lid_benchmark_dedicated_h100.sh --account def-myprof
#   ./sbatch/run_lid_benchmark_dedicated_h100.sh --seeds "0 1 7 23 42" \
#       --tag dedicated_h100_2026_05_01
#   ./sbatch/run_lid_benchmark_dedicated_h100.sh \
#       --output-csv results/dedicated_h100_only.csv
#
# Dev-box note: do NOT run this script on the dev box. It submits SLURM
# jobs and is meant for the HPC login node only.
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT="def-seokbum"
SEEDS_OVERRIDE=""
TAG_OVERRIDE=""
OUTPUT_CSV_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --seeds)
            SEEDS_OVERRIDE="$2"
            shift 2
            ;;
        --tag)
            TAG_OVERRIDE="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--seeds 'S1 S2 ...'] [--tag TAG] [--output-csv PATH]"
            exit 1
            ;;
    esac
done

ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_FLAG="--account=$ACCOUNT"
fi

# ============================================================================
# CONFIGURATION
# ============================================================================

# Matched-protocol triple. SAGE (auto-generated backward), chebyshev-pinn
# (the Chebyshev pseudospectral PINN with torch.autograd backward — labeled
# "Spectral-AD" in the paper's tab:main_results), and plain autodiff PINN.
# These three share forward kernel, optimizer (Adam), precision (fp32), and
# epoch budget (30k); their wall-clock differs only in the gradient engine.
methods_per_problem=(
    sage
    chebyshev-pinn
    autodiff
)

# All three benchmark problems.
problems=(
    cavity
    kovasznay
    elasticity
)

# Both architectures from the headline table.
models=(
    mlp
    pirate-net
)

# Random seeds — the standard 5-seed audit set (matches multiseed_20260427
# and canpinn_hpc_20260428, so the dedicated-H100 ratios can be compared
# row-for-row to the MIG 2g.20gb ratios).
seeds=(
    0
    1
    7
    23
    42
)

if [[ -n "$SEEDS_OVERRIDE" ]]; then
    read -ra seeds <<< "$SEEDS_OVERRIDE"
fi

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Matched-protocol Adam settings — identical to multiseed_20260427 and
# canpinn_hpc_20260428 for the SAGE / chebyshev-pinn / autodiff rows.
EPOCHS=30000
LR="1e-3"
OPTIMIZER="adam"
DTYPE="fp32"

TECHNIQUE="none"
TAG="${TAG_OVERRIDE:-dedicated_h100_$(date +%Y%m%d)}"
TRACK_INTERVAL=100

# ============================================================================
# SLURM RESOURCE ALLOCATION
# ============================================================================

# Dedicated H100 80GB (no MIG). Edit GPU_TYPE if your cluster uses a
# different GRES name (see header comment for common patterns).
#
# Time budget: dedicated H100 is roughly 2-3x faster than MIG 2g.20gb on
# the AutoDiff rows. The MIG runs fit in 3 h; on dedicated H100 they
# should comfortably fit in 2 h. We set 3 h here as a safety margin
# (matches run_lid_benchmark.sh) — over-allocating wall-time is safer
# than risking a kill on the slow PirateNet AutoDiff cells.
GPU_TIME="0-03:00:00"
GPU_TYPE="nvidia_h100_80gb_hbm3:1"
GPU_MEM="32000M"
GPU_CPUS=4

# ============================================================================
# SETUP
# ============================================================================

mkdir -p ./logs
mkdir -p ./results

OUTPUT_CSV="${OUTPUT_CSV_OVERRIDE:-results/lid_benchmark_results.csv}"

# Pre-create the output CSV with its header row (same scheme as
# run_lid_benchmark.sh — keeps parallel jobs from racing on first-write).
CSV_HEADER='timestamp,problem,method,model,optimizer,lr,epochs,seed,grid_size,technique,tag,train_time_s,train_time_min,peak_gpu_memory_mb,gpu_memory_reserved_mb,ms_per_epoch,n_params,pde_rms,continuity_rms,momentum_rms,final_loss,best_epoch,status,device,gpu_name,pytorch_version'

if [[ ! -f "$OUTPUT_CSV" ]] || [[ ! -s "$OUTPUT_CSV" ]]; then
    echo "Pre-creating $OUTPUT_CSV with header row..."
    mkdir -p "$(dirname "$OUTPUT_CSV")"
    if ! printf '%s\n' "$CSV_HEADER" > "$OUTPUT_CSV"; then
        echo "ERROR: failed to write $OUTPUT_CSV"
        exit 1
    fi
    echo "Wrote header ($(awk -F',' '{print NF; exit}' "$OUTPUT_CSV") cols) to $OUTPUT_CSV"
else
    echo "Reusing existing $OUTPUT_CSV ($(wc -l < "$OUTPUT_CSV") lines)."
fi

echo "========================================================================"
echo "EXP-10: Dedicated-H100 wall-clock sanity rerun"
echo "========================================================================"
echo ""
echo "Problems:   ${problems[*]}"
echo "Models:     ${models[*]}"
echo "Methods:    ${methods_per_problem[*]}"
echo "Seeds:      ${seeds[*]} (${#seeds[@]} seeds)"
echo "Epochs:     $EPOCHS  ($OPTIMIZER, lr=$LR, $DTYPE)"
echo "Tag:        $TAG"
echo "Output:     $OUTPUT_CSV"
echo "GPU type:   $GPU_TYPE  (dedicated, no MIG)"
echo "Time/job:   $GPU_TIME"
if [[ -n "$ACCOUNT" ]]; then
    echo "Account:    $ACCOUNT"
fi
echo ""

job_count=0

# ============================================================================
# JOB SUBMISSION LOOP
# ============================================================================

for problem in "${problems[@]}"; do
    case "$problem" in
        cavity)     prefix="cav" ;;
        kovasznay)  prefix="kov" ;;
        elasticity) prefix="ela" ;;
        *)
            echo "ERROR: unknown problem '$problem'"
            exit 1
            ;;
    esac

    echo "=============================================="
    echo "Problem: $problem"
    echo "=============================================="

    for method in "${methods_per_problem[@]}"; do
        for model in "${models[@]}"; do
            for seed in "${seeds[@]}"; do
                job_name="${prefix}_${method}_${model}_s${seed}_dedh100"
                log_file="./logs/${job_name}"

                echo "Submitting: $job_name"
                sbatch \
                    $ACCOUNT_FLAG \
                    --nodes=1 \
                    --ntasks-per-node=1 \
                    --cpus-per-task=$GPU_CPUS \
                    --gpus=$GPU_TYPE \
                    --mem=$GPU_MEM \
                    --time=$GPU_TIME \
                    --job-name=$job_name \
                    --output=${log_file}-%N-%j.out \
                    --error=${log_file}-%N-%j.err \
                    --wrap="
                        module load scipy-stack cuda cudnn
                        module load arrow
                        source ./env/bin/activate
                        echo '========================================'
                        echo 'Job: $job_name'
                        echo 'Problem: $problem'
                        echo 'Method: $method'
                        echo 'Model: $model'
                        echo 'Seed: $seed'
                        echo 'Started: '\$(date)
                        echo '========================================'
                        nvidia-smi
                        python3 -u src/lid_benchmark.py \
                            --problem=$problem \
                            --method=$method \
                            --model=$model \
                            --optimizer=$OPTIMIZER \
                            --lr=$LR \
                            --epochs=$EPOCHS \
                            --dtype=$DTYPE \
                            --seed=$seed \
                            --technique=$TECHNIQUE \
                            --output-csv=$OUTPUT_CSV \
                            --tag=$TAG \
                            --track \
                            --track-interval=$TRACK_INTERVAL
                        echo '========================================'
                        echo 'Finished: '\$(date)
                        echo '========================================'
                    "
                ((job_count++))
            done
        done
    done
done

# ============================================================================
# SUMMARY
# ============================================================================

n_per_problem=$((${#methods_per_problem[@]} * ${#models[@]} * ${#seeds[@]}))
n_total=$(( ${#problems[@]} * n_per_problem ))

echo ""
echo "========================================================================"
echo "ALL JOBS SUBMITTED"
echo "========================================================================"
echo ""
echo "Per problem: ${#methods_per_problem[@]} methods x ${#models[@]} models x ${#seeds[@]} seeds = $n_per_problem jobs"
echo "Problems:    ${#problems[@]} (${problems[*]})"
echo "TOTAL JOBS:  $job_count   (expected $n_total)"
echo ""
echo "Tag:      $TAG"
echo "Results:  ./$OUTPUT_CSV"
echo "Logs:     ./logs/cav_*_dedh100*  ./logs/kov_*_dedh100*  ./logs/ela_*_dedh100*"
echo "========================================================================"
