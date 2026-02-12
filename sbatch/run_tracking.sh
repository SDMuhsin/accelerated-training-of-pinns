#!/bin/bash
# ============================================================================
# PINN Tracking Benchmark - SLURM Submission Script
# ============================================================================
#
# Runs PINN benchmark jobs with --track enabled for ablation study plots.
# Per-epoch stats (train loss, PDE RMS, continuity/momentum RMS) are written
# to individual CSVs in results/tracking_*.csv.
#
# This script mirrors run_lid_benchmark.sh but adds --track and
# --track-interval to every job.
#
# Output tracking CSV columns:
#   problem, method, model, optimizer, lr, seed, grid_size, technique, tag,
#   epoch, train_loss, pde_rms, continuity_rms, momentum_rms,
#   u_rms_error, v_rms_error, p_rms_error
#
# Usage:
#   ./sbatch/run_tracking.sh
#   ./sbatch/run_tracking.sh --account def-myprof
#   ./sbatch/run_tracking.sh --account def-myprof --interval 200
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT="def-seokbum"
TRACK_INTERVAL=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --interval)
            TRACK_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT] [--interval TRACK_INTERVAL]"
            exit 1
            ;;
    esac
done

# Build account flag for sbatch (empty string if not specified)
ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_FLAG="--account=$ACCOUNT"
fi

# ============================================================================
# CONFIGURATION - Modify these arrays to control what runs
# ============================================================================

# Cavity training methods (gradient-based, GPU required)
# These support all three models via --model flag.
cavity_methods=(
    autodiff
    dtpinn
    analytical
    sage
    ropinn
    sk-pinn
)

# Kovasznay training methods (gradient-based, GPU required)
# No analytical or pielm for Kovasznay.
kovasznay_methods=(
    #autodiff
    #dtpinn
    #sage
    #ropinn
    #sk-pinn
)

# Network architectures to benchmark
models=(
    mlp
    tsa-pinn
    pirate-net
)

# Random seeds
seeds=(
    42
    #43
    #44
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

EPOCHS=30000
LR="1e-3"
OPTIMIZER="adam"
TECHNIQUE="none"
TAG=""

# Output CSV (shared summary, same as main benchmark)
OUTPUT_CSV="results/tracking_results.csv"

# ============================================================================
# SLURM RESOURCE ALLOCATION
# ============================================================================

# GPU jobs: all gradient-based methods + sk-pinn
# Tracking adds evaluation overhead (~5-15% depending on interval).
# 3 hours gives generous safety margin.
GPU_TIME="0-03:00:00"
GPU_TYPE="nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_MEM="16000M"
GPU_CPUS=2

# ============================================================================
# SETUP
# ============================================================================

mkdir -p ./logs
mkdir -p ./results

echo "========================================================================"
echo "PINN TRACKING BENCHMARK: SLURM Job Submission"
echo "========================================================================"
echo ""
echo "Cavity methods:    ${cavity_methods[*]}"
echo "Kovasznay methods: ${kovasznay_methods[*]}"
echo "Models:            ${models[*]}"
echo "Seeds:             ${seeds[*]}"
echo "Epochs:            $EPOCHS"
echo "Track interval:    $TRACK_INTERVAL"
echo "LR:                $LR"
echo "Output:            $OUTPUT_CSV"
if [[ -n "$ACCOUNT" ]]; then
    echo "Account:           $ACCOUNT"
fi
echo ""

job_count=0

# ============================================================================
# SECTION 1: CAVITY - GRADIENT-BASED METHODS (GPU)
# ============================================================================

echo "=============================================="
echo "Section 1: Cavity - Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${cavity_methods[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="trk_cav_${method}_${model}_s${seed}"
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
                    echo 'Problem: cavity'
                    echo 'Method: $method'
                    echo 'Model: $model'
                    echo 'Seed: $seed'
                    echo 'Track interval: $TRACK_INTERVAL'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    nvidia-smi
                    python3 -u src/lid_benchmark.py \
                        --problem=cavity \
                        --method=$method \
                        --model=$model \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --epochs=$EPOCHS \
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

# ============================================================================
# SECTION 2: KOVASZNAY FLOW - GRADIENT-BASED METHODS (GPU)
# ============================================================================

echo ""
echo "=============================================="
echo "Section 2: Kovasznay - Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${kovasznay_methods[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="trk_kov_${method}_${model}_s${seed}"
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
                    echo 'Problem: kovasznay'
                    echo 'Method: $method'
                    echo 'Model: $model'
                    echo 'Seed: $seed'
                    echo 'Track interval: $TRACK_INTERVAL'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    nvidia-smi
                    python3 -u src/lid_benchmark.py \
                        --problem=kovasznay \
                        --method=$method \
                        --model=$model \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --epochs=$EPOCHS \
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

# ============================================================================
# SECTION 3: ELASTICITY - GRADIENT-BASED METHODS (GPU)
# ============================================================================

# Elasticity training methods
elasticity_methods=(
    #autodiff
    #dtpinn
    #sage
    #ropinn
    #sk-pinn
)

echo ""
echo "=============================================="
echo "Section 3: Elasticity - Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${elasticity_methods[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="trk_ela_${method}_${model}_s${seed}"
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
                    echo 'Problem: elasticity'
                    echo 'Method: $method'
                    echo 'Model: $model'
                    echo 'Seed: $seed'
                    echo 'Track interval: $TRACK_INTERVAL'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    nvidia-smi
                    python3 -u src/lid_benchmark.py \
                        --problem=elasticity \
                        --method=$method \
                        --model=$model \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --epochs=$EPOCHS \
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

# ============================================================================
# SUMMARY
# ============================================================================

n_cavity_jobs=$((${#cavity_methods[@]} * ${#models[@]} * ${#seeds[@]}))
n_kovasznay_jobs=$((${#kovasznay_methods[@]} * ${#models[@]} * ${#seeds[@]}))
n_elasticity_jobs=$((${#elasticity_methods[@]} * ${#models[@]} * ${#seeds[@]}))

echo ""
echo "========================================================================"
echo "ALL JOBS SUBMITTED"
echo "========================================================================"
echo ""
echo "CAVITY - GRADIENT-BASED (GPU):"
for method in "${cavity_methods[@]}"; do
    echo "  - $method  x  ${#models[@]} models  x  ${#seeds[@]} seeds  =  $((${#models[@]} * ${#seeds[@]})) jobs"
done
echo "  Total: $n_cavity_jobs jobs"
echo ""
echo "KOVASZNAY - GRADIENT-BASED (GPU):"
for method in "${kovasznay_methods[@]}"; do
    echo "  - $method  x  ${#models[@]} models  x  ${#seeds[@]} seeds  =  $((${#models[@]} * ${#seeds[@]})) jobs"
done
echo "  Total: $n_kovasznay_jobs jobs"
echo ""
echo "ELASTICITY - GRADIENT-BASED (GPU):"
for method in "${elasticity_methods[@]}"; do
    echo "  - $method  x  ${#models[@]} models  x  ${#seeds[@]} seeds  =  $((${#models[@]} * ${#seeds[@]})) jobs"
done
echo "  Total: $n_elasticity_jobs jobs"
echo ""
echo "TOTAL JOBS: $job_count"
echo ""
echo "Track interval:  $TRACK_INTERVAL epochs"
echo "Summary CSV:     ./$OUTPUT_CSV"
echo "Tracking CSVs:   ./results/tracking_*.csv"
echo "Logs:            ./logs/trk_cav_*  ./logs/trk_kov_*  ./logs/trk_ela_*"
echo "========================================================================"
