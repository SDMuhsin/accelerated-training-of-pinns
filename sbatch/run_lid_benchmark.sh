#!/bin/bash
# ============================================================================
# Lid-Driven Cavity Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Submits lid-driven cavity NS+Smagorinsky PINN benchmark jobs to SLURM.
# Each job runs src/lid_benchmark.py for one (method, model, seed) combination
# and appends results to a shared CSV with file locking for concurrent safety.
#
# Methods and their characteristics:
#   - autodiff:    Plain autograd PINN (Chebyshev N=50)
#   - dtpinn:      DT-PINN, spectral matrices (Chebyshev N=50)
#   - analytical:  Analytical Jacobian backward (Chebyshev N=50)
#   - sage:        SAGE auto-generated backward (Chebyshev N=50)   [OUR METHOD]
#   - ropinn:      RoPINN, region-optimized (Chebyshev N=50)
#   - sk-pinn:     SK-PINN, sparse RKPM matrices (uniform N=200)
#   - pielm:       PIELM extreme learning machine (own architecture, no --model)
#
# Models:
#   - mlp:         6-layer/64-unit tanh MLP (21,187 params)
#   - tsa-pinn:    Trainable sinusoidal activations (21,571 params)
#   - pirate-net:  Adaptive residual gating (20,983 params)
#
# Output CSV columns:
#   timestamp, method, model, optimizer, lr, epochs, seed, grid_size,
#   technique, tag, train_time_s, train_time_min, peak_gpu_memory_mb,
#   gpu_memory_reserved_mb, ms_per_epoch, n_params, pde_rms,
#   continuity_rms, momentum_rms, final_loss, status, device,
#   gpu_name, pytorch_version
#
# Usage:
#   ./sbatch/run_lid_benchmark.sh
#   ./sbatch/run_lid_benchmark.sh --account def-myprof
#
# ============================================================================

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

ACCOUNT="def-seokbum"

while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--account SLURM_ACCOUNT]"
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

# Training methods (gradient-based, GPU required)
# These support all three models via --model flag.
gradient_methods=(
    #autodiff
    #dtpinn
    #analytical
    sage
    #ropinn
    #sk-pinn
)

# Network architectures to benchmark
models=(
    mlp
    tsa-pinn
    pirate-net
)

# Random seeds for statistical significance (>=3 recommended)
seeds=(
    42
    #43
    #44
    #45
    #46
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

EPOCHS=30000
LR="1e-3"
OPTIMIZER="adam"
TECHNIQUE="none"
TAG=""

# Output CSV (shared across all jobs, concurrent-safe via fcntl locking)
OUTPUT_CSV="results/lid_benchmark_results.csv"

# ============================================================================
# SLURM RESOURCE ALLOCATION
# ============================================================================

# GPU jobs: all gradient-based methods + sk-pinn
# Timing at 30K epochs (MLP / TSA-PINN):
#   sage:        ~2.4 min / ~7 min      (fastest, auto-generated backward)
#   analytical:  ~2.6 min / ~7 min
#   dtpinn:      ~11 min / ~15 min
#   autodiff:    ~22 min / ~48 min
#   ropinn:      ~23 min / ~25 min
#   sk-pinn:     ~22 min / ~27 min     (N=200, sparse RKPM)
# 2 hours gives >3x safety margin for the slowest case.
GPU_TIME="0-02:00:00"
GPU_TYPE="nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_MEM="16000M"
GPU_CPUS=2

# PIELM jobs (CPU-only, numpy-based, takes ~1 min)
CPU_TIME="0-00:30:00"
CPU_MEM="8000M"
CPU_CPUS=4

# ============================================================================
# SETUP
# ============================================================================

mkdir -p ./logs
mkdir -p ./results

echo "========================================================================"
echo "LID-DRIVEN CAVITY BENCHMARK: SLURM Job Submission"
echo "========================================================================"
echo ""
echo "Methods:  ${gradient_methods[*]} pielm"
echo "Models:   ${models[*]}"
echo "Seeds:    ${seeds[*]}"
echo "Epochs:   $EPOCHS"
echo "LR:       $LR"
echo "Output:   $OUTPUT_CSV"
if [[ -n "$ACCOUNT" ]]; then
    echo "Account:  $ACCOUNT"
fi
echo ""

job_count=0

# ============================================================================
# SECTION 1: GRADIENT-BASED METHODS (GPU)
# ============================================================================
# All gradient methods use GPU and support all three model architectures.
# Grid size is determined automatically per method:
#   Chebyshev methods (autodiff, dtpinn, analytical, sage, ropinn): N=50
#   SK-PINN: N=200 (sparse RKPM)

echo "=============================================="
echo "Section 1: Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${gradient_methods[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="lid_${method}_${model}_s${seed}"
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
                    echo 'Method: $method'
                    echo 'Model: $model'
                    echo 'Seed: $seed'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    nvidia-smi
                    python3 -u src/lid_benchmark.py \
                        --method=$method \
                        --model=$model \
                        --optimizer=$OPTIMIZER \
                        --lr=$LR \
                        --epochs=$EPOCHS \
                        --seed=$seed \
                        --technique=$TECHNIQUE \
                        --output-csv=$OUTPUT_CSV \
                        --tag=$TAG
                    echo '========================================'
                    echo 'Finished: '\$(date)
                    echo '========================================'
                "
            ((job_count++))
        done
    done
done

# ============================================================================
# SECTION 2: PIELM (CPU)
# ============================================================================
# PIELM has its own architecture (not compatible with --model flag).
# Very fast (~1 min), CPU-only (numpy-based).

echo ""
echo "=============================================="
echo "Section 2: PIELM (CPU)"
echo "=============================================="

for seed in "${seeds[@]}"; do
    job_name="lid_pielm_s${seed}"
    log_file="./logs/${job_name}"

    echo "Submitting: $job_name"
    sbatch \
        $ACCOUNT_FLAG \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=$CPU_CPUS \
        --mem=$CPU_MEM \
        --time=$CPU_TIME \
        --job-name=$job_name \
        --output=${log_file}-%N-%j.out \
        --error=${log_file}-%N-%j.err \
        --wrap="
            module load scipy-stack
            module load arrow
            source ./env/bin/activate
            echo '========================================'
            echo 'Job: $job_name'
            echo 'Method: pielm'
            echo 'Seed: $seed'
            echo 'Started: '\$(date)
            echo '========================================'
            python3 -u src/lid_benchmark.py \
                --method=pielm \
                --seed=$seed \
                --epochs=$EPOCHS \
                --output-csv=$OUTPUT_CSV \
                --tag=$TAG
            echo '========================================'
            echo 'Finished: '\$(date)
            echo '========================================'
        "
    ((job_count++))
done

# ============================================================================
# SUMMARY
# ============================================================================

n_gradient_jobs=$((${#gradient_methods[@]} * ${#models[@]} * ${#seeds[@]}))
n_pielm_jobs=${#seeds[@]}

echo ""
echo "========================================================================"
echo "ALL JOBS SUBMITTED"
echo "========================================================================"
echo ""
echo "GRADIENT-BASED METHODS (GPU):"
for method in "${gradient_methods[@]}"; do
    echo "  - $method  x  ${#models[@]} models  x  ${#seeds[@]} seeds  =  $((${#models[@]} * ${#seeds[@]})) jobs"
done
echo "  Total GPU jobs: $n_gradient_jobs"
echo ""
echo "PIELM (CPU):"
echo "  - pielm  x  ${#seeds[@]} seeds  =  $n_pielm_jobs jobs"
echo ""
echo "TOTAL JOBS: $job_count"
echo ""
echo "Results:  ./$OUTPUT_CSV"
echo "Logs:     ./logs/lid_*"
echo "========================================================================"
