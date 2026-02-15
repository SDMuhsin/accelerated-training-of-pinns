#!/bin/bash
# ============================================================================
# PINN Benchmark Suite - SLURM Submission Script
# ============================================================================
#
# Submits PINN benchmark jobs to SLURM for two PDE problems:
#   1. Lid-Driven Cavity (NS+Smagorinsky, Re=1000)
#   2. Kovasznay Flow (constant-viscosity NS, Re=40)
#
# Each job runs src/lid_benchmark.py for one (problem, method, model, seed)
# combination and appends results to a shared CSV with file locking.
#
# Cavity methods:
#   - autodiff:    Plain autograd PINN (Chebyshev N=50)
#   - dtpinn:      DT-PINN, spectral matrices (Chebyshev N=50)
#   - analytical:  Analytical Jacobian backward (Chebyshev N=50)
#   - sage:        SAGE auto-generated backward (Chebyshev N=50)   [OUR METHOD]
#   - ropinn:      RoPINN, region-optimized (Chebyshev N=50)
#   - sk-pinn:     SK-PINN, sparse RKPM matrices (uniform N=200)
#   - pielm:       PIELM extreme learning machine (own architecture, no --model)
#
# Kovasznay methods:
#   - autodiff:    Plain autograd PINN (Chebyshev N=30)
#   - dtpinn:      DT-PINN, spectral matrices (Chebyshev N=30)
#   - sage:        SAGE auto-generated backward (Chebyshev N=30)   [OUR METHOD]
#   - ropinn:      RoPINN, region-optimized (Chebyshev N=30)
#   - sk-pinn:     SK-PINN, sparse RKPM matrices (uniform N=150)
#
# Models:
#   - mlp:         6-layer/64-unit tanh MLP (21,187 params)
#   - tsa-pinn:    Trainable sinusoidal activations (21,571 params)
#   - pirate-net:  Adaptive residual gating (20,983 params)
#
# Output CSV columns:
#   timestamp, problem, method, model, optimizer, lr, epochs, seed, grid_size,
#   technique, tag, train_time_s, train_time_min, peak_gpu_memory_mb,
#   gpu_memory_reserved_mb, ms_per_epoch, n_params, pde_rms,
#   continuity_rms, momentum_rms, final_loss, best_epoch, status, device,
#   gpu_name, pytorch_version
#
# Note: --track is enabled so the best model (by PDE RMS) seen during
# training is restored for final evaluation. pde_rms reflects the best
# epoch, not the final epoch. best_epoch records which epoch was used.
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

# Cavity training methods (gradient-based, GPU required)
# These support all three models via --model flag.
cavity_methods=(
    #autodiff
    #dtpinn
    #analytical
    #sage
    #ropinn
    #sk-pinn
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
TRACK_INTERVAL=100

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
echo "PINN BENCHMARK: SLURM Job Submission"
echo "========================================================================"
echo ""
echo "Cavity methods:    ${cavity_methods[*]} pielm"
echo "Kovasznay methods: ${kovasznay_methods[*]}"
echo "Models:            ${models[*]}"
echo "Seeds:             ${seeds[*]}"
echo "Epochs:            $EPOCHS"
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
# All gradient methods use GPU and support all three model architectures.
# Grid size is determined automatically per method:
#   Chebyshev methods (autodiff, dtpinn, analytical, sage, ropinn): N=50
#   SK-PINN: N=200 (sparse RKPM)

echo "=============================================="
echo "Section 1: Cavity - Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${cavity_methods[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="cav_${method}_${model}_s${seed}"
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
# SECTION 2: CAVITY - PIELM (CPU)
# ============================================================================
# PIELM has its own architecture (not compatible with --model flag).
# Very fast (~1 min), CPU-only (numpy-based). Cavity only.

echo ""
echo "=============================================="
echo "Section 2: Cavity - PIELM (CPU)"
echo "=============================================="

for seed in "${seeds[@]}"; do
    job_name="cav_pielm_s${seed}"
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
# SECTION 3: KOVASZNAY FLOW - GRADIENT-BASED METHODS (GPU)
# ============================================================================
# Kovasznay flow (Re=40, constant-viscosity NS, exact analytical solution).
# Non-square domain [-0.5,1.0] x [-0.5,1.5].
# Grid size is determined automatically per method:
#   Chebyshev methods (autodiff, dtpinn, sage, ropinn): N=30
#   SK-PINN: N=150 (sparse RKPM)
# Timing at 30K epochs (MLP):
#   sage:     ~2.4 min  (fastest, auto-generated backward)
#   dtpinn:   ~7.7 min
#   autodiff: ~13.4 min
#   ropinn:   ~16.6 min
#   sk-pinn:  ~TBD      (N=150, sparse RKPM)

echo ""
echo "=============================================="
echo "Section 3: Kovasznay - Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${kovasznay_methods[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="kov_${method}_${model}_s${seed}"
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
# SECTION 4: ELASTICITY - GRADIENT-BASED METHODS (GPU)
# ============================================================================
# 2D Linear Elasticity (Navier-Cauchy, manufactured solution on [0,1]²).
# 2 outputs (displacements ux, uy), no pressure, no Smagorinsky.
# Grid size is determined automatically per method:
#   Chebyshev methods (autodiff, dtpinn, sage, ropinn): N=30
#   SK-PINN: N=100 (sparse RKPM)

# Elasticity training methods
elasticity_methods=(
    autodiff
    dtpinn
    sage
    ropinn
    sk-pinn
)

echo ""
echo "=============================================="
echo "Section 4: Elasticity - Gradient-Based Methods (GPU)"
echo "=============================================="

for method in "${elasticity_methods[@]}"; do
    for model in "${models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="ela_${method}_${model}_s${seed}"
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
n_pielm_jobs=${#seeds[@]}
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
echo "CAVITY - PIELM (CPU):"
echo "  - pielm  x  ${#seeds[@]} seeds  =  $n_pielm_jobs jobs"
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
echo "Results:  ./$OUTPUT_CSV"
echo "Logs:     ./logs/cav_*  ./logs/kov_*  ./logs/ela_*"
echo "========================================================================"
