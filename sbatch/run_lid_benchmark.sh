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
#   - dtpinn:      DT-PINN, RBF-FD + L-BFGS + fp64 (Sharma & Shankar 2022)
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
#   ./sbatch/run_lid_benchmark.sh --seeds "0 1 7 23 42" --tag multi_seed_2026_04_26
#   ./sbatch/run_lid_benchmark.sh --output-csv results/multiseed.csv --seeds "0 1 7 23 42"
#
# Concurrent CSV safety:
#   src/lid_benchmark.py:append_csv_row uses fcntl.LOCK_EX + fstat-inside-lock
#   header decision (fixed 2026-04-26) so parallel jobs writing to the same
#   --output-csv won't double-write headers or interleave rows. This script
#   also pre-creates the CSV with a header before submission as a belt-and-
#   braces precaution against filesystems where fcntl is unreliable (NFS).
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
            # Space-separated list, e.g. --seeds "0 1 7 23 42"
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

# Build account flag for sbatch (empty string if not specified)
ACCOUNT_FLAG=""
if [[ -n "$ACCOUNT" ]]; then
    ACCOUNT_FLAG="--account=$ACCOUNT"
fi

# ============================================================================
# CONFIGURATION - Modify these arrays to control what runs
# ============================================================================

# =====================  DRY-RUN MODE  =====================
# To verify the pipeline end-to-end before launching the full 460-job sweep:
#   - cavity_methods: keep all 5 (one job per method on cavity × mlp × seed 0)
#   - kovasznay_methods + elasticity_methods: empty (loops iterate 0 times)
#   - models: only mlp
#   - seeds: only 0
# Once the dry run succeeds, restore by un-commenting the entries below.
# ===========================================================

# Cavity training methods (gradient-based, GPU required)
# These support all three models via --model flag.
# DRY-RUN: commented out so only PIELM (Section 2) submits.
cavity_methods=(
    # autodiff
    # dtpinn
    # sage
    # ropinn
    # sk-pinn
)

# Kovasznay training methods (gradient-based, GPU required)
# No analytical or pielm for Kovasznay.
kovasznay_methods=(
    # autodiff
    # dtpinn
    # sage
    # ropinn
    # sk-pinn
)

# Network architectures to benchmark
models=(
    mlp
    # tsa-pinn
    # pirate-net
)

# Random seeds for statistical significance.
# Default = 10 seeds. The first five (0, 1, 7, 23, 42) are the 2026-04-26
# audit set, verified locally on A40 to span the per-seed accuracy range
# (AD seed 0 reaches sub-0.012 best-pde-rms on Kov × PirateNet, AD seed 7
# lands at 0.032 — they bracket the paper-era single-seed (42) result).
# The remaining five (11, 19, 31, 53, 89) are arbitrary additional integers
# chosen to push n=10 for tighter mean ± std error bars on tab:main_results.
# Override with --seeds.
seeds=(
    0
    # 1
    # 7
    # 11
    # 19
    # 23
    # 31
    # 42
    # 53
    # 89
)

# Apply --seeds CLI override (space-separated string → bash array)
if [[ -n "$SEEDS_OVERRIDE" ]]; then
    read -ra seeds <<< "$SEEDS_OVERRIDE"
fi

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Default hyperparameters used by Adam-trained methods (autodiff, sage,
# ropinn, sk-pinn). Per-method overrides applied in the loops below.
EPOCHS=30000
LR="1e-3"
OPTIMIZER="adam"
DTYPE="fp32"

# Paper-faithful DT-PINN (Sharma & Shankar 2022) — RBF-FD + raw L-BFGS + lr=0.04
# + fp64 + 5K epochs. Matches temp/dt-pinn/src/dtpinn_cupy_fp64.py:117/372.
# The Python override block in main() also enforces these defaults when no
# CLI flag is passed; we set them explicitly here so the SLURM submission
# remains self-documenting and survives any future change to the override.
DTPINN_EPOCHS=5000
DTPINN_LR="0.04"
DTPINN_OPTIMIZER="lbfgs"
DTPINN_DTYPE="fp64"

TECHNIQUE="none"
TAG="${TAG_OVERRIDE:-multiseed_$(date +%Y%m%d)}"
TRACK_INTERVAL=100

# Helper: emit the four method-specific hyperparameter flags as a single
# string ready to splice into the python invocation.  Adam-trained methods
# get $OPTIMIZER/$LR/$EPOCHS/$DTYPE; --method dtpinn gets the paper-faithful
# DTPINN_* values.  Use as: $(method_hparams "$method")
method_hparams() {
    local m="$1"
    if [[ "$m" == "dtpinn" ]]; then
        echo "--optimizer=$DTPINN_OPTIMIZER --lr=$DTPINN_LR --epochs=$DTPINN_EPOCHS --dtype=$DTPINN_DTYPE"
    else
        echo "--optimizer=$OPTIMIZER --lr=$LR --epochs=$EPOCHS --dtype=$DTYPE"
    fi
}

# Output CSV (shared across all jobs, concurrent-safe via fcntl locking)
OUTPUT_CSV="${OUTPUT_CSV_OVERRIDE:-results/lid_benchmark_results.csv}"

# ============================================================================
# SLURM RESOURCE ALLOCATION
# ============================================================================

# GPU jobs: all gradient-based methods + sk-pinn
# Timing at 30K epochs (MLP / TSA-PINN), measured on H100 MIG 2g.20gb:
#   sage:        ~2.4 min / ~7 min      (fastest, auto-generated backward)
#   autodiff:    ~22 min / ~48 min
#   ropinn:      ~23 min / ~25 min
#   sk-pinn:     ~22 min / ~27 min     (N=200, sparse RKPM)
#   dtpinn:      ~8-30 min / ~25-50 min (RBF-FD + L-BFGS + 5K + fp64; per-seed
#                                       variance is intrinsic to raw L-BFGS)
# A40 reproduction (2026-04-26 audit) is ~1.5–1.8× slower per epoch, so
# autodiff × PirateNet × Kovasznay is ~50 min on A40 vs ~29 min on H100.
# The 2-hour budget gives ≥2× safety margin even on the slower platform.
GPU_TIME="0-02:00:00"
GPU_TYPE="nvidia_h100_80gb_hbm3_2g.20gb:1"
GPU_MEM="16000M"
GPU_CPUS=2

# PIELM jobs (CPU-only, numpy-based). Wall time observed on Compute Canada
# rorqual rc32* CPU nodes: ~82 s per Picard iter at the configured grid
# (max_picard_iter=50 in train_pielm), so worst-case ~68 min. Earlier
# measurements claimed "~1 min" on a faster reference machine; the 30 min
# budget hit the wall on the 2026-04-27 dry-run (cancelled at iter 22/50).
# Set to 4 h for a generous 4× buffer over worst case.
CPU_TIME="0-04:00:00"
CPU_MEM="8000M"
CPU_CPUS=4

# ============================================================================
# SETUP
# ============================================================================

mkdir -p ./logs
mkdir -p ./results

# ----------------------------------------------------------------------------
# Pre-create the output CSV with its header row, so parallel jobs racing on
# a fresh file never both decide to write a header. The Python writer
# (src/lid_benchmark.py:append_csv_row) checks file size inside the fcntl
# lock and skips the header when the file is non-empty.
#
# The header below MUST stay in lockstep with CSV_COLUMNS in
# src/lid_benchmark.py (line ~2436). We hardcode it in bash so the sbatch
# submitter does not need to import the project (numpy/torch on the login
# node is unavailable on standard HPC setups).
# ----------------------------------------------------------------------------
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
echo "PINN BENCHMARK: SLURM Job Submission"
echo "========================================================================"
echo ""
echo "Cavity methods:    ${cavity_methods[*]} pielm"
echo "Kovasznay methods: ${kovasznay_methods[*]}"
echo "Models:            ${models[*]}"
echo "Seeds:             ${seeds[*]} (${#seeds[@]} seeds)"
echo "Epochs:            $EPOCHS"
echo "LR:                $LR"
echo "Tag:               $TAG"
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
#   Chebyshev methods (autodiff, sage, ropinn): N=50
#   DT-PINN: scattered RBF-FD nodes (~N² for size-comparable to Chebyshev)
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
                        $(method_hparams "$method") \
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
                        $(method_hparams "$method") \
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

# Elasticity training methods (commented out for dry-run; un-comment to restore)
elasticity_methods=(
    # autodiff
    # dtpinn
    # sage
    # ropinn
    # sk-pinn
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
                        $(method_hparams "$method") \
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
