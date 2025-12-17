#!/bin/bash

echo "========================================================================"
echo "RERUN: Deep SPECTO-ELM Models Only (with regularization fix)"
echo "========================================================================"
echo ""
echo "This script reruns only the deep layer SPECTO-ELM models (deep2, deep3, deep4)"
echo "which were affected by the ill-conditioned matrix bug on older scipy versions."
echo ""
echo "Fix applied: Regularization increased from 1e-10 to 1e-8 in _solve_lstsq_robust()"
echo "Results are appended to ./results/by_task/<task>.csv"
echo ""

# ============================================================================
# TASK DEFINITIONS (same as main script)
# ============================================================================

# RBF-FD Tasks (scattered points, various domains)
rbf_tasks=(
    poisson-rbf-fd
    nonlinear-poisson
    nonlinear-poisson-rbf-fd
    poisson-disk-sin
    poisson-disk-quadratic
    poisson-square-constant
    poisson-square-sin
    nonlinear-poisson-disk-sin
    nonlinear-poisson-square-constant
    nonlinear-poisson-square-sin
    laplace-disk
    laplace-square
    heat-equation
    heat-fast-decay
)

# Spectral Tasks - 2D Smooth (Chebyshev collocation)
spectral_2d_smooth=(
    spectral-poisson-square
    spectral-laplace-square
    spectral-nonlinear-poisson-square
)

# Spectral Tasks - 3D (test scalability)
spectral_3d=(
    spectral-poisson-cube
    spectral-laplace-cube
    spectral-nonlinear-poisson-cube
)

# Spectral Tasks - Localized Features (favor adaptive methods)
spectral_localized=(
    spectral-poisson-peaked
    spectral-boundary-layer
    spectral-poisson-corner
)

# Combine all tasks
all_tasks=(
    "${rbf_tasks[@]}"
    "${spectral_2d_smooth[@]}"
    "${spectral_3d[@]}"
    "${spectral_localized[@]}"
)

# ============================================================================
# MODEL DEFINITIONS - ONLY DEEP VARIANTS
# ============================================================================

# Only the deep SPECTO-ELM variants that were affected by the bug
deep_models=(
    dt-elm-pinn-deep2    # 2 layers [100, 100] with skip connections
    dt-elm-pinn-deep3    # 3 layers [100, 100, 100] with skip connections
    dt-elm-pinn-deep4    # 4 layers [100, 100, 100, 100] with skip connections
)

# ============================================================================
# CONFIGURATION
# ============================================================================

seeds=(42)

# Time allocation - reduced since deep models should now be fast
CPU_TIME="0-00:30:00"    # 30 minutes (should complete in <5 min with fix)

# Create logs and results directories
mkdir -p ./logs/rerun
mkdir -p ./results/by_task

job_count=0

# ============================================================================
# SUBMIT DEEP SPECTO-ELM JOBS (CPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Submitting Deep SPECTO-ELM Rerun Jobs (CPU)"
echo "=============================================="

for task in "${all_tasks[@]}"; do
    csv_file="./results/by_task/${task}.csv"

    for model in "${deep_models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="${model}_${task}_s${seed}_rerun"
            log_file="./logs/rerun/${job_name}"

            echo "Submitting: $job_name"
            sbatch \
                --nodes=1 \
                --ntasks-per-node=1 \
                --cpus-per-task=4 \
                --mem=16000M \
                --time=$CPU_TIME \
                --output=${log_file}-%N-%j.out \
                --error=${log_file}-%N-%j.err \
                --wrap="
                    module load scipy-stack
                    module load arrow
                    source ./env/bin/activate
                    echo '========================================'
                    echo 'RERUN Job: $job_name'
                    echo 'Model: SPECTO-ELM ($model)'
                    echo 'Task: $task'
                    echo 'Fix: Regularization 1e-10 -> 1e-8'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$model \
                        --seed=$seed \
                        --csv-output=$csv_file \
                        --verbose
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
echo ""
echo "========================================================================"
echo "RERUN JOBS SUBMITTED"
echo "========================================================================"
echo ""
echo "TASK SUMMARY:"
echo "  RBF-FD Tasks:           ${#rbf_tasks[@]}"
echo "  Spectral 2D Smooth:     ${#spectral_2d_smooth[@]}"
echo "  Spectral 3D:            ${#spectral_3d[@]}"
echo "  Spectral Localized:     ${#spectral_localized[@]}"
echo "  ─────────────────────────────"
echo "  TOTAL TASKS:            ${#all_tasks[@]}"
echo ""
echo "MODEL SUMMARY:"
echo "  Deep SPECTO-ELM only:   ${#deep_models[@]} (deep2, deep3, deep4)"
echo ""
echo "SEEDS: ${seeds[@]}"
echo ""
echo "TOTAL JOBS SUBMITTED: $job_count"
echo ""
echo "EXPECTED BEHAVIOR:"
echo "  - Each job should complete in <5 minutes (was 150-200s before fix)"
echo "  - No SVD fallbacks should occur"
echo "  - Results appended to existing CSV files in ./results/by_task/"
echo ""
echo "LOG FILES:"
echo "  ./logs/rerun/<job_name>-<node>-<jobid>.out"
echo "========================================================================"
