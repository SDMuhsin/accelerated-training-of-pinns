#!/bin/bash

echo "========================================================================"
echo "COMPREHENSIVE BENCHMARK: All Models × All Tasks"
echo "========================================================================"
echo ""
echo "This script runs all available models on all available tasks."
echo "Results are saved to ./results/experiments.csv"
echo ""

# ============================================================================
# TASK DEFINITIONS
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
# MODEL DEFINITIONS
# ============================================================================

# SPECTO-ELM variants (CPU, very fast)
specto_elm_models=(
    dt-elm-pinn          # Single-pass (linear tasks only)
    dt-elm-pinn-deep2    # 2 iterations
    dt-elm-pinn-deep3    # 3 iterations
    dt-elm-pinn-deep4    # 4 iterations (recommended for nonlinear)
)

# Other ELM baselines (CPU)
elm_baselines=(
    pielm                # Physics-Informed ELM
)

# Gradient-based methods (GPU recommended)
pinn_models=(
    vanilla-pinn         # Standard PINN
)

# Advanced PINN methods (GPU required)
advanced_pinn_models=(
    ropinn               # Region-Optimized PINN
    das                  # Deep Adaptive Sampling
)

# ============================================================================
# CONFIGURATION
# ============================================================================

seeds=(42)

# Time allocations
CPU_TIME="0-02:00:00"    # 2 hours for CPU jobs
GPU_TIME="0-06:00:00"    # 6 hours for GPU jobs

# DAS hyperparameters
DAS_MAX_STAGE=5
DAS_PDE_EPOCHS=200
DAS_FLOW_EPOCHS=200
DAS_N_TRAIN=1000

# Vanilla PINN epochs
PINN_EPOCHS=2000

# RoPINN settings
ROPINN_EPOCHS=1000

# Create logs and results directories
mkdir -p ./logs
mkdir -p ./results/by_task

job_count=0

# ============================================================================
# SECTION 1: SPECTO-ELM JOBS (CPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 1: SPECTO-ELM Jobs (CPU)"
echo "=============================================="

for task in "${all_tasks[@]}"; do
    csv_file="./results/by_task/${task}.csv"

    for model in "${specto_elm_models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="${model}_${task}_s${seed}"
            log_file="./logs/${job_name}"

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
                    echo 'Job: $job_name'
                    echo 'Model: SPECTO-ELM ($model)'
                    echo 'Task: $task'
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
# SECTION 2: PIELM JOBS (CPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 2: PIELM Jobs (CPU)"
echo "=============================================="

for task in "${all_tasks[@]}"; do
    csv_file="./results/by_task/${task}.csv"

    for model in "${elm_baselines[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="${model}_${task}_s${seed}"
            log_file="./logs/${job_name}"

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
                    echo 'Job: $job_name'
                    echo 'Model: $model'
                    echo 'Task: $task'
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
# SECTION 3: VANILLA PINN JOBS (GPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 3: Vanilla PINN Jobs (GPU)"
echo "=============================================="

for task in "${all_tasks[@]}"; do
    csv_file="./results/by_task/${task}.csv"

    for model in "${pinn_models[@]}"; do
        for seed in "${seeds[@]}"; do
            job_name="${model}_${task}_s${seed}_gpu"
            log_file="./logs/${job_name}"

            echo "Submitting: $job_name"
            sbatch \
                --nodes=1 \
                --ntasks-per-node=1 \
                --cpus-per-task=2 \
                --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 \
                --mem=16000M \
                --time=$GPU_TIME \
                --output=${log_file}-%N-%j.out \
                --error=${log_file}-%N-%j.err \
                --wrap="
                    module load scipy-stack cuda cudnn
                    module load arrow
                    source ./env/bin/activate
                    echo '========================================'
                    echo 'Job: $job_name'
                    echo 'Model: Vanilla PINN'
                    echo 'Task: $task'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    nvidia-smi
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$model \
                        --seed=$seed \
                        --epochs=$PINN_EPOCHS \
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
# SECTION 4: RoPINN JOBS (GPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 4: RoPINN Jobs (GPU)"
echo "=============================================="

for task in "${all_tasks[@]}"; do
    csv_file="./results/by_task/${task}.csv"

    for seed in "${seeds[@]}"; do
        job_name="ropinn_${task}_s${seed}_gpu"
        log_file="./logs/${job_name}"

        echo "Submitting: $job_name"
        sbatch \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=2 \
            --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 \
            --mem=16000M \
            --time=$GPU_TIME \
            --output=${log_file}-%N-%j.out \
            --error=${log_file}-%N-%j.err \
            --wrap="
                module load scipy-stack cuda cudnn
                module load arrow
                source ./env/bin/activate
                echo '========================================'
                echo 'Job: $job_name'
                echo 'Model: RoPINN (Region-Optimized PINN)'
                echo 'Task: $task'
                echo 'Started: '\$(date)
                echo '========================================'
                nvidia-smi
                export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                python3 -m src.experiment_dt_elm_pinn.train_pinn \
                    --task=$task \
                    --model=ropinn \
                    --seed=$seed \
                    --epochs=$ROPINN_EPOCHS \
                    --csv-output=$csv_file \
                    --verbose
                echo '========================================'
                echo 'Finished: '\$(date)
                echo '========================================'
            "
        ((job_count++))
    done
done

# ============================================================================
# SECTION 5: DAS JOBS (GPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 5: DAS Jobs (GPU)"
echo "=============================================="

for task in "${all_tasks[@]}"; do
    csv_file="./results/by_task/${task}.csv"

    for seed in "${seeds[@]}"; do
        job_name="das_${task}_s${seed}_gpu"
        log_file="./logs/${job_name}"

        echo "Submitting: $job_name"
        sbatch \
            --nodes=1 \
            --ntasks-per-node=1 \
            --cpus-per-task=2 \
            --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 \
            --mem=16000M \
            --time=$GPU_TIME \
            --output=${log_file}-%N-%j.out \
            --error=${log_file}-%N-%j.err \
            --wrap="
                module load scipy-stack cuda cudnn
                module load arrow
                source ./env/bin/activate
                echo '========================================'
                echo 'Job: $job_name'
                echo 'Model: DAS (Deep Adaptive Sampling)'
                echo 'Task: $task'
                echo 'Started: '\$(date)
                echo '========================================'
                nvidia-smi
                export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                python3 -m src.experiment_dt_elm_pinn.train_pinn \
                    --task=$task \
                    --model=das \
                    --seed=$seed \
                    --das-max-stage=$DAS_MAX_STAGE \
                    --das-pde-epochs=$DAS_PDE_EPOCHS \
                    --das-flow-epochs=$DAS_FLOW_EPOCHS \
                    --das-n-train=$DAS_N_TRAIN \
                    --das-quantity=residual \
                    --csv-output=$csv_file \
                    --verbose
                echo '========================================'
                echo 'Finished: '\$(date)
                echo '========================================'
            "
        ((job_count++))
    done
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "========================================================================"
echo "ALL JOBS SUBMITTED"
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
echo "  SPECTO-ELM variants:    ${#specto_elm_models[@]} (dt-elm-pinn, deep2, deep3, deep4)"
echo "  ELM baselines:          ${#elm_baselines[@]} (pielm)"
echo "  Vanilla PINN:           ${#pinn_models[@]}"
echo "  RoPINN:                 1"
echo "  DAS:                    1"
echo "  ─────────────────────────────"
echo "  TOTAL MODELS:           $((${#specto_elm_models[@]} + ${#elm_baselines[@]} + ${#pinn_models[@]} + 2))"
echo ""
echo "SEEDS: ${seeds[@]}"
echo ""
echo "TOTAL JOBS SUBMITTED: $job_count"
echo ""
echo "OUTPUT STRUCTURE:"
echo "  Results saved to task-specific CSVs in ./results/by_task/"
echo "  Each task gets its own CSV file, e.g.:"
echo "    ./results/by_task/spectral-poisson-square.csv"
echo "    ./results/by_task/laplace-disk.csv"
echo "    ./results/by_task/spectral-poisson-cube.csv"
echo "    ..."
echo "========================================================================"
