#!/bin/bash

echo "========================================================================"
echo "COMPREHENSIVE BENCHMARK: All Models × Square Domain Tasks"
echo "========================================================================"
echo ""
echo "This script runs all available models on square domain tasks"
echo "(compatible with ALL methods: SPECTO-ELM, DT-PINN, VanillaPINN, PIELM, DAS)"
echo "Results are saved to ./results/experiments.csv"
echo ""

# ============================================================================
# TASK DEFINITIONS - Square domain only (compatible with ALL methods)
# ============================================================================

# Square domain tasks support:
# - SPECTO-ELM (spectral collocation - requires tensor-product domain)
# - DT-PINN (RBF-FD discretization)
# - VanillaPINN (autodiff)
# - PIELM (point collocation)
# - DAS (adaptive sampling)
# - RoPINN (region-optimized)

square_tasks=(
    # Linear Poisson
    poisson-square-constant          # constant source
    poisson-square-sin               # sinusoidal source

    # Laplace
    laplace-square                   # homogeneous Laplace

    # Nonlinear Poisson (exp(u) nonlinearity)
    #nonlinear-poisson-square         # standard nonlinear
    #nonlinear-poisson-square-constant  # constant source variant
    #nonlinear-poisson-square-sin     # sin source variant

    # Heat equation (time-dependent)
    heat-equation                    # standard heat diffusion
    heat-fast-decay                  # fast decay variant

    # Localized features (challenging for spectral)
    #poisson-peaked                   # peaked Gaussian source
    #boundary-layer                   # sharp boundary gradient
    #poisson-corner                   # corner singularity
)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

# SPECTO-ELM variants (CPU, very fast, spectral collocation)
specto_elm_models=(
    dt-elm-pinn          # Single layer [100]
    dt-elm-pinn-deep2    # 2 layers with skip connections
    dt-elm-pinn-deep3    # 3 layers with skip connections
    dt-elm-pinn-deep4    # 4 layers with skip connections (best for nonlinear)
)

# DT-PINN (GPU, RBF-FD discretization + CuPy sparse ops)
dt_pinn_models=(
    dt-pinn              # RBF-FD based PINN
)

# ELM baselines (CPU, fast)
elm_baselines=(
    pielm                # Physics-Informed ELM
    elm                  # Standard ELM baseline
)

# Gradient-based PINN methods (GPU recommended)
pinn_models=(
    vanilla-pinn         # Standard PINN with autodiff
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
DAS_PDE_EPOCHS=1000      # Increased from 500 for complex tasks (sin, peaked, etc.)
DAS_FLOW_EPOCHS=200
DAS_N_TRAIN=1000

# Vanilla PINN epochs
PINN_EPOCHS=2000

# RoPINN settings
ROPINN_EPOCHS=1000

# DT-PINN settings
DT_PINN_EPOCHS=500

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

for task in "${square_tasks[@]}"; do
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
# SECTION 2: DT-PINN JOBS (GPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 2: DT-PINN Jobs (GPU)"
echo "=============================================="

for task in "${square_tasks[@]}"; do
    csv_file="./results/by_task/${task}.csv"

    for model in "${dt_pinn_models[@]}"; do
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
                    echo 'Model: DT-PINN (RBF-FD + GPU)'
                    echo 'Task: $task'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    nvidia-smi
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$model \
                        --seed=$seed \
                        --epochs=$DT_PINN_EPOCHS \
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
# SECTION 3: PIELM/ELM JOBS (CPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 3: PIELM/ELM Jobs (CPU)"
echo "=============================================="

for task in "${square_tasks[@]}"; do
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
# SECTION 4: VANILLA PINN JOBS (GPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 4: Vanilla PINN Jobs (GPU)"
echo "=============================================="

for task in "${square_tasks[@]}"; do
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
# SECTION 5: RoPINN JOBS (GPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 5: RoPINN Jobs (GPU)"
echo "=============================================="

for task in "${square_tasks[@]}"; do
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
# SECTION 6: DAS JOBS (GPU)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 6: DAS Jobs (GPU)"
echo "=============================================="

for task in "${square_tasks[@]}"; do
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
echo "TASKS (${#square_tasks[@]} square domain tasks compatible with ALL methods):"
echo "  Linear Poisson:"
echo "    - poisson-square-constant"
echo "    - poisson-square-sin"
echo "  Laplace:"
echo "    - laplace-square"
echo "  Nonlinear Poisson:"
echo "    - nonlinear-poisson-square"
echo "    - nonlinear-poisson-square-constant"
echo "    - nonlinear-poisson-square-sin"
echo "  Heat Equation:"
echo "    - heat-equation"
echo "    - heat-fast-decay"
echo "  Localized Features:"
echo "    - poisson-peaked"
echo "    - boundary-layer"
echo "    - poisson-corner"
echo ""
echo "MODELS:"
echo "  SPECTO-ELM (CPU):     ${#specto_elm_models[@]} variants (dt-elm-pinn, deep2, deep3, deep4)"
echo "  DT-PINN (GPU):        ${#dt_pinn_models[@]} (RBF-FD discretization + CuPy)"
echo "  ELM baselines (CPU):  ${#elm_baselines[@]} (pielm, elm)"
echo "  Vanilla PINN (GPU):   ${#pinn_models[@]}"
echo "  RoPINN (GPU):         1"
echo "  DAS (GPU):            1 (pde_epochs=$DAS_PDE_EPOCHS)"
echo "  ─────────────────────────────"
n_models=$((${#specto_elm_models[@]} + ${#dt_pinn_models[@]} + ${#elm_baselines[@]} + ${#pinn_models[@]} + 2))
echo "  TOTAL MODELS:         $n_models"
echo ""
echo "SEEDS: ${seeds[@]}"
echo ""
echo "TOTAL JOBS SUBMITTED: $job_count"
echo ""
echo "OUTPUT:"
echo "  Results saved to task-specific CSVs in ./results/by_task/"
echo "  Logs saved to ./logs/"
echo "========================================================================"
