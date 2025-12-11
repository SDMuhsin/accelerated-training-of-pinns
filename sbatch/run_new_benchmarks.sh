#!/bin/bash

echo "Beginning new benchmark task submissions."
echo "================================================================"
echo "Tasks: 3D spectral + localized feature tasks"
echo "Models: dt-elm-pinn, dt-elm-pinn-deep4, vanilla-pinn, das"
echo "================================================================"

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

# SPECTO-ELM variants (dt-elm-pinn = SPECTO-ELM in paper)
elm_models=(
    dt-elm-pinn        # Single-pass SPECTO-ELM (linear tasks only)
    dt-elm-pinn-deep4  # Deep SPECTO-ELM with 4 iterations (all tasks)
)

pinn_models=(
    vanilla-pinn       # Standard PINN baseline
)

das_models=(
    das                # Deep Adaptive Sampling
)

# ============================================================================
# NEW TASKS
# ============================================================================

# 3D Spectral Tasks
tasks_3d=(
    spectral-poisson-cube
    spectral-laplace-cube
    spectral-nonlinear-poisson-cube
)

# Localized Feature Tasks (designed to favor adaptive methods)
tasks_localized=(
    spectral-poisson-peaked
    spectral-boundary-layer
    spectral-poisson-corner
)

# Seeds for reproducibility
seeds=(42 123 456)

# Time allocation
GPU_TIME="0-04:00:00"  # 4 hours

# Create logs directory
mkdir -p ./logs

job_count=0

# ============================================================================
# SUBMIT ELM JOBS (3D + LOCALIZED)
# ============================================================================
echo ""
echo "=============================================="
echo "Submitting ELM Jobs"
echo "=============================================="

for model in "${elm_models[@]}"; do
    # For dt-elm-pinn, skip nonlinear 3D (needs deep variant)
    if [[ "$model" == "dt-elm-pinn" ]]; then
        all_tasks=("${tasks_3d[@]}" "${tasks_localized[@]}")
        # Remove nonlinear from single-pass ELM
        tasks=()
        for task in "${all_tasks[@]}"; do
            if [[ "$task" != *"nonlinear"* ]]; then
                tasks+=("$task")
            fi
        done
    else
        # Deep variants can handle all tasks
        tasks=("${tasks_3d[@]}" "${tasks_localized[@]}")
    fi

    for task in "${tasks[@]}"; do
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
                    echo 'Method: ELM'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    which python3
                    nvidia-smi
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$model \
                        --seed=$seed \
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
# SUBMIT VANILLA PINN JOBS
# ============================================================================
echo ""
echo "=============================================="
echo "Submitting Vanilla PINN Jobs"
echo "=============================================="

PINN_EPOCHS=2000

for model in "${pinn_models[@]}"; do
    all_tasks=("${tasks_3d[@]}" "${tasks_localized[@]}")
    for task in "${all_tasks[@]}"; do
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
                    echo 'Method: Vanilla PINN'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    which python3
                    nvidia-smi
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$model \
                        --seed=$seed \
                        --epochs=$PINN_EPOCHS \
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
# SUBMIT DAS JOBS
# ============================================================================
echo ""
echo "=============================================="
echo "Submitting DAS Jobs"
echo "=============================================="

DAS_MAX_STAGE=5
DAS_PDE_EPOCHS=200
DAS_FLOW_EPOCHS=200
DAS_N_TRAIN=1000
DAS_QUANTITY="residual"

for model in "${das_models[@]}"; do
    all_tasks=("${tasks_3d[@]}" "${tasks_localized[@]}")
    for task in "${all_tasks[@]}"; do
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
                    echo 'Method: DAS (Deep Adaptive Sampling)'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    which python3
                    nvidia-smi
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$model \
                        --seed=$seed \
                        --das-max-stage=$DAS_MAX_STAGE \
                        --das-pde-epochs=$DAS_PDE_EPOCHS \
                        --das-flow-epochs=$DAS_FLOW_EPOCHS \
                        --das-n-train=$DAS_N_TRAIN \
                        --das-quantity=$DAS_QUANTITY \
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
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "SUMMARY:"
echo "  3D Tasks: ${#tasks_3d[@]}"
echo "    - spectral-poisson-cube"
echo "    - spectral-laplace-cube"
echo "    - spectral-nonlinear-poisson-cube"
echo ""
echo "  Localized Tasks: ${#tasks_localized[@]}"
echo "    - spectral-poisson-peaked"
echo "    - spectral-boundary-layer"
echo "    - spectral-poisson-corner"
echo ""
echo "  Models:"
echo "    - dt-elm-pinn (linear tasks only)"
echo "    - dt-elm-pinn-deep4 (all tasks)"
echo "    - vanilla-pinn"
echo "    - das"
echo ""
echo "  Seeds: ${#seeds[@]}"
echo "  TOTAL JOBS: $job_count"
echo ""
echo "Results saved to: ./results/experiments.csv"
echo "=============================================="
