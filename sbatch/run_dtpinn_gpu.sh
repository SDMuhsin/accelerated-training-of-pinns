#!/bin/bash

echo "Beginning DT-PINN GPU benchmark sbatch script submissions."
echo "================================================================"
echo "This script runs DT-PINN on GPU with Adam optimizer (2000 epochs)"
echo "to fix the previous CPU sparse autograd failures on square domains."
echo "================================================================"

# GPU model - DT-PINN only
gpu_model="dt-pinn"

# All 14 tasks to benchmark
tasks=(
    nonlinear-poisson
    poisson-rbf-fd
    poisson-disk-sin
    poisson-disk-quadratic
    poisson-square-constant
    poisson-square-sin
    nonlinear-poisson-rbf-fd
    nonlinear-poisson-disk-sin
    nonlinear-poisson-square-constant
    nonlinear-poisson-square-sin
    laplace-disk
    laplace-square
    heat-equation
    heat-fast-decay
)

# Data file for nonlinear-poisson task (MATLAB data)
file_names=(
    2_2236
)

# Seeds for reproducibility
seeds=(42)

# Time allocation for DT-PINN GPU runs
# Adam with 2000 epochs typically takes 15-30 seconds, but we add buffer
GPU_TIME="0-01:00:00"  # 1 hour buffer

# Network architecture (standard DT-PINN settings)
LAYERS=4
NODES=50
EPOCHS=2000  # Adam needs more epochs

# Create logs directory
mkdir -p ./logs

echo ""
echo "--- Submitting DT-PINN GPU jobs ---"
echo "Model: $gpu_model"
echo "Epochs: $EPOCHS (Adam optimizer auto-selected)"
echo "Network: ${LAYERS} layers x ${NODES} nodes"
echo ""

job_count=0

for task in "${tasks[@]}"; do
    for seed in "${seeds[@]}"; do
        # Only nonlinear-poisson uses file_name; other tasks use Python-generated data
        if [[ "$task" == "nonlinear-poisson" ]]; then
            for file_name in "${file_names[@]}"; do
                job_name="${gpu_model}_${task}_${file_name}_s${seed}_gpu"
                log_file="./logs/${job_name}"
                file_name_arg="--file-name=$file_name"

                echo "Submitting GPU job: $job_name"
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
                        echo 'Started: '\$(date)
                        echo '========================================'
                        which python3
                        nvidia-smi
                        export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                        python3 -m src.experiment_dt_elm_pinn.train_pinn \
                            --task=$task \
                            --model=$gpu_model \
                            $file_name_arg \
                            --seed=$seed \
                            --layers=$LAYERS \
                            --nodes=$NODES \
                            --epochs=$EPOCHS \
                            --verbose
                        echo '========================================'
                        echo 'Finished: '\$(date)
                        echo '========================================'
                    "
                ((job_count++))
            done
        else
            # Tasks without file_name (RBF-FD generated, heat-equation, etc.)
            job_name="${gpu_model}_${task}_s${seed}_gpu"
            log_file="./logs/${job_name}"

            echo "Submitting GPU job: $job_name"
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
                    echo 'Started: '\$(date)
                    echo '========================================'
                    which python3
                    nvidia-smi
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$gpu_model \
                        --seed=$seed \
                        --layers=$LAYERS \
                        --nodes=$NODES \
                        --epochs=$EPOCHS \
                        --verbose
                    echo '========================================'
                    echo 'Finished: '\$(date)
                    echo '========================================'
                "
            ((job_count++))
        fi
    done
done

echo ""
echo "=============================================="
echo "All DT-PINN GPU jobs submitted."
echo ""
echo "Summary:"
echo "  - Tasks: ${#tasks[@]} (14 total)"
echo "  - Model: dt-pinn (GPU + Adam optimizer)"
echo "  - Seeds: ${#seeds[@]}"
echo "  - Total jobs: $job_count"
echo "  - GPU epochs: $EPOCHS (Adam optimizer auto-selected)"
echo "  - Network: ${LAYERS} layers x ${NODES} nodes"
echo ""
echo "Note: DT-PINN now uses Adam optimizer for all runs due to"
echo "      L-BFGS convergence issues with sparse RBF-FD operators."
echo ""
echo "Results will be saved to:"
echo "  - Logs: ./logs/"
echo "  - CSV: ./results/experiments.csv"
echo "=============================================="
