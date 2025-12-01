#!/bin/bash

echo "Beginning DT-PINN experiment sbatch script submissions."

# Define models categorized by resource requirements
# CPU-only models (fast, ELM-based) - no GPU needed
cpu_models=(
    dt-elm-pinn
    dt-elm-pinn-cholesky
    dt-elm-pinn-svd
    pielm
    elm
)

# GPU models (gradient-based, need CUDA)
gpu_models=(
    vanilla-pinn
    dt-pinn
)

# Tasks to run
tasks=(
    nonlinear-poisson
)

# Data files to test
file_names=(
    2_2236
    2_582
)

# Seeds for reproducibility (5 seeds for statistical significance in papers)
seeds=(42) # 123 456 789 1024)

# Time allocations
# CPU models are very fast (<1 minute), but we give buffer
CPU_TIME="0-00:30:00"  # 30 minutes
# GPU models need longer for proper convergence
GPU_TIME="0-01:00:00"  # 6 hours (for 1000 L-BFGS epochs)

# ============================================================================
# PINN training parameters - Paper-quality settings
# ============================================================================
# For L-BFGS: 500-1000 epochs is typically sufficient for convergence
# Papers usually report results at convergence, not at fixed epoch count
EPOCHS=1000

# Network architecture (matching typical PINN paper settings)
LAYERS=4
NODES=50

# Create logs directory
mkdir -p ./logs

# ============================================================================
# Submit CPU jobs (ELM-based models)
# ============================================================================
echo ""
echo "--- Submitting CPU jobs (ELM-based models) ---"

for task in "${tasks[@]}"; do
    for model in "${cpu_models[@]}"; do
        for file_name in "${file_names[@]}"; do
            for seed in "${seeds[@]}"; do
                job_name="${model}_${task}_${file_name}_s${seed}"
                log_file="./logs/${job_name}"

                echo "Submitting CPU job: $job_name"
                sbatch \
                    --nodes=1 \
                    --ntasks-per-node=1 \
                    --cpus-per-task=4 \
                    --mem=8000M \
                    --time=$CPU_TIME \
                    --output=${log_file}-%N-%j.out \
                    --error=${log_file}-%N-%j.err \
                    --wrap="
                        module load scipy-stack
			module load arrow
                        source ./env/bin/activate
                        echo '========================================'
                        echo 'Job: $job_name'
                        echo 'Started: '\$(date)
                        echo '========================================'
                        which python3
                        export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                        python3 -m src.experiment_dt_elm_pinn.train_pinn \
                            --task=$task \
                            --model=$model \
                            --file-name=$file_name \
                            --seed=$seed \
                            --verbose
                        echo '========================================'
                        echo 'Finished: '\$(date)
                        echo '========================================'
                    "
            done
        done
    done
done

# ============================================================================
# Submit GPU jobs (gradient-based PINN models)
# ============================================================================
echo ""
echo "--- Submitting GPU jobs (gradient-based models) ---"

for task in "${tasks[@]}"; do
    for model in "${gpu_models[@]}"; do
        for file_name in "${file_names[@]}"; do
            for seed in "${seeds[@]}"; do
                job_name="${model}_${task}_${file_name}_s${seed}"
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
                            --model=$model \
                            --file-name=$file_name \
                            --seed=$seed \
                            --layers=$LAYERS \
                            --nodes=$NODES \
                            --epochs=$EPOCHS \
                            --verbose
                        echo '========================================'
                        echo 'Finished: '\$(date)
                        echo '========================================'
                    "
            done
        done
    done
done

echo ""
echo "=============================================="
echo "All jobs submitted."
echo ""
echo "Summary:"
echo "  - CPU models: ${#cpu_models[@]} models x ${#tasks[@]} tasks x ${#file_names[@]} files x ${#seeds[@]} seeds"
echo "  - GPU models: ${#gpu_models[@]} models x ${#tasks[@]} tasks x ${#file_names[@]} files x ${#seeds[@]} seeds"
echo "  - GPU epochs: $EPOCHS (L-BFGS)"
echo "  - Network: ${LAYERS} layers x ${NODES} nodes"
echo ""
echo "Results will be saved to:"
echo "  - Logs: ./logs/"
echo "  - CSV: ./results/experiments.csv"
echo "=============================================="
