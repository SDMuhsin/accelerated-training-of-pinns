#!/bin/bash

echo "Beginning RoPINN baseline benchmark submissions."
echo "================================================================"
echo "RoPINN: Region-Optimized Physics-Informed Neural Networks"
echo "Paper defaults: initial_region=1e-4, sample_num=1, past_iterations=10"
echo "================================================================"

# ============================================================================
# MODEL DEFINITIONS - RoPINN variants
# ============================================================================

gpu_models=(
    ropinn        # 4 layers x 50 nodes (comparable to vanilla-pinn)
    ropinn-large  # 4 layers x 512 nodes (original RoPINN paper architecture)
)

# ============================================================================
# SPECTRAL TASKS - Same as SPECTO-ELM benchmarks for comparison
# ============================================================================

spectral_tasks=(
    spectral-poisson-square
    spectral-laplace-square
    spectral-nonlinear-poisson-square
)

# Seeds for reproducibility
seeds=(42)

# Training epochs
EPOCHS=1000

# RoPINN hyperparameters (paper defaults)
INITIAL_REGION="1e-4"
SAMPLE_NUM=1
PAST_ITERATIONS=10

# Time allocation (RoPINN is gradient-based, needs more time than ELM)
GPU_TIME="0-04:00:00"  # 4 hours

# Create logs directory
mkdir -p ./logs

# ============================================================================
# SUBMIT RoPINN GPU JOBS
# ============================================================================
echo ""
echo "=============================================="
echo "Submitting RoPINN GPU Jobs"
echo "=============================================="

job_count=0

for model in "${gpu_models[@]}"; do
    for task in "${spectral_tasks[@]}"; do
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
                    echo 'Method: RoPINN (Region Optimization)'
                    echo 'Started: '\$(date)
                    echo '========================================'
                    which python3
                    nvidia-smi
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 -m src.experiment_dt_elm_pinn.train_pinn \
                        --task=$task \
                        --model=$model \
                        --seed=$seed \
                        --epochs=$EPOCHS \
                        --initial-region=$INITIAL_REGION \
                        --sample-num=$SAMPLE_NUM \
                        --past-iterations=$PAST_ITERATIONS \
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
echo "All RoPINN jobs submitted!"
echo "=============================================="
echo ""
echo "SUMMARY:"
echo "  Models: ${#gpu_models[@]}"
echo "    - ropinn (4 layers x 50 nodes)"
echo "    - ropinn-large (4 layers x 512 nodes)"
echo ""
echo "  Spectral Tasks: ${#spectral_tasks[@]}"
echo "    - spectral-poisson-square"
echo "    - spectral-laplace-square"
echo "    - spectral-nonlinear-poisson-square"
echo ""
echo "  Seeds: ${#seeds[@]}"
echo "  Epochs: $EPOCHS"
echo "  RoPINN params: initial_region=$INITIAL_REGION, sample_num=$SAMPLE_NUM, past_iterations=$PAST_ITERATIONS"
echo ""
echo "  TOTAL JOBS: $job_count"
echo ""
echo "Results saved to: ./results/experiments.csv"
echo "=============================================="
