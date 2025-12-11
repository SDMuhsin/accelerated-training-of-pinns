#!/bin/bash

echo "Beginning DAS baseline benchmark submissions."
echo "================================================================"
echo "DAS: Deep Adaptive Sampling for PDEs"
echo "Comparable settings: 4x50 network, 5 stages × 200 epochs = 1000 total"
echo "(Same total epochs as vanilla-pinn for fair training speed comparison)"
echo "================================================================"

# ============================================================================
# MODEL DEFINITIONS - DAS variants
# ============================================================================

gpu_models=(
    das        # 4 layers x 50 nodes PDE network (matches vanilla-pinn), 6-layer flow
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

# DAS hyperparameters (comparable to vanilla-pinn: 5 stages × 200 = 1000 total epochs)
DAS_MAX_STAGE=5
DAS_PDE_EPOCHS=200
DAS_FLOW_EPOCHS=200
DAS_N_TRAIN=1000
DAS_QUANTITY="residual"

# Time allocation (DAS multi-stage, but comparable total epochs to vanilla-pinn)
GPU_TIME="0-04:00:00"  # 4 hours (same as RoPINN)

# Create logs directory
mkdir -p ./logs

# ============================================================================
# SUBMIT DAS GPU JOBS
# ============================================================================
echo ""
echo "=============================================="
echo "Submitting DAS GPU Jobs"
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
echo "All DAS jobs submitted!"
echo "=============================================="
echo ""
echo "SUMMARY:"
echo "  Models: ${#gpu_models[@]}"
echo "    - das (4 layers x 50 nodes PDE + 6-layer flow, matches vanilla-pinn)"
echo ""
echo "  Spectral Tasks: ${#spectral_tasks[@]}"
echo "    - spectral-poisson-square"
echo "    - spectral-laplace-square"
echo "    - spectral-nonlinear-poisson-square"
echo ""
echo "  Seeds: ${#seeds[@]}"
echo "  DAS params: max_stage=$DAS_MAX_STAGE, pde_epochs=$DAS_PDE_EPOCHS, flow_epochs=$DAS_FLOW_EPOCHS"
echo ""
echo "  TOTAL JOBS: $job_count"
echo ""
echo "Results saved to: ./results/experiments.csv"
echo "=============================================="
